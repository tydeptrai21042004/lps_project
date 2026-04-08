from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .utils import AverageMeter


@dataclass
class EpochMetrics:
    loss: float
    acc: float
    skipped_batches: int = 0
    total_batches: int = 0
    mean_grad_norm: float = 0.0
    max_grad_norm: float = 0.0


@dataclass
class ShiftMetrics:
    mean_logit_l2: float
    mean_prediction_consistency: float


@dataclass
class DebugConfig:
    enabled: bool = False
    debug_every: int = 50
    debug_max_batches: int = 10
    print_parameter_stats: bool = False
    print_activation_stats: bool = False


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()


def _first_nonfinite_gradient(model: nn.Module) -> tuple[str | None, torch.Tensor | None]:
    for name, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return name, p.grad
    return None, None


def _first_nonfinite_parameter(model: nn.Module) -> tuple[str | None, torch.Tensor | None]:
    for name, p in model.named_parameters():
        if p is not None and not torch.isfinite(p).all():
            return name, p
    return None, None


def _tensor_stats(t: torch.Tensor) -> str:
    finite = torch.isfinite(t)
    finite_count = int(finite.sum().item())
    total = t.numel()
    if finite_count == 0:
        return f'finite=0/{total}'
    finite_t = t[finite]
    return (
        f'finite={finite_count}/{total} '
        f'min={finite_t.min().item():+.4e} '
        f'max={finite_t.max().item():+.4e} '
        f'mean={finite_t.mean().item():+.4e} '
        f'std={finite_t.std(unbiased=False).item():+.4e}'
    )


def _nonfinite_channel_hint(param_name: str, t: torch.Tensor) -> str:
    if not param_name.endswith('weight_v') or t.ndim < 2:
        return ''
    flat = t.detach().reshape(t.shape[0], -1)
    bad = ~torch.isfinite(flat).all(dim=1)
    bad_idx = torch.nonzero(bad, as_tuple=False).flatten().tolist()
    if not bad_idx:
        return ''
    return f' bad_output_channels={bad_idx}'


def _global_grad_norm(model: nn.Module) -> float:
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        grad = p.grad.detach()
        if not torch.isfinite(grad).all():
            return float('nan')
        norm = grad.norm(2).item()
        total_sq += norm * norm
    return total_sq ** 0.5


def _max_abs_parameter(model: nn.Module) -> tuple[str, float]:
    best_name = '<none>'
    best_value = 0.0
    for name, p in model.named_parameters():
        if p is None:
            continue
        finite = p.detach()[torch.isfinite(p.detach())]
        if finite.numel() == 0:
            continue
        value = finite.abs().max().item()
        if value > best_value:
            best_name = name
            best_value = value
    return best_name, best_value


def _print_batch_debug(
    *,
    batch_idx: int,
    x: torch.Tensor,
    y: torch.Tensor,
    logits: torch.Tensor,
    loss: torch.Tensor,
    grad_norm: float | None,
    model: nn.Module,
    prefix: str,
    debug_cfg: DebugConfig,
) -> None:
    msg = (
        f"{prefix}[debug] batch={batch_idx} "
        f"loss={loss.item():.6f} "
        f"acc={accuracy_from_logits(logits, y):.4f} "
        f"x=({_tensor_stats(x.detach())}) "
        f"logits=({_tensor_stats(logits.detach())})"
    )
    if grad_norm is not None:
        msg += f" grad_norm={grad_norm:.4e}"
    print(msg)

    if debug_cfg.print_parameter_stats:
        name, max_abs = _max_abs_parameter(model)
        print(f"{prefix}[debug] max_abs_param={name} value={max_abs:.4e}")


def run_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    grad_clip: float | None = None,
    skip_nonfinite_batches: bool = True,
    max_consecutive_skips: int | None = 10,
    log_prefix: str = '',
    debug_cfg: DebugConfig | None = None,
) -> EpochMetrics:
    model.train()
    debug_cfg = debug_cfg or DebugConfig(enabled=False)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    skipped_batches = 0
    grad_norm_sum = 0.0
    grad_norm_count = 0
    grad_norm_max = 0.0
    debug_printed = 0
    total_batches = 0
    consecutive_skips = 0

    for batch_idx, (x, y) in enumerate(loader):
        total_batches += 1
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        if not torch.isfinite(logits).all():
            if skip_nonfinite_batches:
                skipped_batches += 1
                consecutive_skips += 1
                print(f"{log_prefix}[warn] skipped batch {batch_idx}: non-finite logits | {_tensor_stats(logits.detach())}")
                if max_consecutive_skips is not None and consecutive_skips >= max_consecutive_skips:
                    raise RuntimeError(
                        f'Too many consecutive skipped batches ({consecutive_skips}) due to non-finite logits'
                    )
                continue
            raise RuntimeError(f'Non-finite logits at batch {batch_idx}')

        loss = criterion(logits, y)
        if not torch.isfinite(loss):
            if skip_nonfinite_batches:
                skipped_batches += 1
                consecutive_skips += 1
                print(f"{log_prefix}[warn] skipped batch {batch_idx}: non-finite loss={loss.item()}")
                if max_consecutive_skips is not None and consecutive_skips >= max_consecutive_skips:
                    raise RuntimeError(
                        f'Too many consecutive skipped batches ({consecutive_skips}) due to non-finite loss'
                    )
                continue
            raise RuntimeError(f'Non-finite loss at batch {batch_idx}')

        loss.backward()

        grad_name, bad_grad = _first_nonfinite_gradient(model)
        if grad_name is not None:
            if skip_nonfinite_batches:
                skipped_batches += 1
                consecutive_skips += 1
                optimizer.zero_grad(set_to_none=True)
                channel_hint = _nonfinite_channel_hint(grad_name, bad_grad.detach())
                print(
                    f"{log_prefix}[warn] skipped batch {batch_idx}: non-finite gradients after backward | "
                    f"param={grad_name} grad_stats=({_tensor_stats(bad_grad.detach())})"
                    f"{channel_hint} "
                    f"loss={loss.item():.6f} logits_stats=({_tensor_stats(logits.detach())})"
                )
                if max_consecutive_skips is not None and consecutive_skips >= max_consecutive_skips:
                    raise RuntimeError(
                        f'Too many consecutive skipped batches ({consecutive_skips}) due to non-finite gradients at {grad_name}'
                    )
                continue
            raise RuntimeError(f'Non-finite gradients at batch {batch_idx}: {grad_name}')

        grad_norm_before_clip = _global_grad_norm(model)
        if torch.isfinite(torch.tensor(grad_norm_before_clip)):
            grad_norm_sum += grad_norm_before_clip
            grad_norm_count += 1
            grad_norm_max = max(grad_norm_max, grad_norm_before_clip)

        should_debug = (
            debug_cfg.enabled
            and debug_printed < debug_cfg.debug_max_batches
            and ((batch_idx + 1) % max(1, debug_cfg.debug_every) == 0 or batch_idx == 0)
        )
        if should_debug:
            _print_batch_debug(
                batch_idx=batch_idx,
                x=x,
                y=y,
                logits=logits,
                loss=loss,
                grad_norm=grad_norm_before_clip,
                model=model,
                prefix=log_prefix,
                debug_cfg=debug_cfg,
            )
            debug_printed += 1

        if grad_clip is not None:
            clipped_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not torch.isfinite(clipped_grad_norm):
                if skip_nonfinite_batches:
                    skipped_batches += 1
                    consecutive_skips += 1
                    optimizer.zero_grad(set_to_none=True)
                    print(f"{log_prefix}[warn] skipped batch {batch_idx}: non-finite grad norm after clipping ({clipped_grad_norm})")
                    if max_consecutive_skips is not None and consecutive_skips >= max_consecutive_skips:
                        raise RuntimeError(
                            f'Too many consecutive skipped batches ({consecutive_skips}) due to non-finite clipped grad norm'
                        )
                    continue
                raise RuntimeError(f'Non-finite grad norm at batch {batch_idx}: {clipped_grad_norm}')

        optimizer.step()

        bad_param_name, bad_param = _first_nonfinite_parameter(model)
        if bad_param_name is not None:
            if skip_nonfinite_batches:
                skipped_batches += 1
                consecutive_skips += 1
                optimizer.zero_grad(set_to_none=True)
                print(
                    f"{log_prefix}[warn] parameter became non-finite after step: {bad_param_name} "
                    f"param_stats=({_tensor_stats(bad_param.detach())})"
                )
                if max_consecutive_skips is not None and consecutive_skips >= max_consecutive_skips:
                    raise RuntimeError(
                        f'Too many consecutive skipped batches ({consecutive_skips}) due to non-finite parameter {bad_param_name}'
                    )
                continue
            raise RuntimeError(f'Non-finite parameter after step: {bad_param_name}')

        consecutive_skips = 0
        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(accuracy_from_logits(logits, y), x.size(0))

    if skipped_batches > 0:
        print(f"{log_prefix}[info] skipped_batches={skipped_batches}/{max(total_batches, 1)}")

    mean_grad_norm = grad_norm_sum / max(grad_norm_count, 1)
    return EpochMetrics(
        loss=loss_meter.avg,
        acc=acc_meter.avg,
        skipped_batches=skipped_batches,
        total_batches=total_batches,
        mean_grad_norm=mean_grad_norm,
        max_grad_norm=grad_norm_max,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device: torch.device) -> EpochMetrics:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    total_batches = 0

    for x, y in loader:
        total_batches += 1
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        if not torch.isfinite(logits).all():
            raise RuntimeError('Non-finite logits during evaluation')

        loss = criterion(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss during evaluation')

        consecutive_skips = 0
        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(accuracy_from_logits(logits, y), x.size(0))

    return EpochMetrics(loss=loss_meter.avg, acc=acc_meter.avg, total_batches=total_batches)


@torch.no_grad()
def evaluate_shift_stability(
    model: nn.Module,
    loader,
    device: torch.device,
    shifts=(1, 2, 4),
    max_batches: int = 20,
) -> ShiftMetrics:
    model.eval()
    l2_sum = 0.0
    consistency_sum = 0.0
    count = 0
    batches_seen = 0

    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        ref_logits = model(x)
        ref_pred = ref_logits.argmax(dim=1)

        for shift in shifts:
            shifted = torch.roll(x, shifts=shift, dims=-1)
            shifted[..., :shift] = 0.0
            logits = model(shifted)
            pred = logits.argmax(dim=1)

            l2_sum += torch.norm(ref_logits - logits, dim=1).mean().item()
            consistency_sum += (pred == ref_pred).float().mean().item()
            count += 1

        batches_seen += 1
        if batches_seen >= max_batches:
            break

    return ShiftMetrics(
        mean_logit_l2=l2_sum / max(count, 1),
        mean_prediction_consistency=consistency_sum / max(count, 1),
    )


class CSVLogger:
    def __init__(self, path: str | Path, overwrite: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite and self.path.exists():
            self.path.unlink()
        self.header_written = self.path.exists() and self.path.stat().st_size > 0

    def log(self, row: dict) -> None:
        with self.path.open('a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self.header_written:
                writer.writeheader()
                self.header_written = True
            writer.writerow(row)
