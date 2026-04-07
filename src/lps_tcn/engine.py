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


@dataclass
class ShiftMetrics:
    mean_logit_l2: float
    mean_prediction_consistency: float


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    grad_clip: float | None = None,
) -> EpochMetrics:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"Non-finite logits at batch {batch_idx}")

        loss = criterion(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at batch {batch_idx}")

        loss.backward()

        if grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite grad norm at batch {batch_idx}: {grad_norm}")

        optimizer.step()

        for name, p in model.named_parameters():
            if p is not None and not torch.isfinite(p).all():
                raise RuntimeError(f"Non-finite parameter after step: {name}")

        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(accuracy_from_logits(logits, y), x.size(0))

    return EpochMetrics(loss=loss_meter.avg, acc=acc_meter.avg)


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device: torch.device) -> EpochMetrics:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite logits during evaluation")

        loss = criterion(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss during evaluation")

        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(accuracy_from_logits(logits, y), x.size(0))

    return EpochMetrics(loss=loss_meter.avg, acc=acc_meter.avg)


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
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.header_written = self.path.exists() and self.path.stat().st_size > 0

    def log(self, row: dict) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self.header_written:
                writer.writeheader()
                self.header_written = True
            writer.writerow(row)
