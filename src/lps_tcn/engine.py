from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

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


@dataclass
class TrainResult:
    best_epoch: int
    best_val_acc: float
    best_val_loss: float
    test_loss: float
    test_acc: float
    shift_mean_logit_l2: float
    shift_prediction_consistency: float
    parameter_count: int


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

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
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
        loss = criterion(logits, y)
        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(accuracy_from_logits(logits, y), x.size(0))
    return EpochMetrics(loss=loss_meter.avg, acc=acc_meter.avg)


@torch.no_grad()
def evaluate_shift_stability(
    model: nn.Module,
    loader,
    device: torch.device,
    shifts: Sequence[int] = (1, 2, 4),
    max_batches: int = 20,
) -> ShiftMetrics:
    model.eval()
    l2_meter = AverageMeter()
    consistency_meter = AverageMeter()
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
            l2 = torch.norm(ref_logits - logits, dim=1).mean().item()
            consistency = (pred == ref_pred).float().mean().item()
            l2_meter.update(l2, x.size(0))
            consistency_meter.update(consistency, x.size(0))
        batches_seen += 1
        if batches_seen >= max_batches:
            break

    return ShiftMetrics(
        mean_logit_l2=l2_meter.avg,
        mean_prediction_consistency=consistency_meter.avg,
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
