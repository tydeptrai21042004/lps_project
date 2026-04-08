from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset

from train import make_parser

torch.set_num_threads(1)
from src.lps_tcn.engine import run_epoch
from src.lps_tcn.models.factory import ModelConfig, build_model
from src.lps_tcn.utils import set_seed


class NaNGradientIdentity(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return torch.full_like(grad_output, float('nan'))


class BadGradientClassifier(nn.Module):
    """Finite forward pass, intentionally non-finite gradient in backward."""

    def __init__(self, in_features: int = 8, n_classes: int = 2) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, n_classes) * 0.1)
        self.bias = nn.Parameter(torch.zeros(n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1)
        poisoned_weight = NaNGradientIdentity.apply(self.weight)
        return x @ poisoned_weight + self.bias


class TrainingStabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        set_seed(1234)
        self.device = torch.device('cpu')

    def _make_loader(
        self,
        *,
        batches: int = 4,
        batch_size: int = 8,
        channels: int = 1,
        seq_len: int = 64,
        n_classes: int = 3,
    ) -> DataLoader:
        x = torch.randn(batches * batch_size, channels, seq_len)
        y = torch.randint(0, n_classes, (batches * batch_size,))
        return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)

    def test_parser_defaults_use_safe_training_settings(self) -> None:
        args = make_parser().parse_args([])
        self.assertFalse(args.use_weight_norm)
        self.assertAlmostEqual(args.lr, 3e-4)
        self.assertAlmostEqual(args.grad_clip, 0.5)
        self.assertAlmostEqual(args.optimizer_eps, 1e-6)
        self.assertEqual(args.max_consecutive_skips, 10)

    def test_default_tcn_plain_has_no_weight_norm_parameters(self) -> None:
        model = build_model(
            ModelConfig(
                model_name='tcn_plain',
                input_channels=1,
                n_classes=3,
                tcn_channels=(8, 8),
                tcn_kernel_size=5,
                dropout=0.0,
            )
        )
        param_names = [name for name, _ in model.named_parameters()]
        self.assertFalse(any(name.endswith('weight_v') for name in param_names), param_names)

    def test_default_tcn_runs_one_epoch_without_skipped_batches(self) -> None:
        loader = self._make_loader(batches=4, batch_size=8, seq_len=64, n_classes=3)
        model = build_model(
            ModelConfig(
                model_name='tcn_plain',
                input_channels=1,
                n_classes=3,
                tcn_channels=(8, 8),
                tcn_kernel_size=5,
                dropout=0.0,
                use_weight_norm=False,
            )
        ).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-6)

        metrics = run_epoch(
            model,
            loader,
            criterion,
            optimizer,
            self.device,
            grad_clip=0.5,
            skip_nonfinite_batches=True,
            max_consecutive_skips=10,
        )

        self.assertEqual(metrics.skipped_batches, 0)
        self.assertEqual(metrics.total_batches, 4)
        self.assertTrue(torch.isfinite(torch.tensor(metrics.loss)).item())
        self.assertTrue(torch.isfinite(torch.tensor(metrics.acc)).item())
        self.assertTrue(torch.isfinite(torch.tensor(metrics.mean_grad_norm)).item())
        for _, parameter in model.named_parameters():
            self.assertTrue(torch.isfinite(parameter).all().item())

    def test_nonfinite_gradient_streak_raises_fast(self) -> None:
        loader = self._make_loader(batches=4, batch_size=8, channels=1, seq_len=8, n_classes=2)
        model = BadGradientClassifier(in_features=8, n_classes=2).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        with self.assertRaisesRegex(RuntimeError, 'Too many consecutive skipped batches'):
            run_epoch(
                model,
                loader,
                criterion,
                optimizer,
                self.device,
                grad_clip=0.5,
                skip_nonfinite_batches=True,
                max_consecutive_skips=3,
            )


if __name__ == '__main__':
    unittest.main(verbosity=2)
