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

from train import apply_model_family_defaults, make_parser

torch.set_num_threads(1)

from src.lps_tcn.data import build_sequence_loaders
from src.lps_tcn.engine import run_epoch
from src.lps_tcn.models.factory import MODEL_CHOICES, ModelConfig, build_model
from src.lps_tcn.models.frontends import FixedSmoother1d, LPSConvPlus
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

    def test_family_defaults_upgrade_rnn_and_ms_model(self) -> None:
        args = make_parser().parse_args(['--model', 'lps_conv_plus_ms'])
        args = apply_model_family_defaults(args)
        self.assertEqual(args.front_multiscale_kernels, '5,9,17')
        self.assertTrue(args.front_use_se)
        self.assertTrue(args.front_per_channel_gate)

        rnn_args = make_parser().parse_args(['--model', 'lstm', '--rnn-pooling', 'last'])
        rnn_args = apply_model_family_defaults(rnn_args)
        self.assertEqual(rnn_args.rnn_pooling, 'mean')

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

    def test_all_models_forward_shapes(self) -> None:
        x = torch.randn(3, 2, 96)
        for model_name in MODEL_CHOICES:
            cfg = ModelConfig(
                model_name=model_name,
                input_channels=2,
                n_classes=4,
                tcn_channels=(8, 8),
                tcn_kernel_size=5,
                dropout=0.0,
                front_kernel=5,
                front_kernel2=5,
                lstm_hidden_size=16,
                gru_hidden_size=16,
                rnn_proj_channels=4,
                front_multiscale_kernels=(3, 5, 7),
            )
            model = build_model(cfg)
            y = model(x)
            self.assertEqual(tuple(y.shape), (3, 4), msg=f'model {model_name}')

    def test_savgol_dtype_matches_input_dtype(self) -> None:
        module = FixedSmoother1d(channels=1, kernel_size=5, smoother_type='savgol', causal=False)
        x = torch.randn(2, 1, 32, dtype=torch.float32)
        y = module(x)
        self.assertEqual(y.dtype, torch.float32)

    def test_synthetic_dataset_loader_and_standardization(self) -> None:
        bundle = build_sequence_loaders(
            data_root='./data',
            batch_size=16,
            dataset_name='synthetic_multiscale',
            permute=False,
            seed=1234,
            val_ratio=0.1,
            num_workers=0,
        )
        xb, yb = next(iter(bundle.train_loader))
        self.assertEqual(xb.ndim, 3)
        self.assertEqual(xb.shape[1], 3)
        self.assertEqual(bundle.n_classes, 4)
        self.assertEqual(bundle.seq_len, 160)
        self.assertTrue(abs(float(xb.mean())) < 0.5)
        self.assertEqual(yb.ndim, 1)

    def test_multiscale_lps_conv_plus_preserves_shape(self) -> None:
        module = LPSConvPlus(
            channels=3,
            kernel_size1=5,
            kernel_size2=5,
            branch_kernel_sizes=(5, 9, 13),
            use_se=True,
            per_channel_gate=True,
            branch_dropout=0.1,
        )
        x = torch.randn(2, 3, 80)
        y = module(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertEqual(tuple(module.beta.shape), (1, 3, 1))


if __name__ == '__main__':
    unittest.main(verbosity=2)
