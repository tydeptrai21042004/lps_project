from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from compare_models import _resolve_datasets
from train import apply_model_family_defaults, make_parser
from src.lps_tcn.data import DATASET_GROUPS, DATASET_CHOICES
from src.lps_tcn.models.factory import ModelConfig, build_model
from src.lps_tcn.models.model_zoo import MODEL_PAPER_SUPPORT, PAPER_BASELINE_MODELS, PROPOSAL_MODELS


class PaperBaselineTests(unittest.TestCase):
    def test_paper_compare_set_contains_no_ablations(self) -> None:
        for model in PAPER_BASELINE_MODELS + PROPOSAL_MODELS:
            self.assertNotEqual(MODEL_PAPER_SUPPORT[model]['kind'], 'ablation')

    def test_hybrid_dilated_tcn_forward_shape(self) -> None:
        model = build_model(
            ModelConfig(
                model_name='hybrid_dilated_tcn',
                input_channels=2,
                n_classes=4,
                tcn_channels=(8, 8, 8),
                tcn_kernel_size=5,
                dropout=0.0,
            )
        )
        y = model(torch.randn(3, 2, 64))
        self.assertEqual(tuple(y.shape), (3, 4))

    def test_blurpool_tcn_forward_shape(self) -> None:
        model = build_model(
            ModelConfig(
                model_name='blurpool_tcn',
                input_channels=2,
                n_classes=4,
                tcn_channels=(8, 8),
                tcn_kernel_size=5,
                dropout=0.0,
                front_kernel=5,
            )
        )
        y = model(torch.randn(3, 2, 64))
        self.assertEqual(tuple(y.shape), (3, 4))

    def test_archive_defaults_strengthen_proposal_backbone(self) -> None:
        args = make_parser().parse_args(['--model', 'lps_conv_plus', '--dataset', 'ecg5000'])
        args = apply_model_family_defaults(args)
        self.assertEqual(args.class_weighting, 'balanced')
        self.assertEqual(args.norm_type, 'none')
        self.assertEqual(args.gate_init, -1.0)

    def test_new_archive_datasets_are_exposed(self) -> None:
        for dataset_name in ['ecg200', 'gunpoint', 'italy_power_demand', 'coffee']:
            self.assertIn(dataset_name, DATASET_CHOICES)

    def test_dataset_group_resolution_deduplicates_and_preserves_order(self) -> None:
        args = type('Args', (), {
            'dataset': 'ecg5000',
            'datasets': 'gunpoint,ecg5000',
            'dataset_set': 'quick_archive',
        })()
        datasets = _resolve_datasets(args)
        expected = list(DATASET_GROUPS['quick_archive'])
        self.assertEqual(datasets[: len(expected)], expected)
        self.assertEqual(datasets.count('ecg5000'), 1)
        self.assertIn('gunpoint', datasets)
