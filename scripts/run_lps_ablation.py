from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lps_tcn.data import DATASET_CHOICES, DATASET_GROUPS
from src.lps_tcn.models.model_zoo import PAPER_BASELINE_MODELS


def parse_int_list(text: str) -> list[int]:
    values = [int(v.strip()) for v in text.split(',') if v.strip()]
    if not values:
        raise ValueError('Expected at least one integer')
    return values


def parse_choice_list(text: str, allowed: set[str]) -> list[str]:
    values = [v.strip() for v in text.split(',') if v.strip()]
    if not values:
        raise ValueError('Expected at least one choice')
    bad = [v for v in values if v not in allowed]
    if bad:
        raise ValueError(f'Unsupported values: {bad}. Allowed: {sorted(allowed)}')
    return values


def resolve_datasets(dataset: str, datasets: str, dataset_set: str) -> list[str]:
    names: list[str] = []
    if dataset_set:
        names.extend(DATASET_GROUPS[dataset_set])
    if datasets:
        names.extend([v.strip() for v in datasets.split(',') if v.strip()])
    if not names:
        names.append(dataset)
    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name not in DATASET_CHOICES:
            raise ValueError(f'Unknown dataset: {name}')
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def build_variants(args: argparse.Namespace) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for kernel in parse_int_list(args.kernel_sizes):
        for residual_mode in parse_choice_list(args.residual_modes, {'direct', 'residual'}):
            for causal_mode in parse_choice_list(args.causal_modes, {'causal', 'noncausal'}):
                tag = f'lps_conv_k{kernel}_{residual_mode}_{causal_mode}'
                extra = ['--front-k', str(kernel)]
                extra += ['--front-residual'] if residual_mode == 'residual' else ['--no-front-residual']
                extra += ['--causal'] if causal_mode == 'causal' else ['--non-causal']
                variants.append({'model': 'lps_conv', 'tag': tag, 'extra': extra})

    if args.include_unconstrained:
        for causal_mode in parse_choice_list(args.causal_modes, {'causal', 'noncausal'}):
            for residual_mode in parse_choice_list(args.residual_modes, {'direct', 'residual'}):
                for kernel in parse_int_list(args.kernel_sizes):
                    tag = f'learnable_front_k{kernel}_{residual_mode}_{causal_mode}'
                    extra = ['--front-k', str(kernel)]
                    extra += ['--front-residual'] if residual_mode == 'residual' else ['--no-front-residual']
                    extra += ['--causal'] if causal_mode == 'causal' else ['--non-causal']
                    variants.append({'model': 'learnable_front_tcn', 'tag': tag, 'extra': extra})

    if args.include_baselines:
        for model in PAPER_BASELINE_MODELS:
            variants.append({'model': model, 'tag': model, 'extra': []})

    deduped: list[dict[str, Any]] = []
    seen_tags: set[str] = set()
    for variant in variants:
        if variant['tag'] not in seen_tags:
            seen_tags.add(variant['tag'])
            deduped.append(variant)
    return deduped


def save_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    finite = [float(v) for v in values]
    return sum(finite) / max(len(finite), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run LPSConv ablations with optional baseline and unconstrained-front comparisons.')
    parser.add_argument('--project-root', type=str, default=str(ROOT))
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--output-dir', type=str, default='./outputs/lps_ablation')
    parser.add_argument('--dataset', type=str, default='ecg5000', choices=DATASET_CHOICES)
    parser.add_argument('--datasets', type=str, default='')
    parser.add_argument('--dataset-set', type=str, default='', choices=[''] + sorted(DATASET_GROUPS.keys()))
    parser.add_argument('--seeds', type=str, default='1111,2222,3333')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--kernel-sizes', type=str, default='3,5,7,9,11')
    parser.add_argument('--residual-modes', type=str, default='residual,direct')
    parser.add_argument('--causal-modes', type=str, default='causal,noncausal')
    parser.add_argument('--shift-values', type=str, default='1,2,4,8,16')
    parser.add_argument('--include-unconstrained', action='store_true', default=True)
    parser.add_argument('--no-include-unconstrained', dest='include_unconstrained', action='store_false')
    parser.add_argument('--include-baselines', action='store_true', default=False)
    parser.add_argument('--continue-on-error', action='store_true', default=True)
    args, unknown = parser.parse_known_args()

    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = resolve_datasets(args.dataset, args.datasets, args.dataset_set)
    seeds = [int(v.strip()) for v in args.seeds.split(',') if v.strip()]
    variants = build_variants(args)

    manifest = {
        'datasets': datasets,
        'seeds': seeds,
        'variants': variants,
        'unknown_args': unknown,
    }
    (output_dir / 'ablation_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    rows: list[dict[str, Any]] = []
    for dataset_name in datasets:
        for variant in variants:
            for seed in seeds:
                run_dir = output_dir / dataset_name / f"{variant['tag']}_seed{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    str(project_root / 'train.py'),
                    '--data-root',
                    args.data_root,
                    '--output-dir',
                    str(run_dir),
                    '--dataset',
                    dataset_name,
                    '--model',
                    variant['model'],
                    '--seed',
                    str(seed),
                    '--epochs',
                    str(args.epochs),
                    '--batch-size',
                    str(args.batch_size),
                    '--shift-values',
                    args.shift_values,
                    '--run-tag',
                    variant['tag'],
                    *variant['extra'],
                    *unknown,
                ]
                print('Running:', ' '.join(cmd))
                status = 'ok'
                error_message = ''
                try:
                    subprocess.run(cmd, cwd=project_root, check=True)
                except subprocess.CalledProcessError as exc:
                    status = 'failed'
                    error_message = f'command exited with code {exc.returncode}'
                    if not args.continue_on_error:
                        raise

                summary_path = run_dir / 'summary.json'
                if status == 'ok' and summary_path.exists():
                    summary = json.loads(summary_path.read_text(encoding='utf-8'))
                    row = {
                        'dataset': dataset_name,
                        'variant': variant['tag'],
                        'base_model': variant['model'],
                        'seed': seed,
                        'best_val_acc': summary['best_val_acc'],
                        'test_acc': summary['test_acc'],
                        'test_loss': summary['test_loss'],
                        'shift_mean_logit_l2': summary['shift_mean_logit_l2'],
                        'shift_prediction_consistency': summary['shift_prediction_consistency'],
                        'parameter_count': summary['parameter_count'],
                        'status': 'ok',
                    }
                else:
                    row = {
                        'dataset': dataset_name,
                        'variant': variant['tag'],
                        'base_model': variant['model'],
                        'seed': seed,
                        'best_val_acc': float('nan'),
                        'test_acc': float('nan'),
                        'test_loss': float('nan'),
                        'shift_mean_logit_l2': float('nan'),
                        'shift_prediction_consistency': float('nan'),
                        'parameter_count': 0,
                        'status': error_message or status,
                    }
                rows.append(row)

    per_run_fields = [
        'dataset', 'variant', 'base_model', 'seed', 'best_val_acc', 'test_acc', 'test_loss',
        'shift_mean_logit_l2', 'shift_prediction_consistency', 'parameter_count', 'status'
    ]
    save_csv(output_dir / 'ablation_per_run.csv', rows, per_run_fields)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row['dataset'], row['variant']), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (dataset_name, variant), group_rows in grouped.items():
        ok = [row for row in group_rows if row['status'] == 'ok']
        summary_rows.append({
            'dataset': dataset_name,
            'variant': variant,
            'base_model': group_rows[0]['base_model'],
            'runs': len(group_rows),
            'successful_runs': len(ok),
            'best_val_acc_mean': mean([row['best_val_acc'] for row in ok]) if ok else float('nan'),
            'test_acc_mean': mean([row['test_acc'] for row in ok]) if ok else float('nan'),
            'test_loss_mean': mean([row['test_loss'] for row in ok]) if ok else float('nan'),
            'shift_mean_logit_l2_mean': mean([row['shift_mean_logit_l2'] for row in ok]) if ok else float('nan'),
            'shift_prediction_consistency_mean': mean([row['shift_prediction_consistency'] for row in ok]) if ok else float('nan'),
            'parameter_count': max((int(row['parameter_count']) for row in ok), default=0),
        })

    summary_rows.sort(key=lambda row: (row['dataset'], -float(row['shift_prediction_consistency_mean'] if row['successful_runs'] else -1.0)))
    summary_fields = [
        'dataset', 'variant', 'base_model', 'runs', 'successful_runs', 'best_val_acc_mean', 'test_acc_mean',
        'test_loss_mean', 'shift_mean_logit_l2_mean', 'shift_prediction_consistency_mean', 'parameter_count'
    ]
    save_csv(output_dir / 'ablation_summary_by_dataset.csv', summary_rows, summary_fields)

    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        by_variant.setdefault(row['variant'], []).append(row)
    macro_rows: list[dict[str, Any]] = []
    for variant, variant_rows in by_variant.items():
        macro_rows.append({
            'variant': variant,
            'base_model': variant_rows[0]['base_model'],
            'datasets': len(variant_rows),
            'runs': sum(int(row['runs']) for row in variant_rows),
            'successful_runs': sum(int(row['successful_runs']) for row in variant_rows),
            'macro_best_val_acc_mean': mean([float(row['best_val_acc_mean']) for row in variant_rows]),
            'macro_test_acc_mean': mean([float(row['test_acc_mean']) for row in variant_rows]),
            'macro_test_loss_mean': mean([float(row['test_loss_mean']) for row in variant_rows]),
            'macro_shift_mean_logit_l2_mean': mean([float(row['shift_mean_logit_l2_mean']) for row in variant_rows]),
            'macro_shift_prediction_consistency_mean': mean([float(row['shift_prediction_consistency_mean']) for row in variant_rows]),
            'parameter_count': max(int(row['parameter_count']) for row in variant_rows),
        })
    macro_rows.sort(key=lambda row: (-float(row['macro_shift_prediction_consistency_mean']), float(row['macro_shift_mean_logit_l2_mean'])))
    save_csv(
        output_dir / 'ablation_macro_summary.csv',
        macro_rows,
        [
            'variant', 'base_model', 'datasets', 'runs', 'successful_runs', 'macro_best_val_acc_mean',
            'macro_test_acc_mean', 'macro_test_loss_mean', 'macro_shift_mean_logit_l2_mean',
            'macro_shift_prediction_consistency_mean', 'parameter_count'
        ],
    )

    (output_dir / 'ablation_summary_by_dataset.json').write_text(json.dumps(summary_rows, indent=2), encoding='utf-8')
    (output_dir / 'ablation_macro_summary.json').write_text(json.dumps(macro_rows, indent=2), encoding='utf-8')
    print(f'Saved ablation runs to {output_dir}')


if __name__ == '__main__':
    main()
