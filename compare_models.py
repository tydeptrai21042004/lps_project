from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.lps_tcn.data import DATASET_CHOICES, DATASET_GROUPS
from src.lps_tcn.models.model_zoo import ABLATION_MODELS, MODEL_PAPER_SUPPORT, PAPER_BASELINE_MODELS, PROPOSAL_MODELS


PER_RUN_COLUMNS = [
    'dataset',
    'model',
    'seed',
    'best_epoch',
    'best_val_acc',
    'test_acc',
    'test_loss',
    'shift_mean_logit_l2',
    'shift_prediction_consistency',
    'parameter_count',
    'status',
]

AGG_COLUMNS = [
    'model',
    'runs',
    'successful_runs',
    'best_val_acc_mean',
    'best_val_acc_std',
    'test_acc_mean',
    'test_acc_std',
    'test_loss_mean',
    'test_loss_std',
    'shift_mean_logit_l2_mean',
    'shift_prediction_consistency_mean',
    'parameter_count',
]

AGG_DATASET_COLUMNS = [
    'dataset',
    'model',
    'runs',
    'successful_runs',
    'best_val_acc_mean',
    'best_val_acc_std',
    'test_acc_mean',
    'test_acc_std',
    'test_loss_mean',
    'test_loss_std',
    'shift_mean_logit_l2_mean',
    'shift_prediction_consistency_mean',
    'parameter_count',
]

AGG_MACRO_COLUMNS = [
    'model',
    'datasets',
    'runs',
    'successful_runs',
    'macro_best_val_acc_mean',
    'macro_test_acc_mean',
    'macro_test_loss_mean',
    'macro_shift_mean_logit_l2_mean',
    'macro_shift_prediction_consistency_mean',
    'parameter_count_max',
]


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float('nan')
    return result if math.isfinite(result) else float('nan')


def _mean(values: list[float]) -> float:
    finite_values = [v for v in values if math.isfinite(v)]
    if not finite_values:
        return float('nan')
    return statistics.fmean(finite_values)


def _std(values: list[float]) -> float:
    finite_values = [v for v in values if math.isfinite(v)]
    if len(finite_values) < 2:
        return 0.0 if finite_values else float('nan')
    return statistics.stdev(finite_values)


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return 'nan'
        if abs(value) >= 1000:
            return f'{value:.2f}'
        if abs(value) >= 1:
            return f'{value:.4f}'
        return f'{value:.6f}'
    return str(value)


def _print_table(title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    print(f'\n{title}')
    if not rows:
        print('(no rows)')
        return

    widths: dict[str, int] = {}
    for col in columns:
        widths[col] = len(col)
        for row in rows:
            widths[col] = max(widths[col], len(_format_cell(row.get(col, ''))))

    sep = '+-' + '-+-'.join('-' * widths[col] for col in columns) + '-+'
    header = '| ' + ' | '.join(col.ljust(widths[col]) for col in columns) + ' |'

    print(sep)
    print(header)
    print(sep)
    for row in rows:
        print('| ' + ' | '.join(_format_cell(row.get(col, '')).ljust(widths[col]) for col in columns) + ' |')
    print(sep)


def _summarize_group(key_fields: dict[str, Any], model_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [r for r in model_rows if r.get('status') == 'ok']
    best_val_accs = [_safe_float(r.get('best_val_acc')) for r in ok_rows]
    test_accs = [_safe_float(r.get('test_acc')) for r in ok_rows]
    test_losses = [_safe_float(r.get('test_loss')) for r in ok_rows]
    shift_l2s = [_safe_float(r.get('shift_mean_logit_l2')) for r in ok_rows]
    shift_cons = [_safe_float(r.get('shift_prediction_consistency')) for r in ok_rows]
    param_counts = [int(r.get('parameter_count', 0)) for r in ok_rows if r.get('parameter_count') is not None]

    return {
        **key_fields,
        'runs': len(model_rows),
        'successful_runs': len(ok_rows),
        'best_val_acc_mean': _mean(best_val_accs),
        'best_val_acc_std': _std(best_val_accs),
        'test_acc_mean': _mean(test_accs),
        'test_acc_std': _std(test_accs),
        'test_loss_mean': _mean(test_losses),
        'test_loss_std': _std(test_losses),
        'shift_mean_logit_l2_mean': _mean(shift_l2s),
        'shift_prediction_consistency_mean': _mean(shift_cons),
        'parameter_count': max(param_counts) if param_counts else 0,
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row['model']), []).append(row)

    summary_rows = [_summarize_group({'model': model}, model_rows) for model, model_rows in grouped.items()]
    summary_rows.sort(key=lambda row: (_safe_float(row['test_acc_mean']), _safe_float(row['best_val_acc_mean'])), reverse=True)
    return summary_rows


def _aggregate_rows_by_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row['dataset']), str(row['model'])), []).append(row)

    summary_rows = [
        _summarize_group({'dataset': dataset, 'model': model}, model_rows)
        for (dataset, model), model_rows in grouped.items()
    ]
    summary_rows.sort(
        key=lambda row: (str(row['dataset']), -_safe_float(row['test_acc_mean']), -_safe_float(row['best_val_acc_mean']))
    )
    return summary_rows


def _aggregate_macro_across_datasets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_dataset = _aggregate_rows_by_dataset(rows)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in by_dataset:
        grouped.setdefault(str(row['model']), []).append(row)

    macro_rows: list[dict[str, Any]] = []
    for model, model_rows in grouped.items():
        macro_rows.append(
            {
                'model': model,
                'datasets': len({str(r['dataset']) for r in model_rows}),
                'runs': sum(int(r['runs']) for r in model_rows),
                'successful_runs': sum(int(r['successful_runs']) for r in model_rows),
                'macro_best_val_acc_mean': _mean([_safe_float(r['best_val_acc_mean']) for r in model_rows]),
                'macro_test_acc_mean': _mean([_safe_float(r['test_acc_mean']) for r in model_rows]),
                'macro_test_loss_mean': _mean([_safe_float(r['test_loss_mean']) for r in model_rows]),
                'macro_shift_mean_logit_l2_mean': _mean([_safe_float(r['shift_mean_logit_l2_mean']) for r in model_rows]),
                'macro_shift_prediction_consistency_mean': _mean([
                    _safe_float(r['shift_prediction_consistency_mean']) for r in model_rows
                ]),
                'parameter_count_max': max(int(r['parameter_count']) for r in model_rows) if model_rows else 0,
            }
        )

    macro_rows.sort(
        key=lambda row: (_safe_float(row['macro_test_acc_mean']), _safe_float(row['macro_best_val_acc_mean'])),
        reverse=True,
    )
    return macro_rows


def _save_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_datasets(args: argparse.Namespace) -> list[str]:
    dataset_names: list[str] = []
    if args.dataset_set:
        dataset_names.extend(DATASET_GROUPS[args.dataset_set])
    if args.datasets:
        dataset_names.extend([d.strip() for d in args.datasets.split(',') if d.strip()])
    if not dataset_names:
        dataset_names.append(args.dataset)

    deduped: list[str] = []
    seen: set[str] = set()
    for name in dataset_names:
        if name not in seen:
            deduped.append(name)
            seen.add(name)

    unknown = [name for name in deduped if name not in DATASET_CHOICES]
    if unknown:
        raise ValueError('Unknown dataset(s): ' + ', '.join(unknown))
    return deduped


def _print_available_datasets() -> None:
    dataset_rows = [{'dataset': dataset_name} for dataset_name in DATASET_CHOICES]
    _print_table('Available datasets', dataset_rows, ['dataset'])
    group_rows = [
        {'dataset_group': group_name, 'datasets': ', '.join(group_members)}
        for group_name, group_members in sorted(DATASET_GROUPS.items())
    ]
    _print_table('Available dataset groups', group_rows, ['dataset_group', 'datasets'])


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a comparison suite across paper-supported baselines and LPS variants.')
    parser.add_argument('--project-root', type=str, default='.')
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--output-dir', type=str, default='./outputs/compare')
    parser.add_argument('--dataset', type=str, default='ecg5000', choices=DATASET_CHOICES)
    parser.add_argument(
        '--datasets',
        type=str,
        default='',
        help='Comma-separated dataset names. Overrides --dataset and can be combined with --dataset-set.',
    )
    parser.add_argument(
        '--dataset-set',
        type=str,
        default='',
        choices=[''] + sorted(DATASET_GROUPS.keys()),
        help='Predefined dataset group for fairer multi-dataset evaluation.',
    )
    parser.add_argument('--list-datasets', action='store_true', default=False)
    parser.add_argument(
        '--models',
        type=str,
        default='',
        help='Comma-separated model names. Leave empty to use --model-set.',
    )
    parser.add_argument(
        '--model-set',
        type=str,
        default='paper_compare',
        choices=['paper_compare', 'paper_baselines', 'proposals', 'ablations', 'all'],
        help='Predefined model group. paper_compare uses only paper-supported baselines plus proposal models.',
    )
    parser.add_argument('--allow-nonpaper-models', action='store_true', default=False)
    parser.add_argument('--show-model-provenance', action='store_true', default=True)
    parser.add_argument('--hide-model-provenance', dest='show_model_provenance', action='store_false')
    parser.add_argument('--seeds', type=str, default='1111,2222,3333')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--permute', action='store_true')
    parser.add_argument('--continue-on-error', action='store_true', default=True)
    args, unknown = parser.parse_known_args()

    if args.list_datasets:
        _print_available_datasets()
        return

    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_names = _resolve_datasets(args)

    model_sets = {
        'paper_compare': PAPER_BASELINE_MODELS + PROPOSAL_MODELS,
        'paper_baselines': PAPER_BASELINE_MODELS,
        'proposals': PROPOSAL_MODELS,
        'ablations': ABLATION_MODELS,
        'all': PAPER_BASELINE_MODELS + PROPOSAL_MODELS + ABLATION_MODELS,
    }
    models = [m.strip() for m in args.models.split(',') if m.strip()] if args.models else list(model_sets[args.model_set])

    if args.show_model_provenance:
        provenance_rows = []
        for model in models:
            meta = MODEL_PAPER_SUPPORT.get(model, {})
            provenance_rows.append(
                {
                    'model': model,
                    'kind': meta.get('kind', 'unknown'),
                    'paper': meta.get('paper', 'n/a'),
                    'note': meta.get('note', ''),
                }
            )
        _print_table('Model provenance', provenance_rows, ['model', 'kind', 'paper', 'note'])
        with (output_dir / 'model_provenance.json').open('w', encoding='utf-8') as f:
            json.dump(provenance_rows, f, indent=2)

    dataset_manifest = [{'dataset': name, 'dataset_group': args.dataset_set or 'custom'} for name in dataset_names]
    with (output_dir / 'dataset_manifest.json').open('w', encoding='utf-8') as f:
        json.dump(dataset_manifest, f, indent=2)

    if not args.allow_nonpaper_models:
        bad = [m for m in models if MODEL_PAPER_SUPPORT.get(m, {}).get('kind') == 'ablation']
        if bad and args.model_set != 'ablations':
            raise ValueError(
                'The following models are marked as internal ablations rather than paper-supported baselines: '
                + ', '.join(bad)
            )

    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]

    rows: list[dict[str, Any]] = []
    for dataset_name in dataset_names:
        for model in models:
            for seed in seeds:
                run_dir = output_dir / dataset_name / f'{model}_seed{seed}'
                run_dir.parent.mkdir(parents=True, exist_ok=True)
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
                    model,
                    '--seed',
                    str(seed),
                    '--epochs',
                    str(args.epochs),
                    '--batch-size',
                    str(args.batch_size),
                ]
                if args.permute:
                    cmd.append('--permute')
                cmd.extend(unknown)

                print('Running:', ' '.join(cmd))
                status = 'ok'
                error_message = ''
                try:
                    subprocess.run(cmd, check=True, cwd=project_root)
                except subprocess.CalledProcessError as exc:
                    status = 'failed'
                    error_message = f'command exited with code {exc.returncode}'
                    if not args.continue_on_error:
                        raise

                summary_path = run_dir / 'summary.json'
                if status == 'ok' and summary_path.exists():
                    with summary_path.open('r', encoding='utf-8') as f:
                        summary = json.load(f)
                    row = {
                        'dataset': summary['dataset'],
                        'model': summary['model'],
                        'seed': summary['seed'],
                        'best_epoch': summary['best_epoch'],
                        'best_val_acc': summary['best_val_acc'],
                        'test_acc': summary['test_acc'],
                        'test_loss': summary['test_loss'],
                        'shift_mean_logit_l2': summary['shift_mean_logit_l2'],
                        'shift_prediction_consistency': summary['shift_prediction_consistency'],
                        'parameter_count': summary['parameter_count'],
                        'status': status,
                    }
                else:
                    row = {
                        'dataset': dataset_name,
                        'model': model,
                        'seed': seed,
                        'best_epoch': '',
                        'best_val_acc': float('nan'),
                        'test_acc': float('nan'),
                        'test_loss': float('nan'),
                        'shift_mean_logit_l2': float('nan'),
                        'shift_prediction_consistency': float('nan'),
                        'parameter_count': 0,
                        'status': error_message or status,
                    }
                rows.append(row)
                _print_table('Completed runs so far', rows, PER_RUN_COLUMNS)

    if not rows:
        raise RuntimeError('No runs were completed, so there is nothing to summarize.')

    rows.sort(key=lambda row: (str(row['dataset']), -_safe_float(row['test_acc']), -_safe_float(row['best_val_acc'])))
    agg_rows = _aggregate_rows(rows)
    agg_dataset_rows = _aggregate_rows_by_dataset(rows)
    macro_rows = _aggregate_macro_across_datasets(rows)

    _save_csv(output_dir / 'per_run_results.csv', rows, PER_RUN_COLUMNS)
    _save_csv(output_dir / 'aggregate_results.csv', agg_rows, AGG_COLUMNS)
    _save_csv(output_dir / 'aggregate_results_by_dataset.csv', agg_dataset_rows, AGG_DATASET_COLUMNS)
    _save_csv(output_dir / 'macro_results_across_datasets.csv', macro_rows, AGG_MACRO_COLUMNS)
    with (output_dir / 'aggregate_results.json').open('w', encoding='utf-8') as f:
        json.dump(agg_rows, f, indent=2)
    with (output_dir / 'aggregate_results_by_dataset.json').open('w', encoding='utf-8') as f:
        json.dump(agg_dataset_rows, f, indent=2)
    with (output_dir / 'macro_results_across_datasets.json').open('w', encoding='utf-8') as f:
        json.dump(macro_rows, f, indent=2)

    _print_table('Final per-run results', rows, PER_RUN_COLUMNS)
    if len(dataset_names) > 1:
        _print_table('Aggregate results by dataset and model', agg_dataset_rows, AGG_DATASET_COLUMNS)
        _print_table('Macro-average results across datasets', macro_rows, AGG_MACRO_COLUMNS)
    _print_table('Final summary by model', agg_rows, AGG_COLUMNS)

    print(f'\nSaved per-run comparison table to {output_dir / "per_run_results.csv"}')
    print(f'Saved aggregated summary table to {output_dir / "aggregate_results.csv"}')
    print(f'Saved per-dataset summary table to {output_dir / "aggregate_results_by_dataset.csv"}')
    print(f'Saved macro-average summary table to {output_dir / "macro_results_across_datasets.csv"}')


if __name__ == '__main__':
    main()
