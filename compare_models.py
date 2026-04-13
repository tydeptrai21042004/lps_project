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

from src.lps_tcn.data import DATASET_CHOICES
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


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row['model']), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for model, model_rows in grouped.items():
        ok_rows = [r for r in model_rows if r.get('status') == 'ok']
        best_val_accs = [_safe_float(r.get('best_val_acc')) for r in ok_rows]
        test_accs = [_safe_float(r.get('test_acc')) for r in ok_rows]
        test_losses = [_safe_float(r.get('test_loss')) for r in ok_rows]
        shift_l2s = [_safe_float(r.get('shift_mean_logit_l2')) for r in ok_rows]
        shift_cons = [_safe_float(r.get('shift_prediction_consistency')) for r in ok_rows]
        param_counts = [int(r.get('parameter_count', 0)) for r in ok_rows if r.get('parameter_count') is not None]

        summary_rows.append(
            {
                'model': model,
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
        )

    summary_rows.sort(key=lambda row: (_safe_float(row['test_acc_mean']), _safe_float(row['best_val_acc_mean'])), reverse=True)
    return summary_rows


def _save_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a comparison suite across paper-supported baselines and LPS variants.')
    parser.add_argument('--project-root', type=str, default='.')
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--output-dir', type=str, default='./outputs/compare')
    parser.add_argument('--dataset', type=str, default='ecg5000', choices=DATASET_CHOICES)
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

    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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
            provenance_rows.append({
                'model': model,
                'kind': meta.get('kind', 'unknown'),
                'paper': meta.get('paper', 'n/a'),
                'note': meta.get('note', ''),
            })
        _print_table('Model provenance', provenance_rows, ['model', 'kind', 'paper', 'note'])
        with (output_dir / 'model_provenance.json').open('w', encoding='utf-8') as f:
            json.dump(provenance_rows, f, indent=2)
    if not args.allow_nonpaper_models:
        bad = [m for m in models if MODEL_PAPER_SUPPORT.get(m, {}).get('kind') == 'ablation']
        if bad and args.model_set != 'ablations':
            raise ValueError(
                'The following models are marked as internal ablations rather than paper-supported baselines: ' + ', '.join(bad)
            )
    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]

    rows: list[dict[str, Any]] = []
    for model in models:
        for seed in seeds:
            run_dir = output_dir / f'{args.dataset}_{model}_seed{seed}'
            cmd = [
                sys.executable,
                str(project_root / 'train.py'),
                '--data-root',
                args.data_root,
                '--output-dir',
                str(run_dir),
                '--dataset',
                args.dataset,
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
                    'dataset': args.dataset,
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

    rows.sort(key=lambda row: (_safe_float(row['test_acc']), _safe_float(row['best_val_acc'])), reverse=True)
    agg_rows = _aggregate_rows(rows)

    csv_path = output_dir / 'comparison.csv'
    summary_csv_path = output_dir / 'comparison_summary.csv'
    _save_csv(csv_path, rows, PER_RUN_COLUMNS)
    _save_csv(summary_csv_path, agg_rows, AGG_COLUMNS)

    _print_table('Final per-run results', rows, PER_RUN_COLUMNS)
    _print_table('Final summary by model', agg_rows, AGG_COLUMNS)

    print(f'\nSaved per-run comparison table to {csv_path}')
    print(f'Saved aggregated summary table to {summary_csv_path}')


if __name__ == '__main__':
    main()
