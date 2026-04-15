from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def save_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description='Collect per-shift robustness metrics from train.py summary.json files.')
    parser.add_argument('--runs-root', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='')
    args = parser.parse_args()

    runs_root = Path(args.runs_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else runs_root
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_paths = sorted(runs_root.rglob('summary.json'))
    if not summary_paths:
        raise FileNotFoundError(f'No summary.json files found under {runs_root}')

    per_run_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for summary_path in summary_paths:
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        display_name = summary.get('display_name') or summary.get('run_tag') or summary.get('model')
        per_shift = summary.get('shift_by_magnitude', {})
        for shift_text, metrics in per_shift.items():
            shift = int(shift_text)
            row = {
                'dataset': summary['dataset'],
                'display_name': display_name,
                'model': summary['model'],
                'seed': summary['seed'],
                'shift': shift,
                'mean_logit_l2': float(metrics['mean_logit_l2']),
                'mean_prediction_consistency': float(metrics['mean_prediction_consistency']),
            }
            per_run_rows.append(row)
            grouped.setdefault((row['dataset'], row['display_name'], shift), []).append(row)

    save_csv(
        output_dir / 'shift_sweep_per_run.csv',
        per_run_rows,
        ['dataset', 'display_name', 'model', 'seed', 'shift', 'mean_logit_l2', 'mean_prediction_consistency'],
    )

    aggregate_rows: list[dict[str, Any]] = []
    for (dataset, display_name, shift), rows in grouped.items():
        aggregate_rows.append({
            'dataset': dataset,
            'display_name': display_name,
            'shift': shift,
            'runs': len(rows),
            'mean_logit_l2': mean([float(row['mean_logit_l2']) for row in rows]),
            'mean_prediction_consistency': mean([float(row['mean_prediction_consistency']) for row in rows]),
        })
    aggregate_rows.sort(key=lambda row: (row['dataset'], row['display_name'], row['shift']))
    save_csv(
        output_dir / 'shift_sweep_aggregate.csv',
        aggregate_rows,
        ['dataset', 'display_name', 'shift', 'runs', 'mean_logit_l2', 'mean_prediction_consistency'],
    )
    (output_dir / 'shift_sweep_aggregate.json').write_text(json.dumps(aggregate_rows, indent=2), encoding='utf-8')
    print(f'Saved shift sweep tables to {output_dir}')


if __name__ == '__main__':
    main()
