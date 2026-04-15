from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover
    wilcoxon = None


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open('r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def save_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def exact_two_sided_sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return float('nan')
    tail = sum(math.comb(n, k) for k in range(0, min(wins, losses) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run paired significance checks from aggregate_results_by_dataset.csv.')
    parser.add_argument('--input-csv', type=str, required=True)
    parser.add_argument('--baseline', type=str, default='tcn_plain')
    parser.add_argument('--model-column', type=str, default='model')
    parser.add_argument('--metric', type=str, default='test_acc_mean')
    parser.add_argument('--higher-is-better', action='store_true', default=True)
    parser.add_argument('--lower-is-better', dest='higher_is_better', action='store_false')
    parser.add_argument('--targets', type=str, default='')
    parser.add_argument('--output-dir', type=str, default='')
    args = parser.parse_args()

    input_csv = Path(args.input_csv).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(input_csv)
    by_model_dataset: dict[tuple[str, str], dict[str, Any]] = {}
    models: set[str] = set()
    datasets: set[str] = set()
    for row in rows:
        model_name = row[args.model_column]
        by_model_dataset[(model_name, row['dataset'])] = row
        models.add(model_name)
        datasets.add(row['dataset'])

    targets = [t.strip() for t in args.targets.split(',') if t.strip()] if args.targets else sorted(models)
    targets = [model for model in targets if model != args.baseline]

    result_rows: list[dict[str, Any]] = []
    for target in targets:
        paired_diffs: list[float] = []
        wins = 0
        losses = 0
        ties = 0
        used_datasets: list[str] = []
        for dataset in sorted(datasets):
            base_row = by_model_dataset.get((args.baseline, dataset))
            target_row = by_model_dataset.get((target, dataset))
            if base_row is None or target_row is None:
                continue
            base_value = float(base_row[args.metric])
            target_value = float(target_row[args.metric])
            diff = target_value - base_value if args.higher_is_better else base_value - target_value
            paired_diffs.append(diff)
            used_datasets.append(dataset)
            if diff > 1e-12:
                wins += 1
            elif diff < -1e-12:
                losses += 1
            else:
                ties += 1

        wilcoxon_p = float('nan')
        if wilcoxon is not None and any(abs(diff) > 1e-12 for diff in paired_diffs):
            try:
                wilcoxon_p = float(wilcoxon(paired_diffs, alternative='two-sided', zero_method='wilcox').pvalue)
            except Exception:
                wilcoxon_p = float('nan')

        row = {
            'baseline': args.baseline,
            'target': target,
            'metric': args.metric,
            'datasets_compared': len(used_datasets),
            'wins': wins,
            'losses': losses,
            'ties': ties,
            'mean_signed_difference': sum(paired_diffs) / max(len(paired_diffs), 1) if paired_diffs else float('nan'),
            'sign_test_pvalue': exact_two_sided_sign_test(wins, losses),
            'wilcoxon_pvalue': wilcoxon_p,
            'datasets': ','.join(used_datasets),
        }
        result_rows.append(row)

    fieldnames = [
        'baseline', 'target', 'metric', 'datasets_compared', 'wins', 'losses', 'ties',
        'mean_signed_difference', 'sign_test_pvalue', 'wilcoxon_pvalue', 'datasets'
    ]
    save_csv(output_dir / 'significance_results.csv', result_rows, fieldnames)
    (output_dir / 'significance_results.json').write_text(json.dumps(result_rows, indent=2), encoding='utf-8')
    print(f'Saved significance tables to {output_dir}')


if __name__ == '__main__':
    main()
