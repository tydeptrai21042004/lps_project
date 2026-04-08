from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


DATASET_CHOICES = ['seqmnist', 'fashion_mnist', 'kmnist', 'emnist_digits', 'cifar10_gray']


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a comparison suite across strong baselines and LPS variants.')
    parser.add_argument('--project-root', type=str, default='.')
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--output-dir', type=str, default='./outputs/compare')
    parser.add_argument('--dataset', type=str, default='seqmnist', choices=DATASET_CHOICES)
    parser.add_argument(
        '--models',
        type=str,
        default='tcn_plain,smoothed_tcn,gaussian_tcn,savgol_tcn,lstm,gru,fcn,lps_conv_plus',
        help='Comma-separated model names',
    )
    parser.add_argument('--seeds', type=str, default='1111,2222,3333')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--permute', action='store_true')
    args, unknown = parser.parse_known_args()

    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(',') if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]

    rows = []
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
            subprocess.run(cmd, check=True, cwd=project_root)

            with (run_dir / 'summary.json').open('r', encoding='utf-8') as f:
                summary = json.load(f)
            rows.append(
                {
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
                }
            )

    csv_path = output_dir / 'comparison.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f'Saved comparison table to {csv_path}')


if __name__ == '__main__':
    main()
