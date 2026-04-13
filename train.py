from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset

from src.lps_tcn.analysis import collect_frontend_diagnostics
from src.lps_tcn.data import ARCHIVE_DATASETS, DATASET_CHOICES, build_sequence_loaders
from src.lps_tcn.engine import CSVLogger, DebugConfig, evaluate, evaluate_shift_stability, run_epoch
from src.lps_tcn.models.factory import MODEL_CHOICES, ModelConfig, build_model
from src.lps_tcn.utils import count_parameters, ensure_dir, save_json, set_seed


TCN_MODEL_CHOICES = {
    'tcn_plain',
    'tcn_bn',
    'tcn_attn',
    'tcn_strong',
    'smoothed_tcn',
    'gaussian_tcn',
    'hamming_tcn',
    'savgol_tcn',
    'moving_avg_tcn',
    'learnable_front_tcn',
    'lps_conv',
    'lps_conv_plus',
    'lps_conv_plus_ms',
}


def parse_channels(text: str) -> tuple[int, ...]:
    values = [int(v.strip()) for v in text.split(',') if v.strip()]
    if not values:
        raise ValueError('Expected a comma-separated list of integers')
    return tuple(values)


def apply_model_family_defaults(args: argparse.Namespace) -> argparse.Namespace:
    args = argparse.Namespace(**vars(args))

    if args.model in {'lstm', 'bilstm', 'gru', 'bigru'}:
        if args.rnn_pooling == 'last':
            args.rnn_pooling = 'mean'
        if args.rnn_proj_channels <= 0:
            args.rnn_proj_channels = 32
        if args.dropout < 0.1:
            args.dropout = 0.15

    if args.model == 'fcn':
        if args.fcn_channels == '128,256,128':
            args.fcn_channels = '128,256,256'
        if args.fcn_kernel_sizes == '8,5,3':
            args.fcn_kernel_sizes = '11,7,5'
        if args.dropout < 0.1:
            args.dropout = 0.1

    if args.model == 'lps_conv_plus_ms':
        if args.front_multiscale_kernels == '':
            args.front_multiscale_kernels = '5,9,17'
        args.front_use_se = True
        args.front_per_channel_gate = True
        if args.front_branch_dropout < 0.05:
            args.front_branch_dropout = 0.05

    if args.model in {'tcn_bn', 'tcn_attn', 'tcn_strong'}:
        if args.norm_type == 'none':
            args.norm_type = 'batch'
        if args.head_dropout < 0.05:
            args.head_dropout = 0.1

    if args.model == 'tcn_attn' and args.pooling == 'mean':
        args.pooling = 'attention'
        args.causal = False

    if args.model == 'tcn_strong':
        if args.channels == '32,32,32,32,32,32,32,32':
            args.channels = '64,64,64,64,64'
        if args.tcn_kernel_size == 7:
            args.tcn_kernel_size = 5
        if args.pooling == 'mean':
            args.pooling = 'meanmax'
        args.causal = False
        if args.dropout < 0.1:
            args.dropout = 0.1

    if args.dataset in ARCHIVE_DATASETS and args.model in TCN_MODEL_CHOICES:
        if args.class_weighting == 'auto':
            args.class_weighting = 'balanced'
        if args.model in {'lps_conv_plus', 'lps_conv_plus_ms', 'learnable_front_tcn'} and args.gate_init <= -4.0:
            args.gate_init = -1.5
        if args.kernel_init == 'identity' and args.model in {'lps_conv', 'lps_conv_plus', 'lps_conv_plus_ms'}:
            args.kernel_init = 'gaussian'
        if not args.normalize_kernel_dc and args.model in {'lps_conv', 'lps_conv_plus', 'lps_conv_plus_ms'}:
            args.normalize_kernel_dc = True

    if args.dataset in {'ecg5000', 'synthetic_sines', 'synthetic_shiftmix', 'synthetic_multiscale'} and args.epochs < 30:
        args.epochs = 30

    if args.dataset in {'basicmotions'} and args.batch_size > 32:
        args.batch_size = 32

    return args


def _extract_targets_from_dataset(dataset) -> np.ndarray:
    if isinstance(dataset, Subset):
        parent = _extract_targets_from_dataset(dataset.dataset)
        return parent[np.asarray(dataset.indices, dtype=np.int64)]
    if hasattr(dataset, 'y'):
        return np.asarray(dataset.y, dtype=np.int64)
    if hasattr(dataset, 'tensors') and len(dataset.tensors) >= 2:
        return dataset.tensors[1].detach().cpu().numpy().astype(np.int64)
    raise TypeError(f'Unsupported dataset type for target extraction: {type(dataset)!r}')


def compute_class_weights(
    dataset,
    n_classes: int,
    *,
    mode: str = 'auto',
    imbalance_threshold: float = 1.5,
) -> tuple[torch.Tensor | None, list[int]]:
    targets = _extract_targets_from_dataset(dataset)
    counts = np.bincount(targets, minlength=n_classes).astype(np.int64)
    if mode == 'none':
        return None, counts.tolist()

    positive = counts[counts > 0]
    if positive.size == 0:
        return None, counts.tolist()

    if mode == 'auto':
        imbalance_ratio = float(positive.max() / max(1, positive.min()))
        if imbalance_ratio < imbalance_threshold:
            return None, counts.tolist()
    elif mode != 'balanced':
        raise ValueError(f'Unknown class weighting mode: {mode}')

    weights = np.zeros(n_classes, dtype=np.float32)
    scale = float(len(targets)) / float(len(positive))
    for cls_idx, count in enumerate(counts):
        if count > 0:
            weights[cls_idx] = scale / float(count)
    return torch.tensor(weights, dtype=torch.float32), counts.tolist()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train and compare LPS-TCN with stronger baselines and richer datasets.')
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--output-dir', type=str, default='./outputs/run')
    parser.add_argument('--dataset', type=str, default='seqmnist', choices=DATASET_CHOICES)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--permute', action='store_true')

    parser.add_argument('--model', type=str, default='lps_conv_plus', choices=MODEL_CHOICES)
    parser.add_argument('--channels', type=str, default='32,32,32,32,32,32,32,32')
    parser.add_argument('--tcn-kernel-size', type=int, default=7)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--grad-clip', type=float, default=0.5)
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['none', 'cosine'])
    parser.add_argument('--optimizer-eps', type=float, default=1e-6)
    parser.add_argument('--use-weight-norm', action='store_true', default=False)
    parser.add_argument('--no-weight-norm', dest='use_weight_norm', action='store_false')
    parser.add_argument('--skip-nonfinite-batches', action='store_true', default=True)
    parser.add_argument('--no-skip-nonfinite-batches', dest='skip_nonfinite_batches', action='store_false')
    parser.add_argument('--max-consecutive-skips', type=int, default=10)
    parser.add_argument('--class-weighting', type=str, default='auto', choices=['none', 'balanced', 'auto'])
    parser.add_argument('--imbalance-threshold', type=float, default=1.5)

    parser.add_argument('--front-k', type=int, default=9)
    parser.add_argument('--front-k2', type=int, default=9)
    parser.add_argument('--front-h', type=float, default=1.0)
    parser.add_argument('--causal', action='store_true', default=True)
    parser.add_argument('--non-causal', dest='causal', action='store_false')
    parser.add_argument('--front-residual', action='store_true', default=True)
    parser.add_argument('--no-front-residual', dest='front_residual', action='store_false')
    parser.add_argument('--use-relu', action='store_true', default=True)
    parser.add_argument('--no-relu', dest='use_relu', action='store_false')
    parser.add_argument('--dc-mode', type=str, default='project', choices=['none', 'project'])
    parser.add_argument('--kernel-init', type=str, default='identity', choices=['identity', 'gaussian', 'kaiming'])
    parser.add_argument('--normalize-kernel-dc', action='store_true')
    parser.add_argument('--gate-init', type=float, default=-4.0)
    parser.add_argument('--front-multiscale-kernels', type=str, default='')
    parser.add_argument('--front-use-se', action='store_true')
    parser.add_argument('--front-per-channel-gate', action='store_true')
    parser.add_argument('--front-branch-dropout', type=float, default=0.0)

    parser.add_argument('--lstm-hidden-size', type=int, default=160)
    parser.add_argument('--lstm-layers', type=int, default=2)
    parser.add_argument('--lstm-bidirectional', action='store_true')
    parser.add_argument('--gru-hidden-size', type=int, default=160)
    parser.add_argument('--gru-layers', type=int, default=2)
    parser.add_argument('--gru-bidirectional', action='store_true')
    parser.add_argument('--rnn-pooling', type=str, default='mean', choices=['last', 'mean', 'max', 'attention'])
    parser.add_argument('--rnn-proj-channels', type=int, default=32)
    parser.add_argument('--fcn-channels', type=str, default='128,256,256')
    parser.add_argument('--fcn-kernel-sizes', type=str, default='11,7,5')
    parser.add_argument(
        '--smoothed-tcn-smoother',
        type=str,
        default='moving_avg',
        choices=['moving_avg', 'gaussian', 'hamming', 'savgol'],
    )
    parser.add_argument('--smoothed-tcn-kernel-size', type=int, default=5)
    parser.add_argument('--norm-type', type=str, default='none', choices=['none', 'batch', 'group'])
    parser.add_argument('--pooling', type=str, default='mean', choices=['last', 'mean', 'max', 'meanmax', 'attention'])
    parser.add_argument('--head-dropout', type=float, default=0.0)

    parser.add_argument('--shift-batches', type=int, default=20)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug-every', type=int, default=50)
    parser.add_argument('--debug-max-batches', type=int, default=10)
    parser.add_argument('--debug-parameter-stats', action='store_true')
    parser.add_argument('--debug-activation-stats', action='store_true')
    parser.add_argument('--detect-anomaly', action='store_true')
    return parser


def main() -> None:
    args = apply_model_family_defaults(make_parser().parse_args())
    output_dir = ensure_dir(args.output_dir)
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    data = build_sequence_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        dataset_name=args.dataset,
        permute=args.permute,
        seed=args.seed,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
    )

    front_multiscale_kernels = parse_channels(args.front_multiscale_kernels) if args.front_multiscale_kernels else ()

    model_cfg = ModelConfig(
        model_name=args.model,
        input_channels=data.input_channels,
        n_classes=data.n_classes,
        tcn_channels=parse_channels(args.channels),
        tcn_kernel_size=args.tcn_kernel_size,
        dropout=args.dropout,
        front_kernel=args.front_k,
        front_kernel2=args.front_k2,
        front_h=args.front_h,
        causal=args.causal,
        front_residual=args.front_residual,
        gate_init=args.gate_init,
        use_relu=args.use_relu,
        dc_mode=args.dc_mode,
        normalize_kernel_dc=args.normalize_kernel_dc,
        kernel_init=args.kernel_init,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_layers=args.lstm_layers,
        lstm_bidirectional=args.lstm_bidirectional,
        gru_hidden_size=args.gru_hidden_size,
        gru_layers=args.gru_layers,
        gru_bidirectional=args.gru_bidirectional,
        rnn_pooling=args.rnn_pooling,
        rnn_proj_channels=args.rnn_proj_channels,
        fcn_channels=parse_channels(args.fcn_channels),
        fcn_kernel_sizes=parse_channels(args.fcn_kernel_sizes),
        smoothed_tcn_smoother=args.smoothed_tcn_smoother,
        smoothed_tcn_kernel_size=args.smoothed_tcn_kernel_size,
        use_weight_norm=args.use_weight_norm,
        norm_type=args.norm_type,
        pooling=args.pooling,
        head_dropout=args.head_dropout,
        front_multiscale_kernels=front_multiscale_kernels,
        front_use_se=args.front_use_se,
        front_per_channel_gate=args.front_per_channel_gate,
        front_branch_dropout=args.front_branch_dropout,
    )
    model = build_model(model_cfg).to(device)

    class_weights, train_class_counts = compute_class_weights(
        data.train_loader.dataset,
        data.n_classes,
        mode=args.class_weighting,
        imbalance_threshold=args.imbalance_threshold,
    )
    if class_weights is not None:
        class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=args.optimizer_eps,
    )
    scheduler = None
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    param_count = count_parameters(model)
    csv_logger = CSVLogger(output_dir / 'history.csv', overwrite=True)
    checkpoint_path = output_dir / 'best.pt'

    debug_cfg = DebugConfig(
        enabled=args.debug,
        debug_every=args.debug_every,
        debug_max_batches=args.debug_max_batches,
        print_parameter_stats=args.debug_parameter_stats,
        print_activation_stats=args.debug_activation_stats,
    )

    best_val_acc = -1.0
    best_val_loss = float('inf')
    best_epoch = 0

    print(f'model={args.model} device={device} params={param_count:,}')
    print(
        f'dataset={data.dataset_name} permute={args.permute} '
        f'seq_len={data.seq_len} n_classes={data.n_classes} '
        f'train/val/test={data.train_size}/{data.val_size}/{data.test_size}'
    )
    print(
        f'optimizer=Adam lr={args.lr} weight_decay={args.weight_decay} eps={args.optimizer_eps} '
        f'grad_clip={args.grad_clip} use_weight_norm={args.use_weight_norm} '
        f'norm_type={args.norm_type} pooling={args.pooling} causal={args.causal}'
    )
    print(f'class_weighting={args.class_weighting} train_class_counts={train_class_counts}')
    if class_weights is not None:
        print(f'class_weights={[round(float(v), 6) for v in class_weights.detach().cpu().tolist()]}')
    print(f'output_dir={output_dir}')

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            data.train_loader,
            criterion,
            optimizer,
            device,
            grad_clip=args.grad_clip,
            skip_nonfinite_batches=args.skip_nonfinite_batches,
            max_consecutive_skips=args.max_consecutive_skips,
            debug_cfg=debug_cfg,
        )
        val_metrics = evaluate(model, data.val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        row = {
            'epoch': epoch,
            'lr': optimizer.param_groups[0]['lr'],
            'train_loss': train_metrics.loss,
            'train_acc': train_metrics.acc,
            'val_loss': val_metrics.loss,
            'val_acc': val_metrics.acc,
            'train_skipped_batches': train_metrics.skipped_batches,
            'train_total_batches': train_metrics.total_batches,
            'train_mean_grad_norm': train_metrics.mean_grad_norm,
            'train_max_grad_norm': train_metrics.max_grad_norm,
        }
        csv_logger.log(row)

        improved = val_metrics.acc > best_val_acc or (
            abs(val_metrics.acc - best_val_acc) < 1e-12 and val_metrics.loss < best_val_loss
        )
        if improved:
            best_val_acc = val_metrics.acc
            best_val_loss = val_metrics.loss
            best_epoch = epoch
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'model_config': asdict(model_cfg),
                    'epoch': epoch,
                    'best_val_acc': best_val_acc,
                    'best_val_loss': best_val_loss,
                },
                checkpoint_path,
            )

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.acc:.4f} "
            f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.acc:.4f} "
            f"best_val_acc={best_val_acc:.4f} "
            f"skipped={train_metrics.skipped_batches}/{train_metrics.total_batches} "
            f"mean_grad_norm={train_metrics.mean_grad_norm:.4e} max_grad_norm={train_metrics.max_grad_norm:.4e}"
        )

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    test_metrics = evaluate(model, data.test_loader, criterion, device)
    shift_metrics = evaluate_shift_stability(model, data.test_loader, device, max_batches=args.shift_batches)
    diagnostics = collect_frontend_diagnostics(model)

    summary = {
        'model': args.model,
        'dataset': data.dataset_name,
        'seed': args.seed,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss,
        'test_acc': test_metrics.acc,
        'test_loss': test_metrics.loss,
        'shift_mean_logit_l2': shift_metrics.mean_logit_l2,
        'shift_prediction_consistency': shift_metrics.mean_prediction_consistency,
        'parameter_count': param_count,
        'config': vars(args),
        'data': {
            'dataset_name': data.dataset_name,
            'input_channels': data.input_channels,
            'n_classes': data.n_classes,
            'seq_len': data.seq_len,
            'train_size': data.train_size,
            'val_size': data.val_size,
            'test_size': data.test_size,
            'train_class_counts': train_class_counts,
        },
        'class_weights': class_weights.detach().cpu().tolist() if class_weights is not None else None,
        'frontend_diagnostics': diagnostics,
    }
    save_json(output_dir / 'summary.json', summary)


if __name__ == '__main__':
    main()
