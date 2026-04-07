from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from src.lps_tcn.analysis import collect_frontend_diagnostics
from src.lps_tcn.data import build_seqmnist_loaders
from src.lps_tcn.engine import CSVLogger, evaluate, evaluate_shift_stability, run_epoch
from src.lps_tcn.models.factory import ModelConfig, build_model
from src.lps_tcn.utils import count_parameters, ensure_dir, save_json, set_seed


def parse_channels(text: str) -> tuple[int, ...]:
    return tuple(int(v.strip()) for v in text.split(",") if v.strip())


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and compare LPS-TCN with strong baselines.")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs/run")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--permute", action="store_true")

    parser.add_argument(
        "--model",
        type=str,
        default="lps_conv_plus",
        choices=[
            "tcn_plain",
            "smoothed_tcn",
            "lps_conv",
            "lps_conv_plus",
            "gaussian_tcn",
            "hamming_tcn",
            "savgol_tcn",
            "moving_avg_tcn",
            "lstm",
            "bilstm",
            "gru",
            "bigru",
            "fcn",
        ],
    )
    parser.add_argument("--channels", type=str, default="25,25,25,25,25,25,25,25")
    parser.add_argument("--tcn-kernel-size", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine"])

    parser.add_argument("--front-k", type=int, default=9)
    parser.add_argument("--front-h", type=float, default=1.0)
    parser.add_argument("--causal", action="store_true", default=True)
    parser.add_argument("--non-causal", dest="causal", action="store_false")
    parser.add_argument("--front-residual", action="store_true", default=True)
    parser.add_argument("--no-front-residual", dest="front_residual", action="store_false")
    parser.add_argument("--use-relu", action="store_true", default=True)
    parser.add_argument("--no-relu", dest="use_relu", action="store_false")
    parser.add_argument("--dc-mode", type=str, default="project", choices=["none", "project"])
    parser.add_argument("--kernel-init", type=str, default="identity", choices=["identity", "gaussian", "kaiming"])
    parser.add_argument("--normalize-kernel-dc", action="store_true")
    parser.add_argument("--gate-init", type=float, default=-4.0)

    parser.add_argument("--lstm-hidden-size", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--lstm-bidirectional", action="store_true")
    parser.add_argument("--gru-hidden-size", type=int, default=128)
    parser.add_argument("--gru-layers", type=int, default=2)
    parser.add_argument("--gru-bidirectional", action="store_true")
    parser.add_argument("--fcn-channels", type=str, default="128,256,128")
    parser.add_argument("--fcn-kernel-sizes", type=str, default="8,5,3")
    parser.add_argument(
        "--smoothed-tcn-smoother",
        type=str,
        default="moving_avg",
        choices=["moving_avg", "gaussian", "hamming", "savgol"],
    )
    parser.add_argument("--smoothed-tcn-kernel-size", type=int, default=5)

    parser.add_argument("--shift-batches", type=int, default=20)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    output_dir = ensure_dir(args.output_dir)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = build_seqmnist_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        permute=args.permute,
        seed=args.seed,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
    )

    model_cfg = ModelConfig(
        model_name=args.model,
        input_channels=data.input_channels,
        n_classes=data.n_classes,
        tcn_channels=parse_channels(args.channels),
        tcn_kernel_size=args.tcn_kernel_size,
        dropout=args.dropout,
        front_kernel=args.front_k,
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
        fcn_channels=parse_channels(args.fcn_channels),
        fcn_kernel_sizes=parse_channels(args.fcn_kernel_sizes),
        smoothed_tcn_smoother=args.smoothed_tcn_smoother,
        smoothed_tcn_kernel_size=args.smoothed_tcn_kernel_size,
    )
    model = build_model(model_cfg).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    param_count = count_parameters(model)
    csv_logger = CSVLogger(output_dir / "history.csv")
    checkpoint_path = output_dir / "best.pt"

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0

    print(f"model={args.model} device={device} params={param_count:,}")
    print(f"output_dir={output_dir}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            data.train_loader,
            criterion,
            optimizer,
            device,
            grad_clip=args.grad_clip,
        )
        val_metrics = evaluate(model, data.val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics.loss,
            "train_acc": train_metrics.acc,
            "val_loss": val_metrics.loss,
            "val_acc": val_metrics.acc,
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
                    "model_state_dict": model.state_dict(),
                    "model_config": asdict(model_cfg),
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.acc:.4f} "
            f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.acc:.4f} "
            f"best_val_acc={best_val_acc:.4f}"
        )

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, data.test_loader, criterion, device)
    shift_metrics = evaluate_shift_stability(model, data.test_loader, device, max_batches=args.shift_batches)
    diagnostics = collect_frontend_diagnostics(model)

    summary = {
        "model": args.model,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "test_acc": test_metrics.acc,
        "test_loss": test_metrics.loss,
        "shift_mean_logit_l2": shift_metrics.mean_logit_l2,
        "shift_prediction_consistency": shift_metrics.mean_prediction_consistency,
        "parameter_count": param_count,
        "config": vars(args),
        "frontend_diagnostics": diagnostics,
    }
    save_json(output_dir / "summary.json", summary)

    print("\nFinished.")
    print(f"best_epoch={best_epoch}")
    print(f"test_acc={test_metrics.acc:.4f}")
    print(f"test_loss={test_metrics.loss:.4f}")
    print(f"shift_mean_logit_l2={shift_metrics.mean_logit_l2:.4f}")
    print(f"shift_prediction_consistency={shift_metrics.mean_prediction_consistency:.4f}")


if __name__ == "__main__":
    main()
