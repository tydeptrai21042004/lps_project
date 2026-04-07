from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from .baselines import FCNBaseline, GRUClassifier, LSTMClassifier
from .frontends import FixedSmoother1d, FrontendWithResidual, LPSConv, LPSConvPlus
from .tcn import TCNBackboneClassifier


@dataclass
class ModelConfig:
    model_name: str = "lps_conv_plus"
    input_channels: int = 1
    n_classes: int = 10
    tcn_channels: tuple[int, ...] = (25, 25, 25, 25, 25, 25, 25, 25)
    tcn_kernel_size: int = 7
    dropout: float = 0.05
    front_kernel: int = 9
    front_h: float = 1.0
    causal: bool = True
    front_residual: bool = True
    gate_init: float = -4.0
    use_relu: bool = True
    dc_mode: str = "project"
    normalize_kernel_dc: bool = False
    kernel_init: str = "identity"
    fixed_smoother: str = "gaussian"
    lstm_hidden_size: int = 128
    lstm_layers: int = 2
    lstm_bidirectional: bool = False
    gru_hidden_size: int = 128
    gru_layers: int = 2
    gru_bidirectional: bool = False
    fcn_channels: tuple[int, ...] = (128, 256, 128)
    fcn_kernel_sizes: tuple[int, ...] = (8, 5, 3)
    smoothed_tcn_smoother: str = "moving_avg"
    smoothed_tcn_kernel_size: int = 5
    use_weight_norm: bool = True
    pooling: str = "last"


def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.model_name == "tcn_plain":
        return TCNBackboneClassifier(
            input_channels=cfg.input_channels,
            n_classes=cfg.n_classes,
            tcn_channels=list(cfg.tcn_channels),
            tcn_kernel_size=cfg.tcn_kernel_size,
            dropout=cfg.dropout,
            frontend=None,
            use_weight_norm=cfg.use_weight_norm,
            pooling=cfg.pooling,
        )

    if cfg.model_name == "smoothed_tcn":
        return TCNBackboneClassifier(
            input_channels=cfg.input_channels,
            n_classes=cfg.n_classes,
            tcn_channels=list(cfg.tcn_channels),
            tcn_kernel_size=cfg.tcn_kernel_size,
            dropout=cfg.dropout,
            frontend=None,
            use_weight_norm=cfg.use_weight_norm,
            pooling=cfg.pooling,
            smoothed=True,
            smoothed_smoother_type=cfg.smoothed_tcn_smoother,
            smoothed_kernel_size=cfg.smoothed_tcn_kernel_size,
        )

    if cfg.model_name == "lps_conv":
        frontend = LPSConv(
            channels=cfg.input_channels,
            kernel_size=cfg.front_kernel,
            h=cfg.front_h,
            causal=cfg.causal,
            residual=cfg.front_residual,
            init_mode=cfg.kernel_init,
            normalize_kernel_dc=cfg.normalize_kernel_dc,
        )
        return TCNBackboneClassifier(
            input_channels=cfg.input_channels,
            n_classes=cfg.n_classes,
            tcn_channels=list(cfg.tcn_channels),
            tcn_kernel_size=cfg.tcn_kernel_size,
            dropout=cfg.dropout,
            frontend=frontend,
            use_weight_norm=cfg.use_weight_norm,
            pooling=cfg.pooling,
        )

    if cfg.model_name == "lps_conv_plus":
        frontend = LPSConvPlus(
            channels=cfg.input_channels,
            kernel_size1=cfg.front_kernel,
            kernel_size2=cfg.front_kernel,
            h=cfg.front_h,
            causal=cfg.causal,
            use_relu=cfg.use_relu,
            use_pointwise=True,
            dc_mode=cfg.dc_mode,
            residual=cfg.front_residual,
            gate_init=cfg.gate_init,
            kernel_init=cfg.kernel_init,
            normalize_kernel_dc=cfg.normalize_kernel_dc,
        )
        return TCNBackboneClassifier(
            input_channels=cfg.input_channels,
            n_classes=cfg.n_classes,
            tcn_channels=list(cfg.tcn_channels),
            tcn_kernel_size=cfg.tcn_kernel_size,
            dropout=cfg.dropout,
            frontend=frontend,
            use_weight_norm=cfg.use_weight_norm,
            pooling=cfg.pooling,
        )

    if cfg.model_name in {"gaussian_tcn", "hamming_tcn", "savgol_tcn", "moving_avg_tcn"}:
        smoother_name = cfg.fixed_smoother
        if cfg.model_name == "gaussian_tcn":
            smoother_name = "gaussian"
        elif cfg.model_name == "hamming_tcn":
            smoother_name = "hamming"
        elif cfg.model_name == "savgol_tcn":
            smoother_name = "savgol"
        elif cfg.model_name == "moving_avg_tcn":
            smoother_name = "moving_avg"

        smoother = FixedSmoother1d(
            channels=cfg.input_channels,
            kernel_size=cfg.front_kernel,
            smoother_type=smoother_name,
            causal=cfg.causal,
        )
        frontend = FrontendWithResidual(
            frontend=smoother,
            residual=cfg.front_residual,
            gate=False,
        )
        return TCNBackboneClassifier(
            input_channels=cfg.input_channels,
            n_classes=cfg.n_classes,
            tcn_channels=list(cfg.tcn_channels),
            tcn_kernel_size=cfg.tcn_kernel_size,
            dropout=cfg.dropout,
            frontend=frontend,
            use_weight_norm=cfg.use_weight_norm,
            pooling=cfg.pooling,
        )

    if cfg.model_name == "lstm":
        return LSTMClassifier(
            input_channels=cfg.input_channels,
            n_classes=cfg.n_classes,
            hidden_size=cfg.lstm_hidden_size,
            num_layers=cfg.lstm_layers,
            dropout=cfg.dropout,
            bidirectional=cfg.lstm_bidirectional,
        )

    if cfg.model_name == "bilstm":
        return LSTMClassifier(
            input_channels=cfg.input_channels,
            n_classes=cfg.n_classes,
            hidden_size=cfg.lstm_hidden_size,
            num_layers=cfg.lstm_layers,
            dropout=cfg.dropout,
            bidirectional=True,
        )

    if cfg.model_name == "gru":
        return GRUClassifier(
            input_channels=cfg.input_channels,
            n_classes=cfg.n_classes,
            hidden_size=cfg.gru_hidden_size,
            num_layers=cfg.gru_layers,
            dropout=cfg.dropout,
            bidirectional=cfg.gru_bidirectional,
        )

    if cfg.model_name == "bigru":
        return GRUClassifier(
            input_channels=cfg.input_channels,
            n_classes=cfg.n_classes,
            hidden_size=cfg.gru_hidden_size,
            num_layers=cfg.gru_layers,
            dropout=cfg.dropout,
            bidirectional=True,
        )

    if cfg.model_name == "fcn":
        return FCNBaseline(
            input_channels=cfg.input_channels,
            n_classes=cfg.n_classes,
            channels=cfg.fcn_channels,
            kernel_sizes=cfg.fcn_kernel_sizes,
            dropout=cfg.dropout,
        )

    raise ValueError(f"Unknown model_name: {cfg.model_name}")
