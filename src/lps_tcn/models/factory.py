from __future__ import annotations

from dataclasses import dataclass, replace

import torch.nn as nn

from .baselines import FCNBaseline, GRUClassifier, LSTMClassifier
from .frontends import (
    FixedSmoother1d,
    FrontendWithResidual,
    LPSConv,
    LPSConvPlus,
    LearnableDepthwiseConv1d,
)
from .tcn import TCNBackboneClassifier


MODEL_CHOICES = (
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
    'lstm',
    'bilstm',
    'gru',
    'bigru',
    'fcn',
)


@dataclass
class ModelConfig:
    model_name: str = 'lps_conv_plus'
    input_channels: int = 1
    n_classes: int = 10
    tcn_channels: tuple[int, ...] = (32, 32, 32, 32, 32, 32, 32, 32)
    tcn_kernel_size: int = 7
    dropout: float = 0.1
    front_kernel: int = 9
    front_kernel2: int = 9
    front_h: float = 1.0
    causal: bool = True
    front_residual: bool = True
    gate_init: float = -4.0
    use_relu: bool = True
    dc_mode: str = 'project'
    normalize_kernel_dc: bool = False
    kernel_init: str = 'identity'
    fixed_smoother: str = 'gaussian'
    lstm_hidden_size: int = 160
    lstm_layers: int = 2
    lstm_bidirectional: bool = False
    gru_hidden_size: int = 160
    gru_layers: int = 2
    gru_bidirectional: bool = False
    rnn_pooling: str = 'mean'
    rnn_proj_channels: int = 32
    fcn_channels: tuple[int, ...] = (128, 256, 256)
    fcn_kernel_sizes: tuple[int, ...] = (11, 7, 5)
    smoothed_tcn_smoother: str = 'moving_avg'
    smoothed_tcn_kernel_size: int = 5
    use_weight_norm: bool = False
    norm_type: str = 'none'
    pooling: str = 'mean'
    head_dropout: float = 0.0
    front_multiscale_kernels: tuple[int, ...] = ()
    front_use_se: bool = False
    front_per_channel_gate: bool = False
    front_branch_dropout: float = 0.0


def _build_tcn(
    cfg: ModelConfig,
    *,
    frontend: nn.Module | None = None,
    smoothed: bool = False,
    smoothed_smoother_type: str = 'moving_avg',
    smoothed_kernel_size: int = 5,
) -> nn.Module:
    return TCNBackboneClassifier(
        input_channels=cfg.input_channels,
        n_classes=cfg.n_classes,
        tcn_channels=list(cfg.tcn_channels),
        tcn_kernel_size=cfg.tcn_kernel_size,
        dropout=cfg.dropout,
        frontend=frontend,
        causal=cfg.causal,
        use_weight_norm=cfg.use_weight_norm,
        norm_type=cfg.norm_type,
        pooling=cfg.pooling,
        head_dropout=cfg.head_dropout,
        smoothed=smoothed,
        smoothed_smoother_type=smoothed_smoother_type,
        smoothed_kernel_size=smoothed_kernel_size,
    )


def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.model_name == 'tcn_plain':
        return _build_tcn(cfg)

    if cfg.model_name == 'tcn_bn':
        return _build_tcn(replace(cfg, norm_type='batch'))

    if cfg.model_name == 'tcn_attn':
        return _build_tcn(replace(cfg, causal=False, norm_type='batch', pooling='attention', head_dropout=max(cfg.head_dropout, 0.1)))

    if cfg.model_name == 'tcn_strong':
        tuned_channels = cfg.tcn_channels
        if tuned_channels == (32, 32, 32, 32, 32, 32, 32, 32):
            tuned_channels = (64, 64, 64, 64, 64)
        tuned_kernel = cfg.tcn_kernel_size if cfg.tcn_kernel_size != 7 else 5
        tuned_dropout = max(cfg.dropout, 0.1)
        return _build_tcn(
            replace(
                cfg,
                causal=False,
                norm_type='batch',
                pooling='meanmax',
                head_dropout=max(cfg.head_dropout, 0.1),
                tcn_channels=tuned_channels,
                tcn_kernel_size=tuned_kernel,
                dropout=tuned_dropout,
            )
        )

    if cfg.model_name == 'smoothed_tcn':
        return _build_tcn(
            cfg,
            smoothed=True,
            smoothed_smoother_type=cfg.smoothed_tcn_smoother,
            smoothed_kernel_size=cfg.smoothed_tcn_kernel_size,
        )

    if cfg.model_name == 'lps_conv':
        frontend = LPSConv(
            channels=cfg.input_channels,
            kernel_size=cfg.front_kernel,
            h=cfg.front_h,
            causal=cfg.causal,
            residual=cfg.front_residual,
            init_mode=cfg.kernel_init,
            normalize_kernel_dc=cfg.normalize_kernel_dc,
        )
        return _build_tcn(cfg, frontend=frontend)

    if cfg.model_name in {'lps_conv_plus', 'lps_conv_plus_ms'}:
        branch_kernels: tuple[int, ...] = cfg.front_multiscale_kernels if cfg.model_name == 'lps_conv_plus_ms' else ()
        frontend = LPSConvPlus(
            channels=cfg.input_channels,
            kernel_size1=cfg.front_kernel,
            kernel_size2=cfg.front_kernel2,
            h=cfg.front_h,
            causal=cfg.causal,
            use_relu=cfg.use_relu,
            use_pointwise=True,
            dc_mode=cfg.dc_mode,
            residual=cfg.front_residual,
            gate_init=cfg.gate_init,
            kernel_init=cfg.kernel_init,
            normalize_kernel_dc=cfg.normalize_kernel_dc,
            branch_kernel_sizes=branch_kernels,
            use_se=cfg.front_use_se or cfg.model_name == 'lps_conv_plus_ms',
            per_channel_gate=cfg.front_per_channel_gate or cfg.model_name == 'lps_conv_plus_ms',
            branch_dropout=cfg.front_branch_dropout,
        )
        return _build_tcn(cfg, frontend=frontend)

    if cfg.model_name == 'learnable_front_tcn':
        frontend = FrontendWithResidual(
            frontend=LearnableDepthwiseConv1d(
                channels=cfg.input_channels,
                kernel_size=cfg.front_kernel,
                causal=cfg.causal,
                init_mode=cfg.kernel_init,
            ),
            residual=cfg.front_residual,
            gate=True,
            gate_init=cfg.gate_init,
        )
        return _build_tcn(cfg, frontend=frontend)

    if cfg.model_name in {'gaussian_tcn', 'hamming_tcn', 'savgol_tcn', 'moving_avg_tcn'}:
        smoother_name = cfg.fixed_smoother
        if cfg.model_name == 'gaussian_tcn':
            smoother_name = 'gaussian'
        elif cfg.model_name == 'hamming_tcn':
            smoother_name = 'hamming'
        elif cfg.model_name == 'savgol_tcn':
            smoother_name = 'savgol'
        elif cfg.model_name == 'moving_avg_tcn':
            smoother_name = 'moving_avg'

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
        return _build_tcn(cfg, frontend=frontend)

    if cfg.model_name == 'lstm':
        return LSTMClassifier(
            input_channels=cfg.input_channels,
            n_classes=cfg.n_classes,
            hidden_size=cfg.lstm_hidden_size,
            num_layers=cfg.lstm_layers,
            dropout=cfg.dropout,
            bidirectional=cfg.lstm_bidirectional,
            pooling=cfg.rnn_pooling,
            proj_channels=cfg.rnn_proj_channels,
        )

    if cfg.model_name == 'bilstm':
        return LSTMClassifier(
            input_channels=cfg.input_channels,
            n_classes=cfg.n_classes,
            hidden_size=cfg.lstm_hidden_size,
            num_layers=cfg.lstm_layers,
            dropout=cfg.dropout,
            bidirectional=True,
            pooling=cfg.rnn_pooling,
            proj_channels=cfg.rnn_proj_channels,
        )

    if cfg.model_name == 'gru':
        return GRUClassifier(
            input_channels=cfg.input_channels,
            n_classes=cfg.n_classes,
            hidden_size=cfg.gru_hidden_size,
            num_layers=cfg.gru_layers,
            dropout=cfg.dropout,
            bidirectional=cfg.gru_bidirectional,
            pooling=cfg.rnn_pooling,
            proj_channels=cfg.rnn_proj_channels,
        )

    if cfg.model_name == 'bigru':
        return GRUClassifier(
            input_channels=cfg.input_channels,
            n_classes=cfg.n_classes,
            hidden_size=cfg.gru_hidden_size,
            num_layers=cfg.gru_layers,
            dropout=cfg.dropout,
            bidirectional=True,
            pooling=cfg.rnn_pooling,
            proj_channels=cfg.rnn_proj_channels,
        )

    if cfg.model_name == 'fcn':
        return FCNBaseline(
            input_channels=cfg.input_channels,
            n_classes=cfg.n_classes,
            channels=cfg.fcn_channels,
            kernel_sizes=cfg.fcn_kernel_sizes,
            dropout=cfg.dropout,
        )

    raise ValueError(f'Unknown model_name: {cfg.model_name}')
