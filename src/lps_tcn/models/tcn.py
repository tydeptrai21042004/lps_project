from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.utils.parametrizations import weight_norm as apply_weight_norm
except ImportError:
    from torch.nn.utils import weight_norm as apply_weight_norm

from .frontends import FixedSmoother1d


class TemporalAttentionPooling(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.score = nn.Linear(features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, T] -> [N, T, C]
        xt = x.transpose(1, 2)
        weights = torch.softmax(self.score(torch.tanh(xt)), dim=1)
        return torch.sum(weights * xt, dim=1)


class TemporalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        *,
        causal: bool = True,
        use_weight_norm: bool = False,
    ) -> None:
        super().__init__()
        self.causal = causal
        self.left_padding = dilation * (kernel_size - 1) if causal else 0
        same_padding = dilation * (kernel_size - 1) // 2
        conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0 if causal else same_padding,
            dilation=dilation,
        )
        self._init_conv(conv)
        self.conv = apply_weight_norm(conv, name="weight", dim=0) if use_weight_norm else conv

    @staticmethod
    def _init_conv(conv: nn.Conv1d, std: float = 0.01) -> None:
        nn.init.normal_(conv.weight, mean=0.0, std=std)
        if conv.bias is not None:
            nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.causal and self.left_padding > 0:
            x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


def make_downsample(in_channels: int, out_channels: int) -> nn.Module | None:
    if in_channels == out_channels:
        return None
    conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    TemporalConv1d._init_conv(conv, std=0.01)
    return conv


def make_norm(norm_type: str, channels: int) -> nn.Module:
    norm_type = norm_type.lower()
    if norm_type == "none":
        return nn.Identity()
    if norm_type == "batch":
        return nn.BatchNorm1d(channels)
    if norm_type == "group":
        return nn.GroupNorm(1, channels)
    raise ValueError(f"Unknown norm_type: {norm_type}")


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        dropout: float,
        *,
        causal: bool = True,
        use_weight_norm: bool = False,
        norm_type: str = "none",
    ):
        super().__init__()

        self.conv1 = TemporalConv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            dilation=dilation,
            causal=causal,
            use_weight_norm=use_weight_norm,
        )
        self.norm1 = make_norm(norm_type, n_outputs)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = TemporalConv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            dilation=dilation,
            causal=causal,
            use_weight_norm=use_weight_norm,
        )
        self.norm2 = make_norm(norm_type, n_outputs)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = make_downsample(n_inputs, n_outputs)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class SmoothedTemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        dropout: float,
        *,
        causal: bool = True,
        smoother_kernel_size: int = 5,
        smoother_type: str = "moving_avg",
        use_weight_norm: bool = False,
        norm_type: str = "none",
    ) -> None:
        super().__init__()

        self.smooth1 = FixedSmoother1d(
            channels=n_inputs,
            kernel_size=smoother_kernel_size,
            smoother_type=smoother_type,
            causal=causal,
        )
        self.smooth2 = FixedSmoother1d(
            channels=n_outputs,
            kernel_size=smoother_kernel_size,
            smoother_type=smoother_type,
            causal=causal,
        )

        self.conv1 = TemporalConv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            dilation=dilation,
            causal=causal,
            use_weight_norm=use_weight_norm,
        )
        self.norm1 = make_norm(norm_type, n_outputs)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = TemporalConv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            dilation=dilation,
            causal=causal,
            use_weight_norm=use_weight_norm,
        )
        self.norm2 = make_norm(norm_type, n_outputs)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = make_downsample(n_inputs, n_outputs)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.smooth1(x)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.smooth2(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        channels: list[int],
        kernel_size: int = 7,
        dropout: float = 0.05,
        *,
        causal: bool = True,
        use_weight_norm: bool = False,
        norm_type: str = "none",
        block_cls: type[nn.Module] = TemporalBlock,
        block_kwargs: dict | None = None,
        dilation_schedule: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        layers = []
        block_kwargs = block_kwargs or {}

        if dilation_schedule is None:
            dilation_schedule = [2 ** i for i in range(len(channels))]
        if len(dilation_schedule) != len(channels):
            raise ValueError('dilation_schedule must have the same length as channels')

        for i, (out_channels, dilation_size) in enumerate(zip(channels, dilation_schedule)):
            in_channels = num_inputs if i == 0 else channels[i - 1]
            layers.append(
                block_cls(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=int(dilation_size),
                    dropout=dropout,
                    causal=causal,
                    use_weight_norm=use_weight_norm,
                    norm_type=norm_type,
                    **block_kwargs,
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TCNBackboneClassifier(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_classes: int,
        tcn_channels: list[int],
        tcn_kernel_size: int,
        dropout: float,
        frontend: nn.Module | None = None,
        *,
        causal: bool = True,
        use_weight_norm: bool = False,
        norm_type: str = "none",
        pooling: str = "last",
        head_dropout: float = 0.0,
        smoothed: bool = False,
        smoothed_smoother_type: str = "moving_avg",
        smoothed_kernel_size: int = 5,
        dilation_schedule: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.frontend = frontend if frontend is not None else nn.Identity()
        block_cls = SmoothedTemporalBlock if smoothed else TemporalBlock
        block_kwargs = {}

        if smoothed:
            block_kwargs = {
                "smoother_kernel_size": smoothed_kernel_size,
                "smoother_type": smoothed_smoother_type,
            }

        self.tcn = TemporalConvNet(
            num_inputs=input_channels,
            channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dropout=dropout,
            causal=causal,
            use_weight_norm=use_weight_norm,
            norm_type=norm_type,
            block_cls=block_cls,
            block_kwargs=block_kwargs,
            dilation_schedule=dilation_schedule,
        )

        self.pooling = pooling
        valid_pooling = {"last", "mean", "max", "meanmax", "attention"}
        if pooling not in valid_pooling:
            raise ValueError(f"Unknown pooling: {pooling}")

        head_in = tcn_channels[-1]
        self.attn_pool = None
        if pooling == "attention":
            self.attn_pool = TemporalAttentionPooling(head_in)
        elif pooling == "meanmax":
            head_in = head_in * 2

        self.head_dropout = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        self.head = nn.Linear(head_in, n_classes)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x)
        y = self.tcn(x)
        if self.pooling == "last":
            return y[:, :, -1]
        if self.pooling == "mean":
            return y.mean(dim=-1)
        if self.pooling == "max":
            return y.max(dim=-1).values
        if self.pooling == "meanmax":
            return torch.cat([y.mean(dim=-1), y.max(dim=-1).values], dim=1)
        assert self.attn_pool is not None
        return self.attn_pool(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.head_dropout(self.forward_features(x)))
