from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torch.nn.utils.parametrizations import weight_norm as apply_weight_norm
except ImportError:
    from torch.nn.utils import weight_norm as apply_weight_norm

from .frontends import FixedSmoother1d


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


def _init_tcn_conv(conv: nn.Conv1d, std: float = 0.01) -> None:
    nn.init.normal_(conv.weight, mean=0.0, std=std)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


def make_conv1d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    use_weight_norm: bool = False,
) -> nn.Module:
    conv = nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    _init_tcn_conv(conv, std=0.01)
    return apply_weight_norm(conv, name="weight", dim=0) if use_weight_norm else conv


def make_downsample(in_channels: int, out_channels: int) -> nn.Module | None:
    if in_channels == out_channels:
        return None
    conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    _init_tcn_conv(conv, std=0.01)
    return conv


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float,
        use_weight_norm: bool = False,
    ):
        super().__init__()

        self.conv1 = make_conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            use_weight_norm=use_weight_norm,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = make_conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            use_weight_norm=use_weight_norm,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.drop1,
            self.conv2, self.chomp2, self.relu2, self.drop2,
        )

        self.downsample = make_downsample(n_inputs, n_outputs)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
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
        padding: int,
        dropout: float,
        smoother_kernel_size: int = 5,
        smoother_type: str = "moving_avg",
        use_weight_norm: bool = False,
    ) -> None:
        super().__init__()

        self.smooth1 = FixedSmoother1d(
            channels=n_inputs,
            kernel_size=smoother_kernel_size,
            smoother_type=smoother_type,
            causal=True,
        )
        self.smooth2 = FixedSmoother1d(
            channels=n_outputs,
            kernel_size=smoother_kernel_size,
            smoother_type=smoother_type,
            causal=True,
        )

        self.conv1 = make_conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            use_weight_norm=use_weight_norm,
        )
        self.conv2 = make_conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            use_weight_norm=use_weight_norm,
        )

        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = make_downsample(n_inputs, n_outputs)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.smooth1(x)
        out = self.conv1(out)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.smooth2(out)
        out = self.conv2(out)
        out = self.chomp2(out)
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
        use_weight_norm: bool = False,
        block_cls: type[nn.Module] = TemporalBlock,
        block_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        layers = []
        block_kwargs = block_kwargs or {}

        for i, out_channels in enumerate(channels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else channels[i - 1]
            layers.append(
                block_cls(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                    use_weight_norm=use_weight_norm,
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
        use_weight_norm: bool = False,
        pooling: str = "last",
        smoothed: bool = False,
        smoothed_smoother_type: str = "moving_avg",
        smoothed_kernel_size: int = 5,
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
            use_weight_norm=use_weight_norm,
            block_cls=block_cls,
            block_kwargs=block_kwargs,
        )

        self.pooling = pooling
        if pooling not in {"last", "mean"}:
            raise ValueError(f"Unknown pooling: {pooling}")

        head_in = tcn_channels[-1]
        self.head = nn.Linear(head_in, n_classes)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x)
        y = self.tcn(x)
        return y[:, :, -1] if self.pooling == "last" else y.mean(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))
