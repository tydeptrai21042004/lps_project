from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from .frontends import FixedSmoother1d


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


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
        use_weight_norm: bool = True,
    ) -> None:
        super().__init__()
        conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv1 = weight_norm(conv1) if use_weight_norm else conv1
        self.conv2 = weight_norm(conv2) if use_weight_norm else conv2
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.drop1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.drop2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in [self.conv1, self.conv2, self.downsample]:
            if module is None:
                continue
            weight = module.weight_v if hasattr(module, "weight_v") else module.weight
            nn.init.normal_(weight, 0.0, 0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class SmoothedTemporalBlock(nn.Module):
    """TCN block with fixed depthwise smoothing before each dilated conv.

    This baseline is inspired by the smoothed dilated convolution idea: instead of only
    smoothing once at the input, it smooths locally around each dilated convolution.
    """

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
        use_weight_norm: bool = True,
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

        conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv1 = weight_norm(conv1) if use_weight_norm else conv1
        self.conv2 = weight_norm(conv2) if use_weight_norm else conv2
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in [self.conv1, self.conv2, self.downsample]:
            if module is None:
                continue
            weight = module.weight_v if hasattr(module, "weight_v") else module.weight
            nn.init.normal_(weight, 0.0, 0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

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
        use_weight_norm: bool = True,
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
        use_weight_norm: bool = True,
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
        if pooling in {"last", "mean"}:
            head_in = tcn_channels[-1]
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        self.head = nn.Linear(head_in, n_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x)
        y = self.tcn(x)
        if self.pooling == "last":
            return y[:, :, -1]
        return y.mean(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))
