from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        direction_factor = 2 if bidirectional else 1
        self.norm = nn.LayerNorm(hidden_size * direction_factor)
        self.head = nn.Linear(hidden_size * direction_factor, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)
        y, _ = self.lstm(x)
        y = self.norm(y[:, -1, :])
        return self.head(y)


class GRUClassifier(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=gru_dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        direction_factor = 2 if bidirectional else 1
        self.norm = nn.LayerNorm(hidden_size * direction_factor)
        self.head = nn.Linear(hidden_size * direction_factor, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        y, _ = self.gru(x)
        y = self.norm(y[:, -1, :])
        return self.head(y)


class ConvBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FCNBaseline(nn.Module):
    """1D FCN-style baseline inspired by Wang et al. for time-series classification."""

    def __init__(
        self,
        input_channels: int,
        n_classes: int,
        channels: Iterable[int] = (128, 256, 128),
        kernel_sizes: Iterable[int] = (8, 5, 3),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        channels = tuple(int(v) for v in channels)
        kernel_sizes = tuple(int(v) for v in kernel_sizes)
        if len(channels) != len(kernel_sizes):
            raise ValueError("channels and kernel_sizes must have the same length")

        layers: list[nn.Module] = []
        in_ch = input_channels
        for out_ch, k in zip(channels, kernel_sizes):
            layers.append(ConvBlock1d(in_ch, out_ch, k))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(in_ch, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.features(x)
        y = self.pool(y).squeeze(-1)
        return self.head(y)
