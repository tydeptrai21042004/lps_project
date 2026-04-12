from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class AttentionPooling1d(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.score = nn.Linear(features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(torch.tanh(x)), dim=1)
        return torch.sum(weights * x, dim=1)


class SequencePooling(nn.Module):
    def __init__(self, features: int, mode: str = 'mean') -> None:
        super().__init__()
        self.mode = mode
        self.attn = AttentionPooling1d(features) if mode == 'attention' else None
        valid = {'last', 'mean', 'max', 'attention'}
        if mode not in valid:
            raise ValueError(f'Unknown pooling mode {mode!r}. Expected one of {sorted(valid)}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'last':
            return x[:, -1, :]
        if self.mode == 'mean':
            return x.mean(dim=1)
        if self.mode == 'max':
            return x.max(dim=1).values
        assert self.attn is not None
        return self.attn(x)


class RecurrentClassifierBase(nn.Module):
    rnn_cls: type[nn.Module]

    def __init__(
        self,
        input_channels: int,
        n_classes: int,
        hidden_size: int = 160,
        num_layers: int = 2,
        dropout: float = 0.15,
        bidirectional: bool = False,
        pooling: str = 'mean',
        head_dropout: float = 0.1,
        proj_channels: int = 0,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.input_proj = (
            nn.Sequential(
                nn.Conv1d(input_channels, proj_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(proj_channels),
                nn.ReLU(inplace=True),
            )
            if proj_channels > 0 and proj_channels != input_channels
            else nn.Identity()
        )
        rnn_input_channels = proj_channels if proj_channels > 0 else input_channels
        self.rnn = self.rnn_cls(
            input_size=rnn_input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        direction_factor = 2 if bidirectional else 1
        features = hidden_size * direction_factor
        self.pool = SequencePooling(features, mode=pooling)
        self.norm = nn.LayerNorm(features)
        self.drop = nn.Dropout(head_dropout)
        self.head = nn.Linear(features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        y, _ = self.rnn(x)
        y = self.pool(y)
        y = self.norm(y)
        y = self.drop(y)
        return self.head(y)


class LSTMClassifier(RecurrentClassifierBase):
    rnn_cls = nn.LSTM


class GRUClassifier(RecurrentClassifierBase):
    rnn_cls = nn.GRU


class ConvBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class FCNResidualBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock1d(in_channels, out_channels, kernel_size),
            ConvBlock1d(out_channels, out_channels, kernel_size),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        y = self.dropout(y)
        return self.out_act(y + self.shortcut(x))


class FCNBaseline(nn.Module):
    """Stronger 1D FCN-style baseline for time-series classification."""

    def __init__(
        self,
        input_channels: int,
        n_classes: int,
        channels: Iterable[int] = (128, 256, 256),
        kernel_sizes: Iterable[int] = (11, 7, 5),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        channels = tuple(int(v) for v in channels)
        kernel_sizes = tuple(int(v) for v in kernel_sizes)
        if len(channels) != len(kernel_sizes):
            raise ValueError('channels and kernel_sizes must have the same length')

        blocks: list[nn.Module] = []
        in_ch = input_channels
        for out_ch, k in zip(channels, kernel_sizes):
            blocks.append(FCNResidualBlock1d(in_ch, out_ch, k, dropout=dropout))
            in_ch = out_ch

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(in_ch)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(in_ch, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.features(x)
        y = self.pool(y).squeeze(-1)
        y = self.norm(y)
        y = self.drop(y)
        return self.head(y)
