from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.signal import savgol_coeffs
except Exception:  # pragma: no cover
    savgol_coeffs = None


def _make_gaussian_full(kernel_size: int, sigma: Optional[float] = None, device=None, dtype=None) -> torch.Tensor:
    center = kernel_size // 2
    if sigma is None:
        sigma = max(1.0, kernel_size / 6.0)
    x = torch.arange(kernel_size, device=device, dtype=dtype) - center
    w = torch.exp(-(x**2) / (2 * sigma**2))
    w = w / w.sum()
    return w


def _make_hamming_full(kernel_size: int, device=None, dtype=None) -> torch.Tensor:
    n = torch.arange(kernel_size, device=device, dtype=dtype)
    w = 0.54 - 0.46 * torch.cos(2 * math.pi * n / (kernel_size - 1))
    w = w / w.sum()
    return w


def _make_savgol_full(kernel_size: int, polyorder: int = 3, device=None, dtype=None) -> torch.Tensor:
    if savgol_coeffs is None:
        raise ImportError('scipy is required for Savitzky-Golay coefficients')
    coeffs = savgol_coeffs(kernel_size, polyorder=polyorder, use='conv')
    out_dtype = dtype if dtype is not None else torch.float32
    return torch.tensor(coeffs, device=device, dtype=out_dtype)


class SqueezeExcitation1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Conv1d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = x.mean(dim=-1, keepdim=True)
        g = F.relu(self.fc1(g))
        g = torch.sigmoid(self.fc2(g))
        return x * g


class SymmetricConv1d(nn.Module):
    """Learnable Type-I linear-phase 1D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
        h: float = 1.0,
        bias: bool = False,
        causal: bool = True,
        init_mode: str = 'identity',
        normalize_kernel_dc: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError('kernel_size must be odd for Type-I FIR')
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError('groups must divide both in_channels and out_channels')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.groups = groups
        self.h = float(h)
        self.causal = causal
        self.normalize_kernel_dc = normalize_kernel_dc

        half_len = kernel_size // 2 + 1
        self.w_half = nn.Parameter(torch.empty(out_channels, in_channels // groups, half_len))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters(init_mode)

    def reset_parameters(self, init_mode: str) -> None:
        if init_mode == 'identity':
            with torch.no_grad():
                self.w_half.zero_()
                self.w_half[..., -1] = 1.0
        elif init_mode == 'gaussian':
            with torch.no_grad():
                full = _make_gaussian_full(self.kernel_size, device=self.w_half.device, dtype=self.w_half.dtype)
                half = full[self.kernel_size // 2 :]
                self.w_half.copy_(half.view(1, 1, -1).expand_as(self.w_half))
        elif init_mode == 'kaiming':
            nn.init.kaiming_normal_(self.w_half)
        else:
            raise ValueError(f'Unknown init_mode: {init_mode}')

    def build_kernel(self) -> torch.Tensor:
        left = torch.flip(self.w_half[..., :-1], dims=[-1])
        full = torch.cat([left, self.w_half], dim=-1)
        if self.normalize_kernel_dc:
            denom = full.sum(dim=-1, keepdim=True)
            full = full / (denom + 1e-6)
        return full

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.build_kernel().to(dtype=x.dtype, device=x.device)
        bias = self.bias.to(dtype=x.dtype, device=x.device) if self.bias is not None else None
        if self.causal:
            pad_left = self.dilation * (self.kernel_size - 1)
            x = F.pad(x, (pad_left, 0))
            y = F.conv1d(x, w, bias=bias, stride=1, padding=0, dilation=self.dilation, groups=self.groups)
        else:
            pad = self.dilation * (self.kernel_size - 1) // 2
            y = F.conv1d(x, w, bias=bias, stride=1, padding=pad, dilation=self.dilation, groups=self.groups)
        return self.h * y


class LearnableDepthwiseConv1d(nn.Module):
    """Unconstrained learnable front-end baseline with the same causal padding behavior."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 9,
        causal: bool = True,
        init_mode: str = 'identity',
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError('kernel_size must be odd')
        self.channels = channels
        self.kernel_size = kernel_size
        self.causal = causal
        self.weight = nn.Parameter(torch.zeros(channels, 1, kernel_size))

        if init_mode == 'identity':
            with torch.no_grad():
                self.weight.zero_()
                self.weight[:, 0, kernel_size // 2] = 1.0
        elif init_mode == 'gaussian':
            with torch.no_grad():
                base = _make_gaussian_full(kernel_size, device=self.weight.device, dtype=self.weight.dtype)
                self.weight.copy_(base.view(1, 1, -1).repeat(channels, 1, 1))
        elif init_mode == 'kaiming':
            nn.init.kaiming_normal_(self.weight)
        else:
            raise ValueError(f'Unknown init_mode: {init_mode}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(dtype=x.dtype, device=x.device)
        if self.causal:
            x = F.pad(x, (self.kernel_size - 1, 0))
            return F.conv1d(x, w, groups=self.channels)
        pad = self.kernel_size // 2
        return F.conv1d(x, w, padding=pad, groups=self.channels)


class FixedSmoother1d(nn.Module):
    """Paper-supported fixed linear-phase smoother baseline."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        smoother_type: str = 'gaussian',
        causal: bool = True,
        polyorder: int = 3,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError('kernel_size must be odd')
        self.channels = channels
        self.kernel_size = kernel_size
        self.causal = causal
        self.smoother_type = smoother_type

        if smoother_type == 'gaussian':
            base = _make_gaussian_full(kernel_size, dtype=torch.float32)
        elif smoother_type == 'hamming':
            base = _make_hamming_full(kernel_size, dtype=torch.float32)
        elif smoother_type == 'savgol':
            base = _make_savgol_full(kernel_size, polyorder=polyorder, dtype=torch.float32)
        elif smoother_type == 'moving_avg':
            base = torch.ones(kernel_size, dtype=torch.float32) / kernel_size
        else:
            raise ValueError(f'Unknown smoother_type: {smoother_type}')

        weight = base.view(1, 1, -1).repeat(channels, 1, 1)
        self.register_buffer('weight', weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(dtype=x.dtype, device=x.device)
        if self.causal:
            x = F.pad(x, (self.kernel_size - 1, 0))
            return F.conv1d(x, weight, groups=self.channels)
        pad = (self.kernel_size - 1) // 2
        return F.conv1d(x, weight, padding=pad, groups=self.channels)


class LPSConv(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 9,
        h: float = 1.0,
        causal: bool = True,
        residual: bool = False,
        init_mode: str = 'identity',
        normalize_kernel_dc: bool = False,
    ) -> None:
        super().__init__()
        self.front = SymmetricConv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,
            h=h,
            bias=False,
            causal=causal,
            init_mode=init_mode,
            normalize_kernel_dc=normalize_kernel_dc,
        )
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.front(x)
        return x + y if self.residual else y


class _LPSBranch(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size1: int,
        kernel_size2: int,
        *,
        h: float,
        causal: bool,
        use_relu: bool,
        kernel_init: str,
        normalize_kernel_dc: bool,
    ) -> None:
        super().__init__()
        self.sym1 = SymmetricConv1d(
            channels,
            channels,
            kernel_size1,
            groups=channels,
            h=h,
            bias=False,
            causal=causal,
            init_mode=kernel_init,
            normalize_kernel_dc=normalize_kernel_dc,
        )
        self.sym2 = SymmetricConv1d(
            channels,
            channels,
            kernel_size2,
            groups=channels,
            h=h,
            bias=False,
            causal=causal,
            init_mode=kernel_init,
            normalize_kernel_dc=normalize_kernel_dc,
        )
        self.use_relu = use_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.sym1(x)
        if self.use_relu:
            z = F.relu(z)
        z = self.sym2(z)
        return z


class LPSConvPlus(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size1: int = 9,
        kernel_size2: int = 9,
        h: float = 1.0,
        causal: bool = True,
        use_relu: bool = True,
        use_pointwise: bool = True,
        dc_mode: str = 'project',
        residual: bool = True,
        gate_init: float = -4.0,
        kernel_init: str = 'identity',
        normalize_kernel_dc: bool = False,
        branch_kernel_sizes: Iterable[int] | None = None,
        use_se: bool = False,
        per_channel_gate: bool = False,
        branch_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        branch_kernel_sizes = tuple(int(k) for k in branch_kernel_sizes or ())
        if branch_kernel_sizes:
            self.branches = nn.ModuleList(
                [
                    _LPSBranch(
                        channels,
                        kernel_size1=k,
                        kernel_size2=k if kernel_size2 == kernel_size1 else kernel_size2,
                        h=h,
                        causal=causal,
                        use_relu=use_relu,
                        kernel_init=kernel_init,
                        normalize_kernel_dc=normalize_kernel_dc,
                    )
                    for k in branch_kernel_sizes
                ]
            )
            self.sym1 = self.branches[0].sym1
            self.sym2 = self.branches[0].sym2
        else:
            self.branches = None
            self.sym1 = SymmetricConv1d(
                channels,
                channels,
                kernel_size1,
                groups=channels,
                h=h,
                bias=False,
                causal=causal,
                init_mode=kernel_init,
                normalize_kernel_dc=normalize_kernel_dc,
            )
            self.sym2 = SymmetricConv1d(
                channels,
                channels,
                kernel_size2,
                groups=channels,
                h=h,
                bias=False,
                causal=causal,
                init_mode=kernel_init,
                normalize_kernel_dc=normalize_kernel_dc,
            )

        self.use_relu = use_relu
        self.dc_mode = dc_mode
        self.residual = residual
        self.mix = nn.Conv1d(channels, channels, kernel_size=1) if use_pointwise else nn.Identity()
        beta_shape = (1, channels, 1) if per_channel_gate else ()
        self.beta = nn.Parameter(torch.full(beta_shape or (1,), float(gate_init)).reshape(beta_shape or ()))
        self.branch_dropout = nn.Dropout(branch_dropout) if branch_dropout > 0 else nn.Identity()
        self.channel_attention = SqueezeExcitation1d(channels) if use_se else nn.Identity()
        self.reset_mix()

    def reset_mix(self) -> None:
        if isinstance(self.mix, nn.Conv1d):
            with torch.no_grad():
                self.mix.weight.zero_()
                c_out, c_in, _ = self.mix.weight.shape
                for i in range(min(c_out, c_in)):
                    self.mix.weight[i, i, 0] = 1.0
                if self.mix.bias is not None:
                    self.mix.bias.zero_()

    def _stack_branches(self, x: torch.Tensor) -> torch.Tensor:
        if self.branches is None:
            z = self.sym1(x)
            if self.use_relu:
                z = F.relu(z)
            z = self.sym2(z)
            return z

        branch_outputs = [branch(x) for branch in self.branches]
        stacked = torch.stack(branch_outputs, dim=0).mean(dim=0)
        return self.branch_dropout(stacked)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self._stack_branches(x)
        z = self.mix(z)
        z = self.channel_attention(z)

        if self.dc_mode == 'project':
            x_m = x.mean(dim=-1, keepdim=True)
            z_m = z.mean(dim=-1, keepdim=True)
            z = z - z_m + x_m
        elif self.dc_mode == 'none':
            pass
        else:
            raise ValueError(f'Unknown dc_mode: {self.dc_mode}')

        if self.residual:
            alpha = torch.sigmoid(self.beta)
            return x + alpha * (z - x)
        return z


class FrontendWithResidual(nn.Module):
    def __init__(self, frontend: nn.Module, residual: bool = False, gate: bool = False, gate_init: float = -4.0) -> None:
        super().__init__()
        self.frontend = frontend
        self.residual = residual
        self.gate = gate
        self.beta = nn.Parameter(torch.tensor(float(gate_init))) if gate else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.frontend(x)
        if not self.residual:
            return z
        if self.gate and self.beta is not None:
            alpha = torch.sigmoid(self.beta)
            return x + alpha * (z - x)
        return x + z
