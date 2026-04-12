from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _kernel_diagnostics(kernel: torch.Tensor) -> dict[str, float]:
    kernel = kernel.detach().cpu()
    per_channel_sum = kernel.sum(dim=-1).mean().item()
    symmetry_error = torch.mean(torch.abs(kernel - torch.flip(kernel, dims=[-1]))).item()
    freq = torch.fft.rfft(kernel[0, 0]).abs().cpu().numpy()
    return {
        'mean_tap_sum': float(per_channel_sum),
        'symmetry_error': float(symmetry_error),
        'dc_gain_mag': float(freq[0]),
        'nyquist_mag': float(freq[-1]),
    }


@torch.no_grad()
def collect_frontend_diagnostics(model: nn.Module) -> dict[str, Any]:
    diag: dict[str, Any] = {}
    frontend = getattr(model, 'frontend', None)
    if frontend is None:
        return {'frontend': 'none'}

    if hasattr(frontend, 'branches') and frontend.branches:
        diag['frontend'] = frontend.__class__.__name__
        diag['residual'] = bool(getattr(frontend, 'residual', False))
        diag['gate_alpha_mean'] = float(torch.sigmoid(frontend.beta).mean().item()) if hasattr(frontend, 'beta') else None
        for idx, branch in enumerate(frontend.branches):
            diag[f'branch_{idx}'] = {
                'sym1': _kernel_diagnostics(branch.sym1.build_kernel()),
                'sym2': _kernel_diagnostics(branch.sym2.build_kernel()),
            }
        return diag

    if hasattr(frontend, 'sym1') and hasattr(frontend, 'sym2'):
        diag['frontend'] = frontend.__class__.__name__
        diag['residual'] = bool(getattr(frontend, 'residual', False))
        diag['gate_alpha_mean'] = float(torch.sigmoid(frontend.beta).mean().item()) if hasattr(frontend, 'beta') else None
        diag['sym1'] = _kernel_diagnostics(frontend.sym1.build_kernel())
        diag['sym2'] = _kernel_diagnostics(frontend.sym2.build_kernel())
        return diag

    if hasattr(frontend, 'front') and hasattr(frontend.front, 'build_kernel'):
        diag['frontend'] = frontend.__class__.__name__
        diag['residual'] = bool(getattr(frontend, 'residual', False))
        diag['front'] = _kernel_diagnostics(frontend.front.build_kernel())
        return diag

    if hasattr(frontend, 'frontend') and hasattr(frontend.frontend, 'weight'):
        diag['frontend'] = frontend.frontend.__class__.__name__
        weight = frontend.frontend.weight.detach().cpu()
        freq = torch.fft.rfft(weight[0, 0]).abs().cpu().numpy()
        diag['front'] = {
            'mean_tap_sum': float(weight.sum(dim=-1).mean().item()),
            'dc_gain_mag': float(freq[0]),
            'nyquist_mag': float(freq[-1]),
        }
        return diag

    return {'frontend': frontend.__class__.__name__}
