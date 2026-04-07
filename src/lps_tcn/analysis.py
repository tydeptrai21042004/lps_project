from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def collect_frontend_diagnostics(model: nn.Module) -> dict[str, Any]:
    diag: dict[str, Any] = {}
    frontend = getattr(model, "frontend", None)
    if frontend is None:
        return {"frontend": "none"}

    if hasattr(frontend, "sym1") and hasattr(frontend, "sym2"):
        kernels = {
            "sym1": frontend.sym1.build_kernel().detach().cpu(),
            "sym2": frontend.sym2.build_kernel().detach().cpu(),
        }
        diag["frontend"] = frontend.__class__.__name__
        diag["residual"] = bool(getattr(frontend, "residual", False))
        diag["gate_alpha"] = float(torch.sigmoid(frontend.beta).item()) if hasattr(frontend, "beta") else None
    elif hasattr(frontend, "front") and hasattr(frontend.front, "build_kernel"):
        kernels = {"front": frontend.front.build_kernel().detach().cpu()}
        diag["frontend"] = frontend.__class__.__name__
        diag["residual"] = bool(getattr(frontend, "residual", False))
    elif hasattr(frontend, "frontend") and hasattr(frontend.frontend, "weight"):
        kernels = {"fixed": frontend.frontend.weight.detach().cpu()}
        diag["frontend"] = frontend.frontend.__class__.__name__
    else:
        return {"frontend": frontend.__class__.__name__}

    for name, kernel in kernels.items():
        per_channel_sum = kernel.sum(dim=-1).mean().item()
        symmetry_error = torch.mean(torch.abs(kernel - torch.flip(kernel, dims=[-1]))).item()
        freq = torch.fft.rfft(kernel[0, 0]).abs().cpu().numpy()
        diag[name] = {
            "mean_tap_sum": float(per_channel_sum),
            "symmetry_error": float(symmetry_error),
            "dc_gain_mag": float(freq[0]),
            "nyquist_mag": float(freq[-1]),
        }
    return diag
