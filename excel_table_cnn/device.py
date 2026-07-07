"""Device selection: run anywhere, use accelerators when they actually help.

``"auto"`` picks CUDA when available, otherwise CPU. Apple Silicon (MPS) is
fully supported and covered by tests, but it is deliberately **not** part of
auto-selection: with the current small backbone and batch size 1, measured
steady-state training is 2-5x slower on MPS than on the M-series CPU cores
(detection heads launch many tiny kernels; GPU overhead dominates). Pass
``device="mps"`` explicitly to use it — worth re-measuring once the backbone
grows.
"""

from typing import Optional, Union

import torch

DeviceLike = Union[str, torch.device, None]


def resolve_device(device: DeviceLike = "auto") -> torch.device:
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def resolve_amp(amp: Optional[bool], device: torch.device) -> bool:
    """AMP default: on for CUDA, off elsewhere (MPS/CPU run fp32 —
    GradScaler and autocast are only reliably supported on CUDA)."""
    if amp is None:
        return device.type == "cuda"
    return amp and device.type == "cuda"
