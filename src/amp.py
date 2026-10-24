"""
Mixed-precision helpers.

Mixed precision runs the big matrix multiplies in a 16-bit number format and
keeps the weights and the loss in 32-bit:

  * on an NVIDIA GPU we use float16: about 2x faster and half the memory, but
    its smallest positive number is about 6e-8, so tiny gradients underflow to
    zero. GradScaler fixes that: multiply the loss by a big factor S before
    backward (so gradients are S times bigger and do not underflow), then
    divide the gradients by S before the optimiser step. If any gradient became
    inf/NaN, the step is skipped and S is halved.
  * on a CPU we use bfloat16: same exponent range as float32, so nothing
    underflows and no scaler is needed (fewer mantissa bits, less precision).
"""

import contextlib

import torch


def _device_type(device):
    return torch.device(device).type


def autocast(device, enabled):
    """Context manager: ops inside run in 16-bit where it is safe to."""
    if not enabled:
        return contextlib.nullcontext()
    kind = _device_type(device)
    dtype = torch.float16 if kind == "cuda" else torch.bfloat16
    return torch.autocast(device_type=kind, dtype=dtype)


def make_scaler(device, enabled):
    """GradScaler, active only for float16 on a GPU; elsewhere every call is a pass-through."""
    kind = _device_type(device)
    return torch.amp.GradScaler(kind, enabled=enabled and kind == "cuda")
