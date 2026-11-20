"""Shared utilities for the Madrid continual forecasting protocol."""

import torch

def _check_finite(tensor, name, args, epoch=None, batch_idx=None):
    if torch.isfinite(tensor).all():
        return
    location = f"year={getattr(args, 'year', '?')}"
    if epoch is not None:
        location += f", epoch={epoch}"
    if batch_idx is not None:
        location += f", batch={batch_idx}"
    raise FloatingPointError(f"Non-finite values in {name} ({location}).")
