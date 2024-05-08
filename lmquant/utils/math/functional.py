# -*- coding: utf-8 -*-
"""Math utility functions."""

import torch

__all__ = ["is_pow2", "root_"]


def is_pow2(n: int) -> bool:
    """Check if a number is a power of 2."""
    return (n & (n - 1) == 0) and (n > 0)


def root_(y: torch.Tensor, index: float) -> torch.Tensor:
    return y.pow_(1 / index) if index != 2 else y.sqrt_()
