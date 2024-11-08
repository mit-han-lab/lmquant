# -*- coding: utf-8 -*-
"""Math utility functions."""

import torch

__all__ = ["is_pow2", "root_"]


def is_pow2(n: int) -> bool:
    """Check if a number is a power of 2.

    Args:
        n (`int`):
            The number to check.

    Returns:
        `bool`:
            Whether the number is a power of 2.
    """
    return (n & (n - 1) == 0) and (n > 0)


def root_(y: torch.Tensor, index: float) -> torch.Tensor:
    """In-place compute the root of a tensor element-wise.

    Args:
        y (`torch.Tensor`):
            The input tensor.
        index (`float`):
            The root index.

    Returns:
        `torch.Tensor`:
            The output tensor.
    """
    return y.pow_(1 / index) if index != 2 else y.sqrt_()
