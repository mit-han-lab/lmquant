# -*- coding: utf-8 -*-
"""Simple quantization functions."""

import typing as tp

import torch

__all__ = ["ste"]


class STEFunction(torch.autograd.Function):
    """STEFunction for quantization."""

    @staticmethod
    def forward(ctx: tp.Any, tensor: torch.Tensor, fn: tp.Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Forward pass for DtypeSTEFunction."""
        return fn(tensor)

    @staticmethod
    def backward(ctx: tp.Any, grad_output: torch.Tensor) -> tp.Tuple[torch.Tensor, None]:
        """Backward pass for DtypeSTEFunction."""
        return grad_output, None


def ste(tensor: torch.Tensor, fn: tp.Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """STE function."""
    return STEFunction.apply(tensor, fn)
