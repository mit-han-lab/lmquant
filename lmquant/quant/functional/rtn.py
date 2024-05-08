# -*- coding: utf-8 -*-
"""Round-to-nearest (RTN) quantization module."""

import torch

from ..data.dtype import QuantDataType
from ..data.range import QuantRange
from .simple import simple_quantize

__all__ = ["rtn_quantize"]


def rtn_quantize(
    tensor: torch.Tensor,
    *,
    view_shape: torch.Size,
    quant_dtype: QuantDataType,
    scale: torch.Tensor,
    zero: torch.Tensor,
    quant_range: QuantRange = None,
    round_delta: torch.Tensor = None,
) -> torch.Tensor:
    """Quantize the tensor using the RTN quantization kernel.

    Args:
        tensor (torch.Tensor): The input tensor.
        view_shape (torch.Size): The view shape.
        quant_dtype (QuantDataType): The quantization data type.
        scale (torch.Tensor): The scale tensor.
        zero (torch.Tensor): The zero point tensor.
        quant_range (QuantRange, optional): The quantization range. Defaults to ``None``.
        round_delta (torch.Tensor, optional): The rounding delta. Defaults to ``None``.

    Returns:
        torch.Tensor: The quantized tensor in the shape of ``view_shape``.
    """
    requires_grad = tensor.requires_grad or scale.requires_grad
    tensor = tensor.view(view_shape)
    round_delta = round_delta.view(view_shape) if round_delta is not None else None
    q = tensor.div(scale) if requires_grad else tensor.div_(scale)
    q = simple_quantize(q.add_(zero), quant_dtype=quant_dtype, quant_range=quant_range, round_delta=round_delta)
    return q.sub_(zero).mul_(scale)
