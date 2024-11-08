# -*- coding: utf-8 -*-
"""Simple quantization functions."""

import torch

from ...data.dtype import QuantDataType
from ...data.range import LogQuantRange, QuantRange
from .ste import ste

__all__ = ["simple_quantize"]


def simple_quantize(
    tensor: torch.Tensor,
    *,
    quant_dtype: torch.dtype | QuantDataType,
    has_zero_point: bool,
    quant_range: QuantRange | None = None,
    round_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Simple quantization function."""
    requires_grad = tensor.requires_grad
    if isinstance(quant_dtype, torch.dtype):
        dtype = tensor.dtype
        tensor = tensor.to(dtype=quant_dtype).to(dtype=dtype)
        if round_delta is not None:
            tensor = tensor.add_(round_delta)
        if quant_range is not None and quant_range.is_set():
            tensor = torch.clamp(tensor, min=quant_range.min, max=quant_range.max)
        return tensor
    elif isinstance(quant_dtype, QuantDataType):
        if quant_dtype.is_exponent:
            assert round_delta is None, "round_delta is not supported for exponential quantization"
            quant_range = LogQuantRange.construct(quant_dtype, quant_range)
            tensor = ste(tensor.log2(), lambda x: x.floor()) if requires_grad else tensor.log2_().floor_()
            return tensor.clamp_(min=quant_range.min, max=quant_range.max).exp2_()
        elif quant_dtype.is_float_point:
            assert round_delta is None, "round_delta is not supported for float quantization"
            tensor = torch.clamp(tensor, min=quant_dtype.min_value, max=quant_dtype.max_value)
            tensor = ste(tensor, lambda x: quant_dtype.get_codebook(normalize=False, device=tensor.device).quantize(x))
            if quant_range is not None and quant_range.is_set():
                tensor = tensor.clamp_(min=quant_range.min, max=quant_range.max)
            return tensor
        else:
            quant_range = QuantRange.construct(quant_dtype, has_zero_point=has_zero_point, quant_range=quant_range)
            if round_delta is None:
                tensor = ste(tensor, lambda x: torch.round(x)) if requires_grad else tensor.round_()
            else:
                tensor = ste(tensor, lambda x: torch.floor(x)) if requires_grad else tensor.floor_()
                tensor = tensor.add_(round_delta)
            return tensor.clamp_(min=quant_range.min, max=quant_range.max)
    else:
        raise TypeError(
            f"quant_dtype must be either torch.dtype or QuantDataType, got {quant_dtype} ({type(quant_dtype)})"
        )
