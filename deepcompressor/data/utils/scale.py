# -*- coding: utf-8 -*-
"""Utility functions for quantization scale."""

import typing as tp

import torch

from ..dtype import QuantDataType

__all__ = ["infer_scale_dtypes", "infer_scale_quant_spans", "infer_exponent_scale_level"]


def infer_scale_dtypes(
    scale_dtypes: tp.Sequence[torch.dtype | QuantDataType | None], default_dtype: torch.dtype | QuantDataType
) -> list[torch.dtype | QuantDataType]:
    """Get the scale dtypes for the given tensor dtype.

    Args:
        scale_dtypes (`Sequence[torch.dtype | QuantDataType | None]`):
            The scale dtypes.
        default_dtype (`torch.dtype`):
            The default scale dtype.

    Returns:
        `list[torch.dtype | QuantDataType]`:
            The scale dtypes.
    """
    assert isinstance(
        default_dtype, (torch.dtype, QuantDataType)
    ), f"dtype must be torch.dtype or QuantDataType, got {default_dtype}"
    return [s_dtype or default_dtype for s_dtype in scale_dtypes]


def infer_scale_quant_spans(scale_dtypes: tp.Sequence[QuantDataType], base: int = 1) -> list[float]:
    quant_spans: list[float] = [base]
    for s_dtype in reversed(scale_dtypes[1:]):
        assert isinstance(s_dtype, QuantDataType), f"s_dtype must be QuantDataType, got {s_dtype}"
        quant_spans.append(s_dtype.max_value * quant_spans[-1])
    return list(reversed(quant_spans))


def infer_exponent_scale_level(scale_dtypes: tp.Sequence[torch.dtype | QuantDataType]) -> int:
    """Get the exponent scaling level.

    Args:
        scale_dtypes (`Sequence[torch.dtype | QuantDataType]`):
            The scale data types.

    Returns:
        `int`: The exponent scaling level.
    """
    for level, scale_dtype in enumerate(scale_dtypes):
        if isinstance(scale_dtype, QuantDataType) and scale_dtype.is_exponent:
            return level
    return len(scale_dtypes)
