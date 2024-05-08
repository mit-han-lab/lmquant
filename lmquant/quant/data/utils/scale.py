# -*- coding: utf-8 -*-
"""Utility functions for quantization scale."""

import torch

from ..dtype import QuantDataType

__all__ = ["join_scale_tensor", "infer_scale_quant_spans", "infer_exponent_scaling_level"]


def join_scale_tensor(global_scale: torch.Tensor, local_scale: torch.Tensor) -> torch.Tensor:
    """Multiply the local scale tensor by the global scale tensor.

    Args:
        global_scale (torch.Tensor): Global scale tensor.
        local_scale (torch.Tensor): Local scale tensor.

    Returns:
        torch.Tensor: The compounded scale tensor.
    """
    # gs: (#gs_g0, 1, #gs_g1, 1, #gs_g2, 1, ...)
    # ss: (#ss_g0, 1, #ss_g1, 1, #ss_g2, 1, ...) -> (#gs_g0, rs0, #gs_g1, rs1, #gs_g2, rs2, ...)
    shape = local_scale.shape
    return (
        local_scale
        if global_scale is None
        else local_scale.view(
            tuple(
                global_scale.shape[i] if j == 0 else local_scale.shape[i] // global_scale.shape[i]
                for i in range(0, len(global_scale.shape), 2)
                for j in range(2)
            )
        ).mul(global_scale)
    ).view(shape)


def infer_scale_quant_spans(scale_dtypes: list[QuantDataType]) -> list[float]:
    quant_spans = [1]
    for s_dtype in reversed(scale_dtypes[1:]):
        assert isinstance(s_dtype, QuantDataType), f"s_dtype must be QuantDataType, got {s_dtype}"
        quant_spans.append(s_dtype.max_value * quant_spans[-1])
    return list(reversed(quant_spans))


def infer_exponent_scaling_level(scale_dtypes: list[QuantDataType]) -> int:
    """Get the exponent scaling level.

    Args:
        scale_dtypes (list[QuantDataType]): The scale data types.

    Returns:
        int: The exponent scaling level.
    """
    for level, scale_dtype in enumerate(scale_dtypes):
        if isinstance(scale_dtype, QuantDataType) and scale_dtype.is_exp:
            return level
    return len(scale_dtypes)
