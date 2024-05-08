# -*- coding: utf-8 -*-
"""Quantization function."""

import torch
import torch.utils.hooks

from ...dataset import ActivationsCache
from ..data.range import DynamicRange, QuantRange, RangeBound
from ..data.tensor import QuantTensor
from ..functional.config import QuantConfig, QuantKernelConfig
from ..functional.progressive import progressive_quantize

__all__ = ["quantize"]


def quantize(
    tensor: torch.Tensor,
    config: QuantConfig,
    kernel_config: QuantKernelConfig | None = None,
    *,
    channels_dim: int | None = None,
    scale: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = None,
    zero: torch.Tensor = None,
    dynamic_range: DynamicRange | tuple[DynamicRange, DynamicRange] = None,
    quant_range: QuantRange = None,
    range_bound: RangeBound = None,
    round_delta: torch.Tensor = None,
    inputs: ActivationsCache = None,
    return_with_dequant: bool = True,
    return_with_quant: bool = False,
    default_dtype: torch.dtype | None = None,
    develop_dtype: torch.dtype = torch.float32,
) -> QuantTensor:
    """Quantize a floating point tensor.

    Args:
        tensor (torch.Tensor): The floating-point tensor to be quantized.
        config (QuantConfig): The quantization configuration.
        kernel_config (QuantKernelConfig, optional): The kernel configuration. Defaults to ``None``.
        channels_dim (int, optional): The dimension of channels in activations. Defaults to ``None``.
        scale (torch.Tensor | tuple[torch.Tensor, torch.Tensor], optional): The scale tensor.
            Defaults to ``None``.
        zero (torch.Tensor, optional): The zero point tensor. Defaults to ``None``.
        dynamic_range (DynamicRange | tuple[DynamicRange, DynamicRange], optional): The dynamic range.
            Defaults to ``None``.
        quant_range (QuantRange, optional): The quantization range. Defaults to ``None``.
        range_bound (RangeBound, optional): The dynamic range bound. Defaults to ``None``.
        round_delta (torch.Tensor, optional): The rounding delta. Defaults to ``None``.
        inputs (ActivationsCache, optional): The inputs cache. Defaults to ``None``.
        return_with_dequant (bool, optional): Whether to return with dequantized tensor. Defaults to ``True``.
        return_with_quant (bool, optional): Whether to return with quantized tensor. Defaults to ``False``.
        default_dtype (torch.dtype | None, optional): The default dtype. Defaults to ``None``.
        develop_dtype (torch.dtype, optional): The develop dtype. Defaults to ``torch.float32``.

    Returns:
        QuantTensor: The quantized tensor.
    """
    shape = tensor.shape
    if channels_dim is not None:
        tensor = tensor.view(-1, *shape[channels_dim:])
    result = progressive_quantize(
        tensor,
        config,
        kernel_config=kernel_config,
        scale=scale,
        zero=zero,
        dynamic_range=dynamic_range,
        quant_range=quant_range,
        range_bound=range_bound,
        round_delta=round_delta,
        inputs=inputs,
        return_with_dequant=return_with_dequant,
        return_with_quant=return_with_quant,
        default_dtype=tensor.dtype if default_dtype is None else default_dtype,
        develop_dtype=develop_dtype,
    )
    if result.data is not None:
        result._dequantized = result.data.view(shape)
    if result.qdata is not None:
        result._quantized = result.qdata.view(shape)
    return result
