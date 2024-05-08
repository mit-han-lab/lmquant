# -*- coding: utf-8 -*-
"""One-step quantization kernel."""
import functools

import torch

from ...dataset import ActivationsCache
from ..data.range import DynamicRange, QuantRange, RangeBound
from ..data.scale import QuantScale
from ..data.tensor import QuantTensor
from ..data.utils import ShapeUtils
from .config import QuantConfig, QuantKernelConfig, QuantKernelType
from .gptq import gptq_quantize
from .rtn import rtn_quantize
from .scale import infer_scale_and_zero
from .ste import ste

__all__ = ["onestep_quantize"]


def onestep_quantize(  # noqa: C901
    tensor: torch.Tensor,
    config: QuantConfig,
    kernel_config: QuantKernelConfig | None = None,
    *,
    scale: torch.Tensor = None,
    zero: torch.Tensor = None,
    dynamic_range: DynamicRange = None,
    quant_range: QuantRange = None,
    range_bound: RangeBound = None,
    round_delta: torch.Tensor = None,
    inputs: ActivationsCache = None,
    return_with_dequant: bool = True,
    return_with_quant: bool = False,
    default_dtype: torch.dtype = torch.float16,
    develop_dtype: torch.dtype = torch.float32,
) -> QuantTensor:
    """Quantize a floating point tensor. tensor ~ tensor_hat = s * (q - z).

    Args:
        tensor (torch.Tensor): The floating-point tensor to be quantized.
        config (QuantConfig): The quantization configuration.
        kernel_config (QuantKernelConfig, optional): The quantization kernel configuration.
        scale (torch.Tensor, optional): The scale tensor. Defaults to ``None``.
        zero (torch.Tensor, optional): The zero point tensor. Defaults to ``None``.
        dynamic_range (DynamicRange, optional): The dynamic range of the tensor. Defaults to ``None``.
        quant_range (QuantRange, optional): The quantization range. Defaults to ``None``.
        range_bound (RangeBound, optional): The dynamic range bound. Defaults to ``None``.
        round_delta (torch.Tensor, optional): The rounding delta. Defaults to ``None``.
        inputs (ActivationsCache, optional): The inputs cache. Defaults to ``None``.
        return_with_dequant (bool, optional): Whether to return with dequantized tensor. Defaults to ``True``.
        return_with_quant (bool, optional): Whether to return with quantized tensor. Defaults to ``False``.
        default_dtype (torch.dtype, optional): The default dtype. Defaults to ``torch.float16``.
        develop_dtype (torch.dtype, optional): The develop dtype. Defaults to ``torch.float32``.

    Returns:
        QuantTensor: The quantized tensor.
    """
    if config is None or config.dtype is None:
        return QuantTensor(dequantized=tensor, quantized=tensor, view_shape=tensor.shape)
    dtype, shape = tensor.dtype, tensor.shape
    requires_grad = tensor.requires_grad
    requires_grad = requires_grad or (scale is not None and scale.requires_grad)
    requires_grad = requires_grad or (round_delta is not None and round_delta.requires_grad)
    if requires_grad:
        assert return_with_dequant, "The dequantized tensor must be returned when requires_grad is True."
    develop_tensor = tensor.to(dtype=develop_dtype) if dtype != develop_dtype else tensor.clone()
    group_shapes = ShapeUtils.infer_group_shapes(group_shapes=config.group_shapes, shape=shape)
    view_shape = ShapeUtils.infer_view_shape(shape, group_shape=group_shapes[-1])
    # region step 1: get the scale and zero point
    scale, zero = infer_scale_and_zero(
        tensor=develop_tensor,
        quant_dtype=config.dtype,
        group_shapes=group_shapes,
        scale_quant_dtypes=config.get_scale_dtypes(default_dtype=default_dtype),
        exponent_scaling_level=config.exponent_scaling_level,
        view_shape=view_shape,
        dynamic_range=dynamic_range,
        quant_range=quant_range,
        scale=scale,
        zero=zero,
    )
    assert isinstance(scale, QuantScale), "The scale must be a QuantScale instance."
    # endregion
    # region step 2: quantize the tensor
    if kernel_config is None or kernel_config.kernel == QuantKernelType.RTN:
        # ! we cannot wrap the `rtn_quantize` in `ste` because `round_delta` may require grad
        tensor_hat = rtn_quantize(
            tensor=develop_tensor,
            view_shape=view_shape,
            quant_dtype=config.dtype,
            scale=scale.data,
            zero=zero,
            quant_range=quant_range,
            round_delta=round_delta,
        )
    elif kernel_config.kernel == QuantKernelType.GPTQ:
        assert not scale.data.requires_grad, "scale must not require gradient."
        assert round_delta is None or not round_delta.requires_grad, "round_delta must not require gradient."
        tensor_hat = ste(
            develop_tensor,
            fn=functools.partial(
                gptq_quantize,
                view_shape=view_shape,
                quant_dtype=config.dtype,
                gptq_config=kernel_config,
                scale=scale.data,
                zero=zero,
                inputs=inputs,
                quant_range=quant_range,
                range_bound=range_bound,
                round_delta=round_delta,
            ),
        )
    else:
        raise ValueError(f"Invalid quantization kernel: {kernel_config.kernel}")
    del develop_tensor
    # endregion
    assert tensor_hat.requires_grad == requires_grad, "requires_grad must be consistent."
    assert not tensor_hat.isnan().any(), "Quantized tensor contains NaN."
    assert not tensor_hat.isinf().any(), "Quantized tensor contains Inf."
    dequantized = None
    if return_with_dequant:
        dequantized = tensor_hat.view(shape).to(dtype=dtype)
    quantized = None
    if return_with_quant:
        quantized = tensor_hat.detach()
        if return_with_dequant:
            quantized = tensor_hat.clone()
        quantized = quantized.view(view_shape).div_(scale.data).add_(zero).view(shape)
    return QuantTensor(
        dequantized=dequantized,
        quantized=quantized,
        scale=scale if return_with_quant else None,
        zero=zero if return_with_quant else None,
        view_shape=view_shape if return_with_quant else None,
    )
