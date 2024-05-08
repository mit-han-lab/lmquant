# -*- coding: utf-8 -*-
"""Progressive quantization kernel."""

import torch

from ...dataset import ActivationsCache
from ..data.dtype import QuantDataType
from ..data.range import DynamicRange, QuantRange, RangeBound
from ..data.tensor import QuantTensor
from ..data.utils import ShapeUtils
from .config import QuantConfig, QuantKernelConfig
from .onestep import onestep_quantize
from .scale import infer_scale_and_zero

__all__ = ["progressive_quantize"]


def progressive_quantize(  # noqa: C901
    tensor: torch.Tensor,
    config: QuantConfig,
    kernel_config: QuantKernelConfig | None = None,
    *,
    scale: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = None,
    zero: torch.Tensor = None,
    dynamic_range: DynamicRange | tuple[DynamicRange, DynamicRange] = None,
    quant_range: QuantRange = None,
    range_bound: RangeBound = None,
    round_delta: torch.Tensor = None,
    inputs: ActivationsCache = None,
    return_with_dequant: bool = True,
    return_with_quant: bool = False,
    default_dtype: torch.dtype = torch.float16,
    develop_dtype: torch.dtype = torch.float32,
) -> QuantTensor:
    """Quantize a floating point tensor using two-step quantization if compute_dtype is specified.

    Args:
        tensor (torch.Tensor): The floating-point tensor to be quantized.
        config (QuantConfig): The quantization configuration.
        kernel_config (QuantKernelConfig, optional): The quantization kernel configuration.
        scale (torch.Tensor | tuple[torch.Tensor, torch.Tensor], optional): The scale tensor.
            Defaults to ``None``.
        zero (torch.Tensor, optional): The zero point tensor. Defaults to ``None``.
        dynamic_range (DynamicRange | tuple[DynamicRange, DynamicRange], optional): The dynamic range
            of the tensor. Defaults to ``None``.
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
    if config.compute_dtype is None:
        if isinstance(scale, (tuple, list)):
            scale = scale[0]
        if isinstance(dynamic_range, (list, tuple)):
            dynamic_range = dynamic_range[0]
        return onestep_quantize(
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
            default_dtype=default_dtype,
            develop_dtype=develop_dtype,
        )
    assert isinstance(config.compute_dtype, QuantDataType), "compute_dtype must be a QuantDataType."
    assert not config.compute_dtype.has_zero_point, "compute_dtype must not have zero point."
    assert range_bound is None, "range_bound must be None when compute_dtype is specified."
    shape, dtype = tensor.shape, tensor.dtype
    requires_grad = tensor.requires_grad or (round_delta is not None and round_delta.requires_grad)
    if requires_grad:
        assert return_with_dequant, "The dequantized tensor must be returned when requires_grad is True."
    # region step 0: check the dynamic range and scale
    if scale is None:
        scale = (None, None)
    assert isinstance(scale, (tuple, list)), "scale must be a tuple or list."
    assert len(scale) == 2, "scale must have two elements."
    compute_scale, store_scale = scale
    if dynamic_range is None:
        dynamic_range = (None, None)
    elif isinstance(dynamic_range, DynamicRange):
        assert not dynamic_range.is_set(), "dynamic_range must not be set."
        dynamic_range = (dynamic_range, dynamic_range)
    else:
        assert isinstance(dynamic_range, (tuple, list)), "dynamic_range must be a tuple or list."
        if all(d is None or not d.is_set() for d in dynamic_range):
            dynamic_range = (None, None)
    assert isinstance(dynamic_range, (tuple, list)), "dynamic_range must be a tuple or list."
    assert len(dynamic_range) == 2, "dynamic_range must have two elements."
    compute_dynamic_range, store_dynamic_range = dynamic_range
    # endregion
    if config.saturate_compute_dtype:
        compute_quant_range = QuantRange.build(config.compute_dtype)
    else:
        compute_quant_range = QuantRange.build_protective(config.compute_dtype, config.dtype)
    develop_tensor = tensor.to(dtype=develop_dtype) if dtype != develop_dtype else tensor.clone()
    # region step 1: get the compute level sale
    compute_config = config.get_compute_level_config()
    compute_group_shapes = ShapeUtils.infer_group_shapes(group_shapes=compute_config.group_shapes, shape=shape)
    compute_view_shape = ShapeUtils.infer_view_shape(shape, group_shape=compute_group_shapes[-1])
    compute_scale, _ = infer_scale_and_zero(
        tensor=develop_tensor,
        quant_dtype=compute_config.dtype,
        group_shapes=compute_group_shapes,
        scale_quant_dtypes=compute_config.get_scale_dtypes(default_dtype=default_dtype),
        exponent_scaling_level=compute_config.exponent_scaling_level,
        view_shape=compute_view_shape,
        dynamic_range=compute_dynamic_range,
        quant_range=compute_quant_range,
        scale=compute_scale,
    )
    del compute_group_shapes, compute_config
    # endregion
    # region step 2: quantize the tensor after divinding by the compute level scale
    develop_tensor = develop_tensor.view(compute_view_shape).div_(compute_scale.data).view(shape)
    develop_tensor = develop_tensor.clamp_(min=compute_quant_range.min, max=compute_quant_range.max)
    result = onestep_quantize(
        develop_tensor,
        config.get_store_level_config(),
        kernel_config=kernel_config,
        scale=store_scale,
        zero=zero,
        dynamic_range=store_dynamic_range,
        quant_range=quant_range,
        range_bound=compute_quant_range,
        round_delta=round_delta,
        inputs=inputs,
        return_with_dequant=return_with_dequant,
        return_with_quant=return_with_quant,
        default_dtype=default_dtype,
        develop_dtype=develop_dtype,
    )
    # endregion
    if return_with_dequant:
        tensor_hat = result.data
        if config.saturate_compute_dtype:
            tensor_hat = tensor_hat.clamp_(min=config.compute_dtype.max_value, max=config.compute_dtype.min_value)
        else:
            assert tensor_hat.max() <= config.compute_dtype.max_value, "Quantized tensor exceeds maximum value."
            assert tensor_hat.min() >= config.compute_dtype.min_value, "Quantized tensor exceeds minimum value."
        result._dequantized = tensor_hat.view(compute_view_shape).mul_(compute_scale.data).view(shape).to(dtype)
        assert result.data.requires_grad == requires_grad, "requires_grad must be consistent."
    if return_with_quant:
        result.scale = compute_scale.extend(result.scale)
    return result
