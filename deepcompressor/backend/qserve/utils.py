# -*- coding: utf-8 -*-
"""QServe backend utilities."""

import torch

__all__ = ["convert_to_qserve_w4x8y16_linear_weight", "convert_to_qserve_w8x8y16_linear_weight"]


def pack_w4(weight: torch.Tensor):
    assert weight.dtype == torch.int8, f"quantized weight should be torch.int8, but got {weight.dtype}."
    oc, ic = weight.shape
    # pack to M_2, [M/32, K/32, (M_8, K_4), (K_2, M_2, K_4)]
    _weight = weight.reshape(oc // 32, 2, 2, 8, ic // 32, 2, 4, 4)
    _weight = _weight.permute(1, 0, 4, 3, 6, 5, 2, 7).contiguous()
    _weight = (_weight[1] << 4) + _weight[0]
    assert _weight.shape == (oc // 32, ic // 32, 8, 4, 2, 2, 4)
    return _weight.view(oc // 32, ic // 32, 32, 16)


def convert_to_qserve_w4x8y16_linear_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    group_size: int = -1,
    subscale: torch.Tensor | None = None,
    zero_pre_scaled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Convert a weight tensor to QServe W4-X8-Y16 linear weight format.

    Args:
        weight (`torch.Tensor`):
            weight tensor to be converted.
        scale (`torch.Tensor`):
            scale tensor for the weight tensor.
        zero (`torch.Tensor`):
            zero point tensor for the weight tensor.
        group_size (`int`, *optional*, defaults to `-1`):
            quantization group size.
        subscale (`torch.Tensor` or `None`, *optional*, defaults to `None`):
            subscale tensor for the weight tensor.
        zero_pre_scaled (`bool`, *optional*, defaults to `False`):
            whether zero point tensor is pre-scaled.

    Returns:
        `tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]`:
            packed quantized weight tensor, scale tensor, zero point tensor, and subscale tensor.
    """
    dtype = weight.dtype
    assert dtype == torch.float16, "currently qserve only supports fp16."
    assert scale is not None, "scale tensor is required for quantization."
    assert zero is not None, "zero point tensor is required for quantization."
    weight = weight.to(dtype=torch.float32)
    scale = scale.to(dtype=torch.float32, device=weight.device)
    zero = zero.to(dtype=torch.float32, device=weight.device)
    oc, ic = weight.shape
    group_size = ic if group_size <= 0 else group_size
    assert group_size <= ic, "group size should be less than or equal to input channel size."
    assert ic % group_size == 0, "input channel size should be divisible by group size."
    if group_size < ic:  # per-group quantization
        assert subscale is not None, "subscale tensor is required for per-group quantization."
        subscale = subscale.to(dtype=weight.dtype, device=weight.device)
        gc = ic // group_size
        # region reshape scale and zero point
        if scale.numel() == 1:
            scale = scale.view(-1).expand(oc)
        scale = scale.reshape(oc).contiguous().view(oc, 1)
        if zero.numel() == 1:
            zero = zero.view(1, 1).expand(oc, gc)
        zero = zero.reshape(oc, gc).contiguous().view(oc, gc, 1).round_()
        if subscale.numel() == 1:
            subscale = subscale.view(1, 1).expand(oc, gc)
        subscale = subscale.reshape(oc, gc).contiguous().view(oc, gc, 1).round_()
        # endregion
        # region quantize weight tensor
        weight = weight.div_(scale).round_()
        assert weight.min() >= -128, "first-level quantized weight should be greater than or equal to -128."
        assert weight.max() <= 127, "first-level quantized weight should be less than or equal to 127."
        weight = weight.view(oc, gc, group_size)
        if not zero_pre_scaled:  # zero point is int8
            weight = weight.add_(zero)
        weight = weight.div_(subscale)
        if zero_pre_scaled:  # zero point is int4
            if zero.min() < 0:  # sint4 zero point
                zero = zero.add_(8)  # convert to uint4 zero point
            assert zero.min() >= 0, "quantized zero point should be non-negative."
            assert zero.max() <= 15, "quantized zero point should be less than 16."
            weight = weight.add_(zero)
            zero = zero.mul_(subscale)
        else:
            if weight.min() < 0:  # sint4 weight
                weight = weight.add_(8)  # convert to uint4 weight
                zero = zero.add_(8 * subscale)
        _weight = weight.mul(subscale)
        assert _weight.min() >= 0, "first-level dequantize weight should be non-negative."
        assert _weight.max() <= 255, "first-level dequantize weight should be less than 256."
        del _weight
        assert subscale.min() >= 0, "subscale should be non-negative."
        assert subscale.max() <= 127, "subscale should be less than or equal to 127."
        assert zero.min() >= 0, "quantized zero point should be non-negative."
        assert zero.max() <= 255, "quantized zero point should be less than 256."
        assert weight.min() >= 0, "quantized weight should be non-negative."
        assert weight.max() <= 15, "quantized weight should be less than 16."
        # endregion
        subscale = subscale.to(torch.int8).view(oc // 32, 4, 8, gc).permute(3, 0, 2, 1).contiguous().view(gc, oc)
        zero = -zero  # ! for group quant, qserve uses q*s+z=r instead of q*s-z=r
        zero = zero.to(torch.int8).view(oc // 32, 4, 8, gc).permute(3, 0, 2, 1).contiguous().view(gc, oc)
    else:  # per-channel quantization
        assert subscale is None, "subscale tensor is not required for per-channel quantization."
        # region reshape scale and zero point
        if scale.numel() == 1:
            scale = scale.view(-1).expand(oc)
        scale = scale.reshape(oc).contiguous().view(oc, 1)
        if zero.numel() == 1:
            zero = zero.view(-1).expand(oc)
        zero = zero.reshape(oc).contiguous().view(oc, 1)
        # endregion
        # region quantize weight tensor
        if not zero_pre_scaled:  # zero point is fp16
            weight = weight.add_(zero)
        weight = weight.div_(scale).round_()
        if zero_pre_scaled:  # zero point is int4
            zero = zero.round_()
            if zero.min() < 0:  # sint4 zero point
                zero = zero.add_(8)  # convert to uint4 zero point
            assert zero.min() >= 0, "quantized zero point should be non-negative."
            assert zero.max() <= 15, "quantized zero point should be less than 16."
            weight = weight.add_(zero)
            zero = zero.mul_(scale)
        else:
            if weight.min() < 0:  # sint4 weight
                weight = weight.add_(8)  # convert to uint4 weight
                zero = zero.add_(8 * scale)
        assert weight.min() >= 0, "quantized weight should be non-negative."
        assert weight.max() <= 15, "quantized weight should be less than 16."
        # endregion
        zero = zero.view(oc).to(dtype=dtype)
    weight = pack_w4(weight.view(oc, ic).to(torch.int8))
    weight = weight.view(oc, ic // 2)
    scale = scale.view(oc).to(dtype=dtype)
    return weight, scale, zero, subscale


def convert_to_qserve_w8x8y16_linear_weight(
    weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a weight tensor to QServe W8-X8-Y16 linear weight format.

    Args:
        weight (`torch.Tensor`):
            weight tensor to be converted.
        scale (`torch.Tensor`):
            scale tensor for the weight tensor.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`:
            packed quantized weight tensor and scale tensor.
    """
    dtype = weight.dtype
    assert dtype == torch.float16, "currently qserve only supports fp16."
    assert scale is not None, "scale tensor is required for quantization."
    weight = weight.to(dtype=torch.float32)
    scale = scale.to(dtype=torch.float32, device=weight.device)
    oc = weight.shape[0]
    if scale.numel() == 1:
        scale = scale.view(-1).expand(oc)
    scale = scale.reshape(oc).contiguous().view(oc, 1)
    weight = weight.div_(scale).round_()
    assert weight.min() >= -128, "quantized weight should be greater than or equal to -128."
    assert weight.max() <= 127, "quantized weight should be less than or equal to 127."
    weight = weight.contiguous().to(torch.int8)
    scale = scale.view(oc).to(dtype=dtype)
    return weight, scale
