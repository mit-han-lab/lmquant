# -*- coding: utf-8 -*-
"""Round-to-nearest (RTN) quantization module."""

import torch

from ...data.dtype import QuantDataType
from ...data.range import QuantRange
from ...data.zero import ZeroPointDomain
from ..config.kernel import BaseQuantKernel
from ..impl.simple import simple_quantize

__all__ = ["QuantRtnKernel", "rtn_quantize"]


class QuantRtnKernel(BaseQuantKernel):
    """Round-to-nearest (RTN) Quantization kernel."""

    def quantize(
        self,
        tensor: torch.Tensor,
        *,
        view_shape: torch.Size,
        quant_dtype: QuantDataType,
        zero_domain: ZeroPointDomain | None,
        scale: torch.Tensor,
        zero: torch.Tensor,
        quant_range: QuantRange | None = None,
        round_delta: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Quantize the tensor.

        Args:
            tensor (`torch.Tensor`):
                The tensor to quantize.
            view_shape (`torch.Size`):
                The view shape when quantizing the tensor.
            quant_dtype (`QuantDataType`):
                The quantization data type.
            zero_domain (`ZeroPointDomain` or `None`):
                The zero point domain.
            scale (`torch.Tensor`):
                The scale tensor.
            zero (`torch.Tensor`):
                The zero point tensor.
            quant_range (`QuantRange` or `None`, *optional*, defaults to `None`):
                The quantization range.
            round_delta (`torch.Tensor` or `None`, *optional*, defaults to `None`):
                The rounding delta.
            **kwargs: Other keyword arguments.

        Returns:
            `torch.Tensor`:
                The quantized tensor in the shape of ``view_shape``.
        """
        return rtn_quantize(
            tensor,
            view_shape=view_shape,
            quant_dtype=quant_dtype,
            zero_domain=zero_domain,
            scale=scale,
            zero=zero,
            quant_range=quant_range,
            round_delta=round_delta,
        )


def rtn_quantize(
    tensor: torch.Tensor,
    *,
    view_shape: torch.Size,
    quant_dtype: QuantDataType,
    zero_domain: ZeroPointDomain | None,
    scale: torch.Tensor,
    zero: torch.Tensor,
    quant_range: QuantRange | None = None,
    round_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Quantize the tensor using the RTN quantization kernel.

    Args:
        tensor (`torch.Tensor`):
            The tensor to quantize.
        view_shape (`torch.Size`):
            The view shape when quantizing the tensor.
        quant_dtype (`QuantDataType`):
            The quantization data type.
        zero_domain (`ZeroPointDomain` or `None`):
            The zero point domain.
        scale (`torch.Tensor`):
            The scale tensor.
        zero (`torch.Tensor`):
            The zero point tensor.
        quant_range (`QuantRange` or `None`, *optional*, defaults to `None`):
            The quantization range.
        round_delta (`torch.Tensor` or `None`, *optional*, defaults to `None`):
            The rounding delta.

    Returns:
        `torch.Tensor`:
            The quantized tensor in the shape of ``view_shape``.
    """
    qtensor = tensor.view(view_shape)
    round_delta = round_delta.view(view_shape) if round_delta is not None else None
    if zero_domain == ZeroPointDomain.PostScale:
        qtensor = qtensor.add_(zero)
    qtensor = qtensor.div(scale)
    if zero_domain == ZeroPointDomain.PreScale:
        qtensor = qtensor.add_(zero)
    qtensor = simple_quantize(
        qtensor,
        quant_dtype=quant_dtype,
        has_zero_point=zero_domain is not None,
        quant_range=quant_range,
        round_delta=round_delta,
    )
    return qtensor
