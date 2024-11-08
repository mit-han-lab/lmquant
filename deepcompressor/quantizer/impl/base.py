# -*- coding: utf-8 -*-
"""Quantizer."""

import typing as tp
from dataclasses import dataclass, field

import torch

from ...data.range import DynamicRange, QuantRange, RangeBound
from ...data.scale import QuantScale
from ...data.tensor import QuantTensor
from ...data.zero import ZeroPointDomain
from ...utils.config import KeyEnableConfig
from ..config.base import BaseQuantizerConfig
from ..config.kernel import BaseQuantKernel, BaseQuantKernelConfig
from ..kernel.rtn import QuantRtnKernel
from .info import QuantInfo

__all__ = ["QuantizerImpl"]


@dataclass
class QuantizerImpl:
    """Quantizer implementation.

    Args:
        config (`BasicQuantizerConfig` or `None`):
            The quantizer configuration.
        key (`str`, *optional*, defaults to `""`):
            The key of the quantizer.

    Attributes:
        info (`QuantInfo` or `None`):
            The quantization information.
    """

    config: BaseQuantizerConfig | None
    key: str = ""
    info: QuantInfo | None = field(init=False, default=None)

    def is_enabled(self) -> bool:
        """Whether the quantizer is enabled."""
        if self.config is None:
            return False
        if isinstance(self.config, KeyEnableConfig):
            return self.config.is_enabled_for(self.key)
        return self.config.is_enabled()

    def quantize(
        self,
        tensor: torch.Tensor,
        *,
        kernel: BaseQuantKernel | BaseQuantKernelConfig | None = None,
        channels_dim: int | None = None,
        # scale-based quantization arguments
        scale: torch.Tensor | tp.Sequence[torch.Tensor] | None = None,
        zero: torch.Tensor | None = None,
        # range-based quantization arguments
        dynamic_range: DynamicRange | tp.Sequence[DynamicRange] | None = None,
        # other arguments
        range_bound: RangeBound | None = None,
        quant_range: QuantRange | None = None,
        return_with_dequant: bool = True,
        return_with_quant: bool = False,
        default_dtype: torch.dtype | None = torch.float16,
        develop_dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> QuantTensor:
        """Quantize a floating point tensor.

        Args:
            tensor (`torch.Tensor`):
                The floating-point tensor to be quantized.
            kernel (`QuantKernel` or `QuantKernelConfig` or `None`, *optional*, defaults to `None`):
                The quantization kernel or its configuration.
            channels_dim (`int` or `None`, *optional*, defaults to `None`):
                The dimension of (input) channels.
            scale (`torch.Tensor` or `Sequence[torch.Tensor]` or `None`, *optional*, defaults to `None`):
                The scale tensor.
            zero (`torch.Tensor` or `None`, *optional*, defaults to `None`):
                The zero point tensor.
            dynamic_range (`DynamicRange` or `Sequence[DynamicRange]` or `None`, *optional*, defaults to `None`):
                The dynamic range.
            range_bound (`RangeBound` or `None`, *optional*, defaults to `None`):
                The dynamic range bound.
            quant_range (`QuantRange` or `None`, *optional*, defaults to `None`):
                The quantization range.
            return_with_dequant (`bool`, *optional*, defaults to `True`):
                Whether to return with dequantized tensor.
            return_with_quant (`bool`, *optional*, defaults to `False`):
                Whether to return with quantized tensor.
            default_dtype (`torch.dtype` or `None`, *optional*, defaults to `torch.float16`):
                The default dtype for scale.
            develop_dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                The develop dtype.
            **kwargs:
                Other keyword arguments for the quantization kernel. For example,
                ``inputs`` for the input tensors in GPTQ kernel,
                ``round_delta`` for the rounding delta in the RTN kernel.

        Returns:
            `QuantTensor`:
                The quantized tensor.
        """
        shape = tensor.shape
        if channels_dim is not None:
            tensor = tensor.reshape(-1, *shape[channels_dim:])
        round_delta = kwargs.pop("round_delta", None)
        if round_delta is not None:
            round_delta = round_delta.view(-1, *shape[channels_dim:])
        result = self._quantize(
            tensor,
            kernel=kernel,
            scale=scale,
            zero=zero,
            dynamic_range=dynamic_range,
            range_bound=range_bound,
            quant_range=quant_range,
            round_delta=round_delta,
            return_with_dequant=return_with_dequant,
            return_with_quant=return_with_quant,
            default_dtype=default_dtype or tensor.dtype,
            develop_dtype=develop_dtype,
            **kwargs,
        )
        if result.data is not None:
            result._dequantized = result.data.view(shape)
        if result.qdata is not None:
            result._quantized = result.qdata.view(shape)
        return result

    def _quantize(  # noqa: C901
        self,
        tensor: torch.Tensor,
        *,
        kernel: BaseQuantKernel | BaseQuantKernelConfig | None = None,
        # scale-based quantization arguments
        scale: torch.Tensor | tp.Sequence[torch.Tensor | None] | None = None,
        zero: torch.Tensor | None = None,
        # range-based quantization arguments
        dynamic_range: DynamicRange | tp.Sequence[DynamicRange | None] | None = None,
        # other arguments
        range_bound: RangeBound | None = None,
        quant_range: QuantRange | None = None,
        round_delta: torch.Tensor | None = None,
        return_with_dequant: bool = True,
        return_with_quant: bool = False,
        default_dtype: torch.dtype = torch.float16,
        develop_dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> QuantTensor:
        """Quantize a floating point tensor.

        Args:
            tensor (`torch.Tensor`):
                The floating-point tensor to be quantized.
            kernel (`QuantKernel` or `QuantKernelConfig` or `None`, *optional*, defaults to `None`):
                The quantization kernel or its configuration.
            scale (`torch.Tensor` or `Sequence[torch.Tensor]` or `None`, *optional*, defaults to `None`):
                The scale tensor.
            zero (`torch.Tensor` or `None`, *optional*, defaults to `None`):
                The zero point tensor.
            dynamic_range (`DynamicRange` or `Sequence[DynamicRange]` or `None`, *optional*, defaults to `None`):
                The dynamic range.
            range_bound (`RangeBound` or `None`, *optional*, defaults to `None`):
                The dynamic range bound.
            quant_range (`QuantRange` or `None`, *optional*, defaults to `None`):
                The quantization range.
            return_with_dequant (`bool`, *optional*, defaults to `True`):
                Whether to return with dequantized tensor.
            return_with_quant (`bool`, *optional*, defaults to `False`):
                Whether to return with quantized tensor.
            default_dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
                The default dtype for scale.
            develop_dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                The develop dtype.
            **kwargs:
                Other keyword arguments for the quantization kernel. For example,
                ``inputs`` for the input tensors in GPTQ kernel,
                ``round_delta`` for the rounding delta in the RTN kernel.

        Returns:
            `QuantTensor`:
                The quantized tensor.
        """
        shape, dtype = tensor.shape, tensor.dtype
        self.update(shape, default_dtype, quant_range, range_bound)
        if self.info is None or self.info.num_steps == 0:
            return QuantTensor(dequantized=tensor, quantized=tensor, view_shape=shape)
        # region check scale and dynamic_range arguments
        num_steps = self.info.num_steps
        if scale is None:
            scale = (None,) * num_steps
        elif not isinstance(scale, tp.Sequence):
            scale = (scale,)
        if dynamic_range is None:
            dynamic_range = (None,) * num_steps
        elif isinstance(dynamic_range, DynamicRange):
            if not dynamic_range.is_set():
                dynamic_range = (None,) * num_steps
            else:
                dynamic_range = (dynamic_range,)
        assert isinstance(scale, (tuple, list)), "scale must be a tuple or list."
        assert len(scale) == num_steps, "scale must have the same length as infos."
        assert isinstance(dynamic_range, (tuple, list)), "dynamic_range must be a tuple or list."
        assert len(dynamic_range) == num_steps, "dynamic_range must have the same length as infos."
        # endregion
        # region compute and quantize the scales and zero point for quantization
        quant_scale = QuantScale()
        develop_tensor = tensor.to(dtype=develop_dtype) if dtype != develop_dtype else tensor.clone()
        for step, (step_info, step_scale, step_dynamic_range) in enumerate(
            zip(self.info.steps, scale, dynamic_range, strict=True)
        ):
            step_scale, step_zero = step_info.scale.quantize(
                scale=step_scale,
                zero=None if step < num_steps - 1 else zero,
                tensor=develop_tensor,
                dynamic_range=step_dynamic_range,
            )
            quant_scale.append(step_scale)
            if step < num_steps - 1:
                step_quant_range = step_info.tensor_quant_range
                develop_tensor = develop_tensor.view(step_info.tensor_view_shape).div_(step_scale.data).view(shape)
                develop_tensor = develop_tensor.clamp_(min=step_quant_range.min, max=step_quant_range.max)
        quant_zero = step_zero
        # endregion
        # region quantize the tensor
        assert isinstance(step_scale, QuantScale), "The last scale must be a QuantScale."
        assert isinstance(step_zero, torch.Tensor), "The last zero point must be a tensor."
        if round_delta is not None:
            if round_delta.shape[0] == 1:
                round_delta = round_delta.view(1, 1, *step_info.tensor_view_shape[2:])
            else:
                round_delta = round_delta.view(step_info.tensor_view_shape)
        if isinstance(kernel, BaseQuantKernelConfig):
            kernel = kernel.build()
        kernel = kernel or QuantRtnKernel()
        develop_tensor = kernel.quantize(
            tensor=develop_tensor,
            view_shape=step_info.tensor_view_shape,
            quant_dtype=step_info.quant_dtype,
            zero_domain=step_info.zero_domain,
            scale=step_scale.data,
            zero=step_zero,
            quant_range=step_info.quant_range,
            range_bound=step_info.range_bound,
            round_delta=round_delta,
            **kwargs,
        )
        assert not develop_tensor.isnan().any(), "Quantized tensor contains NaN."
        assert not develop_tensor.isinf().any(), "Quantized tensor contains Inf."
        # endregion
        # region update the quantized tensor
        quantized = None
        if return_with_quant:
            quantized = develop_tensor.detach()
            if return_with_dequant:
                quantized = develop_tensor.clone()
            quantized = develop_tensor.view(shape)
        # endregion
        # region update the dequantized tensor
        dequantized = None
        if return_with_dequant:
            dequantized = develop_tensor
            if self.config.zero_domain == ZeroPointDomain.PreScale:
                dequantized = dequantized.sub_(step_zero)
            dequantized = dequantized.mul_(step_scale.data)
            if self.config.zero_domain == ZeroPointDomain.PostScale:
                dequantized = dequantized.sub_(step_zero)
            for step in range(num_steps - 2, -1, -1):
                step_info, step_scale = self.info.get_child(step), quant_scale.get_child(step)
                step_min, step_max = step_info.quant_dtype.min_value, step_info.quant_dtype.max_value
                if self.info.needs_dequant_saturation or step < num_steps - 2:
                    dequantized = dequantized.clamp_(min=step_min, max=step_max)
                else:
                    assert dequantized.max() <= step_max, "Quantized tensor exceeds maximum value."
                    assert dequantized.min() >= step_min, "Quantized tensor exceeds minimum value."
                dequantized = dequantized.view(step_info.tensor_view_shape).mul_(step_scale.data)
            dequantized = dequantized.view(shape).to(dtype=dtype)
        # endregion
        return QuantTensor(
            dequantized=dequantized,
            quantized=quantized,
            scale=quant_scale if return_with_quant else None,
            zero=quant_zero if return_with_quant else None,
            view_shape=self.info.steps[-1].tensor_view_shape if return_with_quant else None,
        )

    def update(
        self,
        tensor_shape: torch.Size,
        default_dtype: torch.dtype | None,
        quant_range: QuantRange | None,
        range_bound: RangeBound | None,
    ) -> QuantInfo | None:
        """Update the quantization information.

        Args:
            tensor_shape (`torch.Size`):
                The shape of the tensor.
            default_dtype (`torch.dtype` or `None`):
                The default data type of the scale.
            quant_range (`QuantRange` or `None`):
                The quantization range.
            range_bound (`RangeBound` or `None`):
                The range bound.

        Returns:
            `QuantInfo` or `None`:
                The updated quantization. If the quantizer is disabled, return `None`.
        """
        if not self.is_enabled():
            self.info = None
        else:
            config = self.config.decompose()
            assert default_dtype is not None, "default_dtype must be set."
            if self.info is None or self.info.is_outdated(
                config, tensor_shape, default_dtype, quant_range, range_bound
            ):
                self.info = QuantInfo.construct(
                    config, tensor_shape, default_dtype, quant_range=quant_range, range_bound=range_bound
                )
        return self.info
