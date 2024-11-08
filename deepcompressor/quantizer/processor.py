# -*- coding: utf-8 -*-
"""Quantizer."""

import typing as tp
from dataclasses import _MISSING_TYPE, MISSING, dataclass

import torch

from ..data.range import DynamicRange, QuantRange, RangeBound
from ..data.tensor import QuantTensor
from ..nn.patch.lowrank import LowRankBranch
from ..utils.common import tree_map
from ..utils.config import KeyEnableConfig
from ..utils.hooks import BaseInputPackager, BaseOutputPackager, BaseTensorProcessor
from .config.kernel import BaseKeyEnableQuantKernelConfig, BaseQuantKernel, BaseQuantKernelConfig
from .config.lowrank import QuantLowRankConfig
from .impl.base import QuantizerImpl
from .impl.info import QuantInfo

__all__ = ["Quantizer"]


@dataclass
class Quantizer(QuantizerImpl, BaseTensorProcessor):
    """Quantizer class.

    Args:
        config (`BasicQuantizerConfig` or `None`):
            The quantizer configuration.
        key (`str`, *optional*, defaults to `""`):
            The key of the quantizer.
        kernel (`BaseKeyEnableQuantKernelConfig` or `BaseQuantKernelConfig` or `BaseQuantKernel` or `None`,
                *optional*, defaults to `None`):
            The quantizer kernel configuration.
        channels_dim (`int` or `None`, *optional*, defaults to `None`):
            The dimension of channels.
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
        default_dtype (`torch.dtype` or `None`, *optional*, defaults to `None`):
            The default scale dtype
        develop_dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            The quantization development dtype.

        low_rank (`QuantLowRankConfig` or `None`, *optional*, defaults to `None`):
            The quantization low-rank branch configuration.
        input_packager (`BaseInputPackager` or `None`, *optional*, defaults to `None`):
            The input packager, used for unpacking and repacking the input tensor(s).
        output_packager (`BaseOutputPackager` or `None`, *optional*, defaults to `None`):
            The output packager, used for unpacking and repacking the output tensor(s).
    """

    # region keyword arguments' defaults
    kernel: BaseKeyEnableQuantKernelConfig | BaseQuantKernelConfig | BaseQuantKernel | None = None
    channels_dim: int | None = None
    scale: torch.Tensor | tp.Sequence[torch.Tensor] | None = None
    zero: torch.Tensor | None = None
    dynamic_range: DynamicRange | tp.Sequence[DynamicRange] | None = None
    range_bound: RangeBound | None = None
    quant_range: QuantRange | None = None
    default_dtype: torch.dtype | None = None
    develop_dtype: torch.dtype = torch.float32
    # endregion
    # region hook-related attributes
    low_rank: QuantLowRankConfig | None = None
    input_packager: BaseInputPackager | None = None
    output_packager: BaseOutputPackager | None = None
    # endregion

    def is_enabled_low_rank(self) -> bool:
        if self.low_rank is None:
            return False
        if isinstance(self.low_rank, KeyEnableConfig):
            return self.low_rank.is_enabled_for(self.key)
        return self.low_rank.is_enabled()

    def get_input_packager(self) -> BaseInputPackager | None:
        return self.input_packager

    def get_output_packager(self) -> BaseOutputPackager | None:
        return self.output_packager

    def process(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.quantize(tensor).data

    def quantize(
        self,
        tensor: torch.Tensor,
        /,
        *,
        return_with_dequant: bool = True,
        return_with_quant: bool = False,
        kernel: (
            BaseKeyEnableQuantKernelConfig | BaseQuantKernelConfig | BaseQuantKernel | None | _MISSING_TYPE
        ) = MISSING,
        channels_dim: int | None | _MISSING_TYPE = MISSING,
        # scale-based quantization arguments
        scale: torch.Tensor | tp.Sequence[torch.Tensor] | None | _MISSING_TYPE = MISSING,
        zero: torch.Tensor | None | _MISSING_TYPE = MISSING,
        # range-based quantization arguments
        dynamic_range: DynamicRange | tp.Sequence[DynamicRange] | None | _MISSING_TYPE = MISSING,
        range_bound: RangeBound | None | _MISSING_TYPE = MISSING,
        # other arguments
        quant_range: QuantRange | None | _MISSING_TYPE = MISSING,
        default_dtype: torch.dtype | None | _MISSING_TYPE = MISSING,
        develop_dtype: torch.dtype | _MISSING_TYPE = MISSING,
        **kwargs,
    ) -> QuantTensor:
        """Quantize a tensor.

        Args:
            tensor (`torch.Tensor`):
                The tensor to quantize.
            return_with_dequant (`bool`, *optional*, defaults to `True`):
                Whether to return the dequantized tensor.
            return_with_quant (`bool`, *optional*, defaults to `False`):
                Whether to return the quantized tensor.
            kernel (`BaseKeyEnableQuantKernelConfig` or `BaseQuantKernelConfig` or `BaseQuantKernel`
                    or `None` or `_MISSING_TYPE`, *optional*, defaults to `MISSING`):
                The quantization kernel configuration.
            channels_dim (`int` or `None` or `_MISSING_TYPE`, *optional*, defaults to `MISSING`):
                The dimension of channels.
            scale (`torch.Tensor` or `Sequence[torch.Tensor]` or `None` or `_MISSING_TYPE`,
                   *optional*, defaults to `MISSING`):
                The scale tensor.
            zero (`torch.Tensor` or `None` or `_MISSING_TYPE`, *optional*, defaults to `MISSING`):
                The zero point tensor.
            dynamic_range (`DynamicRange` or `Sequence[DynamicRange]` or `None` or `_MISSING_TYPE`,
                           *optional*, defaults to `MISSING`):
                The dynamic range.
            range_bound (`RangeBound` or `None` or `_MISSING_TYPE`, *optional*, defaults to `MISSING`):
                The dynamic range bound.
            quant_range (`QuantRange` or `None` or `_MISSING_TYPE`, *optional*, defaults to `MISSING`):
                The quantization range.
            default_dtype (`torch.dtype` or `None` or `_MISSING_TYPE`, *optional*, defaults to `MISSING`):
                The default scale dtype.
            develop_dtype (`torch.dtype` or `_MISSING_TYPE`, *optional*, defaults to `MISSING`):
                The quantization development dtype.
            **kwargs:
                Other keyword arguments for the quantization kernel. For example,
                ``inputs`` for the input tensors in GPTQ kernel,
                ``round_delta`` for the rounding delta in the RTN kernel.

        Returns:
            QuantTensor: The quantized tensor.
        """
        channels_dim = self.channels_dim if channels_dim is MISSING else channels_dim
        scale = self.scale if scale is MISSING else scale
        zero = self.zero if zero is MISSING else zero
        dynamic_range = self.dynamic_range if dynamic_range is MISSING else dynamic_range
        range_bound = self.range_bound if range_bound is MISSING else range_bound
        quant_range = self.quant_range if quant_range is MISSING else quant_range
        default_dtype = self.default_dtype if default_dtype is MISSING else default_dtype
        develop_dtype = self.develop_dtype if develop_dtype is MISSING else develop_dtype
        if kernel is MISSING:
            kernel = self.kernel
        if isinstance(kernel, BaseKeyEnableQuantKernelConfig):
            kernel = kernel.specialize_for(self.key)
        elif isinstance(kernel, KeyEnableConfig):
            kernel = kernel if kernel.is_enabled_for(self.key) else None
        assert isinstance(kernel, (BaseQuantKernel, BaseQuantKernelConfig, type(None)))
        return super().quantize(
            tensor,
            kernel=kernel,
            channels_dim=channels_dim,
            scale=scale,
            zero=zero,
            dynamic_range=dynamic_range,
            range_bound=range_bound,
            quant_range=quant_range,
            return_with_dequant=return_with_dequant,
            return_with_quant=return_with_quant,
            default_dtype=default_dtype,
            develop_dtype=develop_dtype,
            **kwargs,
        )

    def update(
        self,
        tensor_shape: torch.Size,
        default_dtype: torch.dtype | _MISSING_TYPE = MISSING,
        quant_range: QuantRange | None | _MISSING_TYPE = MISSING,
        range_bound: RangeBound | None | _MISSING_TYPE = MISSING,
    ) -> QuantInfo | None:
        """Update the quantization information.

        Args:
            tensor_shape (`torch.Size`):
                The shape of the tensor.
            default_dtype (`torch.dtype` or `None` or `_MISSING_TYPE`, *optional*, defaults to `MISSING`):
                The default scale dtype.
            quant_range (`QuantRange` or `None` or `_MISSING_TYPE`, *optional*, defaults to `MISSING`):
                The quantization range.
            range_bound (`RangeBound` or `None` or `_MISSING_TYPE`, *optional*, defaults to `MISSING`):
                The dynamic range bound

        Returns:
            `QuantInfo` or `None`:
                The updated quantization. If the quantizer is disabled, return `None`.
        """
        return super().update(
            tensor_shape,
            default_dtype=self.default_dtype if default_dtype is MISSING else default_dtype,
            quant_range=self.quant_range if quant_range is MISSING else quant_range,
            range_bound=self.range_bound if range_bound is MISSING else range_bound,
        )

    def quantize_with_low_rank(
        self,
        tensors: torch.Tensor | tp.Sequence[torch.Tensor],
        /,
        *,
        return_with_dequant: bool = True,
        return_with_quant: bool = False,
        kernel: (
            BaseKeyEnableQuantKernelConfig | BaseQuantKernelConfig | BaseQuantKernel | None | _MISSING_TYPE
        ) = MISSING,
        channels_dim: int | None | _MISSING_TYPE = MISSING,
        # scale-based quantization arguments
        scale: torch.Tensor | tp.Sequence[torch.Tensor] | None | _MISSING_TYPE = MISSING,
        zero: torch.Tensor | None | _MISSING_TYPE = MISSING,
        # range-based quantization arguments
        dynamic_range: DynamicRange | tp.Sequence[DynamicRange] | None | _MISSING_TYPE = MISSING,
        range_bound: RangeBound | None | _MISSING_TYPE = MISSING,
        # other arguments
        quant_range: QuantRange | None | _MISSING_TYPE = MISSING,
        default_dtype: torch.dtype | None | _MISSING_TYPE = MISSING,
        develop_dtype: torch.dtype | _MISSING_TYPE = MISSING,
        **kwargs,
    ) -> tuple[list[QuantTensor], list[LowRankBranch] | None]:
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]
        qkwargs = dict(
            return_with_dequant=return_with_dequant,
            return_with_quant=return_with_quant,
            kernel=kernel,
            channels_dim=channels_dim,
            scale=scale,
            zero=zero,
            dynamic_range=dynamic_range,
            range_bound=range_bound,
            quant_range=quant_range,
            default_dtype=default_dtype,
            develop_dtype=develop_dtype,
            **kwargs,
        )
        if self.is_enabled_low_rank():
            qtensors: list[QuantTensor] = []
            branches: list[LowRankBranch] = []
            if len(tensors) == 1 or self.low_rank.exclusive:
                if self.low_rank.compensate:
                    qkwargs["return_with_dequant"] = True
                    for t in tensors:
                        qt = self.quantize(t.data, **qkwargs)
                        lb = LowRankBranch(t.shape[1], t.shape[0], rank=self.low_rank.rank, weight=t.data - qt.data)
                        qtensors.append(qt)
                        branches.append(lb)
                else:
                    for t in tensors:
                        lb = LowRankBranch(t.shape[1], t.shape[0], rank=self.low_rank.rank, weight=t.data)
                        qt = self.quantize(t.data - lb.get_effective_weight(), **qkwargs)
                        qtensors.append(qt)
                        branches.append(lb)
                return qtensors, branches
            else:
                st = torch.cat([t.data for t in tensors], dim=0)
                if self.low_rank.compensate:
                    qkwargs["return_with_dequant"] = True
                    for t in tensors:
                        qt = self.quantize(t.data, **qkwargs)
                        qtensors.append(qt)
                    sl = LowRankBranch(
                        st.shape[1],
                        st.shape[0],
                        rank=self.low_rank.rank,
                        weight=st - torch.cat([q.data for q in qtensors], dim=0),
                    )
                    del st
                    i = 0
                    for t in tensors:
                        lb = LowRankBranch(t.shape[1], t.shape[0], rank=self.low_rank.rank)
                        lb.a = sl.a
                        lb.b.to(dtype=t.dtype, device=t.device)
                        lb.b.weight.copy_(sl.b.weight[i : i + t.shape[0]])
                        branches.append(lb)
                        i += t.shape[0]
                    return qtensors, branches
                else:
                    sl = LowRankBranch(st.shape[1], st.shape[0], rank=self.low_rank.rank, weight=st)
                    del st
                    i = 0
                    for t in tensors:
                        lb = LowRankBranch(t.shape[1], t.shape[0], rank=self.low_rank.rank)
                        lb.a = sl.a
                        lb.b.to(dtype=t.dtype, device=t.device)
                        lb.b.weight.copy_(sl.b.weight[i : i + t.shape[0]])
                        qt = self.quantize(t.data - lb.get_effective_weight(), **qkwargs)
                        qtensors.append(qt)
                        branches.append(lb)
                        i += t.shape[0]
                    return qtensors, branches
        else:
            return [self.quantize(t.data, **qkwargs) for t in tensors], None

    def state_dict(self, device: torch.device | str = "cpu") -> dict[str, tp.Any]:
        """Get the state dictionary of the quantizer.

        Args:
            device (`torch.device` or `str`, *optional*, defaults to `"cpu"`):
                The device to store the state dictionary.

        Returns:
            `dict[str, Any]`:
                The state dictionary.
        """
        state_dict = {}

        def _copy_to(x):
            return x.to(device).clone()

        state_dict["channels_dim"] = self.channels_dim
        state_dict["scale"] = tree_map(_copy_to, self.scale)
        state_dict["zero"] = _copy_to(self.zero) if self.zero is not None else None
        if self.dynamic_range is None:
            state_dict["dynamic_range"] = None
        elif isinstance(self.dynamic_range, DynamicRange):
            state_dict["dynamic_range"] = tree_map(_copy_to, self.dynamic_range.to_dict())
        else:
            state_dict["dynamic_range"] = tree_map(_copy_to, tuple(d.to_dict() for d in self.dynamic_range))
        state_dict["range_bound"] = self.range_bound.to_dict() if self.range_bound is not None else None
        state_dict["quant_range"] = self.quant_range.to_dict() if self.quant_range is not None else None
        return state_dict

    def load_state_dict(self, state_dict: dict[str, tp.Any], device: torch.device | str = "cpu"):
        """Load the state dictionary.

        Args:
            state_dict (`dict[str, Any]`):
                The state dictionary.
            device (`torch.device` or `str`, *optional*, defaults to `"cpu"`):
                The device to load the state dictionary.
        """

        def _move_to(x):
            return x.to(device)

        self.channels_dim = state_dict["channels_dim"]
        self.scale = tree_map(_move_to, state_dict["scale"])
        self.zero = _move_to(state_dict["zero"]) if state_dict["zero"] is not None else None
        if state_dict["dynamic_range"] is None:
            self.dynamic_range = None
        elif isinstance(state_dict["dynamic_range"], dict):
            self.dynamic_range = DynamicRange.from_dict(tree_map(_move_to, state_dict["dynamic_range"]))
        else:
            self.dynamic_range = tuple(
                DynamicRange.from_dict(tree_map(_move_to, d)) for d in state_dict["dynamic_range"]
            )
        self.range_bound = RangeBound.from_dict(state_dict["range_bound"])
        self.quant_range = QuantRange.from_dict(state_dict["quant_range"])
