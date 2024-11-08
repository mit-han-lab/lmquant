# -*- coding: utf-8 -*-
"""Quantization scale module."""

import math
import typing as tp
from dataclasses import dataclass, field

import torch

from ...data.dtype import QuantDataType
from ...data.range import DynamicRange, QuantRange, RangeBound
from ...data.scale import QuantScale
from ...data.utils import ScaleUtils
from ...data.zero import ZeroPointDomain
from .simple import simple_quantize

__all__ = ["quantize_scale", "QuantScaleInfo"]


def quantize_scale(
    s: torch.Tensor,
    /,
    *,
    quant_dtypes: tp.Sequence[QuantDataType],
    quant_spans: tp.Sequence[float],
    view_shapes: tp.Sequence[torch.Size],
) -> QuantScale:
    """Quantize the scale tensor.

    Args:
        s (`torch.Tensor`):
            The scale tensor.
        quant_dtypes (`Sequence[QuantDataType]`):
            The quantization dtypes of the scale tensor.
        quant_spans (`Sequence[float]`):
            The quantization spans of the scale tensor.
        view_shapes (`Sequence[torch.Size]`):
            The view shapes of the scale tensor.

    Returns:
        `QuantScale`:
            The quantized scale tensor.
    """
    scale = QuantScale()
    s = s.abs()
    for view_shape, quant_dtype, quant_span in zip(view_shapes[:-1], quant_dtypes[:-1], quant_spans[:-1], strict=True):
        s = s.view(view_shape)  # (#g0, rs0, #g1, rs1, #g2, rs2, ...)
        ss = s.amax(dim=list(range(1, len(view_shape), 2)), keepdim=True)  # i.e., s_dynamic_span
        ss = simple_quantize(
            ss / quant_span, has_zero_point=False, quant_dtype=quant_dtype
        )  # i.e., s_scale = s_dynamic_span / s_quant_span
        s = s / ss
        scale.append(ss)
    view_shape = view_shapes[-1]
    s = s.view(view_shape)
    if any(v != 1 for v in view_shape[1::2]):
        ss = s.amax(dim=list(range(1, len(view_shape), 2)), keepdim=True)
        ss = simple_quantize(ss / quant_spans[-1], has_zero_point=False, quant_dtype=quant_dtypes[-1])
    else:
        assert quant_spans[-1] == 1, "The last quant span must be 1."
        ss = simple_quantize(s, has_zero_point=False, quant_dtype=quant_dtypes[-1])
    scale.append(ss)
    scale.remove_zero()
    return scale


@dataclass
class QuantScaleInfo:
    # region tensor information
    tensor_view_shape: torch.Size
    tensor_quant_dtype: torch.dtype | QuantDataType
    tensor_zero_domain: ZeroPointDomain | None
    tensor_quant_range: QuantRange
    tensor_range_bound: RangeBound | None
    # endregion
    default_quant_dtype: torch.dtype | QuantDataType
    scale_view_shapes: list[torch.Size]
    scale_quant_dtypes: list[torch.dtype | QuantDataType]
    exponent_scale_level: int = field(init=False)
    zero_quant_dtype: torch.dtype | QuantDataType = field(init=False)
    # region linear scale information
    linear_tensor_quant_span: float = field(init=False)
    linear_scale_quant_dtypes: list[torch.dtype | QuantDataType] = field(init=False)
    linear_scale_view_shapes: list[torch.Size] = field(init=False)
    linear_scale_quant_spans: list[float] = field(init=False)
    # endregion
    # region exponent scale information
    exponent_tensor_quant_span: float = field(init=False)
    exponent_scale_quant_dtypes: list[torch.dtype | QuantDataType] = field(init=False)
    exponent_scale_view_shapes: list[torch.Size] = field(init=False)
    exponent_scale_quant_spans: list[float] = field(init=False)
    # endregion

    @property
    def has_zero_point(self) -> bool:
        return self.tensor_zero_domain is not None

    def __post_init__(self):
        if isinstance(self.tensor_quant_dtype, torch.dtype):
            raise NotImplementedError("torch.dtype is not supported yet.")
        self.tensor_quant_range = QuantRange.construct(
            self.tensor_quant_dtype, has_zero_point=self.has_zero_point, quant_range=self.tensor_quant_range
        )
        self.scale_quant_dtypes = ScaleUtils.infer_scale_dtypes(self.scale_quant_dtypes, self.default_quant_dtype)
        self.exponent_scale_level = ScaleUtils.infer_exponent_scale_level(self.scale_quant_dtypes)
        if self.has_zero_point:
            if self.tensor_zero_domain == ZeroPointDomain.PreScale:
                self.zero_quant_dtype = self.tensor_quant_dtype
            elif self.tensor_zero_domain == ZeroPointDomain.PostScale:
                # TODO: fix zero quant dtype (signed or unsigned)
                self.zero_quant_dtype = self.scale_quant_dtypes[-1]
                if isinstance(self.zero_quant_dtype, QuantDataType) and self.zero_quant_dtype.is_exponent:
                    self.zero_quant_dtype = self.default_quant_dtype
            else:
                raise ValueError(f"Unsupported zero point domain: {self.tensor_zero_domain}")
            self.linear_tensor_quant_span = self.tensor_quant_range.max - self.tensor_quant_range.min
            self.exponent_tensor_quant_span = 2 ** int(
                math.log2(self.tensor_quant_range.max) + int(self.tensor_quant_dtype.signed)
            )
        else:
            self.zero_quant_dtype = None
            self.linear_tensor_quant_span = self.tensor_quant_range.max
            self.exponent_tensor_quant_span = 2 ** int(math.log2(self.tensor_quant_range.max))
        if self.exponent_scale_level >= 0 and self.exponent_scale_level < len(self.scale_quant_dtypes):
            lin_s_dtypes = self.scale_quant_dtypes[: self.exponent_scale_level]
            exp_s_dtypes = self.scale_quant_dtypes[self.exponent_scale_level :]
            lin_s_view_shapes = self.scale_view_shapes[: self.exponent_scale_level]
            exp_s_view_shapes = self.scale_view_shapes[self.exponent_scale_level :]
            exp_s_spans = ScaleUtils.infer_scale_quant_spans(exp_s_dtypes)
            lin_s_spans = ScaleUtils.infer_scale_quant_spans(lin_s_dtypes, base=exp_s_spans[-1]) if lin_s_dtypes else []
        else:
            lin_s_dtypes, exp_s_dtypes = self.scale_quant_dtypes, []
            lin_s_view_shapes, exp_s_view_shapes = self.scale_view_shapes, []
            lin_s_spans, exp_s_spans = ScaleUtils.infer_scale_quant_spans(lin_s_dtypes), []
        self.linear_scale_quant_dtypes = lin_s_dtypes
        self.linear_scale_view_shapes = lin_s_view_shapes
        self.linear_scale_quant_spans = lin_s_spans
        self.exponent_scale_quant_dtypes = exp_s_dtypes
        self.exponent_scale_view_shapes = exp_s_view_shapes
        self.exponent_scale_quant_spans = exp_s_spans

    def quantize(
        self,
        *,
        # scale-based quantization related arguments
        scale: torch.Tensor | None = None,
        zero: torch.Tensor | None = None,
        # range-based quantization related arguments
        tensor: torch.Tensor | None = None,
        dynamic_range: DynamicRange | None = None,
    ) -> tuple[QuantScale, torch.Tensor]:
        """Get the quantization scale and zero point of the tensor to be quantized.

        Args:
            scale (`torch.Tensor` or `None`, *optional*, defaults to `None`):
                The scale tensor.
            zero (`torch.Tensor` or `None`, *optional*, defaults to `None`):
                The zero point tensor.
            tensor (`torch.Tensor` or `None`, *optional*, defaults to `None`):
                Ten tensor to be quantized. This is only used for range-based quantization.
            dynamic_range (`DynamicRange` or `None`, *optional*, defaults to `None`):
                The dynamic range of the tensor to be quantized.

        Returns:
            `tuple[QuantScale, torch.Tensor]`:
                The scale and the zero point.
        """
        # region step 1: get the dynamic span for range-based scale or the scale tensor
        if scale is None:
            range_based = True
            assert isinstance(tensor, torch.Tensor), "View tensor must be a tensor."
            dynamic_range = dynamic_range or DynamicRange()
            dynamic_range = dynamic_range.measure(
                tensor.view(self.tensor_view_shape),
                zero_domain=self.tensor_zero_domain,
                is_float_point=self.tensor_quant_dtype.is_float_point,
            )
            dynamic_range = dynamic_range.intersect(self.tensor_range_bound)
            dynamic_span = (dynamic_range.max - dynamic_range.min) if self.has_zero_point else dynamic_range.max
        else:
            range_based = False
            scale = scale.view(self.scale_view_shapes[-1])
            assert isinstance(scale, torch.Tensor), "Scale must be a tensor."
        # endregion
        # region step 2: get the scale
        if self.linear_scale_quant_dtypes:
            if range_based:
                linear_scale = dynamic_span / self.linear_tensor_quant_span
            elif self.exponent_scale_quant_dtypes:
                linear_scale = scale.mul(self.exponent_tensor_quant_span).div(self.linear_tensor_quant_span)
            else:
                linear_scale = scale
            lin_s = quantize_scale(
                linear_scale,
                quant_dtypes=self.linear_scale_quant_dtypes,
                quant_spans=self.linear_scale_quant_spans,
                view_shapes=self.linear_scale_view_shapes,
            )
            assert lin_s.data is not None, "Linear scale tensor is None."
            assert not lin_s.data.isnan().any(), "Linear scale tensor contains NaN."
            assert not lin_s.data.isinf().any(), "Linear scale tensor contains Inf."
        else:
            lin_s = QuantScale()
        if self.exponent_scale_quant_dtypes:
            if range_based:
                exp_scale = dynamic_span / self.exponent_tensor_quant_span
            else:
                exp_scale = scale
            if lin_s.data is not None:
                lin_s.data = lin_s.data.expand(self.linear_scale_view_shapes[-1]).reshape(self.scale_view_shapes[-1])
                exp_scale = exp_scale / lin_s.data
            exp_s = quantize_scale(
                exp_scale,
                quant_dtypes=self.exponent_scale_quant_dtypes,
                quant_spans=self.exponent_scale_quant_spans,
                view_shapes=self.exponent_scale_view_shapes,
            )
            assert exp_s.data is not None, "Exponential scale tensor is None."
            assert not exp_s.data.isnan().any(), "Exponential scale tensor contains NaN."
            assert not exp_s.data.isinf().any(), "Exponential scale tensor contains Inf."
            s = exp_s if lin_s.data is None else lin_s.extend(exp_s)
        else:
            s = lin_s
        assert s.data is not None, "Scale tensor is None."
        assert not s.data.isnan().any(), "Scale tensor contains NaN."
        assert not s.data.isinf().any(), "Scale tensor contains Inf."
        # endregion
        # region step 3: get the zero point
        if self.has_zero_point:
            if range_based:
                if self.tensor_zero_domain == ZeroPointDomain.PreScale:
                    zero = self.tensor_quant_range.min - dynamic_range.min / s.data
                else:
                    zero = self.tensor_quant_range.min * s.data - dynamic_range.min
            assert isinstance(zero, torch.Tensor), "Zero point must be a tensor."
            z = simple_quantize(zero, has_zero_point=True, quant_dtype=self.zero_quant_dtype)
        else:
            z = torch.tensor(0, dtype=s.data.dtype, device=s.data.device)
        assert not z.isnan().any(), "Zero point tensor contains NaN."
        assert not z.isinf().any(), "Zero point tensor contains Inf."
        # endregion
        return s, z
