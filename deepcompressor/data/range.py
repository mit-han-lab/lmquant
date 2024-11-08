# -*- coding: utf-8 -*-
"""Dynamic range calculation for quantization."""

import math
import typing as tp
from dataclasses import dataclass

import torch

from .dtype import QuantDataType
from .zero import ZeroPointDomain

__all__ = ["RangeBound", "QuantRange", "LogQuantRange", "ProtectiveQuantRange", "DynamicRange"]


@dataclass
class RangeBound:
    """Range bound data class."""

    min: float | None = None
    max: float | None = None

    def is_set(self) -> bool:
        """Return whether the range bound is set."""
        return self.min is not None or self.max is not None

    def to_dict(self) -> dict[str, tp.Any]:
        """Return the dictionary representation of the range bound."""
        return {"min": self.min, "max": self.max}

    @classmethod
    def from_dict(cls, data: dict[str, tp.Any] | None) -> tp.Optional[tp.Self]:
        """Return the range bound from the given dictionary."""
        return cls(min=data["min"], max=data["max"]) if data is not None else None


class QuantRange(RangeBound):
    """Quantization range data class."""

    def log2(self) -> "LogQuantRange":
        """Return the log-scale of the current quantization range."""
        log2_abs_min = int(math.log2(min(abs(self.min or 0), abs(self.max or 0))))
        return LogQuantRange(
            min=None,
            max=None if self.max is None else log2_abs_min,
        )

    def intersect(self, quant_dtype: QuantDataType, *, has_zero_point: bool) -> "QuantRange":
        """Return the intersection of the current quantization range and the given data type.

        Args:
            quant_dtype (`QuantDataType`):
                The quantization data type.
            has_zero_point (`bool`):
                Whether the quantization range has zero-point.

        Returns:
            `QuantRange`:
                The intersection of the current quantization range and the given data type.
        """
        max_value = quant_dtype.max_value if self.max is None else min(self.max, quant_dtype.max_value)
        min_value = quant_dtype.min_value if self.min is None else max(self.min, quant_dtype.min_value)
        if quant_dtype.signed and not has_zero_point:
            max_value = min(abs(min_value), abs(max_value))
            min_value = -max_value
        return QuantRange(min=min_value, max=max_value)

    def intersect_log2(self, quant_dtype: QuantDataType) -> "LogQuantRange":
        """Return the intersection of the current quantization range and the given data type in log2 space.

        Args:
            quant_dtype (`QuantDataType`):
                The quantization data type.

        Returns:
            `LogQuantRange`:
                The intersection of the current quantization range and the given data type in log2 space.
        """
        return self.log2().intersect_log2(quant_dtype)

    @staticmethod
    def construct(
        dtype: QuantDataType, *, has_zero_point: bool, quant_range: tp.Optional["QuantRange"] = None
    ) -> "QuantRange":
        """Return the intersection of the given quantization range and the given data type.

        Args:
            dtype (`QuantDataType`):
                The quantization data type.
            has_zero_point (`bool`):
                Whether the quantization range has zero-point.
            quant_range (`QuantRange` or `None`, *optional*, defaults to `None`):
                The extra quantization range.

        Returns:
            `QuantRange`:
                The intersection of the given quantization range and the given data type.
        """
        return (quant_range or QuantRange()).intersect(dtype, has_zero_point=has_zero_point)


class LogQuantRange(QuantRange):
    """Log-scale quantization range data class."""

    def log2(self) -> "LogQuantRange":
        """Return the log-scale of the quantization range."""
        return self

    def intersect(self, quant_dtype: QuantDataType, *, has_zero_point: bool) -> "QuantRange":
        """Return the intersection of the current quantization range and the given data type.

        Args:
            quant_dtype (`QuantDataType`):
                The quantization data type.
            has_zero_point (`bool`):
                Whether the quantization range has zero-point.

        Returns:
            `QuantRange`:
                The intersection of the current quantization range and the given data type.
        """
        raise NotImplementedError("LogQuantRange does not support intersect method")

    def intersect_log2(self, quant_dtype: QuantDataType) -> "LogQuantRange":
        """Return the intersection of the current quantization range and the given data type in log2 space.

        Args:
            quant_dtype (`QuantDataType`):
                The quantization data type.

        Returns:
            `LogQuantRange`:
                The intersection of the current quantization range and the given data type in log2 space.
        """
        max_value = (
            quant_dtype.max_exponent_value if self.max is None else min(self.max, quant_dtype.max_exponent_value)
        )
        min_value = (
            quant_dtype.min_exponent_value if self.min is None else max(self.min, quant_dtype.min_exponent_value)
        )
        return LogQuantRange(min=min_value, max=max_value)

    @staticmethod
    def construct(
        dtype: QuantDataType, quant_range: tp.Optional[tp.Union["LogQuantRange", QuantRange]] = None
    ) -> "LogQuantRange":
        """Return the intersection of the given quantization range and the given data type in log2 space.

        Args:
            dtype (`QuantDataType`):
                The quantization data type.
            quant_range (`LogQuantRange` or `QuantRange` or `None`, *optional*, defaults to `None`):
                The extra quantization range.

        Returns:
            `LogQuantRange`:
                The intersection of the given quantization range and the given data type in log2 space.
        """
        return (quant_range or LogQuantRange()).intersect_log2(dtype)


class ProtectiveQuantRange(QuantRange):
    _instances: tp.ClassVar[
        dict[tuple[QuantDataType, QuantDataType, tuple[float, float], ZeroPointDomain], "ProtectiveQuantRange"]
    ] = {}

    @staticmethod
    def construct(
        outer_dtype: QuantDataType,
        inner_dtype: QuantDataType,
        zero_domain: ZeroPointDomain | None,
        inner_quant_range: QuantRange | None = None,
    ) -> QuantRange:
        """Return the protective quantization range.

        Args:
            outer_dtype (`QuantDataType`):
                The data type of the outer level in the quantization hierarchy.
            inner_dtype (`QuantDataType`):
                The data type of the inner level in the quantization hierarchy.
            zero_domain (`ZeroPointDomain` or `None`):
                The zero-point domain.
            inner_quant_range (`QuantRange` or `None`, *optional*, defaults to `None`):
                The inner quantization range.

        Returns:
            `QuantRange`:
                The protective quantization range.
        """
        assert outer_dtype.is_integer, "outer_dtype must be integer data type"
        assert inner_dtype.is_integer, "inner_dtype must be integer data type"
        assert zero_domain is not None or outer_dtype.signed == inner_dtype.signed
        if zero_domain is None:
            return QuantRange.construct(outer_dtype, has_zero_point=False)

        inner_quant_range = QuantRange.construct(inner_dtype, has_zero_point=True, quant_range=inner_quant_range)
        qmax, qmin = int(inner_quant_range.max), int(inner_quant_range.min)  # type: ignore
        key = (outer_dtype, inner_dtype, (qmin, qmax), zero_domain)
        if key not in ProtectiveQuantRange._instances:
            outer_quant_range = QuantRange.construct(outer_dtype, has_zero_point=False)
            vrmax, vrmin = int(outer_quant_range.max), int(outer_quant_range.min)  # type: ignore
            qrmax, qrmin = int(inner_dtype.max_value), int(inner_dtype.min_value)
            vranges: set[tuple[int, int]] = set()
            for vmax in range(0, vrmax + 1):
                for vmin in range(vrmin, vmax + 1):
                    s = round((vmax - vmin) / (qmax - qmin))
                    assert s >= 0, "s must be non-negative"
                    s = 1 if s == 0 else s
                    s = min(s, vrmax)
                    if zero_domain == ZeroPointDomain.PreScale:
                        z = max(min(round(qmin - vmin / s), qrmax), qrmin)
                        m = (max(min(round(vmax / s + z), qmax), qmin) - z) * s
                        n = (max(min(round(vmin / s + z), qmax), qmin) - z) * s
                    elif zero_domain == ZeroPointDomain.PostScale:
                        z = max(min(round(qmin * s - vmin), vrmax), vrmin)
                        m = max(min(round((vmax + z) / s), qmax), qmin) * s - z
                        n = max(min(round((vmin + z) / s), qmax), qmin) * s - z
                    else:
                        raise ValueError(f"unsupported zero-point domain {zero_domain}")
                    if vrmin <= m <= vrmax and vrmin <= n <= vrmax:
                        vranges.add((vmin, vmax))
            found_pmax = None
            for pmax in range(vrmax, 0, -1):
                pmin = -pmax
                valid = True
                for vmax in range(0, pmax + 1):
                    for vmin in range(pmin, vmax + 1):
                        if (vmin, vmax) not in vranges:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    found_pmax = pmax
                    break
            assert found_pmax is not None, "failed to find the protective quantization range"
            ProtectiveQuantRange._instances[key] = ProtectiveQuantRange(min=-found_pmax, max=found_pmax)
        return ProtectiveQuantRange._instances[key]


@dataclass
class DynamicRange:
    """Dynamic range data class."""

    min: torch.Tensor | None = None
    max: torch.Tensor | None = None
    ratio: float | torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.max is None:
            assert self.min is None, "min must be None if max is None"

    def is_set(self) -> bool:
        """Return whether the dynamic range is set."""
        return self.min is not None or self.max is not None or self.ratio is not None

    def intersect(self, range_bound: RangeBound | None) -> "DynamicRange":
        """Return the intersection of the current dynamic range and the given range bound.

        Args:
            range_bound (`RangeBound` or `None`):
                The range bound.

        Returns:
            `DynamicRange`:
                The intersection of the current dynamic range and the given range bound.
        """
        assert self.max is not None, "max must be specified"
        vmax, vmin = self.max, self.min
        if range_bound is not None:
            if range_bound.max is not None:
                vmax = vmax.clamp(max=range_bound.max)
            if vmin is not None and range_bound.min is not None:
                vmin = vmin.clamp(min=range_bound.min)
        return DynamicRange(min=vmin, max=vmax)

    def measure(  # noqa: C901
        self,
        tensors: torch.Tensor | list[torch.Tensor],
        /,
        *,
        zero_domain: ZeroPointDomain | None,
        is_float_point: bool,
    ) -> "DynamicRange":
        """Return a dynamic range of the given tensor.

        Args:
            tensors (`torch.Tensor` or `list[torch.Tensor]`):
                The tensor in the shape of (#g0, gs0, #g1, gs1, ..., #gn, gsn).
            zero_domain (`ZeroPointDomain` or `None`):
                The zero-point domain.
            is_float_point (`bool`):
                Whether the data type is floating-point.

        Returns:
            `DynamicRange`:
                The dynamic range. If the max value is already specified, return the current object.
        """
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]
        if self.ratio is None and self.max is not None:  # static range
            tensor = tensors[0]
            shape = torch.Size([s if i % 2 == 0 else 1 for i, s in enumerate(tensor.shape)])
            vmax = self._format_m_(self.max, shape=shape, dtype=tensor.dtype, device=tensor.device)
            vmin = self._format_m_(self.min, shape=shape, dtype=tensor.dtype, device=tensor.device)
        else:
            if self.max is None:
                assert self.min is None, "min must be None if max is None"
            reduced = list(range(1, tensors[0].ndim, 2))
            # region step 1: determine the value range (i.e., vmax and vmin)
            if zero_domain is None:
                vmin = None
                vmax = tensors[0].abs().amax(dim=reduced, keepdim=True)
                for tensor in tensors[1:]:
                    vmax = torch.maximum(vmax, tensor.abs().amax(dim=reduced, keepdim=True).to(vmax.device))
            else:
                vmax = tensors[0].amax(dim=reduced, keepdim=True)
                for tensor in tensors[1:]:
                    vmax = torch.maximum(vmax, tensor.amax(dim=reduced, keepdim=True).to(vmax.device))
                vmin = tensors[0].amin(dim=reduced, keepdim=True)
                for tensor in tensors[1:]:
                    vmin = torch.minimum(vmin, tensor.amin(dim=reduced, keepdim=True).to(vmin.device))
                if is_float_point:  # ! we adapt the zero-point to be the mean of the data
                    vavg = tensors[0].mean(dim=reduced, keepdim=True)
                    if len(tensors) > 1:
                        for tensor in tensors[1:]:
                            vavg = vavg + tensor.mean(dim=reduced, keepdim=True).to(vavg.device)
                        vavg = vavg / len(tensors)
            # endregion
            # region step 2: scale the value range by self.ratio
            if zero_domain is None:
                if self.ratio is not None:
                    vmax = vmax * self.ratio
            else:
                assert vmin is not None, "vmin must be specified"
                if is_float_point:
                    vmag = torch.maximum(vmax - vavg, vavg - vmin)
                    if self.ratio is not None:
                        vmag = vmag * self.ratio
                    vmax = vavg + vmag
                    vmin = vavg - vmag
                else:
                    if self.ratio is not None:
                        vmin = vmin * self.ratio
                        vmax = vmax * self.ratio
                if zero_domain == ZeroPointDomain.PreScale:
                    vmax = vmax.clamp(min=0)
                    vmin = vmin.clamp(max=0)
            # endregion
            # region step 3: clamp the value range by (self.min, self.max)
            if self.max is not None:
                vmax = vmax.clamp(max=self.max.to(vmax.device))
                if vmin is not None and self.min is not None:
                    vmin = vmin.clamp(min=self.min.to(vmin.device))
            # endregion
        return DynamicRange(min=vmin, max=vmax)

    def scale(
        self, ratio: float | torch.Tensor, zero_domain: ZeroPointDomain | None, is_float_point: bool
    ) -> "DynamicRange":
        """Return new dynamic range by scaling the current range.

        Args:
            ratio (`float` or `torch.Tensor`):
                The scaling ratio.
            zero_domain (`ZeroPointDomain` or `None`):
                The zero-point domain.
            is_float_point (`bool`):
                Whether the data type is floating-point.

        Returns:
            `DynamicRange`:
                The new dynamic range.
        """
        assert ratio is not None, "ratio must be specified"
        if zero_domain is None:
            assert self.max is not None, "self.max must be specified"
            assert self.min is None, "self.min must be None for data type without zero-point"
            max_value = self.max * ratio
            min_value = None
        else:
            assert self.min is not None, "self.min must be specified"
            assert self.max is not None, "self.max must be specified"
            if is_float_point:
                centroid_value = (self.min + self.max) / 2
                vmag = (self.max - centroid_value) * ratio
                max_value = centroid_value + vmag
                min_value = centroid_value - vmag
            else:
                min_value = self.min * ratio
                max_value = self.max * ratio
            if zero_domain == ZeroPointDomain.PreScale:
                max_value = max_value.clamp(min=0)
                min_value = min_value.clamp(max=0)
        return DynamicRange(min=min_value, max=max_value)

    @staticmethod
    def construct(
        tensors: torch.Tensor | list[torch.Tensor],
        /,
        *,
        zero_domain: ZeroPointDomain | None,
        is_float_point: bool,
    ) -> "DynamicRange":
        return DynamicRange().measure(tensors, zero_domain=zero_domain, is_float_point=is_float_point)

    @staticmethod
    def _format_m_(
        value: torch.Tensor | float | None,
        *,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if value is None:
            return None
        elif isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.view(-1).to(dtype=dtype, device=device).expand(shape)
            elif value.numel() == shape.numel():
                return value.view(shape).to(dtype=dtype, device=device)
            elif value.shape[1:] == shape[1:] and value.shape[0] == 1:
                return value.to(dtype=dtype, device=device).expand(shape)
            else:
                raise ValueError(f"Invalid value shape: {value.shape}")
        else:
            return torch.full(shape, value, dtype=dtype, device=device)

    def to_dict(self) -> dict[str, tp.Any]:
        """Return the dictionary representation of the dynamic range."""
        return {"min": self.min, "max": self.max, "ratio": self.ratio}

    @classmethod
    def from_dict(cls, data: dict[str, tp.Any] | None) -> tp.Optional[tp.Self]:
        """Return the dynamic range from the given dictionary."""
        return cls(min=data["min"], max=data["max"], ratio=data["ratio"]) if data is not None else None
