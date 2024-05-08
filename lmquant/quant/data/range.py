# -*- coding: utf-8 -*-
"""Dynamic range calculation for quantization."""

import math
import typing as tp
from dataclasses import asdict, dataclass
from types import MappingProxyType

import torch

from .dtype import QDType, QuantDataType

__all__ = ["RangeBound", "DynamicRange", "QuantRange", "LogQuantRange"]


_protective_range = MappingProxyType(
    {
        (QDType.sint8, QDType.zint8): (-127, 127),
        (QDType.sint8, QDType.zint7): (-126, 126),
        (QDType.sint8, QDType.zint6): (-125, 125),
        (QDType.sint8, QDType.zint5): (-123, 123),
        (QDType.sint8, QDType.zint4): (-119, 119),
        (QDType.sint8, QDType.zint3): (-111, 111),
        (QDType.sint8, QDType.zint2): (-96, 96),
        (QDType.sint8, QDType.nint8): (-127, 127),
        (QDType.sint8, QDType.nint7): (-126, 126),
        (QDType.sint8, QDType.nint6): (-125, 125),
        (QDType.sint8, QDType.nint5): (-123, 123),
        (QDType.sint8, QDType.nint4): (-119, 119),
        (QDType.sint8, QDType.nint3): (-111, 111),
        (QDType.sint8, QDType.nint2): (-96, 96),
        (QDType.sint8, QDType.sint8): (-127, 127),
        (QDType.sint8, QDType.sint7): (-126, 126),
        (QDType.sint8, QDType.sint6): (-125, 125),
        (QDType.sint8, QDType.sint5): (-123, 123),
        (QDType.sint8, QDType.sint4): (-119, 119),
        (QDType.sint8, QDType.sint3): (-111, 111),
        (QDType.sint8, QDType.sint2): (-96, 96),
        # TODO: add more safe quantization range
    }
)


T = tp.TypeVar("T")


@dataclass
class BaseRange(tp.Generic[T]):
    """Base range data class."""

    min: T | None = None
    max: T | None = None

    def is_set(self) -> bool:
        """Return whether the range is set."""
        return self.min is not None or self.max is not None

    def to_dict(self) -> dict[str, T | None]:
        """Return the dictionary representation of the range."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, T | None]):
        """Create a range from the dictionary representation."""
        return cls(**data)


@dataclass
class RangeBound(BaseRange[float]):
    """Range bound data class."""

    @property
    def lower_bound(self) -> float | None:
        """Return the lower bound."""
        return self.min

    @property
    def upper_bound(self) -> float | None:
        """Return the upper bound."""
        return self.max


@dataclass
class QuantRange(RangeBound):
    """Quantization range data class."""

    def log2(self) -> "LogQuantRange":
        """Return the log-scale of the current quantization range."""
        log2_abs_min = int(math.log2(min(abs(self.min or 0), abs(self.max or 0))))
        return LogQuantRange(
            min=None if self.min is None else -log2_abs_min,
            max=None if self.max is None else log2_abs_min,
        )

    def intersect(self, dtype: QuantDataType) -> "QuantRange":
        """Return the intersection of the current quantization range and the given data type.

        Args:
            dtype (QuantDataType): The quantization data type.

        Returns:
            QuantRange: The intersection of the current quantization range and the given data type.
        """
        min_value = dtype.min_value if self.min is None else max(self.min, dtype.min_value)
        max_value = dtype.max_value if self.max is None else min(self.max, dtype.max_value)
        return QuantRange(min=min_value, max=max_value)

    def intersect_log2(self, dtype: QuantDataType) -> "LogQuantRange":
        """Return the intersection of the current quantization range and the given data type in log2 space.

        Args:
            dtype (QuantDataType): The quantization data type.

        Returns:
            LogQuantRange: The intersection of the current quantization range and the given data type in log2 space.
        """
        return self.log2().intersect(dtype)

    @staticmethod
    def build(dtype: QuantDataType) -> "QuantRange":
        """Return the quantization range of the given data type.

        Args:
            dtype (QuantDataType): The quantization data type.

        Returns:
            QuantRange: The quantization range.
        """
        return QuantRange(min=dtype.min_value, max=dtype.max_value)

    @staticmethod
    def build_protective(compute_dtype: "QuantDataType", storage_dtype: "QuantDataType") -> "QuantRange":
        """Get the protective quantization range.

        Args:
            compute_dtype (QuantDataType): The compute data type.
            storage_dtype (QuantDataType): The storage data type.

        Returns:
            QuantRange: The quantization range.
        """
        min_value, max_value = _protective_range[(compute_dtype, storage_dtype)]
        return QuantRange(min=min_value, max=max_value)

    @staticmethod
    def build_intersect(dtype: QuantDataType, quant_range: tp.Optional["QuantRange"] = None) -> "QuantRange":
        """Return the intersection of the given quantization range and the given data type.

        Args:
            dtype (QuantDataType): The quantization data type.
            quant_range (QuantRange | None, optional): The quantization range. Defaults to ``None``.

        Returns:
            QuantRange: The intersection of the given quantization range and the given data type.
        """
        return QuantRange.build(dtype) if quant_range is None else quant_range.intersect(dtype)


@dataclass
class LogQuantRange(QuantRange):
    """Log-scale quantization range data class."""

    def log2(self) -> "LogQuantRange":
        """Return the log-scale of the quantization range."""
        return self

    def intersect(self, dtype: QuantDataType) -> "LogQuantRange":
        """Return the intersection of the current quantization range and the given data type in log2 space.

        Args:
            dtype (QuantDataType): The quantization data type.

        Returns:
            LogQuantRange: The intersection of the current quantization range and the given data type in log2 space.
        """
        max_value = dtype.max_exponent_value if self.max is None else min(self.max, dtype.max_exponent_value)
        min_value = dtype.min_exponent_value if self.min is None else max(self.min, dtype.min_exponent_value)
        return LogQuantRange(min=min_value, max=max_value)

    @staticmethod
    def build(dtype: QuantDataType) -> QuantRange:
        """Return the quantization range in log2 space.

        Args:
            dtype (QuantDataType): The quantization data type.

        Returns:
            LogQuantRange: The quantization range in log2 space.
        """
        return LogQuantRange(min=dtype.min_exponent_value, max=dtype.max_exponent_value)

    @staticmethod
    def build_protective(compute_dtype: "QuantDataType", storage_dtype: "QuantDataType") -> "LogQuantRange":
        """Get the protective quantization range in log2 space.

        Args:
            compute_dtype (QuantDataType): The compute data type.
            storage_dtype (QuantDataType): The storage data type.

        Returns:
            LogQuantRange: The quantization range in log2 space.
        """
        raise NotImplementedError("The protective quantization range in log2 space is not implemented yet.")

    @staticmethod
    def build_intersect(dtype: QuantDataType, quant_range: tp.Optional["LogQuantRange"] = None) -> "LogQuantRange":
        """Return the intersection of the given quantization range and the given data type in log2 space.

        Args:
            dtype (QuantDataType): The quantization data type.
            quant_range (LogQuantRange | None, optional): The quantization range. Defaults to ``None``.

        Returns:
            LogQuantRange: The intersection of the given quantization range and the given data type in log2 space.
        """
        return LogQuantRange.build(dtype) if quant_range is None else quant_range.intersect_log2(dtype)


@dataclass
class DynamicRange(BaseRange[torch.Tensor]):
    """Dynamic range data class."""

    ratio: float | torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.max is None:
            assert self.min is None, "min must be None if max is None"
        else:
            assert self.ratio is None, "ratio must be None if max is specified"

    def is_set(self) -> bool:
        """Return whether the dynamic range is set."""
        return super().is_set() or self.ratio is not None

    def measure(
        self,
        tensors: torch.Tensor | list[torch.Tensor],
        /,
        *,
        has_zero_point: bool = False,
        is_float: bool = False,
    ) -> "DynamicRange":
        """Return a dynamic range of the given tensor.

        Args:
            tensor (torch.Tensor): The tensor in the shape of (#g0, gs0, #g1, gs1, ..., #gn, gsn).
            qdtype (QuantDataType): The quantization data type.
            has_zero_point (bool): Whether the data type has non-zero zero-point.
            is_float (bool): Whether the data type is floating-point.

        Returns:
            DynamicRange: The dynamic range. If the max value is already specified, return the current object.
        """
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]
        if self.max is not None:
            tensor = tensors[0]
            shape = torch.Size([s if i % 2 == 0 else 1 for i, s in enumerate(tensor.shape)])
            max_value = self._format_m_(self.max, shape=shape, dtype=tensor.dtype, device=tensor.device)
            if self.min is not None:
                min_value = self._format_m_(self.min, shape=shape, dtype=tensor.dtype, device=tensor.device)
            else:
                min_value = None
        else:
            assert self.min is None, "min must be None if max is None"
            reduced = list(range(1, tensors[0].ndim, 2))
            # region step 1: determine the value range (i.e., vmax and vmin)
            if has_zero_point:
                vmax = tensors[0].amax(dim=reduced, keepdim=True)
                for tensor in tensors[1:]:
                    vmax = torch.maximum(vmax, tensor.amax(dim=reduced, keepdim=True).to(vmax.device))
                vmin = tensors[0].amin(dim=reduced, keepdim=True)
                for tensor in tensors[1:]:
                    vmin = torch.minimum(vmin, tensor.amin(dim=reduced, keepdim=True).to(vmin.device))
                if is_float:  # ! we adapt the zero-point to be the mean of the data
                    centroid_value = tensors[0].mean(dim=reduced, keepdim=True)
                    if len(tensors) > 1:
                        for tensor in tensors[1:]:
                            centroid_value = centroid_value + tensor.mean(dim=reduced, keepdim=True).to(
                                centroid_value.device
                            )
                        centroid_value = centroid_value / len(tensors)
            else:
                vmin = None
                vmax = tensors[0].abs().amax(dim=reduced, keepdim=True)
                for tensor in tensors[1:]:
                    vmax = torch.maximum(vmax, tensor.abs().amax(dim=reduced, keepdim=True).to(vmax.device))
            # endregion
            # region step 2: scale the value range by range_scale
            if has_zero_point:
                if is_float:
                    vmag = torch.maximum(vmax - centroid_value, centroid_value - vmin)
                    if self.ratio is not None:
                        vmag = vmag * self.ratio
                    max_value = centroid_value + vmag
                    min_value = centroid_value - vmag
                else:
                    min_value, max_value = vmin, vmax
                    if self.ratio is not None:
                        min_value = min_value * self.ratio
                        max_value = max_value * self.ratio
            else:
                min_value, max_value = vmin, vmax
                if self.ratio is not None:
                    max_value = max_value * self.ratio
            # endregion
        return DynamicRange(min=min_value, max=max_value)

    def scale(self, ratio: float | torch.Tensor, has_zero_point: bool, is_float: bool) -> "DynamicRange":
        """Return new dynamic range by scaling the current range.

        Args:
            ratio (float | torch.Tensor): The scaling ratio.

        Returns:
            DynamicRange: The new dynamic range.
        """
        assert ratio is not None, "ratio must be specified"
        if has_zero_point:
            assert self.min is not None, "self.min must be specified"
            assert self.max is not None, "self.max must be specified"
            if is_float:
                centroid_value = (self.min + self.max) / 2
                vmag = (max_value - centroid_value) * ratio
                max_value = centroid_value + vmag
                min_value = centroid_value - vmag
            else:
                min_value = self.min * ratio
                max_value = self.max * ratio
        else:
            assert self.min is None, "self.min must be None for data type without zero-point"
            max_value = self.max * ratio
            min_value = None
        return DynamicRange(min=min_value, max=max_value)

    @staticmethod
    def build(tensors: torch.Tensor | list[torch.Tensor], /, *, has_zero_point: bool, is_float: bool) -> "DynamicRange":
        return DynamicRange().measure(tensors, has_zero_point=has_zero_point, is_float=is_float)

    @staticmethod
    def _format_m_(
        value: torch.Tensor | float,
        *,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            if value.numel() == shape.numel():
                return value.view(shape).to(dtype=dtype, device=device)
            elif value.numel() == 1:
                return value.view(-1).to(dtype=dtype, device=device).expand(shape)
            else:
                raise ValueError(f"Invalid value shape: {value.shape}")
        else:
            return torch.full(shape, value, dtype=dtype, device=device)
