# -*- coding: utf-8 -*-
"""Quantization kernel config."""

import typing as tp
from dataclasses import dataclass, field

import omniconfig
import torch
from omniconfig import configclass

from ...data.dtype import QuantDataType
from ...data.utils import DtypeUtils, ScaleUtils, ShapeUtils

__all__ = ["QuantConfig"]


@configclass
@dataclass
class QuantConfig:
    """Quantization configuration.

    Args:
        dtype (QuantDataType): The quantization data type. Defaults to ``None``.
        group_shapes (list[list[int]] | list[int]): The shapes for per-group quantization.
            Defaults to ``((-1, -1, -1),)``.
        group_scale_dtypes (list[torch.dtype | QuantDataType | None] | torch.dtype | QuantDataType | None): The
            quantization scale data type for per-group quantization. Defaults to ``(None,)``.
        compute_dtype (QuantDataType | None): The quantization data type for compute. Defaults to ``None``.
        compute_group_level (int): The group level for compute. Defaults to ``-1``.
        saturate_compute_dtype (bool): Whether to saturate the compute dtype. Defaults to ``False``.
    """

    dtype: QuantDataType | None = None
    # group quantization parameters
    group_shapes: tuple[tuple[int]] = ((-1, -1, -1),)
    group_scale_dtypes: tuple[torch.dtype | QuantDataType | None] = (None,)
    # compute parameters
    compute_dtype: QuantDataType | None = None
    compute_group_level: int = -1
    saturate_compute_dtype: bool = False
    # scaling parameters
    exponent_scaling_level: int = field(init=False, default=None)

    @property
    def enabled(self) -> bool:
        """Whether the quantization configuration is enabled."""
        return self.dtype is not None

    @property
    def enabled_progressive_quant(self) -> bool:
        """Whether progressive quantization is enabled."""
        return self.compute_dtype is not None

    @property
    def enabled_exponent_scaling(self) -> bool:
        """Whether exponent scaling is enabled."""
        return self.exponent_scaling_level >= len(self.group_scale_dtypes)

    @property
    def largest_group_shape(self) -> tuple[int]:
        """The shape of the largest group."""
        return self.group_shapes[0]

    @property
    def smallest_group_shape(self) -> tuple[int]:
        """The shape of the smallest group."""
        return self.group_shapes[-1]

    def __post_init__(self) -> None:
        self.group_shapes, self.group_scale_dtypes = ShapeUtils.format_group_configs(
            group_shapes=self.group_shapes, group_scale_dtypes=self.group_scale_dtypes
        )
        self.exponent_scaling_level = ScaleUtils.infer_exponent_scaling_level(self.group_scale_dtypes)
        if (self.compute_group_level + 1) % len(self.group_shapes) == 0:
            self.compute_group_level = -1
            self.compute_dtype = None
            self.saturate_compute_dtype = False
        else:
            self.compute_group_level = self.compute_group_level % len(self.group_shapes)
        if self.dtype is None:
            self.group_shapes, self.group_scale_dtypes = ((-1, -1, -1),), (None,)
            self.compute_dtype, self.compute_group_level, self.saturate_compute_dtype = None, -1, False
        if self.compute_dtype is None:
            self.compute_group_level = -1
            self.saturate_compute_dtype = False
        elif self.compute_dtype == self.dtype:
            self.compute_dtype = None
            self.compute_group_level = -1
            self.saturate_compute_dtype = False
        if self.compute_dtype is not None:
            assert isinstance(
                self.compute_dtype, QuantDataType
            ), f"compute_dtype must be QuantDataType, got {self.compute_dtype}"
            assert not self.compute_dtype.has_zero_point, "compute_dtype must not have zero point"
        if self.enabled_progressive_quant:
            assert self.compute_group_level >= 0, "compute_group_level must be >= 0 if compute_dtype is not None"
            if self.enabled_exponent_scaling:
                assert self.compute_group_level < self.exponent_scaling_level, (
                    "compute_group_level must be < exponent_scaling_level "
                    f"({self.compute_group_level} >= {self.exponent_scaling_level})"
                )

    def enabled_for(self, key: str) -> bool:
        """Whether the quantization configuration is enabled for the given module key.

        Args:
            key (str): The key.

        Returns:
            bool: Whether the quantization configuration is enabled for the given key.
        """
        return self.enabled

    def __str__(self) -> str:
        s = f"(dtype={self.dtype}"
        if self.dtype is not None:
            s += f", group={list(zip(self.group_shapes, self.group_scale_dtypes))}"
            if self.compute_dtype is not None:
                s += f", compute=[{self.compute_dtype}, level={self.compute_group_level}"
                s += f", saturate={self.saturate_compute_dtype}]"
        return s + ")"

    def _get_effective_bits(
        self, *, shape: torch.Size = torch.Size((4096, 4096)), default_dtype: torch.dtype = torch.float16
    ) -> float:
        """Get the effective bits of the quantization.

        Args:
            shape (torch.Size): The shape of the tensor to be quantized. Defaults to ``torch.Size((4096, 4096))``.
            dtype (torch.dtype): The dtype of the tensor to be quantized. Defaults to ``torch.float16``.

        Returns:
            float: The effective bits.
        """
        if self.dtype is None:
            return DtypeUtils.infer_dtype_bits(default_dtype)
        bits = self.dtype.total_bits
        group_shapes = ShapeUtils.infer_group_shapes(self.group_shapes, shape=shape)
        scale_dtypes = self.get_scale_dtypes(default_dtype=default_dtype)
        for group_shape, scale_dtype in zip(group_shapes, scale_dtypes):
            bits += DtypeUtils.infer_dtype_bits(scale_dtype) / group_shape.numel()
        if self.dtype.has_zero_point:
            bits += self.dtype.total_bits / group_shapes[-1].numel()
        return bits

    def _get_dtype_name(self, default_dtype: torch.dtype = torch.float16) -> str:
        """Get the name of the quantization data type.

        Args:
            default_dtype (torch.dtype, optional): The default dtype. Defaults to ``torch.float16``.

        Returns:
            str: The name of the quantization data type.
        """
        return DtypeUtils.infer_dtype_name(self.dtype or default_dtype)

    def _get_group_shapes_name(self, default_dtype: torch.dtype = torch.float16) -> str:
        """Get the name of the group shapes.

        Args:
            default_dtype (torch.dtype, optional): The default_dtype dtype of the input tensor.
                Defaults to ``torch.float16``.

        Returns:
            str: The name of the group shapes.
        """
        group_shapes_name = ""
        for group_shape, sdtype in zip(self.group_shapes, self.group_scale_dtypes):
            group_shapes_name += f".{ShapeUtils.infer_group_shape_name(group_shape)}"
            group_shapes_name += f".{DtypeUtils.infer_dtype_name(sdtype or default_dtype)}"
        if self.compute_dtype is not None:
            group_shapes_name += f".[{ShapeUtils.infer_group_shape_name(self.group_shapes[self.compute_group_level])}"
            group_shapes_name += f".{DtypeUtils.infer_dtype_name(self.compute_dtype)}"
            if self.saturate_compute_dtype:
                group_shapes_name += ".sat"
            group_shapes_name += "]"
        return group_shapes_name[1:]

    def generate_dirnames(
        self,
        shape: torch.Size = torch.Size((4096, 4096, 16, 16)),
        default_dtype: torch.dtype = torch.float16,
        prefix: str = "",
    ) -> list[str]:
        """Generate the directory names of the quantization configuration.

        Args:
            shape (torch.Size, optional): The shape of the input tensor.
                Defaults to ``torch.Size((4096, 4096, 16, 16))``.
            default_dtype (torch.dtype, optional): The dtype of the input tensor.
                Defaults to ``torch.float16``.

        Returns:
            list[str]: The names of the quantization configuration.
                - The number of effective bits.
                - The name of the quantization data type.
                - The name of the group shapes.
        """
        if self.dtype is None:
            dtype_str = DtypeUtils.infer_dtype_name(default_dtype)
            names = [str(DtypeUtils.infer_dtype_bits(default_dtype)), dtype_str, f"tnsr.{dtype_str}"]
        else:
            names = [
                str(int(self._get_effective_bits(shape=shape, default_dtype=default_dtype))),
                self._get_dtype_name(self.dtype or default_dtype),
                self._get_group_shapes_name(default_dtype=default_dtype),
            ]
        if prefix:
            names = [f"{prefix}.{name}" for name in names]
        return names

    def get_scale_dtypes(self, default_dtype: torch.dtype | QuantDataType) -> list[torch.dtype | QuantDataType]:
        """Get the scale dtypes for the given tensor dtype.

        Args:
            default_dtype (torch.dtype): The dtype of the tensor to be quantized.

        Returns:
            List[torch.dtype | QuantDataType]: The scale dtypes.
        """
        assert isinstance(
            default_dtype, (torch.dtype, QuantDataType)
        ), f"dtype must be torch.dtype or QuantDataType, got {default_dtype}"
        return [s_dtype or default_dtype for s_dtype in self.group_scale_dtypes]

    def get_group_shapes(self, shape: torch.Size) -> list[torch.Size]:
        """Get the group shapes for the given tensor shape.

        Args:
            shape (torch.Size): The shape of the tensor to be quantized.

        Returns:
            List[torch.Size]: The group shapes.
        """
        return ShapeUtils.infer_group_shapes(group_shapes=self.group_shapes, shape=shape)

    def get_compute_level_config(self) -> tp.Optional["QuantConfig"]:
        """Get the quantization configuration for the compute level."""
        if self.compute_dtype is None:
            return None
        return QuantConfig(
            dtype=self.compute_dtype,
            group_shapes=self.group_shapes[: self.compute_group_level + 1],
            group_scale_dtypes=self.group_scale_dtypes[: self.compute_group_level + 1],
            compute_dtype=None,
            compute_group_level=-1,
            saturate_compute_dtype=False,
        )

    def get_store_level_config(self) -> tp.Optional["QuantConfig"]:
        """Get the quantization configuration for the store level."""
        if self.compute_dtype is None:
            return self
        return QuantConfig(
            dtype=self.dtype,
            group_shapes=self.group_shapes[self.compute_group_level + 1 :],
            group_scale_dtypes=self.group_scale_dtypes[self.compute_group_level + 1 :],
            compute_dtype=None,
            compute_group_level=-1,
            saturate_compute_dtype=False,
        )

    @classmethod
    def update_get_arguments(
        cls: type["QuantConfig"],
        *,
        overwrites: dict[str, tp.Callable[[omniconfig.Arguments], None] | None] | None = None,
        defaults: dict[str, tp.Any] | None = None,
    ) -> tuple[dict[str, tp.Callable[[omniconfig.Arguments], None] | None], dict[str, tp.Any]]:
        """Get the arguments for the quantization configuration."""
        overwrites = overwrites or {}
        defaults = defaults or {}
        overwrites.setdefault(
            "group_shapes",
            lambda parser: parser.add_argument(
                "--group-shapes",
                nargs="+",
                type=lambda s: [int(n) for n in s.split(",")],
                default=defaults.get("group_shapes", [[-1, -1]]),
                help="Group shapes",
            ),
        )
        overwrites.setdefault(
            "group_scale_dtypes",
            lambda parser: parser.add_argument(
                "--group-scale-dtypes",
                nargs="+",
                type=QuantDataType.from_str,
                default=defaults.get("group_scale_dtypes", [None]),
                help="Group scale dtypes",
            ),
        )
        return overwrites, defaults
