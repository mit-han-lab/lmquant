# -*- coding: utf-8 -*-
"""Quantization kernel config."""

import typing as tp
from abc import abstractmethod
from dataclasses import dataclass, field

import omniconfig
import torch
from omniconfig import configclass

from ...data.dtype import QuantDataType
from ...data.utils import DtypeUtils, ScaleUtils, ShapeUtils
from ...data.zero import ZeroPointDomain
from ...utils.config import EnableConfig

__all__ = [
    "BaseQuantizerConfig",
    "DecomposedQuantizerConfig",
    "QuantizerConfig",
    "ProgressiveQuantizerConfig",
]


class BaseQuantizerConfig(EnableConfig):
    """Base Quantizer configuration."""

    @property
    @abstractmethod
    def quant_dtype(self) -> QuantDataType | None:
        """The quantization data type."""
        ...

    @property
    @abstractmethod
    def zero_domain(self) -> ZeroPointDomain | None:
        """The zero-point domain."""
        ...

    @property
    @abstractmethod
    def largest_group_shape(self) -> tp.Sequence[int]:
        """The shape of the largest group."""
        ...

    @property
    @abstractmethod
    def smallest_group_shape(self) -> tp.Sequence[int]:
        """The shape of the smallest group."""
        ...

    def is_enabled(self) -> bool:
        """Whether the quantization configuration is enabled."""
        return self.quant_dtype is not None

    @abstractmethod
    def decompose(self) -> "DecomposedQuantizerConfig":
        """Decompose the configuration to a list of simple configurations."""
        ...

    def generate_dirnames(
        self,
        *,
        prefix: str = "",
        shape: torch.Size | tuple[int, ...] = (4096, 4096),
        default_dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> list[str]:
        """Generate the directory names of the quantization configuration.

        Args:
            prefix (`str`, *optional*, defaults to `""`):
                The prefix for the directory names.
            shape (`torch.Size` or `tuple[int, ...]`, *optional*, defaults to `(4096, 4096)`):
                The shape of the tensor to be quantized.

        Returns:
            `list[str]`:
                The names of the quantization configuration.
                    - The number of effective bits.
                    - The name of the quantization data type.
                    - The name of the group shapes.
        """
        return self.decompose().generate_dirnames(
            prefix=prefix, shape=torch.Size(shape), default_dtype=default_dtype, **kwargs
        )


@dataclass(frozen=True)
class DecomposedQuantizerConfig(BaseQuantizerConfig):
    steps: tuple["QuantizerConfig", ...]
    needs_dequant_saturation: bool = False

    @property
    def quant_dtype(self) -> QuantDataType | None:
        return self.steps[-1].dtype if self.steps else None

    @property
    def zero_domain(self) -> ZeroPointDomain | None:
        return self.steps[-1].zero_point if self.steps else None

    @property
    def largest_group_shape(self) -> tp.Sequence[int]:
        return self.steps[0].largest_group_shape if self.steps else (-1, -1, -1)

    @property
    def smallest_group_shape(self) -> tp.Sequence[int]:
        return self.steps[-1].smallest_group_shape if self.steps else (-1, -1, -1)

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    def decompose(self) -> "DecomposedQuantizerConfig":
        return self

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, DecomposedQuantizerConfig):
            return False
        if self.num_steps != value.num_steps:
            return False
        for rhs, lhs in zip(self.steps, value.steps, strict=True):
            # ! we only compare the dtype, group_shapes, and scale_dtypes
            if rhs.dtype != lhs.dtype:
                return False
            if rhs.group_shapes != lhs.group_shapes:
                return False
            if rhs.scale_dtypes != lhs.scale_dtypes:
                return False
        if self.num_steps > 1:
            if self.needs_dequant_saturation != value.needs_dequant_saturation:
                return False
        return True

    def _get_effective_bits(
        self, *, shape: torch.Size | tuple[int, ...] = (4096, 4096), default_dtype: torch.dtype = torch.float16
    ) -> float:
        """Get the effective bits of the quantization.

        Args:
            shape (`torch.Size` or `tuple[int, ...]`, *optional*, defaults to `(4096, 4096)`):
                The shape of the tensor to be quantized.
            dtype (torch.dtype, *optional*, defaults to `torch.float16`):
                The dtype of the tensor to be quantized.

        Returns:
            `float`:
                The effective bits.
        """
        shape = torch.Size(shape)
        if self.quant_dtype is None:
            return DtypeUtils.infer_dtype_bits(default_dtype)
        bits = self.quant_dtype.total_bits
        for step_config in self.steps:
            group_shapes = ShapeUtils.infer_group_shapes(step_config.group_shapes, shape=shape)
            scale_dtypes = ScaleUtils.infer_scale_dtypes(step_config.scale_dtypes, default_dtype=default_dtype)
            for group_shape, scale_dtype in zip(group_shapes, scale_dtypes, strict=True):
                bits += DtypeUtils.infer_dtype_bits(scale_dtype) / group_shape.numel()
        if self.zero_domain == ZeroPointDomain.PreScale:
            bits += self.quant_dtype.total_bits / group_shapes[-1].numel()
        elif self.zero_domain == ZeroPointDomain.PostScale:
            bits += DtypeUtils.infer_dtype_bits(scale_dtype) / group_shape.numel()
        return bits

    def _get_dtype_name(self, default_dtype: torch.dtype = torch.float16) -> str:
        """Get the name of the quantization data type.

        Args:
            default_dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
                The default_dtype dtype of the input tensor.

        Returns:
            `str`:
                The name of the quantization data type.
        """
        if self.quant_dtype is None:
            return DtypeUtils.infer_dtype_name(default_dtype)
        name = DtypeUtils.infer_dtype_name(self.quant_dtype)
        if self.zero_domain == ZeroPointDomain.PreScale:
            name += ".z"
        elif self.zero_domain == ZeroPointDomain.PostScale:
            name += ".zp"
        return name

    def _get_group_shapes_name(self, default_dtype: torch.dtype = torch.float16) -> str:
        """Get the name of the group shapes.

        Args:
            default_dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
                The default_dtype dtype of the input tensor.

        Returns:
            str: The name of the group shapes.
        """
        if self.quant_dtype is None:
            return f"tnsr.{DtypeUtils.infer_dtype_name(default_dtype)}"
        num_steps = len(self.steps)
        names = []
        step_default_dtype = default_dtype
        for step, step_config in enumerate(self.steps):
            step_names = []
            for group_shape, sdtype in zip(step_config.group_shapes, step_config.scale_dtypes, strict=True):
                name = f"{ShapeUtils.infer_group_shape_name(group_shape)}"
                name += f".{DtypeUtils.infer_dtype_name(sdtype or step_default_dtype)}"
                step_names.append(name)
            step_name = ".".join(reversed(step_names))
            names.append(f"[{step_name}]" if step < num_steps - 2 else step_name)
            step_default_dtype = step_config.dtype
            assert step_default_dtype is not None, "step_default_dtype must not be None"
        return ".".join(reversed(names))

    def generate_dirnames(
        self,
        *,
        prefix: str = "",
        shape: torch.Size | tuple[int, ...] = (4096, 4096),
        default_dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> list[str]:
        """Generate the directory names of the quantization configuration.

        Args:
            prefix (`str`, *optional*, defaults to `""`):
                The prefix for the directory names.
            shape (`torch.Size` or `tuple[int, ...]`, *optional*, defaults to `(4096, 4096)`):
                The shape of the tensor to be quantized.
            default_dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
                The dtype of the tensor to be quantized.

        Returns:
            `list[str]`:
                The names of the quantization configuration.
                    - The number of effective bits.
                    - The name of the quantization data type.
                    - The name of the group shapes.
        """
        shape = torch.Size(shape)
        bits_str = str(int(self._get_effective_bits(shape=shape, default_dtype=default_dtype)))
        dtype_str = self._get_dtype_name(default_dtype=default_dtype)
        group_str = self._get_group_shapes_name(default_dtype=default_dtype)
        names = [bits_str, dtype_str, group_str]
        if prefix:
            names = [f"{prefix}.{name}" for name in names]
        return names


@configclass
@dataclass
class QuantizerConfig(BaseQuantizerConfig):
    """Quantizer configuration.

    Args:
        dtype (`QuantDataType` or `None`, *optional*, defaults to `None`):
            The quantization data type.
        zero_point (`ZeroPointDomain` or `None`, *optional*, defaults to `None`):
            The zero-point domain.
        group_shapes (`Sequence[Sequence[int]]`, *optional*, defaults to `((-1, -1, -1),)`):
            The shapes for per-group quantization.
        scale_dtypes (`Sequence[torch.dtype | QuantDataType | None]`, *optional*, defaults to `(None,)`):
            The quantization scale data type for per-group quantization.
    """

    dtype: QuantDataType | None = None
    zero_point: ZeroPointDomain | None = None
    group_shapes: tp.Sequence[tp.Sequence[int]] = field(
        default=((-1, -1, -1),),
        metadata={omniconfig.ARGPARSE_KWARGS: {"nargs": "+", "type": lambda s: [int(n) for n in s.split(",")]}},
    )
    scale_dtypes: tp.Sequence[torch.dtype | QuantDataType | None] = field(
        default=(None,), metadata={omniconfig.ARGPARSE_KWARGS: {"nargs": "+", "type": DtypeUtils.eval_dtype}}
    )

    def __post_init__(self) -> None:
        self.group_shapes, self.scale_dtypes = ShapeUtils.format_group_configs(
            group_shapes=self.group_shapes, scale_dtypes=self.scale_dtypes
        )
        if self.dtype is None:
            self.group_shapes, self.scale_dtypes = ((-1, -1, -1),), (None,)

    @property
    def quant_dtype(self) -> QuantDataType | None:
        """The final quantization data type."""
        return self.dtype

    @property
    def zero_domain(self) -> ZeroPointDomain | None:
        """The final zero-point domain."""
        return self.zero_point

    @property
    def largest_group_shape(self) -> tp.Sequence[int]:
        """The shape of the largest group."""
        return self.group_shapes[0]

    @property
    def smallest_group_shape(self) -> tp.Sequence[int]:
        """The shape of the smallest group."""
        return self.group_shapes[-1]

    def decompose(self) -> DecomposedQuantizerConfig:
        """Decompose the configuration to a list of simple configurations."""
        return DecomposedQuantizerConfig(steps=(self,) if self.dtype is not None else ())


@configclass
@dataclass
class ProgressiveQuantizerConfig(QuantizerConfig):
    """Progressive Quantizer configuration.

    Args:
        dtype (`QuantDataType` or `None`, *optional*, defaults to `None`):
            The quantization data type.
        zero_point (`ZeroPointDomain` or `None`, *optional*, defaults to `None`):
            The zero-point domain.
        group_shapes (`Sequence[Sequence[int]]`, *optional*, defaults to `((-1, -1, -1),)`):
            The shapes for per-group quantization.
        scale_dtypes (`Sequence[torch.dtype | QuantDataType | None]`, *optional*, defaults to `(None,)`):
            The quantization scale data type for per-group quantization.
        intermediate_dtypes (`Sequence[QuantDataType]`, *optional*, defaults to `()`):
            The intermediate quantization data types.
        intermediate_levels (Sequence[int], *optional*, defaults to `()`):
            The intermediate quantization levels.
        needs_dequant_saturation (`bool`, *optional*, defaults to `False`):
            Whether the dequantization needs saturation.
    """

    intermediate_dtypes: tp.Sequence[QuantDataType] = field(
        default_factory=tuple, metadata={omniconfig.ARGPARSE_KWARGS: {"nargs": "+", "type": QuantDataType.from_str}}
    )
    intermediate_levels: tp.Sequence[int] = field(
        default_factory=tuple, metadata={omniconfig.ARGPARSE_KWARGS: {"nargs": "+", "type": int}}
    )
    needs_dequant_saturation: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dtype is None:
            self.intermediate_dtypes = ()
            self.intermediate_levels = ()
            self.needs_dequant_saturation = False
            return
        num_levels = len(self.group_shapes)
        if isinstance(self.intermediate_dtypes, QuantDataType):
            self.intermediate_dtypes = (self.intermediate_dtypes,)
        if isinstance(self.intermediate_levels, int):
            self.intermediate_levels = (self.intermediate_levels,)
        self.intermediate_dtypes = tuple(self.intermediate_dtypes)
        self.intermediate_levels = tuple(level % num_levels for level in self.intermediate_levels)
        if len(self.intermediate_dtypes) == 0:
            self.intermediate_levels = ()
            self.needs_dequant_saturation = False
        assert len(self.intermediate_dtypes) == len(self.intermediate_levels)
        assert len(self.intermediate_levels) < num_levels
        assert all(isinstance(dtype, QuantDataType) for dtype in self.intermediate_dtypes)
        assert all(level < num_levels - 1 for level in self.intermediate_levels)

    def decompose(self) -> DecomposedQuantizerConfig:
        """Decompose the configuration to a list of simple configurations."""
        if self.dtype is None:
            return DecomposedQuantizerConfig(steps=())
        elif len(self.intermediate_dtypes) == 0:
            return DecomposedQuantizerConfig(steps=(self,))
        else:
            steps = []
            prev_level = 0
            for level, dtype in zip(self.intermediate_levels, self.intermediate_dtypes, strict=True):
                steps.append(
                    QuantizerConfig(
                        dtype=dtype,
                        zero_point=None,
                        group_shapes=self.group_shapes[prev_level : level + 1],
                        scale_dtypes=self.scale_dtypes[prev_level : level + 1],
                    )
                )
                prev_level = level + 1
            steps.append(
                QuantizerConfig(
                    dtype=self.dtype,
                    zero_point=self.zero_point,
                    group_shapes=self.group_shapes[prev_level:],
                    scale_dtypes=self.scale_dtypes[prev_level:],
                )
            )
            return DecomposedQuantizerConfig(steps=tuple(steps), needs_dequant_saturation=self.needs_dequant_saturation)
