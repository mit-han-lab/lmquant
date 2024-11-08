# -*- coding: utf-8 -*-
"""Quantizatizer config."""

import typing as tp
from dataclasses import dataclass, field

import torch
from omniconfig import configclass

from deepcompressor.calib.config import SkipBasedDynamicRangeCalibConfig
from deepcompressor.data.dtype import QuantDataType
from deepcompressor.quantizer.config import ProgressiveQuantizerConfig
from deepcompressor.quantizer.kernel import QuantGptqConfig
from deepcompressor.utils.config import EnableConfig, SkipBasedConfig

__all__ = ["LlmQuantizerConfig", "LlmWeightQuantizerConfig", "LlmActivationQuantizerConfig", "LlmModuleQuantizerConfig"]


@configclass
@dataclass
class LlmQuantizerConfig(SkipBasedConfig, ProgressiveQuantizerConfig):
    """Llm Quantizer Configuration.

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
        skips (`Sequence[str]`, *optional*, defaults to `[]`):
            The keys of the modules to skip.
        static (`bool`, *optional*, defaults to `False`):
            Whether to use static quantization.
        kernel_gptq (`QuantGptqConfig` or `None`, *optional*, defaults to `None`):
            The GPTQ kernel configuration.
        calib_range (`SkipBasedDynamicRangeCalibConfig` or `None`, *optional*, defaults to `None`):
            The dynamic range calibration configuration.
    """

    static: bool = False
    kernel_gptq: QuantGptqConfig | None = None
    calib_range: SkipBasedDynamicRangeCalibConfig | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.quant_dtype is None:
            self.static = False
            self.kernel_gptq = None
            self.calib_range = None
        if self.static and self.calib_range is None:
            self.calib_range = SkipBasedDynamicRangeCalibConfig()

    @property
    def enabled_gptq(self) -> bool:
        """Whether quantization kernel calibration is enabled."""
        return self.kernel_gptq is not None

    @property
    def enabled_calib_range(self) -> bool:
        """Whether quantization dynamic range calibration is enabled."""
        return self.calib_range is not None

    def generate_calib_dirname(self) -> str:
        """Generate the name for quantization calibration.

        Returns:
            str: The name.
        """
        name = ""
        if self.static:
            name += ".static"
        if self.enabled_gptq:
            name += ".gptq"
        if self.enabled_calib_range and (self.calib_range.needs_search or self.calib_range.ratio != 1):
            name += ".range"
        return name[1:] if name else ""


@configclass
@dataclass
class LlmWeightQuantizerConfig(LlmQuantizerConfig):
    """Llm Weight Quantizer Configuration.

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
        skips (`Sequence[str]`, *optional*, defaults to `[]`):
            The keys of the modules to skip.
        kernel_gptq (`QuantGptqConfig` or `None`, *optional*, defaults to `None`):
            The GPTQ kernel configuration.
        calib_range (`SkipBasedDynamicRangeCalibConfig` or `None`, *optional*, defaults to `None`):
            The dynamic range calibration configuration.
    """

    static: bool = field(init=False, default=True)

    @property
    def needs_calib_data(self) -> bool:
        return self.enabled_gptq or (self.enabled_calib_range and self.calib_range.needs_search)


@configclass
@dataclass
class LlmActivationQuantizerConfig(LlmQuantizerConfig):
    """Llm Activation quantization configuration.

    Args:
        dtype (`QuantDataType` or `None`, *optional*, defaults to `None`):
            The quantization data type.
        zero_point (`ZeroPointDomain` or `None`, *optional*, defaults to `None`):
            The zero-point domain.
        group_shapes (`Sequence[Sequence[int]]`, *optional*, defaults to `((-1, -1, -1),)`):
            The shapes for per-group quantization.
        scale_dtypes (`Sequence[torch.dtype | QuantDataType | None]`, *optional*, defaults to `(None,)`):
            The quantization scale data type for per-group quantization.
        skips (`Sequence[str]`, *optional*, defaults to `[]`):
            The keys of the modules to skip.
        static (`bool`, *optional*, defaults to `False`):
            Whether to use static quantization.
        calib_range (`SkipBasedDynamicRangeCalibConfig` or `None`, *optional*, defaults to `None`):
            The dynamic range calibration configuration.
    """

    intermediate_dtypes: tp.Sequence[QuantDataType] = field(init=False, default=())
    intermediate_levels: tp.Sequence[int] = field(init=False, default=())
    needs_dequant_saturation: bool = field(init=False, default=False)
    kernel_gptq: None = field(init=False, default=None)

    @property
    def needs_calib_data(self) -> bool:
        return self.enabled_calib_range and (self.calib_range.needs_search or self.static)


@configclass
@dataclass
class LlmModuleQuantizerConfig(EnableConfig):
    """Llm Module quantization configuration.

    Args:
        wgts (`LlmWeightQuantizerConfig`):
            The weight quantization configuration.
        ipts (`LlmActivationQuantizerConfig`):
            The input activation quantization configuration.
        opts (`LlmActivationQuantizerConfig`):
            The output activation quantization configuration.
    """

    wgts: LlmWeightQuantizerConfig
    ipts: LlmActivationQuantizerConfig
    opts: LlmActivationQuantizerConfig

    def is_enabled(self) -> bool:
        """Whether the quantization is enabled."""
        return self.enabled_wgts or self.enabled_ipts or self.enabled_opts

    @property
    def enabled_wgts(self) -> bool:
        """Whether to enable weight quantization."""
        return self.wgts is not None and self.wgts.is_enabled()

    @property
    def enabled_ipts(self) -> bool:
        """Whether to enable activation quantization."""
        return self.ipts is not None and self.ipts.is_enabled()

    @property
    def enabled_opts(self) -> bool:
        """Whether to enable activation quantization."""
        return self.opts is not None and self.opts.is_enabled()

    def generate_dirnames(
        self,
        *,
        prefix: str = "",
        shape: torch.Size | tuple[int, ...] = (1024, 1024, 16, 16),
        default_dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> list[str]:
        """Get the directory names of the quantization configuration.

        Args:
            prefix (`str`, *optional*, defaults to `""`):
                The prefix for the directory names.
            shape (`torch.Size` or `tuple[int, ...]`, *optional*, defaults to `(1024, 1024, 16, 16)`):
                The shape of the tensor to be quantized.
            default_dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
                The dtype of the tensor to be quantized.

        Returns:
            `list[str]`:
                The directory names of the quantization configuration.
                    - The number of effective bits.
                    - The name of the quantization data type.
                    - The name of the group shapes.
                    - The name of the modules to skip.
        """
        wgts_names = self.wgts.generate_dirnames(prefix="w", shape=shape, default_dtype=default_dtype)
        ipts_names = self.ipts.generate_dirnames(prefix="x", shape=shape, default_dtype=default_dtype)
        opts_names = self.opts.generate_dirnames(prefix="y", shape=shape, default_dtype=default_dtype)
        names = [
            f"{wgts_name}-{ipts_name}-{opts_name}"
            for wgts_name, ipts_name, opts_name in zip(wgts_names, ipts_names, opts_names, strict=True)
        ]
        if prefix:
            names = [f"{prefix}.[{name}]" for name in names]
        return names

    def generate_calib_dirname(self) -> str:
        """Generate the name for quantization calibration.

        Returns:
            `str`:
                The name.
        """
        name = ""
        if self.enabled_wgts:
            calib_name = self.wgts.generate_calib_dirname()
            if calib_name:
                name += f"-w.{calib_name}"
        if self.enabled_ipts:
            calib_name = self.ipts.generate_calib_dirname()
            if calib_name:
                name += f"-x.{calib_name}"
        if self.enabled_opts:
            calib_name = self.opts.generate_calib_dirname()
            if calib_name:
                name += f"-y.{calib_name}"
        return name[1:] if name else name
