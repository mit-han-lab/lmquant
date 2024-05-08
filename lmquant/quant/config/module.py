# -*- coding: utf-8 -*-
"""Quantization config."""

from dataclasses import dataclass

import torch
from omniconfig import configclass

from ..calib.config import QuantReorderConfig, QuantRotationConfig, QuantSmoothConfig, SearchBasedCalibGranularity
from ..quantizer.config import ActivationQuantizerConfig, WeightQuantizerConfig

__all__ = ["ModuleQuantizerConfig", "ModuleQuantConfig"]


@configclass
@dataclass
class ModuleQuantizerConfig:
    """Basic Module quantization configuration.

    Args:
        wgts (WeightQuantizerConfig): The weight quantization configuration.
        ipts (ActivationQuantizerConfig): The input activation quantization configuration.
        opts (ActivationQuantizerConfig): The output activation quantization configuration.
    """

    wgts: WeightQuantizerConfig
    ipts: ActivationQuantizerConfig
    opts: ActivationQuantizerConfig

    @property
    def enabled_wgts(self) -> bool:
        """Whether to enable weight quantization."""
        return self.wgts.dtype is not None

    @property
    def enabled_ipts(self) -> bool:
        """Whether to enable activation quantization."""
        return self.ipts.dtype is not None

    @property
    def enabled_opts(self) -> bool:
        """Whether to enable activation quantization."""
        return self.opts.dtype is not None

    def generate_dirnames(
        self,
        shape: torch.Size = torch.Size((1024, 1024, 16, 16)),
        default_dtype: torch.dtype = torch.float16,
        prefix: str = "",
    ) -> list[str]:
        """Get the directory names of the quantization configuration.

        Args:
            shape (torch.Size, optional): The shape of the input tensor.
                                          Defaults to ``torch.Size((1024, 1024, 16, 16))``.
            default_dtype (torch.dtype, optional): The dtype of the input tensor. Defaults to ``torch.float16.`

        Returns:
            list[str]: The directory names of the quantization configuration.
                - The number of effective bits.
                - The name of the quantization data type.
                - The name of the group shapes.
                - The name of the modules to skip.
        """
        wgts_names = self.wgts.generate_dirnames(shape=shape, default_dtype=default_dtype, prefix="w")
        ipts_names = self.ipts.generate_dirnames(shape=shape, default_dtype=default_dtype, prefix="x")
        opts_names = self.opts.generate_dirnames(shape=shape, default_dtype=default_dtype, prefix="y")
        names = [
            f"{wgts_name}-{ipts_name}-{opts_name}"
            for wgts_name, ipts_name, opts_name in zip(wgts_names, ipts_names, opts_names)
        ]
        if prefix and names:
            names = [f"{prefix}.[{name}]" for name in names]
        return names

    def generate_calib_name(self) -> str:
        """Generate the name for quantization calibration.

        Returns:
            str: The name.
        """
        name = ""
        if self.enabled_wgts:
            calib_name = self.wgts.generate_calib_name()
            if calib_name:
                name += f"-w.{calib_name}"
        if self.enabled_ipts:
            calib_name = self.ipts.generate_calib_name()
            if calib_name:
                name += f"-x.{calib_name}"
        if self.enabled_opts:
            calib_name = self.opts.generate_calib_name()
            if calib_name:
                name += f"-y.{calib_name}"
        return name[1:] if name else name


@configclass
@dataclass
class ModuleQuantConfig(ModuleQuantizerConfig):
    """Module quantization configuration.

    Args:
        wgts (WeightQuantizerConfig): The weight quantization configuration.
        ipts (ActivationQuantizerConfig): The input activation quantization configuration.
        opts (ActivationQuantizerConfig): The output activation quantization configuration.
        rotation (QuantRotationConfig): The rotation configuration. Defaults to ``None``.
        reorder (QuantReorderConfig): The channel reorder configuration. Defaults to ``None``.
        smooth (QuantSmoothConfig): The smooth quantization configuration. Defaults to ``None``.
        bias_correction (bool): Whether to correct the bias. Defaults to ``False``.
    """

    rotation: QuantRotationConfig | None = None
    reorder: QuantReorderConfig | None = None
    smooth: QuantSmoothConfig | None = None
    bias_correction: bool = False

    @property
    def enabled_smooth(self) -> bool:
        """Whether to enable smooth quantization."""
        return self.smooth is not None

    @property
    def enabled_smooth_xw(self) -> bool:
        """Whether to enable xw smooth quantization."""
        return self.enabled_smooth and self.smooth.enabled_smooth_xw

    @property
    def enabled_smooth_yx(self) -> bool:
        """Whether to enable yy smooth quantization."""
        return self.enabled_smooth and self.smooth.enabled_smooth_yx

    @property
    def enabled_reorder(self) -> bool:
        """Whether to enable channel reorder."""
        return self.reorder is not None

    @property
    def enabled_rotation(self) -> bool:
        """Whether to enable rotation."""
        return self.rotation is not None

    @property
    def enabled_bias_correction(self) -> bool:
        """Whether to correct the bias."""
        return self.bias_correction

    @property
    def needs_orig_wgts(self) -> bool:
        """Whether to keep the original weights."""
        if self.enabled_ipts and self.ipts.enabled_calib_range:
            return True
        if self.enabled_opts and self.opts.enabled_calib_range:
            return True
        return self.enabled_wgts and self.enabled_bias_correction

    def __post_init__(self) -> None:  # noqa: C901
        if self.smooth is not None:
            if not self.smooth.enabled_smooth_xw and not self.smooth.enabled_smooth_yx:
                self.smooth = None
        if self.rotation is not None and self.reorder is not None:
            self.reorder.skips.append("residual")
            if self.rotation.with_hadamard_transform:
                self.reorder.skips.extend(self.rotation.transforms)
                self.reorder.skips = sorted(set(self.reorder.skips))
        if self.enabled_ipts:
            if self.ipts.enabled_calib_range and self.ipts.calib_range.granularity == SearchBasedCalibGranularity.Group:
                self.ipts.calib_range.granularity = SearchBasedCalibGranularity.ChannelGroup
            if self.ipts.static:
                assert self.ipts.smallest_group_shape[0] == -1, "static quantization requires batch group size to be -1"
        if self.enabled_opts:
            if self.opts.enabled_calib_range and self.opts.calib_range.granularity == SearchBasedCalibGranularity.Group:
                self.opts.calib_range.granularity = SearchBasedCalibGranularity.ChannelGroup
            if self.opts.static:
                assert self.opts.smallest_group_shape[0] == -1, "static quantization requires batch group size to be -1"

    def generate_calib_name(self) -> str:
        name = ""
        if self.enabled_rotation:
            name += "-rot"
            if self.rotation.random:
                name += ".rnd"
        if self.enabled_reorder:
            name += "-reorder"
            if self.reorder.dynamic:
                name += ".dyn"
        if self.enabled_smooth:
            name += f"-smooth"
            if self.enabled_smooth_xw:
                name += f".xw"
            if self.enabled_smooth_yx:
                name += f".yx"
        if self.enabled_bias_correction:
            name += "-bias"
        calib_name = super().generate_calib_name()
        if calib_name:
            name += f"-{calib_name}"
        return name[1:] if name else name
