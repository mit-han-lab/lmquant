# -*- coding: utf-8 -*-
"""Module quantizer config."""

from dataclasses import dataclass

import torch
from omniconfig import configclass

from .tensor import ActivationQuantizerConfig, WeightQuantizerConfig

__all__ = ["ModuleQuantizerConfig"]


@configclass
@dataclass
class ModuleQuantizerConfig:
    """Module quantization configuration.

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
