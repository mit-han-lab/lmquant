# -*- coding: utf-8 -*-
"""Model Quantization config."""

import os
import typing as tp
from dataclasses import dataclass, field

import torch.nn as nn
from omniconfig import IGNORE_FIELD, Arguments, configclass

from .module import ModuleQuantConfig

__all__ = ["QuantCachePath", "ModelQuantConfig"]


@dataclass
class QuantCachePath:
    """Module quantization cache path."""

    rotation: str = ""
    reorder: str = ""
    smooth: str = ""
    wgts: str = ""
    acts: str = ""

    def clone(self) -> "QuantCachePath":
        """Clone the cache paths.

        Returns:
            ModuleQuantCachePath: The cloned cache paths.
        """
        return QuantCachePath(
            rotation=self.rotation,
            reorder=self.reorder,
            smooth=self.smooth,
            wgts=self.wgts,
            acts=self.acts,
        )

    def add_parent_dirs(self, *parent_dirs: str) -> "QuantCachePath":
        """Add the parent directories to the cache paths.

        Args:
            parent_dirs (str): The parent directories.
        """
        if self.rotation:
            self.rotation = os.path.join(*parent_dirs, self.rotation)
        if self.reorder:
            self.reorder = os.path.join(*parent_dirs, self.reorder)
        if self.smooth:
            self.smooth = os.path.join(*parent_dirs, self.smooth)
        if self.wgts:
            self.wgts = os.path.join(*parent_dirs, self.wgts)
        if self.acts:
            self.acts = os.path.join(*parent_dirs, self.acts)
        return self

    def add_chidren(self, *children: str) -> "QuantCachePath":
        """Add the children to the cache paths.

        Args:
            children (str): The children.
        """
        if self.rotation:
            self.rotation = os.path.join(self.rotation, *children)
        if self.reorder:
            self.reorder = os.path.join(self.reorder, *children)
        if self.smooth:
            self.smooth = os.path.join(self.smooth, *children)
        if self.wgts:
            self.wgts = os.path.join(self.wgts, *children)
        if self.acts:
            self.acts = os.path.join(self.acts, *children)
        return self


@configclass
@dataclass
class ModelQuantConfig(ModuleQuantConfig):
    """Module quantization configuration.

    Args:
        wgts (WeightQuantizerConfig): The weight quantization configuration.
        ipts (ActivationQuantizerConfig): The input activation quantization configuration.
        opts (ActivationQuantizerConfig): The output activation quantization configuration.
        rotation (QuantRotationConfig): The rotation configuration. Defaults to ``None``.
        smooth (SmoothQuantConfig): The smooth quantization configuration. Defaults to ``None``.
        reorder (QuantReorderConfig): The channel reorder configuration. Defaults to ``None``.
        bias_correction (bool): Whether to correct the bias. Defaults to ``False``.
        keywords_i (dict[str, list[str]]): The module name keywords for the input quantization.
            Defaults to ``{}``.
        keywords_w (dict[str, list[str]]): The param name keywords for the weight quantization.
            Defaults to ``{}``.
        keywords_o (dict[str, list[str]]): The module name keywords for the output quantization.
            Defaults to ``{}``.
        module_types_i (list[type[nn.Module]] | type[nn.Module]): The module types for the input quantization.
            Defaults to ``[]``.
        module_types_w (list[type[nn.Module]] | type[nn.Module]): The module types for the weight quantization.
            Defaults to ``[]``.
        module_types_o (list[type[nn.Module]] | type[nn.Module]): The module types for the output quantization.
            Defaults to ``[]``.
        channels_dims (dict[type[nn.Module], tuple[int, int]]): The channel dimensions of the inputs and outputs
            of the modules. Defaults to ``{}``.
    """

    # module quantization parameters
    keywords_i: dict[str, list[str]] = field(default_factory=dict)
    keywords_w: dict[str, list[str]] = field(default_factory=dict)
    keywords_o: dict[str, list[str]] = field(default_factory=dict)
    module_types_i: list[type[nn.Module]] | type[nn.Module] = field(default_factory=list)
    module_types_w: list[type[nn.Module]] | type[nn.Module] = field(default_factory=list)
    module_types_o: list[type[nn.Module]] | type[nn.Module] = field(default_factory=list)
    # general parameters
    channels_dims: dict[type[nn.Module], tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:  # noqa: C901
        super().__post_init__()
        self.channels_dims[nn.Linear] = (-1, -1)

    def needs_quant_weights(self, param_name: str, module: nn.Module = None) -> bool:
        """Whether to quantize the weight of a module.

        Args:
            param_name (str): The name of the parameter.

        Returns:
            bool: Whether to quantize the weight of the module.
        """
        if not self.enabled_wgts:
            return False
        if module is None or isinstance(module, self.module_types_w):
            needs_quant = False
            for key, keywords in self.keywords_w.items():
                for k in keywords:
                    if k in param_name:
                        needs_quant = self.wgts.enabled_for(key)
                        break
            return needs_quant
        return False

    def needs_quant_inputs(self, module_name: str, module: nn.Module) -> bool:
        """Whether to quantize the input of a module.

        Args:
            module_name (str): The name of the module.
            module (nn.Module): The module.

        Returns:
            bool: Whether to quantize the input of the module.
        """
        if not self.enabled_ipts:
            return False
        if isinstance(module, self.module_types_i):
            needs_quant = False
            for key, keywords in self.keywords_i.items():
                for k in keywords:
                    if module_name.endswith(k):
                        needs_quant = self.ipts.enabled_for(key)
                        break
            return needs_quant
        return False

    def needs_quant_outputs(self, module_name: str, module: nn.Module) -> bool:
        """Whether to quantize the output of a module.

        Args:
            module_name (str): The name of the module.
            module (nn.Module): The module.

        Returns:
            bool: Whether to quantize the output of the module.
        """
        if not self.enabled_opts:
            return False
        if isinstance(module, self.module_types_o):
            needs_quant = False
            for key, keywords in self.keywords_o.items():
                for k in keywords:
                    if module_name.endswith(k):
                        needs_quant = self.opts.enabled_for(key)
                        break
            return needs_quant
        return False

    def set_keywords(
        self,
        keywords_i: dict[str, list[str]] = None,
        keywords_w: dict[str, list[str]] = None,
        keywords_o: dict[str, list[str]] = None,
    ):
        """Set the keywords for the module quantization configuration."""
        if keywords_i is not None:
            self.keywords_i = keywords_i
        if keywords_w is not None:
            self.keywords_w = keywords_w
        if keywords_o is not None:
            self.keywords_o = keywords_o

    @classmethod
    def update_get_arguments(
        cls: type["ModelQuantConfig"],
        *,
        overwrites: dict[str, tp.Callable[[Arguments], None] | None] | None = None,
        defaults: dict[str, tp.Any] | None = None,
    ) -> tuple[dict[str, tp.Any], dict[str, tp.Any]]:
        """Get the arguments for the module quantization configuration."""
        overwrites = overwrites or {}
        defaults = defaults or {}
        overwrites.setdefault("keywords_i", IGNORE_FIELD)
        overwrites.setdefault("keywords_w", IGNORE_FIELD)
        overwrites.setdefault("keywords_o", IGNORE_FIELD)
        overwrites.setdefault("module_types_i", IGNORE_FIELD)
        overwrites.setdefault("module_types_w", IGNORE_FIELD)
        overwrites.setdefault("module_types_o", IGNORE_FIELD)
        overwrites.setdefault("channels_dims", IGNORE_FIELD)
        return overwrites, defaults

    def generate_cache_dirpath(self) -> QuantCachePath:  # noqa: C901
        """Generate the cache paths for the module quantization configuration."""
        quant_names = self.generate_dirnames()
        w_kernel_names = self.wgts.generate_calib_kernel_dirnames(prefix="w.kernel")
        if self.enabled_rotation:
            quant_names.extend(self.rotation.generate_dirnames(prefix="rotate"))
        reorder_dirpath = ""
        if self.enabled_reorder:
            reorder_names = self.reorder.generate_dirnames(prefix="reorder")
            if self.reorder.allow_kernel_calib:
                reorder_names.extend(w_kernel_names)
                w_kernel_names = []
            quant_names.extend(reorder_names)
            reorder_dirpath = os.path.join("reorder", *quant_names)
        smooth_dirpath = ""
        if self.enabled_smooth:
            smooth_names = self.smooth.generate_dirnames(prefix="smooth")
            if (self.smooth.enabled_smooth_xw and self.smooth.xw.allow_kernel_calib) or (
                self.smooth.enabled_smooth_yx and self.smooth.yx.allow_kernel_calib
            ):
                smooth_names.extend(w_kernel_names)
                w_kernel_names = []
            quant_names.extend(smooth_names)
            smooth_dirpath = os.path.join("smooth", *quant_names)
        quant_names.extend(w_kernel_names)
        wgts_dirpath = ""
        if self.enabled_wgts and self.wgts.enabled_calib_range:
            quant_names.extend(self.wgts.generate_calib_range_dirnames(prefix="w.range"))
            wgts_dirpath = os.path.join("wgts", *quant_names)
        needs_acts_cache, acts_dirpath = False, ""
        if self.enabled_ipts and self.ipts.enabled_calib_range:
            quant_names.extend(self.ipts.generate_calib_range_dirnames(prefix="x.range"))
            needs_acts_cache = True
        if self.enabled_opts and self.opts.enabled_calib_range:
            quant_names.extend(self.opts.generate_calib_range_dirnames(prefix="y.range"))
            needs_acts_cache = True
        if needs_acts_cache:
            acts_dirpath = os.path.join("acts", *quant_names)
        return QuantCachePath(
            reorder=reorder_dirpath,
            smooth=smooth_dirpath,
            wgts=wgts_dirpath,
            acts=acts_dirpath,
        )
