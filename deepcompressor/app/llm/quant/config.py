# -*- coding: utf-8 -*-
"""Quantization config."""

import os
from dataclasses import dataclass, field

import torch
from omniconfig import configclass

from deepcompressor.calib.config import (
    QuantRotationConfig,
    SearchBasedCalibGranularity,
    SearchBasedCalibObjective,
    SearchBasedCalibStrategy,
    SkipBasedChannelOrderConfig,
    SmoothTransfomerConfig,
)
from deepcompressor.data.utils.dtype import eval_dtype
from deepcompressor.utils.common import num2str

from ..cache.config import LlmQuantCacheConfig
from ..nn.struct import LlmFeedForwardStruct, LlmSelfAttentionStruct
from .dataset import LlmCalibDataLoaderConfig
from .quantizer import LlmModuleQuantizerConfig

__all__ = ["LlmQuantConfig"]


@configclass
@dataclass
class LlmQuantConfig(LlmModuleQuantizerConfig):
    """Large Language Model Module quantization configuration.

    Args:
        wgts (`LlmWeightQuantizerConfig`):
            The weight quantization configuration.
        ipts (`LlmActivationQuantizerConfig`):
            The input activation quantization configuration.
        opts (`LlmActivationQuantizerConfig`):
            The output activation quantization configuration.
        calib (`LlmCalibDataLoaderConfig`):
            The calibration dataset configuration.
        rotation (`QuantRotationConfig` or `None`, *optional*, defaults to `None`):
            The quantization rotation configuration.
        reorder (`SkipBasedChannelOrderConfig` or `None`, *optional*, defaults to `None`):
            The quantization reordering configuration.
        smooth (`SmoothTransfomerConfig`, *optional*, defaults to `None`):
            The quantization smoothing configuration.
        develop_dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            The development data type during quantization.
    """

    calib: LlmCalibDataLoaderConfig
    rotation: QuantRotationConfig | None = None
    reorder: SkipBasedChannelOrderConfig | None = None
    smooth: SmoothTransfomerConfig | None = None
    develop_dtype: torch.dtype = field(default_factory=lambda s=torch.float32: eval_dtype(s, with_quant_dtype=False))

    def __post_init__(self) -> None:  # noqa: C901
        if self.smooth is not None:
            if not self.smooth.enabled_proj and not self.smooth.enabled_attn:
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
        if self.enabled_reorder:
            if not self.reorder.dynamic:
                qkv_proj_rkey, up_proj_rkey = LlmSelfAttentionStruct.qkv_proj_rkey, LlmFeedForwardStruct.up_proj_rkey
                skips_to_remove = []
                for skip in self.reorder.skips:
                    if skip.startswith(qkv_proj_rkey) or skip.endswith(f"_{qkv_proj_rkey}"):
                        self.reorder.skips.append("residual")
                        skips_to_remove.append(skip)
                    elif skip.startswith(up_proj_rkey) or skip.endswith(f"_{up_proj_rkey}"):
                        self.reorder.skips.append("residual")
                        skips_to_remove.append(skip)
                self.reorder.skips = sorted(set(self.reorder.skips))
                for skip in skips_to_remove:
                    self.reorder.skips.remove(skip)
            self.reorder.skips = sorted(set(self.reorder.skips))

    @property
    def enabled_smooth(self) -> bool:
        """Whether to enable smooth quantization."""
        return self.smooth is not None

    @property
    def enabled_smooth_proj(self) -> bool:
        """Whether to enable xw smooth quantization."""
        return self.enabled_smooth and self.smooth.enabled_proj

    @property
    def enabled_smooth_attn(self) -> bool:
        """Whether to enable yy smooth quantization."""
        return self.enabled_smooth and self.smooth.enabled_attn

    @property
    def enabled_reorder(self) -> bool:
        """Whether to enable channel reorder."""
        return self.reorder is not None and self.reorder.is_enabled()

    @property
    def enabled_rotation(self) -> bool:
        """Whether to enable rotation."""
        return self.rotation is not None

    @property
    def needs_acts_quantizer_cache(self) -> bool:
        """Whether to cache the activations quantizer settings."""
        if self.enabled_ipts and self.ipts.needs_calib_data:
            return True
        if self.enabled_opts and self.opts.needs_calib_data:
            return True
        return False

    def generate_calib_dirname(self) -> str:
        name = ""
        if self.enabled_rotation:
            name += "-rotate"
            if self.rotation.random:
                name += ".rnd"
        if self.enabled_reorder:
            name += "-reorder"
            if self.reorder.dynamic:
                name += ".dyn"
        if self.enabled_smooth:
            name += "-smooth"
            if self.enabled_smooth_proj:
                name += ".proj"
            if self.enabled_smooth_attn:
                name += ".attn"
        calib_name = super().generate_calib_dirname()
        if calib_name:
            name += f"-{calib_name}"
        return name[1:] if name else name

    def generate_default_dirname(self) -> str:  # noqa: C901
        """Generate directory name for a large language model quantization configuration."""
        w_names = x_names = {"qkv_proj": "qkv", "out_proj": "out", "up_proj": "fc1", "down_proj": "fc2"}
        y_names = {"attn_q": "q", "attn_k": "k", "attn_v": "v"}
        skip_name = ""
        if self.enabled_opts:
            skip_y_name = "+".join(y_names[y] for y in self.opts.skips if y in y_names)
            if skip_y_name:
                skip_name += f".y.[{skip_y_name}]"
        if self.enabled_wgts:
            skip_w_name = "+".join(w_names[w] for w in self.wgts.skips if w in w_names)
            if skip_w_name:
                skip_name += f".w.[{skip_w_name}]"
        if self.enabled_ipts:
            skip_x_name = "+".join(x_names[x] for x in self.ipts.skips if x in x_names)
            if skip_x_name:
                skip_name += f".x.[{skip_x_name}]"
        if skip_name:
            skip_name = "-skip" + skip_name
        if self.enabled_wgts and self.wgts.enabled_gptq:
            skip_name += "-gptq"
        rotation_name = ""
        if self.enabled_rotation:
            rotation_name = "-rot"
            if self.rotation.random:
                rotation_name += ".rnd"
            if self.rotation.with_hadamard_transform:
                rotation_name += ".[+{}]".format("+".join(w_names[w] for w in self.rotation.transforms))
        reorder_name = ""
        if self.enabled_reorder:
            reorder_name = "-rodr"
            if self.reorder.strategy == SearchBasedCalibStrategy.Manual:
                if self.reorder.channel_metric.value != "xMax":
                    reorder_name += f".{self.reorder.channel_metric.value}"
                if self.reorder.channel_index.value != "Seq":
                    reorder_name += f".{self.reorder.channel_index.value}"
            else:
                reorder_name += f".{self.reorder.strategy.name}"
            reorders, skips = [], []
            for k in w_names.keys() if self.reorder.dynamic else ("residual", "out_proj", "down_proj"):
                v = w_names.get(k, "res")
                if k in self.reorder.skips:
                    skips.append(v)
                else:
                    reorders.append(v)
            if len(reorders) <= len(skips):
                reorder_name += ".[{}]".format("+".join(reorders))
            elif skips:
                reorder_name += ".skip.[{}]".format("+".join(skips))
        smooth_name = ""
        if self.enabled_smooth:
            smooth_name = "-smth"
            if self.smooth.enabled_proj:
                smooth_name += ".proj"
                if self.smooth.proj.granularity != SearchBasedCalibGranularity.Layer:
                    smooth_name += f".{self.smooth.proj.granularity.name}"
                if self.smooth.proj.strategy != SearchBasedCalibStrategy.Manual:
                    smooth_name += f".{self.smooth.proj.strategy.name}"
                    if self.smooth.proj.alpha <= 0:
                        smooth_name += f".a{num2str(self.smooth.proj.alpha)}"
                    if self.smooth.proj.beta <= 0:
                        smooth_name += f".b{num2str(self.smooth.proj.beta)}"
                else:
                    smooth_name += f".a{num2str(self.smooth.proj.alpha)}"
                    smooth_name += f".b{num2str(self.smooth.proj.beta)}"
                xspan_eq_wspan = True
                for xspan, wspan in self.smooth.proj.spans:
                    if xspan != wspan:
                        xspan_eq_wspan = False
                        break
                if xspan_eq_wspan:
                    smooth_name += ".[{}]".format("+".join(xspan.name for xspan, _ in self.smooth.proj.spans))
                else:
                    smooth_name += ".[{}]".format(
                        "+".join(f"x.{xspan.name}.w.{wspan.name}" for xspan, wspan in self.smooth.proj.spans)
                    )
                smooths, skips = [], []
                for k, v in w_names.items():
                    if k in self.smooth.proj.skips:
                        skips.append(v)
                    else:
                        smooths.append(v)
                if len(smooths) <= len(skips):
                    smooth_name += ".[{}]".format("+".join(smooths))
                elif skips:
                    smooth_name += ".skip.[{}]".format("+".join(skips))
            if self.smooth.enabled_attn:
                smooth_name += ".attn"
                if self.smooth.attn.granularity != SearchBasedCalibGranularity.Layer:
                    smooth_name += f".{self.smooth.attn.granularity.name}"
                if self.smooth.attn.strategy != SearchBasedCalibStrategy.Manual:
                    smooth_name += f".{self.smooth.attn.strategy.name}"
                    if self.smooth.attn.alpha <= 0:
                        smooth_name += f".a{num2str(self.smooth.attn.alpha)}"
                    if self.smooth.attn.beta <= 0:
                        smooth_name += f".b{num2str(self.smooth.attn.beta)}"
                else:
                    smooth_name += f".a{num2str(self.smooth.attn.alpha)}"
                    smooth_name += f".b{num2str(self.smooth.attn.beta)}"
                xspan_eq_yspan = True
                for xspan, yspan in self.smooth.attn.spans:
                    if xspan != yspan:
                        xspan_eq_yspan = False
                        break
                if xspan_eq_yspan:
                    smooth_name += ".[{}]".format("+".join(xspan.name for xspan, _ in self.smooth.attn.spans))
                else:
                    smooth_name += ".[{}]".format(
                        "+".join(f"x.{xspan.name}.y.{yspan.name}" for xspan, yspan in self.smooth.attn.spans)
                    )
        wrange_name = ""
        if (
            self.enabled_wgts
            and self.wgts.enabled_calib_range
            and (self.wgts.calib_range.needs_search or self.wgts.calib_range.ratio != 1)
        ):
            wrange_name = "-w.range"
            if self.wgts.calib_range.needs_search:
                if self.wgts.calib_range.granularity != SearchBasedCalibGranularity.Group:
                    wrange_name += f".{self.wgts.calib_range.granularity.name}"
                if self.wgts.calib_range.objective != SearchBasedCalibObjective.OutputsError:
                    wrange_name += f".{self.wgts.calib_range.objective.name}"
                if self.wgts.calib_range.degree != 2:
                    wrange_name += f".d{num2str(self.wgts.calib_range.degree)}"
                wrange_name += f".[{num2str(self.wgts.calib_range.max_shrink)}"
                wrange_name += f".{num2str(self.wgts.calib_range.max_expand)}"
                wrange_name += f".g{self.wgts.calib_range.num_grids}]"
            else:
                wrange_name += f".r{num2str(self.wgts.calib_range.ratio)}"
            if self.wgts.calib_range.skips:
                wrange_name += ".skip.[{}]".format("+".join(w_names[w] for w in self.wgts.calib_range.skips))
        xrange_name = ""
        if (
            self.enabled_ipts
            and self.ipts.enabled_calib_range
            and (self.ipts.calib_range.needs_search or self.ipts.calib_range.ratio != 1)
        ):
            xrange_name = "-x.range"
            if self.ipts.calib_range.needs_search:
                if self.ipts.calib_range.granularity != SearchBasedCalibGranularity.Group:
                    xrange_name += f".{self.ipts.calib_range.granularity.name}"
                if self.ipts.calib_range.objective != SearchBasedCalibObjective.OutputsError:
                    xrange_name += f".{self.ipts.calib_range.objective.name}"
                if self.ipts.calib_range.degree != 2:
                    xrange_name += f".d{num2str(self.ipts.calib_range.degree)}"
                xrange_name += f".[{num2str(self.ipts.calib_range.max_shrink)}"
                xrange_name += f".{num2str(self.ipts.calib_range.max_expand)}"
                xrange_name += f".g{self.ipts.calib_range.num_grids}]"
            else:
                xrange_name += f".r{num2str(self.ipts.calib_range.ratio)}"
            if self.ipts.calib_range.skips:
                xrange_name += ".skip.[{}]".format("+".join(w_names[w] for w in self.ipts.calib_range.skips))
        yrange_name = ""
        if (
            self.enabled_opts
            and self.opts.enabled_calib_range
            and (self.opts.calib_range.needs_search or self.opts.calib_range.ratio != 1)
        ):
            yrange_name = "-y.range"
            if self.opts.calib_range.needs_search:
                if self.opts.calib_range.granularity != SearchBasedCalibGranularity.Group:
                    yrange_name += f".{self.opts.calib_range.granularity.name}"
                if self.opts.calib_range.objective != SearchBasedCalibObjective.OutputsError:
                    yrange_name += f".{self.opts.calib_range.objective.name}"
                if self.opts.calib_range.degree != 2:
                    yrange_name += f".d{num2str(self.opts.calib_range.degree)}"
                yrange_name += f".[{num2str(self.opts.calib_range.max_shrink)}"
                yrange_name += f".{num2str(self.opts.calib_range.max_expand)}"
                yrange_name += f".g{self.opts.calib_range.num_grids}]"
            else:
                yrange_name += f".r{num2str(self.opts.calib_range.ratio)}"
            if self.opts.calib_range.skips:
                yrange_name += ".skip.[{}]".format("+".join(y_names[y] for y in self.opts.calib_range.skips))
        name = skip_name + rotation_name + reorder_name + smooth_name + wrange_name + xrange_name + yrange_name
        name = name[1:] if name else "default"
        name += f"-{self.calib.generate_dirnames()[0]}"
        return name

    def generate_cache_dirpath(
        self, *, root: str, seed: int, default_dtype: torch.dtype = torch.float16
    ) -> LlmQuantCacheConfig:  # noqa: C901
        """Generate the cache paths for the module quantization configuration."""
        quant_names = self.generate_dirnames(default_dtype=default_dtype)
        w_kernel_names = []
        if self.enabled_wgts and self.wgts.enabled_gptq:
            w_kernel_names = self.wgts.kernel_gptq.generate_dirnames(prefix="w.kernel")
        if self.enabled_rotation:
            quant_names.extend(self.rotation.generate_dirnames(prefix="rotate"))
        reorder_dirpath = ""
        if self.enabled_reorder:
            reorder_names = self.reorder.generate_dirnames(prefix="reorder")
            quant_names.extend(reorder_names)
            reorder_dirpath = os.path.join("reorder", *quant_names)
        smooth_dirpath = ""
        if self.enabled_smooth:
            smooth_names = self.smooth.generate_dirnames(prefix="smooth")
            quant_names.extend(smooth_names)
            smooth_dirpath = os.path.join("smooth", *quant_names)
        quant_names.extend(w_kernel_names)
        wgts_dirpath = ""
        if self.enabled_wgts and self.wgts.enabled_calib_range:
            quant_names.extend(self.wgts.calib_range.generate_dirnames(prefix="w.range"))
            wgts_dirpath = os.path.join("wgts", *quant_names)
        acts_dirpath = ""
        if self.needs_acts_quantizer_cache:
            if self.enabled_ipts and self.ipts.enabled_calib_range:
                quant_names.extend(self.ipts.calib_range.generate_dirnames(prefix="x.range"))
            if self.enabled_opts and self.opts.enabled_calib_range:
                quant_names.extend(self.opts.calib_range.generate_dirnames(prefix="y.range"))
            acts_dirpath = os.path.join("acts", *quant_names)
        cache_dirpath = LlmQuantCacheConfig(
            reorder=reorder_dirpath,
            smooth=smooth_dirpath,
            wgts=wgts_dirpath,
            acts=acts_dirpath,
        ).add_parent_dirs(*self.calib.generate_dirnames())
        if self.enabled_rotation:
            cache_dirpath.rotation = os.path.join(
                "rotation",
                f"seed.{seed}" if self.rotation.random else "hadamard",
            )
        cache_dirpath.add_parent_dirs(root, "llm", "cache", "quant")
        return cache_dirpath
