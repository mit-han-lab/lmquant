# -*- coding: utf-8 -*-
"""Quantization config."""

import os
from dataclasses import dataclass, field

import torch
from omniconfig import configclass

from deepcompressor.calib.config import (
    SearchBasedCalibGranularity,
    SearchBasedCalibObjective,
    SearchBasedCalibStrategy,
    SmoothTransfomerConfig,
)
from deepcompressor.data.utils.dtype import eval_dtype
from deepcompressor.quantizer.config import QuantLowRankConfig
from deepcompressor.utils.common import num2str

from ..cache.config import DiffusionQuantCacheConfig
from ..dataset.calib import DiffusionCalibCacheLoaderConfig
from .quantizer.config import DiffusionModuleQuantizerConfig

__all__ = ["DiffusionQuantConfig"]


@configclass
@dataclass
class DiffusionQuantConfig(DiffusionModuleQuantizerConfig):
    """Diffusion model quantization configuration.

    Args:
        wgts (`DiffusionWeightQuantizerConfig`):
            The weight quantization configuration.
        ipts (`DiffusionActivationQuantizerConfig`):
            The input activation quantization configuration.
        opts (`DiffusionActivationQuantizerConfig`):
            The output activation quantization configuration.
        calib (`DiffusionCalibDatasetConfig`):
            The calibration dataset configuration.
        smooth (`TransfomerQuantSmoothConfig` or `None`, *optional*, defaults to `None`):
            The smooth quantization configuration.
        develop_dtype (`torch.dtype`, *optional*, defaults to `None`):
            The development data type.
    """

    calib: DiffusionCalibCacheLoaderConfig
    smooth: SmoothTransfomerConfig | None = None
    develop_dtype: torch.dtype = field(default_factory=lambda s=torch.float32: eval_dtype(s, with_quant_dtype=False))

    def __post_init__(self) -> None:  # noqa: C901
        super().__post_init__()
        if self.smooth is not None:
            if not self.smooth.enabled_proj and not self.smooth.enabled_attn:
                self.smooth = None
        if self.enabled_smooth and self.smooth.enabled_proj and self.smooth.proj.allow_low_rank:
            if self.enabled_wgts:
                self.smooth.proj.allow_low_rank = self.wgts.enabled_low_rank
                if self.smooth.proj.allow_low_rank:
                    self.smooth.proj.granularity = SearchBasedCalibGranularity.Layer
            else:
                self.smooth.proj.allow_low_rank = False
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
        self.organize()
        self.unsigned_ipts = self.ipts.for_unsigned()

    @property
    def enabled_smooth(self) -> bool:
        """Whether to enable smooth quantization."""
        return self.smooth is not None

    @property
    def enabled_smooth_proj(self) -> bool:
        """Whether to enable smooth quantization for projections."""
        return self.enabled_smooth and self.smooth.enabled_proj

    @property
    def enabled_smooth_attn(self) -> bool:
        """Whether to enable smooth quantization for attentions."""
        return self.enabled_smooth and self.smooth.enabled_attn

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

    def generate_cache_dirpath(
        self, *, root: str, shift: bool, default_dtype: torch.dtype = torch.float16
    ) -> DiffusionQuantCacheConfig:  # noqa: C901
        """Generate the cache paths for the module quantization configuration."""
        quant_names = self.generate_dirnames(default_dtype=default_dtype)
        if shift:
            quant_names.append("shift")
        if self.enabled_wgts and self.wgts.enabled_low_rank:
            quant_names.extend(QuantLowRankConfig.generate_dirnames(self.wgts.low_rank, prefix="lowrank"))
        smooth_dirpath = ""
        if self.enabled_smooth:
            quant_names.extend(self.smooth.generate_dirnames(prefix="smooth"))
            smooth_dirpath = os.path.join("smooth", *quant_names)
        branch_dirpath = ""
        if self.enabled_wgts and self.wgts.enabled_low_rank:
            quant_names.extend(self.wgts.low_rank.generate_dirnames(prefix="lowrank"))
            branch_dirpath = os.path.join("branch", *quant_names)
        wgts_dirpath = ""
        if self.enabled_wgts and self.wgts.needs_calib_data:
            quant_names.extend(self.wgts.calib_range.generate_dirnames(prefix="w.range"))
            wgts_dirpath = os.path.join("wgts", *quant_names)
        acts_dirpath = ""
        if self.needs_acts_quantizer_cache:
            if self.enabled_ipts and self.ipts.needs_calib_data:
                quant_names.extend(self.ipts.calib_range.generate_dirnames(prefix="x.range"))
            if self.enabled_opts and self.opts.needs_calib_data:
                quant_names.extend(self.opts.calib_range.generate_dirnames(prefix="y.range"))
            acts_dirpath = os.path.join("acts", *quant_names)
        cache_dirpath = DiffusionQuantCacheConfig(
            smooth=smooth_dirpath, branch=branch_dirpath, wgts=wgts_dirpath, acts=acts_dirpath
        ).simplify(type(self)._key_map)
        cache_dirpath = cache_dirpath.add_parent_dirs(*self.calib.generate_dirnames())
        cache_dirpath = cache_dirpath.add_parent_dirs(root, "diffusion", "cache", "quant")
        return cache_dirpath

    def generate_default_dirname(self) -> str:  # noqa: C901
        """Generate output directory name for evaluating a large language model."""
        key_map = type(self)._key_map

        def simplify_skips(skips):
            return set(
                DiffusionQuantCacheConfig.simplify_path("skip.[{}]".format("+".join(skips)), key_map=key_map)[
                    6:-1
                ].split("+")
            )

        skip_name, y_skips, w_skips, x_skips = "", set(), set(), set()
        if self.enabled_opts and self.opts.skips:
            y_skips = simplify_skips(self.opts.skips)
        if self.enabled_ipts and self.ipts.skips:
            x_skips = simplify_skips(self.ipts.skips)
        if self.enabled_wgts and self.wgts.skips:
            w_skips = simplify_skips(self.wgts.skips)
        skips_map = {}
        if y_skips or x_skips or w_skips:
            skip_name = "-skip"
            skip_name_list: list[tuple[str, set]] = []
            if y_skips:
                skip_name_list.append(("y", y_skips))
            if x_skips:
                skip_name_list.append(("x", x_skips))
            if w_skips:
                skip_name_list.append(("w", w_skips))
            # sort the keys by the number of elements in the set
            skip_name_list = sorted(skip_name_list, key=lambda x: (len(x[1]), x[0]), reverse=False)
            skips_map = {k: v for k, v in skip_name_list}  # noqa: C416
            skip_name_map: dict[str, set] = {}
            skip_0, skip_0_names = skip_name_list[0]
            skip_name_map[skip_0] = skip_0_names
            if len(skip_name_list) > 1:
                skip_1, skip_1_names = skip_name_list[1]
                if skip_1_names.issuperset(skip_0_names):
                    skip_1_names = skip_1_names - skip_0_names
                    skip_1_names.add(f"[{skip_0}]")
                skip_name_map[skip_1] = skip_1_names
                if len(skip_name_list) > 2:
                    skip_2, skip_2_names = skip_name_list[2]
                    if skip_2_names.issuperset(skip_name_list[1][1]):  # skip_1_names may be modified
                        skip_2_names = skip_2_names - skip_name_list[1][1]
                        skip_2_names.add(f"[{skip_1}]")
                    if skip_2_names.issuperset(skip_0_names):
                        skip_2_names = skip_2_names - skip_0_names
                        skip_2_names.add(f"[{skip_0}]")
                    skip_name_map[skip_2] = skip_2_names
            if "y" in skip_name_map:
                skip_name += f".y.[{'+'.join(sorted(skip_name_map['y']))}]"
            if "x" in skip_name_map:
                skip_name += f".x.[{'+'.join(sorted(skip_name_map['x']))}]"
            if "w" in skip_name_map:
                skip_name += f".w.[{'+'.join(sorted(skip_name_map['w']))}]"
            del skip_name_list, skip_name_map
        lowrank_name = ""
        if self.enabled_wgts and self.wgts.enabled_low_rank:
            lowrank_name = f"-low.r{num2str(self.wgts.low_rank.rank)}"
            if self.wgts.low_rank.num_iters > 1:
                lowrank_name += f".i{num2str(self.wgts.low_rank.num_iters)}"
                if self.wgts.low_rank.early_stop:
                    lowrank_name += ".e"
            if self.wgts.low_rank.exclusive:
                lowrank_name += ".s"
            if self.wgts.low_rank.compensate:
                lowrank_name += ".c"
            if self.wgts.low_rank.objective != SearchBasedCalibObjective.OutputsError:
                lowrank_name += f".{self.wgts.low_rank.objective.name}"
            if self.wgts.low_rank.skips:
                lowrank_skips = simplify_skips(self.wgts.low_rank.skips)
                if "w" in skips_map and lowrank_skips.issuperset(skips_map["w"]):
                    lowrank_skips = lowrank_skips - skips_map["w"]
                    lowrank_skips.add("[w]")
                lowrank_name += ".skip.[{}]".format("+".join(sorted(lowrank_skips)))
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
                if self.smooth.proj.allow_low_rank:
                    smooth_name += ".lr"
                if not self.smooth.proj.allow_b_quant or not self.smooth.proj.allow_a_quant:
                    smooth_name += ".no.["
                    if not self.smooth.proj.allow_a_quant:
                        smooth_name += "a+"
                    if not self.smooth.proj.allow_b_quant:
                        smooth_name += "b+"
                    smooth_name = smooth_name[:-1] + "]"
                if self.smooth.proj.skips:
                    smooth_skips = simplify_skips(self.smooth.proj.skips)
                    if "w" in skips_map and smooth_skips.issuperset(skips_map["w"]):
                        smooth_skips = smooth_skips - skips_map["w"]
                        smooth_skips.add("[w]")
                    smooth_name += ".skip.[{}]".format("+".join(sorted(smooth_skips)))
            if self.smooth.enabled_attn:
                smooth_name += ".yx"
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
                wrange_skips = simplify_skips(self.wgts.calib_range.skips)
                if "w" in skips_map and wrange_skips.issuperset(skips_map["w"]):
                    wrange_skips = wrange_skips - skips_map["w"]
                    wrange_skips.add("[w]")
                wrange_name += ".skip.[{}]".format("+".join(sorted(wrange_skips)))
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
                xrange_skips = simplify_skips(self.ipts.calib_range.skips)
                if "x" in skips_map and xrange_skips.issuperset(skips_map["x"]):
                    xrange_skips = xrange_skips - skips_map["x"]
                    xrange_skips.add("[x]")
                xrange_name += ".skip.[{}]".format("+".join(sorted(xrange_skips)))
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
                yrange_skips = simplify_skips(self.opts.calib_range.skips)
                if "y" in skips_map and yrange_skips.issuperset(skips_map["y"]):
                    yrange_skips = yrange_skips - skips_map["y"]
                    yrange_skips.add("[y]")
                yrange_name += ".skip.[{}]".format("+".join(sorted(yrange_skips)))
        name = skip_name + lowrank_name + smooth_name + wrange_name + xrange_name + yrange_name
        name = name[1:] if name else "default"
        name += f"-{self.calib.generate_dirnames()[0]}"
        return name

    @classmethod
    def set_key_map(cls, key_map: dict[str, set[str]]) -> None:
        """Set the key map for the language model quantization configuration.

        Args:
            key_map (dict[str, set[str]]): The key map.
        """
        cls._key_map = key_map

    def organize(self) -> dict[str, bool]:  # noqa: C901
        """Organize the flags for the diffusion model quantization configuration.

        Returns:
            dict[str, bool]: The organized flags.
        """
        key_map = type(self)._key_map
        wgts_skip_set, ipts_skip_set, opts_skip_set = set(), set(), set()
        if self.wgts is not None:
            wgts_skips = []
            for skip in self.wgts.skips:
                wgts_skips.extend(list(key_map[skip]))
            wgts_skip_set = set(wgts_skips)
            self.wgts.skips = sorted(wgts_skip_set)
            if self.wgts.low_rank is not None:
                wgts_low_rank_skips = []
                for skip in self.wgts.low_rank.skips:
                    wgts_low_rank_skips.extend(list(key_map[skip]))
                self.wgts.low_rank.skips = sorted(set(wgts_low_rank_skips) - wgts_skip_set)
            if self.wgts.calib_range is not None:
                wgts_calib_range_skips = []
                for skip in self.wgts.calib_range.skips:
                    wgts_calib_range_skips.extend(list(key_map[skip]))
                self.wgts.calib_range.skips = sorted(set(wgts_calib_range_skips) - wgts_skip_set)
        if self.ipts is not None:
            ipts_skips = []
            for skip in self.ipts.skips:
                ipts_skips.extend(list(key_map[skip]))
            ipts_skip_set = set(ipts_skips)
            self.ipts.skips = sorted(ipts_skip_set)
            if self.ipts.calib_range is not None:
                ipts_calib_range_skips = []
                for skip in self.ipts.calib_range.skips:
                    ipts_calib_range_skips.extend(list(key_map[skip]))
                self.ipts.calib_range.skips = sorted(set(ipts_calib_range_skips) - ipts_skip_set)
        if self.opts is not None:
            opts_skips = []
            for skip in self.opts.skips:
                opts_skips.extend(list(key_map[skip]))
            opts_skip_set = set(opts_skips)
            self.opts.skips = sorted(opts_skip_set)
            if self.opts.calib_range is not None:
                opts_calib_range_skips = []
                for skip in self.opts.calib_range.skips:
                    opts_calib_range_skips.extend(list(key_map[skip]))
                self.opts.calib_range.skips = sorted(set(opts_calib_range_skips) - opts_skip_set)
        if self.smooth is not None and self.smooth.proj is not None:
            smooth_proj_skips = []
            for skip in self.smooth.proj.skips:
                smooth_proj_skips.extend(list(key_map[skip]))
            self.smooth.proj.skips = sorted(set(smooth_proj_skips) - (wgts_skip_set & ipts_skip_set))
