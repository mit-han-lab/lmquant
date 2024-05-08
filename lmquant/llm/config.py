# -*- coding: utf-8 -*-
"""Configurations for evaluating a large language model."""

import os
import random
import typing as tp
from dataclasses import dataclass, field

import numpy as np
import omniconfig
import torch
from omniconfig import ConfigParser, configclass

from lmquant.quant.config import (
    QuantCachePath,
    QuantChannelOrderCalibConfig,
    SearchBasedCalibGranularity,
    SearchBasedCalibObjective,
    SearchBasedCalibStrategy,
)
from lmquant.utils import num2str

from .dataset import LlmCalibConfig
from .eval import LlmEvalConfig
from .model import LlmModelConfig
from .quant import LlmQuantConfig

__all__ = ["LlmRunConfig"]


@configclass
@dataclass
class LlmRunConfig:
    """Top-level config for evaluating a large language model.

    Args:
        model (LlmModelConfig): Arguments for creating a large language model.
        eval (LlmEvalConfig): Arguments for evaluating a large language model.
        calib (LlmCalibrationConfig): Arguments for collecting calibration inputs.
        quant (ModuleQuantConfig): Arguments for quantizing a large language model.
        seed (int): Random seed. Defaults to ``12345``.
        save_model (bool): Whether to save the quantized model. Defaults to ``False``.
    """

    model: LlmModelConfig
    eval: LlmEvalConfig
    calib: LlmCalibConfig
    quant: LlmQuantConfig = field(metadata={omniconfig.ARGPARSE_KWARGS: {"prefix": ""}})
    seed: int = 12345
    save_model: bool = False
    output_dirpath: str = field(init=False)
    cache_dirpath: QuantCachePath = field(init=False, default_factory=QuantCachePath)
    cache_path: QuantCachePath = field(init=False, default_factory=QuantCachePath)

    def __post_init__(self):  # noqa: C901
        # region set num_gpus and batch_size for auto parallelism of large models
        self.eval.num_gpus = min(torch.cuda.device_count(), self.eval.num_gpus)
        if self.model.size < 50:
            self.eval.batch_size = min(8, self.eval.batch_size)
        elif self.model.size < 100:
            self.eval.batch_size = min(4, self.eval.batch_size)
        else:
            self.eval.batch_size = min(1, self.eval.batch_size)
        # endregion
        if self.quant.enabled_wgts or self.quant.enabled_ipts or self.quant.enabled_opts:
            self.cache_dirpath = self.quant.generate_cache_dirpath().add_parent_dirs(self.calib.cache_dirpath)
            if self.quant.enabled_rotation:
                self.cache_dirpath.rotation = os.path.join(
                    self.calib.cache_root,
                    "llm",
                    "cache",
                    "rotation",
                    f"seed.{self.seed}" if self.quant.rotation.random else "hadamard",
                )
            self.cache_path = self.cache_dirpath.clone().add_chidren(f"{self.model.name}.pt")
        if self.eval.output_dirname_without_timestamp == "default":
            self.eval.output_dirname_without_timestamp = self.generate_output_dirname()
            if self.eval.attach_timestamp:
                self.eval.output_dirname = f"{self.eval.output_dirname_without_timestamp}-{self.eval.timestamp}"
            else:
                self.eval.output_dirname = self.eval.output_dirname_without_timestamp
        self.output_dirpath = os.path.join(
            self.eval.output_root,
            "llm",
            self.model.family,
            self.model.name,
            *self.quant.generate_dirnames()[:-1],
            self.quant.generate_calib_name(),
            self.eval.output_dirname,
        )
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)

    def generate_output_dirname(self) -> str:  # noqa: C901
        """Generate output directory name for evaluating a large language model."""
        xw_names = {"proj_qkv": "qkv", "proj_out": "out", "proj_1st": "fc1", "proj_2nd": "fc2"}
        if "mixtral" in self.model.name.lower():
            xw_names["router"] = "r"
        y_names = {"attn_q": "q", "attn_k": "k", "attn_v": "v"}
        yx_names = {"attn_qk": "qk"}
        skip_name = ""
        if self.quant.enabled_opts:
            skip_y_name = ".y.[{}]".format("+".join(y_names[y] for y in self.quant.opts.skips if y in y_names))
            if skip_y_name != ".y.[]":
                skip_name += skip_y_name
        if self.quant.enabled_wgts:
            skip_w_name = ".w.[{}]".format("+".join(xw_names[w] for w in self.quant.wgts.skips if w in xw_names))
            if skip_w_name != ".w.[]":
                skip_name += skip_w_name
        if self.quant.enabled_ipts:
            skip_x_name = ".x.[{}]".format("+".join(xw_names[x] for x in self.quant.ipts.skips if x in xw_names))
            if skip_x_name != ".x.[]":
                skip_name += skip_x_name
        if skip_name:
            skip_name = "-skip" + skip_name
        if self.quant.enabled_wgts and self.quant.wgts.calib_kernel:
            skip_name += "-krnl"
        rotation_name = ""
        if self.quant.enabled_rotation:
            rotation_name = "-rot"
            if self.quant.rotation.random:
                rotation_name += ".rnd"
            if self.quant.rotation.with_hadamard_transform:
                rotation_name += ".[+{}]".format("+".join(xw_names[w] for w in self.quant.rotation.transforms))
        reorder_name = ""
        if self.quant.enabled_reorder:
            reorder_name = "-rodr"
            metric_names = {
                QuantChannelOrderCalibConfig.ChannelMetric.InputsAbsMax: "xMax",
                QuantChannelOrderCalibConfig.ChannelMetric.InputsAbsMean: "xAvg",
                QuantChannelOrderCalibConfig.ChannelMetric.InputsRootMeanSquare: "xRms",
                QuantChannelOrderCalibConfig.ChannelMetric.WeightsAbsMax: "wMax",
                QuantChannelOrderCalibConfig.ChannelMetric.WeightsAbsMean: "wAvg",
                QuantChannelOrderCalibConfig.ChannelMetric.WeightsRootMeanSquare: "wRms",
                QuantChannelOrderCalibConfig.ChannelMetric.AbsMaxProduct: "pMax",
                QuantChannelOrderCalibConfig.ChannelMetric.AbsMeanProduct: "pAvg",
                QuantChannelOrderCalibConfig.ChannelMetric.RootMeanSquareProduct: "pRms",
            }
            index_names = {
                QuantChannelOrderCalibConfig.ChannelIndex.Sequential: "Seq",
                QuantChannelOrderCalibConfig.ChannelIndex.Transpose: "Trp",
            }
            if self.quant.reorder.strategy == SearchBasedCalibStrategy.Manual:
                if self.quant.reorder.channel_metric != QuantChannelOrderCalibConfig.ChannelMetric.InputsAbsMax:
                    reorder_name += f".{metric_names[self.quant.reorder.channel_metric]}"
                reorder_name += f".{index_names[self.quant.reorder.channel_index]}"
            else:
                reorder_name += f".{self.quant.reorder.strategy.name}"
            reorders, skips = [], []
            for k in xw_names.keys() if self.quant.reorder.dynamic else ("residual", "proj_out", "proj_2nd"):
                v = xw_names.get(k, "res")
                if k in self.quant.reorder.skips:
                    skips.append(v)
                else:
                    reorders.append(v)
            if len(reorders) <= len(skips):
                reorder_name += ".[{}]".format("+".join(reorders))
            elif skips:
                reorder_name += ".skip.[{}]".format("+".join(skips))
            if self.quant.reorder.allow_kernel_calib:
                reorder_name += ".krnl"
        smooth_name = ""
        if self.quant.enabled_smooth:
            smooth_name = "-smth"
            if self.quant.smooth.enabled_smooth_xw:
                smooth_name += ".xw"
                if self.quant.smooth.xw.granularity != SearchBasedCalibGranularity.Layer:
                    smooth_name += f".{self.quant.smooth.xw.granularity.name}"
                if self.quant.smooth.xw.strategy != SearchBasedCalibStrategy.Manual:
                    smooth_name += f".{self.quant.smooth.xw.strategy.name}"
                    if self.quant.smooth.xw.alpha <= 0:
                        smooth_name += f".a{num2str(self.quant.smooth.xw.alpha)}"
                    if self.quant.smooth.xw.beta <= 0:
                        smooth_name += f".b{num2str(self.quant.smooth.xw.beta)}"
                else:
                    smooth_name += f".a{num2str(self.quant.smooth.xw.alpha)}"
                    smooth_name += f".b{num2str(self.quant.smooth.xw.beta)}"
                xrange_eq_wrange = True
                for xrange, wrange in self.quant.smooth.xw.ranges:
                    if xrange != wrange:
                        xrange_eq_wrange = False
                        break
                if xrange_eq_wrange:
                    smooth_name += ".[{}]".format("+".join(xrange.name for xrange, _ in self.quant.smooth.xw.ranges))
                else:
                    smooth_name += ".[{}]".format(
                        "+".join(f"x.{xrange.name}.w.{wrange.name}" for xrange, wrange in self.quant.smooth.xw.ranges)
                    )
                smooths, skips = [], []
                for k, v in xw_names.items():
                    if k in self.quant.smooth.xw.skips:
                        skips.append(v)
                    else:
                        smooths.append(v)
                if len(smooths) <= len(skips):
                    smooth_name += ".[{}]".format("+".join(smooths))
                elif skips:
                    smooth_name += ".skip.[{}]".format("+".join(skips))
                if (
                    self.quant.smooth.xw.strategy != SearchBasedCalibStrategy.Manual
                    and self.quant.smooth.xw.allow_kernel_calib
                ):
                    smooth_name += ".krnl"
            if self.quant.smooth.enabled_smooth_yx:
                smooth_name += ".yx"
                if self.quant.smooth.yx.granularity != SearchBasedCalibGranularity.Layer:
                    smooth_name += f".{self.quant.smooth.yx.granularity.name}"
                if self.quant.smooth.yx.strategy != SearchBasedCalibStrategy.Manual:
                    smooth_name += f".{self.quant.smooth.yx.strategy.name}"
                    if self.quant.smooth.yx.alpha <= 0:
                        smooth_name += f".a{num2str(self.quant.smooth.yx.alpha)}"
                    if self.quant.smooth.yx.beta <= 0:
                        smooth_name += f".b{num2str(self.quant.smooth.yx.beta)}"
                else:
                    smooth_name += f".a{num2str(self.quant.smooth.yx.alpha)}"
                    smooth_name += f".b{num2str(self.quant.smooth.yx.beta)}"
                xrange_eq_yrange = True
                for xrange, yrange in self.quant.smooth.yx.ranges:
                    if xrange != yrange:
                        xrange_eq_yrange = False
                        break
                if xrange_eq_yrange:
                    smooth_name += ".[{}]".format("+".join(xrange.name for xrange, _ in self.quant.smooth.yx.ranges))
                else:
                    smooth_name += ".[{}]".format(
                        "+".join(f"x.{xrange.name}.y.{yrange.name}" for xrange, yrange in self.quant.smooth.yx.ranges)
                    )
                smooths = []
                for k, v in yx_names.items():
                    if k not in self.quant.smooth.yx.skips:
                        smooths.append(v)
                smooth_name += ".[{}]".format("+".join(smooths))
        wrange_name = ""
        if (
            self.quant.enabled_wgts
            and self.quant.wgts.enabled_calib_range
            and (self.quant.wgts.calib_range.needs_search or self.quant.wgts.calib_range.ratio != 1)
        ):
            wrange_name = "-w.range"
            if self.quant.wgts.calib_range.needs_search:
                if self.quant.wgts.calib_range.granularity != SearchBasedCalibGranularity.Group:
                    wrange_name += f".{self.quant.wgts.calib_range.granularity.name}"
                if self.quant.wgts.calib_range.objective != SearchBasedCalibObjective.OutputsError:
                    wrange_name += f".{self.quant.wgts.calib_range.objective.name}"
                if self.quant.wgts.calib_range.degree != 2:
                    wrange_name += f".d{num2str(self.quant.wgts.calib_range.degree)}"
                wrange_name += f".[{num2str(self.quant.wgts.calib_range.max_shrink)}"
                wrange_name += f".{num2str(self.quant.wgts.calib_range.max_expand)}"
                wrange_name += f".g{self.quant.wgts.calib_range.num_grids}]"
                if self.quant.wgts.calib_range.allow_kernel_calib:
                    wrange_name += ".krnl"
            else:
                wrange_name += f".r{num2str(self.quant.wgts.calib_range.ratio)}"
            if self.quant.wgts.calib_range.skips:
                wrange_name += ".skip.[{}]".format("+".join(xw_names[w] for w in self.quant.wgts.calib_range.skips))
        xrange_name = ""
        if (
            self.quant.enabled_ipts
            and self.quant.ipts.enabled_calib_range
            and (self.quant.ipts.calib_range.needs_search or self.quant.ipts.calib_range.ratio != 1)
        ):
            xrange_name = "-x.range"
            if self.quant.ipts.calib_range.needs_search:
                if self.quant.ipts.calib_range.granularity != SearchBasedCalibGranularity.Group:
                    xrange_name += f".{self.quant.ipts.calib_range.granularity.name}"
                if self.quant.ipts.calib_range.objective != SearchBasedCalibObjective.OutputsError:
                    xrange_name += f".{self.quant.ipts.calib_range.objective.name}"
                if self.quant.ipts.calib_range.degree != 2:
                    xrange_name += f".d{num2str(self.quant.ipts.calib_range.degree)}"
                xrange_name += f".[{num2str(self.quant.ipts.calib_range.max_shrink)}"
                xrange_name += f".{num2str(self.quant.ipts.calib_range.max_expand)}"
                xrange_name += f".g{self.quant.ipts.calib_range.num_grids}]"
                if self.quant.ipts.calib_range.allow_kernel_calib:
                    xrange_name += ".krnl"
            else:
                xrange_name += f".r{num2str(self.quant.ipts.calib_range.ratio)}"
            if self.quant.ipts.calib_range.skips:
                xrange_name += ".skip.[{}]".format("+".join(xw_names[w] for w in self.quant.ipts.calib_range.skips))
        yrange_name = ""
        if (
            self.quant.enabled_opts
            and self.quant.opts.enabled_calib_range
            and (self.quant.opts.calib_range.needs_search or self.quant.opts.calib_range.ratio != 1)
        ):
            yrange_name = "-y.range"
            if self.quant.opts.calib_range.needs_search:
                if self.quant.opts.calib_range.granularity != SearchBasedCalibGranularity.Group:
                    yrange_name += f".{self.quant.opts.calib_range.granularity.name}"
                if self.quant.opts.calib_range.objective != SearchBasedCalibObjective.OutputsError:
                    yrange_name += f".{self.quant.opts.calib_range.objective.name}"
                if self.quant.opts.calib_range.degree != 2:
                    yrange_name += f".d{num2str(self.quant.opts.calib_range.degree)}"
                yrange_name += f".[{num2str(self.quant.opts.calib_range.max_shrink)}"
                yrange_name += f".{num2str(self.quant.opts.calib_range.max_expand)}"
                yrange_name += f".g{self.quant.opts.calib_range.num_grids}]"
                if self.quant.opts.calib_range.allow_kernel_calib:
                    yrange_name += ".krnl"
            else:
                yrange_name += f".r{num2str(self.quant.opts.calib_range.ratio)}"
            if self.quant.opts.calib_range.skips:
                yrange_name += ".skip.[{}]".format("+".join(yx_names[y] for y in self.quant.opts.calib_range.skips))
        name = skip_name + rotation_name + reorder_name + smooth_name + wrange_name + xrange_name + yrange_name
        name = name[1:] if name else "default"
        name += f"-{self.calib.generate_dirnames()[0]}"
        return name

    @staticmethod
    def parse_args(args: tp.Any = None) -> tuple["LlmRunConfig", dict[str, dict], list[str]]:
        """Parse arguments for evaluating a large language model.

        Args:
            args (list[str], optional): Arguments to parse. Defaults to ``None``.

        Returns:
            tuple[Config, dict[str, dict], list[str]]: Configs from the parsed arguments,
                                                       parsed yaml configs, and unknown arguments.
        """
        parser = ConfigParser("Evaluate a large language model")
        parser.add_config(LlmRunConfig, **LlmQuantConfig.generate_flags())
        config, parsed_args, unknown_args = parser.parse_known_args(args)
        assert isinstance(config, LlmRunConfig)
        return config, parsed_args, unknown_args

    @staticmethod
    def dump_default(path: str = "__default__.yaml") -> None:
        """Dump default configurations for evaluating a large language model."""
        parser = ConfigParser("Evaluate a large language model")
        parser.add_config(LlmRunConfig, **LlmQuantConfig.generate_flags())
        parser.dump_default(path)
