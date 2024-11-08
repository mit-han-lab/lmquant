# -*- coding: utf-8 -*-
"""Configurations for evaluating a large language model."""

import argparse
import os
import random
import typing as tp
from dataclasses import dataclass, field

import numpy as np
import omniconfig
import torch
from omniconfig import ConfigParser, configclass

from deepcompressor.data.utils import ScaleUtils
from deepcompressor.utils.config.output import OutputConfig

from .cache.config import LlmCacheConfig, LlmQuantCacheConfig
from .eval.config import LlmEvalConfig
from .model.config import LlmModelConfig
from .quant.config import LlmQuantConfig

__all__ = [
    "LlmPtqRunConfig",
    "LlmCacheConfig",
    "LlmQuantCacheConfig",
    "LlmEvalConfig",
    "LlmModelConfig",
    "LlmQuantConfig",
]


@configclass
@dataclass
class LlmPtqRunConfig:
    """Top-level config of post-training quantization for a large language model.

    Args:
        cache (`LlmCacheConfig`):
            Large language model quantization cache path configuration.
        output (`OutputConfig`):
            Output directory configuration.
        model (`LlmModelConfig`):
            Large language model configuration.
        eval (`LlmEvalConfig`):
            Large language model evaluation configuration.
        quant (`LlmQuantConfig`):
            Large language model quantization configuration.
        seed (`int`, *optional*, defaults to `12345`):
            Random seed.
        skip_eval (`bool`, *optional*, defaults to `False`):
            Whether to skip evaluation.
        load_model (`str`, *optional*, defaults to `""`):
            Directory path to load the model checkpoint.
        save_model (`str`, *optional*, defaults to `""`):
            Directory path to save the model checkpoint.
        copy_on_save (`bool`, *optional*, defaults to `False`):
            Whether to copy the quantization cache on save.
    """

    cache: LlmCacheConfig
    output: OutputConfig
    model: LlmModelConfig
    eval: LlmEvalConfig
    quant: LlmQuantConfig = field(metadata={omniconfig.ARGPARSE_KWARGS: {"prefix": ""}})
    seed: int = 12345
    skip_eval: bool = False
    load_from: str = ""
    save_model: str = ""
    copy_on_save: bool = False

    def __post_init__(self):  # noqa: C901
        # region set scale default dtype
        if self.quant.enabled_wgts:
            self.quant.wgts.scale_dtypes = tuple(
                ScaleUtils.infer_scale_dtypes(self.quant.wgts.scale_dtypes, default_dtype=self.model.dtype)
            )
        if self.quant.enabled_ipts:
            self.quant.ipts.scale_dtypes = tuple(
                ScaleUtils.infer_scale_dtypes(self.quant.ipts.scale_dtypes, default_dtype=self.model.dtype)
            )
        if self.quant.enabled_opts:
            self.quant.opts.scale_dtypes = tuple(
                ScaleUtils.infer_scale_dtypes(self.quant.opts.scale_dtypes, default_dtype=self.model.dtype)
            )
        # endregion
        # region set num_gpus and batch_size for auto parallelism of large models
        self.eval.num_gpus = min(torch.cuda.device_count(), self.eval.num_gpus)
        if self.model.size < 50:
            self.eval.batch_size = min(8, self.eval.batch_size)
        elif self.model.size < 100:
            self.eval.batch_size = min(4, self.eval.batch_size)
        else:
            self.eval.batch_size = min(1, self.eval.batch_size)
        # endregion
        if self.quant.is_enabled():
            if self.cache.path.is_all_empty():
                self.cache.dirpath = self.quant.generate_cache_dirpath(
                    root=self.cache.root, seed=self.seed, default_dtype=self.model.dtype
                )
                self.cache.path = self.cache.dirpath.clone().add_children(f"{self.model.name}.pt")
            else:
                self.cache.dirpath = self.cache.path.clone().to_dirpath()
        if self.output.dirname == "default":
            self.output.dirname = self.quant.generate_default_dirname()
        self.output.dirpath = os.path.join(
            self.output.root,
            "llm",
            self.model.family,
            self.model.name,
            *self.quant.generate_dirnames(default_dtype=self.model.dtype)[:-1],
            self.quant.generate_calib_dirname(),
            self.output.dirname,
        )
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)

    @classmethod
    def get_parser(cls) -> ConfigParser:
        """Get a parser for evaluating a large language model.

        Returns:
            `ConfigParser`: A parser for evaluating a large language model.
        """
        parser = ConfigParser("Evaluate a large language model")
        parser.add_config(cls)
        return parser
