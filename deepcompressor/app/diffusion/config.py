# -*- coding: utf-8 -*-
"""Top-level config of post-training quantization for a diffusion model."""

import os
from dataclasses import dataclass, field

import diffusers.training_utils
import omniconfig
import torch
from omniconfig import ConfigParser, configclass

from deepcompressor.app.llm.config import LlmCacheConfig, LlmQuantConfig
from deepcompressor.data.utils import ScaleUtils
from deepcompressor.utils.config.output import OutputConfig

from .cache import DiffusionPtqCacheConfig, DiffusionQuantCacheConfig
from .eval import DiffusionEvalConfig
from .nn.struct import DiffusionModelStruct
from .pipeline import DiffusionPipelineConfig
from .quant import DiffusionQuantConfig

__all__ = [
    "DiffusionPtqRunConfig",
    "DiffusionPtqCacheConfig",
    "DiffusionQuantCacheConfig",
    "DiffusionEvalConfig",
    "DiffusionPipelineConfig",
    "DiffusionQuantConfig",
]


@configclass
@dataclass
class DiffusionPtqRunConfig:
    """Top-level config of post-training quantization for a diffusion model.

    Args:
        cache (`DiffusionPtqCacheConfig`):
            The cache configuration.
        output (`OutputConfig`):
            The output directory configuration.
        pipeline (`DiffusionPipelineConfig`):
            The diffusion pipeline configuration
        eval (`DiffusionEvalConfig`):
            The evaluation configuration.
        quant (`DiffusionQuantConfig`):
            The post-training quantization configuration.
        seed (`int`, *optional*, defaults to `12345`):
            The seed for reproducibility.
        skip_gen (`bool`, *optional*, defaults to `False`):
            Whether to skip generation.
        skip_eval (`bool`, *optional*, defaults to `False`):
            Whether to skip evaluation.
        load_model (`str`, *optional*, defaults to `""`):
            Directory path to load the model checkpoint.
        save_model (`str`, *optional*, defaults to `""`):
            Directory path to save the model checkpoint.
        copy_on_save (`bool`, *optional*, defaults to `False`):
            Whether to copy the quantization cache on save.
    """

    cache: DiffusionPtqCacheConfig | None
    output: OutputConfig
    pipeline: DiffusionPipelineConfig
    eval: DiffusionEvalConfig
    quant: DiffusionQuantConfig = field(metadata={omniconfig.ARGPARSE_KWARGS: {"prefix": ""}})
    text: LlmQuantConfig | None = None
    text_cache: LlmCacheConfig = field(default_factory=LlmCacheConfig)
    seed: int = 12345
    skip_gen: bool = False
    skip_eval: bool = False
    load_from: str = ""
    save_model: str = ""
    copy_on_save: bool = False

    def __post_init__(self):
        # region set text encoder quanatization scale default dtype
        if self.text is not None and self.text.enabled_wgts:
            self.text.wgts.scale_dtypes = tuple(
                ScaleUtils.infer_scale_dtypes(self.text.wgts.scale_dtypes, default_dtype=self.pipeline.dtype)
            )
        if self.text is not None and self.text.enabled_ipts:
            self.text.ipts.scale_dtypes = tuple(
                ScaleUtils.infer_scale_dtypes(self.text.ipts.scale_dtypes, default_dtype=self.pipeline.dtype)
            )
        if self.text is not None and self.text.enabled_opts:
            self.text.opts.scale_dtypes = tuple(
                ScaleUtils.infer_scale_dtypes(self.text.opts.scale_dtypes, default_dtype=self.pipeline.dtype)
            )
        # endregion
        self.eval.num_gpus = min(torch.cuda.device_count(), self.eval.num_gpus)
        if self.eval.batch_size_per_gpu is None:
            self.eval.batch_size_per_gpu = max(1, self.eval.batch_size // self.eval.num_gpus)
            self.eval.batch_size = self.eval.batch_size_per_gpu * self.eval.num_gpus
        else:
            self.eval.batch_size = self.eval.batch_size_per_gpu * self.eval.num_gpus
        # region setup calib dataset path
        self.quant.calib.path = self.quant.calib.path.format(
            dtype=self.pipeline.dtype,
            family=self.pipeline.family,
            model=self.pipeline.name,
            protocol=self.eval.protocol,
            data=self.quant.calib.data,
        )
        if self.quant.calib.path:
            self.quant.calib.path = os.path.abspath(os.path.expanduser(self.quant.calib.path))
        # endregion
        # region setup eval reference root
        self.eval.ref_root = self.eval.ref_root.format(
            dtype=self.pipeline.dtype,
            family=self.pipeline.family,
            model=self.pipeline.name,
            protocol=self.eval.protocol,
        )
        if self.eval.ref_root:
            self.eval.ref_root = os.path.abspath(os.path.expanduser(self.eval.ref_root))
        # endregion
        # region setup cache directory
        if self.cache is not None:
            if self.quant.enabled_wgts or self.quant.enabled_ipts or self.quant.enabled_opts:
                self.cache.dirpath = self.quant.generate_cache_dirpath(
                    root=self.cache.root, shift=self.pipeline.shift_activations, default_dtype=self.pipeline.dtype
                )
                self.cache.path = self.cache.dirpath.clone().add_children(f"{self.pipeline.name}.pt")
            else:
                self.cache.dirpath = self.cache.path = None
        if self.text is not None and self.text.is_enabled():
            if not self.text_cache.root:
                self.text_cache.root = os.path.join(self.cache.root, "diffusion")
            self.text_cache.dirpath = self.text.generate_cache_dirpath(root=self.text_cache.root, seed=self.seed)
            self.text_cache.path = self.text_cache.dirpath.clone().add_children(f"{self.pipeline.name}.pt")
        # endregion
        # region setup output directory
        if self.output.dirname == "reference":
            assert self.eval.ref_root
            self.output.job = f"run-{self.eval.num_samples}"
            self.output.dirpath = self.eval.ref_root
            self.eval.ref_root = ""
            self.eval.gen_root = "{output}"
        else:
            if self.output.dirname == "default":
                self.output.dirname = self.generate_default_dirname()
            calib_dirname = self.quant.generate_calib_dirname() or "-"
            self.output.dirpath = os.path.join(
                self.output.root,
                "diffusion",
                self.pipeline.family,
                self.pipeline.name,
                *self.quant.generate_dirnames(default_dtype=self.pipeline.dtype)[:-1],
                calib_dirname,
                self.output.dirname,
            )
        if (self.eval.chunk_start > 0 or self.eval.chunk_step > 1) and not self.eval.chunk_only:
            self.output.job += f".c{self.eval.chunk_start}.{self.eval.chunk_step}"
        # endregion
        diffusers.training_utils.set_seed(self.seed)

    def generate_default_dirname(self) -> str:
        name = "-shift" if self.pipeline.shift_activations else ""
        if self.quant.is_enabled():
            name += f"-{self.quant.generate_default_dirname()}"
        if self.text is not None and self.text.is_enabled():
            name += f"-text-{self.text.generate_default_dirname()}"
        size_name = ""
        if self.eval.height:
            size_name += f".h{self.eval.height}"
        if self.eval.width:
            size_name += f".w{self.eval.width}"
        if size_name:
            name += f"-{size_name[1:]}"
        sampling_name = ""
        if self.eval.num_steps is not None:
            sampling_name += f".t{self.eval.num_steps}"
        if self.eval.guidance_scale is not None:
            sampling_name += f".g{self.eval.guidance_scale}"
        if sampling_name:
            name += f"-{sampling_name[1:]}"
        if self.eval.num_samples != -1:
            name += f"-s{self.eval.num_samples}"
            if self.eval.chunk_only:
                name += f".c{self.eval.chunk_start}.{self.eval.chunk_step}"
        assert name[0] == "-"
        return name[1:]

    @classmethod
    def get_parser(cls) -> ConfigParser:
        """Get a parser for post-training quantization of a diffusion model.

        Returns:
            `ConfigParser`:
                A parser for post-training quantization of a diffusion model.
        """
        parser = ConfigParser("Diffusion Run configuration")
        DiffusionQuantConfig.set_key_map(DiffusionModelStruct._get_default_key_map())
        parser.add_config(cls)
        return parser
