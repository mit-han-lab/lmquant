# -*- coding: utf-8 -*-
"""Diffusion model evaluation."""

import logging
import os
import typing as tp
from dataclasses import dataclass, field

import datasets
import diffusers
import omniconfig
import torch
from diffusers import DiffusionPipeline
from omniconfig import configclass
from torch import multiprocessing as mp
from tqdm import tqdm

from deepcompressor.utils.common import hash_str_to_int

from .benchmarks import get_dataset
from .metrics import compute_image_metrics

__all__ = ["DiffusionEvalConfig"]


@configclass
@dataclass
class DiffusionEvalConfig:
    """Diffusion model evaluation configuration.

    Args:
        protocol (`str`):
            The protocol of the evaluation pipeline.
        num_gpus (`int`, *optional*, defaults to `1`):
            The number of GPUs to use.
        batch_size (`int`, *optional*, defaults to `1`):
            The batch size used for inference.
        batch_size_per_gpu (`int`, *optional*, defaults to `None`):
            The batch size per GPU.
        height (`int`, *optional*, defaults to `None`):
            The height of the generated images.
        width (`int`, *optional*, defaults to `None`):
            The width of the generated images.
        num_steps (`int`, *optional*, defaults to `None`):
            The number of inference steps.
        guidance_scale (`float`, *optional*, defaults to `None`):
            The guidance scale.
        num_samples (`int`, *optional*, defaults to `1024`):
            The number of samples to generate.
        benchmarks (`list[str]`, *optional*, defaults to `["COCO", "DCI", "MJHQ", "GenEval"]`):
            The benchmark datasets to evaluate on.
        gt_metrics (`list[str]`, *optional*, defaults to `["clip_iqa", "clip_score", "psnr", "lpips", "ssim", "fid"]`):
            The ground truth metrics to compute.
        ref_metrics (`list[str]`, *optional*, defaults to `["psnr", "lpips", "ssim", "fid"]`):
            The reference metrics to compute.
        ref_root (`str`, *optional*, defaults to `""`):
            The root directory path to the reference images.
        gt_stats_root (`str`, *optional*, defaults to `""`):
            The root directory path to the ground truth statistics.
        chunk_start (`int`, *optional*, defaults to `0`):
            The starting chunk index.
        chunk_step (`int`, *optional*, defaults to `1`):
            The chunk step size.
    """

    protocol: str

    num_gpus: int = field(default=1, metadata={omniconfig.ARGPARSE_ARGS: ("--num-gpus", "-n")})
    batch_size: int = 1
    batch_size_per_gpu: int | None = None

    height: int | None = None
    width: int | None = None
    num_steps: int | None = None
    guidance_scale: float | None = None
    num_samples: int = 1024

    benchmarks: list[str] = field(
        default_factory=lambda: ["COCO", "DCI", "MJHQ", "GenEval"],
        metadata={omniconfig.ARGPARSE_KWARGS: {"nargs": "+", "type": str}},
    )
    gt_metrics: list[str] = field(
        default_factory=lambda: ["clip_iqa", "clip_score", "image_reward", "fid"],
        metadata={omniconfig.ARGPARSE_KWARGS: {"nargs": "+", "type": str}},
    )
    ref_metrics: list[str] = field(
        default_factory=lambda: ["psnr", "lpips", "ssim", "fid"],
        metadata={omniconfig.ARGPARSE_KWARGS: {"nargs": "+", "type": str}},
    )
    gen_root: str = ""
    ref_root: str = ""
    gt_stats_root: str = ""

    chunk_start: int = 0
    chunk_step: int = 1
    chunk_only: bool = False

    def __post_init__(self):
        assert self.protocol
        self.protocol = self.protocol.lower().format(num_steps=self.num_steps, guidance_scale=self.guidance_scale)
        assert 0 <= self.chunk_start < self.chunk_step
        if self.chunk_start == 0 and self.chunk_step == 1:
            self.chunk_only = False

    def get_pipeline_kwargs(self) -> dict[str, tp.Any]:
        kwargs = {}
        if self.height is not None:
            kwargs["height"] = self.height
        if self.width is not None:
            kwargs["width"] = self.width
        if self.num_steps is not None:
            kwargs["num_inference_steps"] = self.num_steps
        if self.guidance_scale is not None:
            kwargs["guidance_scale"] = self.guidance_scale
        return kwargs

    def _generate(
        self,
        rank: int,
        dataset: datasets.Dataset,
        pipeline: DiffusionPipeline,
        dirpath: str,
        logger: logging.Logger,
        dataset_name: str | None = None,
    ) -> None:
        if self.num_gpus > 1:
            pipeline = pipeline.to(rank)
        if rank == 0:
            logger.info(
                f"  {dataset.config_name} has {len(dataset)} samples "
                f"(chunk_start={dataset._chunk_start}, chunk_step={dataset._chunk_step},"
                f" unchunk_size={dataset._unchunk_size})"
            )
        pipeline.set_progress_bar_config(
            desc="Sampling", leave=False, dynamic_ncols=True, position=1, disable=self.num_gpus > 1
        )
        if dataset_name is None:
            dataset_name = dataset.config_name
        for batch in tqdm(
            dataset.iter(batch_size=self.batch_size, drop_last_batch=False),
            desc=dataset_name if self.num_gpus == 1 else f"{dataset_name} (GPU {rank})",
            leave=False,
            dynamic_ncols=True,
            position=rank,
            total=(len(dataset) + self.batch_size - 1) // self.batch_size,
        ):
            filenames = batch["filename"][rank :: self.num_gpus]
            if len(filenames) == 0:
                continue
            if all(os.path.exists(os.path.join(dirpath, f"{filename}.png")) for filename in filenames):
                continue
            prompts = batch["prompt"][rank :: self.num_gpus]
            seeds = [hash_str_to_int(name) for name in filenames]
            diffusers.training_utils.set_seed(seeds[0])
            generators = [torch.Generator().manual_seed(seed) for seed in seeds]
            output = pipeline(prompts, generator=generators, **self.get_pipeline_kwargs())
            images = output.images
            for filename, image in zip(filenames, images, strict=True):
                image.save(os.path.join(dirpath, f"{filename}.png"))

    def generate(self, pipeline: DiffusionPipeline, gen_root: str = "") -> None:
        logger = logging.getLogger(f"{__name__}.DiffusionEval")
        gen_root = gen_root or self.gen_root
        for benchmark in self.benchmarks:
            dataset = get_dataset(
                benchmark, max_dataset_size=self.num_samples, chunk_start=self.chunk_start, chunk_step=self.chunk_step
            )
            if benchmark.endswith(".yaml") or benchmark.endswith(".yml"):
                dataset_name = os.path.splitext(os.path.basename(benchmark))[0]
                dirpath = os.path.join(gen_root, "samples", "YAML", f"{dataset_name}-{dataset._unchunk_size}")
            else:
                dataset_name = dataset.config_name
                dirpath = os.path.join(gen_root, "samples", benchmark, f"{dataset.config_name}-{dataset._unchunk_size}")
            if self.chunk_only:
                dirpath += f".{dataset._chunk_start}.{dataset._chunk_step}"
            os.makedirs(dirpath, exist_ok=True)
            args = (dataset, pipeline, dirpath, logger, dataset_name)
            if self.num_gpus == 1:
                self._generate(0, *args)
            else:
                mp.spawn(self._generate, args=args, nprocs=self.num_gpus, join=True)

    def evaluate(
        self, pipeline: DiffusionPipeline, gen_root: str = "", skip_gen: bool = False
    ) -> dict[str, tp.Any] | None:
        gen_root = gen_root or self.gen_root
        if not skip_gen:
            self.generate(pipeline, gen_root=gen_root)
        return compute_image_metrics(
            gen_root=gen_root,
            benchmarks=self.benchmarks,
            max_dataset_size=self.num_samples,
            chunk_start=self.chunk_start,
            chunk_step=self.chunk_step,
            ref_root=self.ref_root,
            gt_stats_root=self.gt_stats_root,
            gt_metrics=self.gt_metrics,
            ref_metrics=self.ref_metrics,
        )
