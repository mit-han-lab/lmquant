# -*- coding: utf-8 -*-
"""Collect calibration dataset."""

import os
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import yaml
from omniconfig import configclass
from tqdm import trange

from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
from deepcompressor.utils.common import hash_str_to_int, tree_map

from .utils import CollectHook


def process(x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    return torch.from_numpy(x.float().numpy()).to(dtype)


def collect(config: DiffusionPtqRunConfig, filenames: list[str], dataset: dict[str, str]):
    samples_dirpath = os.path.join(config.output.root, "samples")
    caches_dirpath = os.path.join(config.output.root, "caches")
    os.makedirs(samples_dirpath, exist_ok=True)
    os.makedirs(caches_dirpath, exist_ok=True)
    caches = []

    pipeline = config.pipeline.build()
    model = pipeline.unet if hasattr(pipeline, "unet") else pipeline.transformer
    assert isinstance(model, nn.Module)
    model.register_forward_hook(CollectHook(caches=caches), with_kwargs=True)

    batch_size = config.eval.batch_size
    print(f"In total {len(filenames)} samples")
    print(f"Evaluating with batch size {batch_size}")
    pipeline.set_progress_bar_config(desc="Sampling", leave=False, dynamic_ncols=True, position=1)
    num_batches = (len(filenames) + batch_size - 1) // batch_size
    for i in trange(num_batches, desc="Images", leave=False, dynamic_ncols=True, position=0):
        batch = filenames[i * batch_size : (i + 1) * batch_size]
        prompts = [dataset[name] for name in batch]
        seeds = [hash_str_to_int(name) for name in batch]
        generators = [torch.Generator(device=pipeline.device).manual_seed(seed) for seed in seeds]
        images = pipeline(prompts, generator=generators, **config.eval.get_pipeline_kwargs()).images
        if len(caches) == batch_size * config.eval.num_steps:
            num_guidances = 1
        elif len(caches) == 2 * batch_size * config.eval.num_steps:
            num_guidances = 2
        else:
            raise ValueError(f"Unexpected number of caches: {len(caches)} != {batch_size} * {config.eval.num_steps}")
        for j, (filename, image) in enumerate(zip(batch, images, strict=True)):
            image.save(os.path.join(samples_dirpath, f"{filename}.png"))
            for s in range(config.eval.num_steps):
                for g in range(num_guidances):
                    c = caches[s * batch_size * num_guidances + g * batch_size + j]
                    c["filename"] = filename
                    c["step"] = s
                    c["guidance"] = g
                    c = tree_map(lambda x: process(x), c)
                    torch.save(c, os.path.join(caches_dirpath, f"{filename}-{s:05d}-{g}.pt"))
        caches.clear()


@configclass
@dataclass
class CollectConfig:
    """Configuration for collecting calibration dataset.

    Args:
        root (`str`, *optional*, defaults to `"datasets"`):
            Root directory to save the collected dataset.
        dataset_name (`str`, *optional*, defaults to `"qdiff"`):
            Name of the collected dataset.
        prompt_path (`str`, *optional*, defaults to `"prompts/qdiff.yaml"`):
            Path to the prompt file.
        num_samples (`int`, *optional*, defaults to `128`):
            Number of samples to collect.
    """

    root: str = "datasets"
    dataset_name: str = "qdiff"
    prompt_path: str = "prompts/qdiff.yaml"
    num_samples: int = 128


if __name__ == "__main__":
    parser = DiffusionPtqRunConfig.get_parser()
    parser.add_config(CollectConfig, scope="collect", prefix="collect")
    configs, _, unused_cfgs, unused_args, unknown_args = parser.parse_known_args()
    ptq_config, collect_config = configs[""], configs["collect"]
    assert isinstance(ptq_config, DiffusionPtqRunConfig)
    assert isinstance(collect_config, CollectConfig)
    if len(unused_cfgs) > 0:
        print(f"Warning: unused configurations {unused_cfgs}")
    if unused_args is not None:
        print(f"Warning: unused arguments {unused_args}")
    assert len(unknown_args) == 0, f"Unknown arguments: {unknown_args}"

    collect_dirpath = os.path.join(
        collect_config.root,
        str(ptq_config.pipeline.dtype),
        ptq_config.pipeline.name,
        ptq_config.eval.protocol,
        collect_config.dataset_name,
        f"s{collect_config.num_samples}",
    )
    print(f"Saving caches to {collect_dirpath}")

    dataset = yaml.safe_load(open(collect_config.prompt_path, "r"))
    filenames = list(dataset.keys())
    if collect_config.num_samples > 0:
        random.Random(0).shuffle(filenames)
        filenames = filenames[: collect_config.num_samples]
        filenames = sorted(filenames)

    ptq_config.output.root = collect_dirpath
    os.makedirs(ptq_config.output.root, exist_ok=True)
    collect(ptq_config, filenames=filenames, dataset=dataset)
