# -*- coding: utf-8 -*-
"""Functions for collecting calibration dataset for quantization."""

import os
import random
import typing as tp
from dataclasses import dataclass

import torch
import torch.nn as nn
from datasets import load_dataset
from omniconfig import configclass
from transformers.cache_utils import DynamicCache
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from lmquant.dataset.cache.action import AverageCache, CacheAction, ConcatCache
from lmquant.dataset.cache.activation import ActivationCache, IOActivationsCache
from lmquant.dataset.cache.calibration import CalibrationCache
from lmquant.dataset.config import BaseCalibDatasetConfig
from lmquant.dataset.transform import LinearTransformFn

from .nn import LlmDecoderLayerStruct, LlmModelStruct, RotaryEmbedding

__all__ = ["LlmCalibConfig", "LlmCalibrationCache", "LlmConcatCache", "LlmAverageCache"]


@configclass
@dataclass(kw_only=True)
class LlmCalibConfig(BaseCalibDatasetConfig):
    """Configuration for collecting calibration dataset for quantization.

    Args:
        data (str): Dataset name.
        num_samples (int): Number of samples to collect.
        cache_root (str): Root directory for caching.
        dataset_path (str): Path to the dataset.
        seq_length (int): Sequence length of each sample. Defaults to ``512``.
        local_dataset_path (str): Local path to the dataset. Defaults to ``""``.

    Attributes:
        data (str): Dataset name.
        num_samples (int): Number of samples to collect.
        seq_length (int): Sequence length of each sample.
        num_tokens (int): Number of tokens in each sample.
        cache_root (str): Root directory for caching the calibration results the calibration results.
        dataset_path (str): Path to the dataset.

    """

    dataset_path: str
    seq_length: int = 512
    min_seq_length: int = 0
    max_seq_length: int = 0
    local_dataset_path: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.min_seq_length = max(0, self.min_seq_length)
        self.max_seq_length = max(0, self.max_seq_length)
        self.cache_dirpath = os.path.join(self.cache_root, "llm", "cache", *self.generate_dirnames())
        self.dataset_path = os.path.expanduser(self.dataset_path)
        self.local_dataset_path = os.path.expanduser(self.local_dataset_path)
        if os.path.exists(self.local_dataset_path):
            self.dataset_path = self.local_dataset_path

    @property
    def num_tokens(self) -> int:
        """Number of tokens in each sample."""
        return self.num_samples * self.seq_length

    def generate_dirnames(self) -> list[str]:
        """Get the names of the configuration fields."""
        return [f"{self.data}.{self.num_samples}x{self.seq_length}.[{self.min_seq_length}-{self.max_seq_length}]"]


class LlmConcatCache(ConcatCache):
    """Action for concatenating cached activations."""

    def _unpack(
        self, name: str, module: nn.Module, args: tuple[torch.Tensor, ...], kwargs: dict[str, tp.Any] | None
    ) -> tuple[torch.Tensor, ...]:
        if len(args) == 0:
            assert "hidden_states" in kwargs, "hidden_states should be in kwargs if args is empty"
            args = (kwargs["hidden_states"],)
        return args


class LlmAverageCache(AverageCache):
    """Action for averaging cached activations."""

    def _unpack(
        self, name: str, module: nn.Module, args: tuple[torch.Tensor, ...], kwargs: dict[str, tp.Any] | None
    ) -> tuple[torch.Tensor, ...]:
        if len(args) == 0:
            assert "hidden_states" in kwargs, "hidden_states should be in kwargs if args is empty"
            args = (kwargs["hidden_states"],)
        return args


class LlmCalibrationCache(CalibrationCache):
    """Cache for collecting calibration dataset for quantizing large language models."""

    config: LlmCalibConfig

    def __init__(self, config: LlmCalibConfig) -> None:
        """Initialize LlmCalibrationCache.

        Args:
            config (LlmCalibrationCache): Configuration for collecting calibration dataset.
        """
        super().__init__(config)

    def _init_cache(self, m: nn.Module, /) -> IOActivationsCache:
        """Initialize cache.

        Args:
            m (nn.Module): Module.

        Returns:
            IOCacheInfo: Cache information for inputs and outputs.

        Raises:
            NotImplementedError: If the module is not supported.
        """
        if isinstance(m, (nn.Linear, RotaryEmbedding, MixtralSparseMoeBlock)) or m.__class__.__name__.endswith(
            ("DecoderLayer", "Attention", "MLP")
        ):
            return IOActivationsCache(
                inputs=ActivationCache(channels_dim=-1, transform=LinearTransformFn()),
                outputs=ActivationCache(channels_dim=-1, transform=LinearTransformFn()),
            )
        else:
            raise ValueError(f"Module {m.__class__.__name__} is not supported")

    def _pre_layer_kwargs_hook(
        self,
        m: nn.Module,
        args: tuple[torch.Tensor, ...],
        kwargs: dict[str, tp.Any],
        kwargs_cache: dict[str, tp.Any],
    ) -> None:
        def _check_equality(_k, _v, _cached):
            if isinstance(_v, DynamicCache):
                assert _cached is None, f"kwargs_cache[{_k}] should be None"
            elif isinstance(_v, torch.Tensor):
                assert _v.allclose(_cached), f"kwargs_cache[{_k}] should be the same as kwargs[{_k}]"
            elif isinstance(_v, tuple):
                assert len(_v) == len(
                    _cached), f"kwargs_cache[{_k}] is a tuple, and should have the same length as kwargs[{_k}]"
                for i in range(len(_v)):
                    _check_equality(_k, _v[i], _cached[i])
            else:
                assert _v == _cached, f"kwargs_cache[{_k}] should be the same as {_v}"

        if kwargs_cache:
            assert len(kwargs_cache) == len(kwargs), "kwargs_cache should have the same length as kwargs"
            for k, v in kwargs.items():
                assert k in kwargs_cache, f"kwargs_cache should have the same keys as kwargs, but missing {k}"
                cached = kwargs_cache[k]
                _check_equality(k, v, cached)
        else:
            for k, v in kwargs.items():
                if isinstance(v, DynamicCache):
                    kwargs_cache[k] = None
                else:
                    kwargs_cache[k] = v

    def _iter_samples(self, tokenizer: nn.Module) -> tp.Generator[torch.Tensor, None, None]:
        """Iterate over samples.

        Args:
            tokenizer (nn.Module): Tokenizer for encoding text.

        Yields:
            Generator[torch.Tensor, None, None]: Generator for iterating over samples.
                Each sample is a tensor of shape (1, seq_length).
        """
        if self.config.data == "pileval":
            assert tokenizer is not None, "tokenizer is required for pileval dataset"
            dataset = load_dataset(self.config.dataset_path, split="validation")
            dataset = dataset.shuffle(seed=42)
            rng = random.Random(42)
            samples, num_tokens = [], 0
            for _data in dataset:
                line = _data["text"]
                line = line.strip()
                # line_encoded is a list of token ids
                line_encoded = tokenizer.encode(line)
                seq_length = len(line_encoded)
                if seq_length == 0:
                    continue
                if self.config.min_seq_length > 0 and seq_length < self.config.min_seq_length:
                    continue
                if self.config.max_seq_length > 0 and seq_length > self.config.max_seq_length:
                    continue
                # sample is a tensor of shape (1, seq_length)
                sample = torch.tensor([line_encoded])
                if seq_length > self.config.seq_length:
                    tok = rng.randint(0, seq_length - self.config.seq_length)
                    sample = sample[:, tok : tok + self.config.seq_length]
                samples.append(sample)
                num_tokens += sample.shape[1]
                if len(samples) >= self.config.num_samples and num_tokens >= self.config.num_tokens:
                    break
            # now concatenate all samples and split according to seq_length
            samples = torch.cat(samples, dim=1).split(self.config.seq_length, dim=1)
            if num_tokens > self.config.num_tokens:
                samples = samples[:-1]
            samples = samples[: self.config.num_samples]
            for sample in samples:
                yield sample
        else:
            raise NotImplementedError(f"Calibration dataset {self.config.data} is not supported")

    def iter_layer_activations(  # noqa: C901
        self,
        model: nn.Module | LlmModelStruct,
        *args,
        needs_inputs_fn: tp.Callable[[str, nn.Module], bool],
        action: CacheAction | None = None,
        needs_outputs_fn: tp.Callable[[str, nn.Module], bool] | None = None,
        needs_samples_caching: bool = True,
        **kwargs,
    ) -> tp.Generator[
        tuple[
            str,
            tuple[
                LlmDecoderLayerStruct,
                dict[str, IOActivationsCache],
                dict[str, tp.Any],
            ],
        ],
        None,
        None,
    ]:
        """Iterate over model activations for each layer.

        Args:
            model (nn.Module | LlmModelStruct): Model.
            action (CacheAction): Action for caching activations. If ``None``, ``LlmConcatCache("cpu")`` is used.
                Defaults to ``None``.
            needs_inputs (Callable[[str, nn.Module], bool]): Function for determining whether to cache inputs
                for a module given its name and itself.
            needs_outputs (Callable[[str, nn.Module], bool], optional): Function for determining whether to
                cache outputs for a module given its name and itself. Defaults to ``None``. If ``None``,
                ``False`` is always returned.
            needs_samples_caching (bool, optional): Whether to cache input samples. Defaults to ``True``.
            *args: Arguments for ``_iter_samples``.
            **kwargs: Keyword arguments for ``_iter_samples``.

        Yields:
            Generator[tuple[str, tuple[LlmDecoderLayerStruct, dict[str, IOActivationsCache], dict[str, Any]]],
                    None, None]: Generator of tuple of
                - layer name
                - a tuple of
                    - layer struct,
                    - input and output caches for each module in the layer,
                    - layer input keyword arguments.
        """
        if isinstance(model, LlmModelStruct):
            model_struct = model
            model = model_struct.module
        else:
            model_struct = LlmModelStruct.build(model)
        backbone_struct = model_struct.backbone_struct
        layer_structs = backbone_struct.layer_structs
        action = LlmConcatCache("cpu") if action is None else action
        for layer_idx, (layer_name, (layer, layer_cache, layer_kwargs)) in enumerate(
            self._iter_layer_activations(
                model,
                *args,
                action=action,
                layers=backbone_struct.layers,
                needs_inputs_fn=needs_inputs_fn,
                needs_outputs_fn=needs_outputs_fn,
                needs_samples_caching=needs_samples_caching,
                **kwargs,
            )
        ):
            layer_struct = layer_structs[layer_idx]
            assert layer_idx == layer_struct.idx
            assert layer_name == layer_struct.full_name
            assert layer is layer_struct.module
            if layer_struct.proj_v_full_name in layer_cache:
                cache = layer_cache[layer_struct.proj_v_full_name]
                layer_cache[layer_struct.proj_q_full_name] = cache
                layer_cache[layer_struct.proj_k_full_name] = cache
            if layer_struct.proj_1st_full_names[0] in layer_cache:
                for expert_idx in range(layer_struct.config.num_experts):
                    cache = layer_cache[layer_struct.proj_1st_full_names[expert_idx]]
                    for name in layer_struct.proj_1st_full_names[expert_idx :: layer_struct.config.num_experts]:
                        layer_cache[name] = cache
                if layer_struct.config.num_experts == 1 and layer_struct.ffn_block_full_name not in layer_cache:
                    layer_cache[layer_struct.ffn_block_full_name] = layer_cache[layer_struct.proj_1st_full_names[0]]
            if layer_struct.config.num_experts > 1 and layer_struct.ffn_block_full_name in layer_cache:
                layer_cache[layer_struct.router_full_name] = layer_cache[layer_struct.ffn_block_full_name]
            yield layer_name, (layer_struct, layer_cache, layer_kwargs)
