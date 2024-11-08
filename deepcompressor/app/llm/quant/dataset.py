# -*- coding: utf-8 -*-
"""Functions for collecting calibration dataset for quantization."""

import os
import random
import typing as tp
from dataclasses import MISSING, dataclass, field

import torch
import torch.nn as nn
import torch.utils.data
from datasets import load_dataset
from omniconfig import configclass
from transformers import PreTrainedTokenizer
from transformers.cache_utils import Cache
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from transformers.models.t5.modeling_t5 import T5DenseActDense, T5DenseGatedActDense

from deepcompressor.data.cache import IOTensorsCache, ModuleForwardInput, TensorCache
from deepcompressor.data.utils.reshape import LinearReshapeFn
from deepcompressor.dataset.action import CacheAction, ConcatCacheAction
from deepcompressor.dataset.cache import BaseCalibCacheLoader
from deepcompressor.dataset.config import BaseDataLoaderConfig

from ..nn.rope import RotaryEmbedding
from ..nn.struct import LlmModelStruct, LlmTransformerBlockStruct

__all__ = ["LlmCalibDataLoaderConfig", "LlmCalibCacheLoader"]


@configclass
@dataclass(kw_only=True)
class LlmCalibDataLoaderConfig(BaseDataLoaderConfig):
    """Configuration for collecting calibration dataset for quantization.

    Args:
        data (`str`):
            Dataset name.
        num_samples (`int`):
            Number of dataset samples.
        path (`str`):
            Path to the dataset.
        seq_length (`int`):
            Sequence length of each sample.
        min_seq_length (`int`, *optional*, defaults to `0`):
            Minimum sequence length of each sample.
        max_seq_length (`int`, *optional*, defaults to `0`):
            Maximum sequence length of each sample.
        local_path (`str`, *optional*, defaults to `""`):
            Local path to the dataset.
    """

    path: str
    seq_length: int
    min_seq_length: int = 0
    max_seq_length: int = 0
    local_path: str = ""
    batch_size: int = field(init=False, default=1)

    def __post_init__(self) -> None:
        self.min_seq_length = max(0, self.min_seq_length)
        self.max_seq_length = max(0, self.max_seq_length)
        self.path = os.path.expanduser(self.path)
        self.local_path = os.path.expanduser(self.local_path)
        if os.path.exists(self.local_path):
            self.path = self.local_path

    def generate_dirnames(self, *, prefix: str = "", **kwargs) -> list[str]:
        """Get the names of the configuration fields."""
        name = f"{self.data}.{self.num_samples}x{self.seq_length}.[{self.min_seq_length}-{self.max_seq_length}]"
        return [f"{prefix}.{name}" if prefix else name]

    def build_dataset(self, tokenizer: PreTrainedTokenizer) -> "LlmCalibDataset":
        """Build calibration dataset.

        Args:
            tokenizer (`PreTrainedTokenizer`):
                Tokenizer for encoding text.

        Returns:
            `LlmCalibDataset`:
                Calibration dataset.
        """
        return LlmCalibDataset(
            tokenizer,
            data=self.data,
            path=self.path,
            num_samples=self.num_samples,
            seq_length=self.seq_length,
            max_seq_length=self.max_seq_length,
            min_seq_length=self.min_seq_length,
        )

    def build_loader(self, tokenizer: PreTrainedTokenizer) -> "LlmCalibCacheLoader":
        """Build calibration data cache.

        Args:
            tokenizer (`PreTrainedTokenizer`):
                Tokenizer for encoding text.

        Returns:
            `LlmCalibDataCache`:
                Calibration data cache.
        """
        return LlmCalibCacheLoader(config=self, tokenizer=tokenizer)


class LlmCalibDataset(torch.utils.data.Dataset):
    data: list[torch.Tensor]

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data: str,
        path: str,
        num_samples: int,
        seq_length: int,
        max_seq_length: int = -1,
        min_seq_length: int = -1,
        seed: int = 42,
    ) -> None:
        assert num_samples > 0, "num_samples should be positive"
        assert seq_length > 0, "seq_length should be positive"
        num_tokens = num_samples * seq_length
        assert tokenizer is not None, "tokenizer is required"
        if data == "pileval":
            dataset = load_dataset(path, split="validation")
        else:
            raise NotImplementedError(f"Calibration dataset {data} is not supported")
        dataset = dataset.shuffle(seed=seed)
        rng = random.Random(seed)
        seqs, toks = [], 0
        for sample in dataset:
            line = tokenizer.encode(sample["text"].strip())
            length = len(line)
            if length == 0:
                continue
            if min_seq_length > 0 and length < min_seq_length:
                continue
            if max_seq_length > 0 and length > max_seq_length:
                continue
            # sample is a tensor of shape (seq_length, )
            seq = torch.tensor(line)
            if length > seq_length:
                tok = rng.randint(0, length - seq_length)
                seq = seq[tok : tok + seq_length]
            seqs.append(seq)
            toks += seq.numel()
            if len(seqs) >= num_samples and toks >= num_tokens:
                break
        # now concatenate all samples and split according to seq_length
        seqs = torch.cat(seqs).split(seq_length)
        if toks > num_tokens:
            seqs = seqs[:-1]
        seqs = seqs[:num_samples]
        self.data = seqs

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class LlmCalibCacheLoader(BaseCalibCacheLoader):
    """Cache for collecting calibration dataset for quantizing large language models."""

    config: LlmCalibDataLoaderConfig
    dataset: LlmCalibDataset

    def __init__(self, config: LlmCalibDataLoaderConfig, tokenizer: PreTrainedTokenizer) -> None:
        """Initialize large language model calibration cache loader.

        Args:
            config (`LlmCalibDataLoaderConfig`):
                Configuration for loading calibration dataset.
            tokenizer (`PreTrainedTokenizer`):
                Tokenizer for encoding text.
        """
        super().__init__(dataset=config.build_dataset(tokenizer=tokenizer), batch_size=config.batch_size)
        self.batch_size = min(self.batch_size, len(self.dataset))
        self.config = config

    def _init_cache(self, name: str, module: nn.Module) -> IOTensorsCache:
        """Initialize cache.

        Args:
            name (`str`):
                Module name.
            module (`nn.Module`):
                Module.

        Returns:
            `IOTensorsCache`:
                Input and output tensors cache.
        """
        if isinstance(
            module, (nn.Linear, RotaryEmbedding, MixtralSparseMoeBlock, T5DenseActDense, T5DenseGatedActDense)
        ) or module.__class__.__name__.endswith(("DecoderLayer", "Attention", "MLP")):
            return IOTensorsCache(
                inputs=TensorCache(channels_dim=-1, reshape=LinearReshapeFn()),
                outputs=TensorCache(channels_dim=-1, reshape=LinearReshapeFn()),
            )
        else:
            super()._init_cache(name, module)

    def _convert_layer_inputs(
        self, m: nn.Module, args: tuple[tp.Any, ...], kwargs: dict[str, tp.Any], save_all: bool = False
    ) -> ModuleForwardInput:
        """Convert layer inputs to module forward input.

        Args:
            m (`nn.Module`):
                Layer.
            args (`tuple[Any, ...]`):
                Layer input arguments.
            kwargs (`dict[str, Any]`):
                Layer input keyword arguments.
            save_all (`bool`, *optional*, defaults to `False`):
                Whether to save all inputs.

        Returns:
            `ModuleForwardInput`:
                Module forward input.
        """
        x = args[0].detach().cpu() if save_all else MISSING
        return ModuleForwardInput(
            args=[x, *args[1:]], kwargs={k: None if isinstance(v, Cache) else v for k, v in kwargs.items()}
        )

    def iter_samples(self) -> tp.Generator[ModuleForwardInput, None, None]:
        """Iterate over model input samples.

        Args:
            tokenizer (`nn.Module`):
                Tokenizer for encoding text.

        Yields:
            `ModuleForwardInput`:
                Module forward input.
        """
        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False, drop_last=True
        )
        for data in dataloader:
            yield ModuleForwardInput(args=(data,))

    def iter_layer_activations(  # noqa: C901
        self,
        model: nn.Module | LlmModelStruct,
        *args,
        action: CacheAction | None = None,
        needs_inputs_fn: tp.Callable[[str, nn.Module], bool] | bool | None = True,
        needs_outputs_fn: tp.Callable[[str, nn.Module], bool] | bool | None = None,
        **kwargs,
    ) -> tp.Generator[
        tuple[
            str,
            tuple[
                LlmTransformerBlockStruct,
                dict[str, IOTensorsCache],
                dict[str, tp.Any],
            ],
        ],
        None,
        None,
    ]:
        """Iterate over model activations for each layer.

        Args:
            model (`nn.Module`):
                Model.
            action (`CacheAction`, *optional*, defaults to `None`):
                Action for caching activations. If ``None``, ``ConcatCacheAction("cpu")`` is used.
            needs_inputs_fn (`Callable[[str, nn.Module], bool]` or `bool` or `None`, *optional*, defaults to `True`):
                Function for determining whether to cache inputs for a module given its name and itself.
            needs_outputs_fn (`Callable[[str, nn.Module], bool]` or `bool` or `None`, *optional*, defaults to `None`):
                Function for determining whether to cache outputs for a module given its name and itself.
            *args: Arguments for ``_iter_samples``.
            **kwargs: Keyword arguments for ``_iter_samples``.

        Yields:
            Generator[
                tuple[str, tuple[LlmTransformerBlockStruct, dict[str, IOTensorsCache], dict[str, Any]]],
                None,
                None
            ]:
                Generator of tuple of
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
            model_struct = LlmModelStruct.construct(model)
        layers, layer_structs, recomputes, use_prev_layer_outputs = model_struct.get_iter_layer_activations_args()
        action = ConcatCacheAction("cpu") if action is None else action
        for layer_idx, (layer_name, (layer, layer_cache, layer_inputs)) in enumerate(
            self._iter_layer_activations(
                model,
                *args,
                action=action,
                layers=layers,
                needs_inputs_fn=needs_inputs_fn,
                needs_outputs_fn=needs_outputs_fn,
                recomputes=recomputes,
                use_prev_layer_outputs=use_prev_layer_outputs,
                **kwargs,
            )
        ):
            layer_kwargs = layer_inputs[0].kwargs
            for layer_input in layer_inputs:
                for key, value in layer_input.kwargs.items():
                    if isinstance(value, torch.Tensor):
                        assert torch.equal(value, layer_kwargs[key])
                    else:
                        assert value == layer_kwargs[key]
            layer_struct = layer_structs[layer_idx]
            assert layer_name == layer_struct.name, f"Expected {layer_struct.name}, got {layer_name}"
            assert layer is layer_struct.module
            for transformer_block_struct in layer_struct.iter_transformer_block_structs():
                for attn_struct in transformer_block_struct.iter_attention_structs():
                    if attn_struct.v_proj_name in layer_cache:
                        cache = layer_cache[attn_struct.v_proj_name]
                        layer_cache[attn_struct.q_proj_name] = cache
                        layer_cache[attn_struct.k_proj_name] = cache
                ffn_struct = transformer_block_struct.ffn_struct
                up_proj_names = ffn_struct.up_proj_names
                if up_proj_names[0] in layer_cache:
                    for expert_idx in range(ffn_struct.config.num_experts):
                        cache = layer_cache[up_proj_names[expert_idx]]
                        for name in up_proj_names[expert_idx :: ffn_struct.config.num_experts]:
                            layer_cache[name] = cache
                    if ffn_struct.config.num_experts == 1 and ffn_struct.name not in layer_cache:
                        layer_cache[ffn_struct.name] = layer_cache[up_proj_names[0]]
                if ffn_struct.config.num_experts > 1 and ffn_struct.name in layer_cache:
                    layer_cache[ffn_struct.moe_gate_name] = layer_cache[ffn_struct.name]
            yield layer_name, (layer_struct, layer_cache, layer_kwargs)
