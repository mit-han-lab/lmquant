# -*- coding: utf-8 -*-
"""Calibration dataset for diffusion models."""

import random
import typing as tp
from collections import OrderedDict
from dataclasses import MISSING, dataclass

import torch
import torch.nn as nn
import torch.utils.data
from diffusers.models.attention import JointTransformerBlock
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock, FluxTransformerBlock
from omniconfig import configclass

from deepcompressor.data.cache import IOTensorsCache, ModuleForwardInput, TensorCache, TensorsCache
from deepcompressor.data.utils.reshape import AttentionInputReshapeFn, LinearReshapeFn, ReshapeFn
from deepcompressor.dataset.action import CacheAction, ConcatCacheAction
from deepcompressor.dataset.cache import BaseCalibCacheLoader
from deepcompressor.dataset.config import BaseDataLoaderConfig

from ..nn.struct import DiffusionBlockStruct, DiffusionModelStruct
from .base import DiffusionDataset

__all__ = [
    "DiffusionCalibCacheLoaderConfig",
    "DiffusionCalibDataset",
    "DiffusionConcatCacheAction",
    "DiffusionCalibCacheLoader",
]


@configclass
@dataclass(kw_only=True)
class DiffusionCalibCacheLoaderConfig(BaseDataLoaderConfig):
    """Configuration for collecting calibration dataset for quantization.

    Args:
        data (`str`):
            Dataset name.
        num_samples (`int`):
            Number of dataset samples.
        batch_size (`int`):
            Batch size when loading dataset.
        path (`str`):
            Path to the dataset directory.
        num_workers (`int`):
            Number of workers for data loading.
    """

    path: str
    num_workers: int = 8

    def build_dataset(self) -> "DiffusionCalibDataset":
        """Build the calibration dataset."""
        return DiffusionCalibDataset(self.path, num_samples=self.num_samples)

    def build_loader(self) -> "DiffusionCalibCacheLoader":
        """Build the data loader."""
        return DiffusionCalibCacheLoader(self)


class DiffusionCalibDataset(DiffusionDataset):
    data: list[dict[str, tp.Any]]

    def __init__(self, path: str, num_samples: int = -1, seed: int = 0) -> None:
        super().__init__(path, num_samples=num_samples, seed=seed, ext=".pt")
        data = [torch.load(path) for path in self.filepaths]
        random.Random(seed).shuffle(data)
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> dict[str, tp.Any]:
        return self.data[idx]


class DiffusionConcatCacheAction(ConcatCacheAction):
    def info(
        self,
        name: str,
        module: nn.Module,
        tensors: dict[int | str, torch.Tensor],
        cache: TensorsCache,
    ) -> None:
        """Update cache information.

        Args:
            name (`str`):
                Module name.
            module (`nn.Module`):
                Module.
            tensors (`dict[int | str, torch.Tensor]`):
                Tensors to cache.
            cache (`TensorsCache`):
                Cache.
        """
        if isinstance(module, Attention):
            encoder_hidden_states = tensors.get("encoder_hidden_states", None)
            if encoder_hidden_states is None:
                tensors.pop("encoder_hidden_states", None)
                cache.tensors.pop("encoder_hidden_states", None)
            else:
                encoder_hidden_states_cache = cache.tensors["encoder_hidden_states"]
                encoder_channels_dim = 1 if encoder_hidden_states.dim() == 4 else -1
                if encoder_hidden_states_cache.channels_dim is None:
                    encoder_hidden_states_cache.channels_dim = encoder_channels_dim
                    if encoder_channels_dim == -1:
                        encoder_hidden_states_cache.reshape = LinearReshapeFn()
                    else:
                        encoder_hidden_states_cache.reshape = AttentionInputReshapeFn(encoder_channels_dim)
                else:
                    assert encoder_hidden_states_cache.channels_dim == encoder_channels_dim
            if tensors["image_rotary_emb"] is None:
                tensors.pop("image_rotary_emb")
                cache.tensors.pop("image_rotary_emb")
            hidden_states, hidden_states_cache = tensors["hidden_states"], cache.tensors["hidden_states"]
            channels_dim = 1 if hidden_states.dim() == 4 else -1
            if hidden_states_cache.channels_dim is None:
                hidden_states_cache.channels_dim = channels_dim
                if channels_dim == -1:
                    hidden_states_cache.reshape = LinearReshapeFn()
                else:
                    hidden_states_cache.reshape = AttentionInputReshapeFn(channels_dim)
            else:
                assert hidden_states_cache.channels_dim == channels_dim
        return super().info(name, module, tensors, cache)


class DiffusionCalibCacheLoader(BaseCalibCacheLoader):
    config: DiffusionCalibCacheLoaderConfig
    dataset: DiffusionCalibDataset

    def __init__(self, config: DiffusionCalibCacheLoaderConfig) -> None:
        """Initialize the cache for the diffusion calibration dataset.

        Args:
            config (`DiffusionCalibCacheLoaderConfig`):
                Configuration for the calibration cache loader.
        """
        super().__init__(dataset=config.build_dataset(), batch_size=config.batch_size)
        self.batch_size = min(config.batch_size, len(self.dataset))
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
                Cache for inputs and outputs.
        """
        if isinstance(module, FluxSingleTransformerBlock):
            return IOTensorsCache(
                inputs=TensorsCache(
                    OrderedDict(
                        hidden_states=TensorCache(channels_dim=-1, reshape=LinearReshapeFn()),
                        temb=TensorCache(channels_dim=1, reshape=LinearReshapeFn()),
                        image_rotary_emb=TensorCache(channels_dim=1, reshape=ReshapeFn()),
                    )
                ),
                outputs=TensorCache(channels_dim=-1, reshape=LinearReshapeFn()),
            )
        elif isinstance(module, Attention):
            return IOTensorsCache(
                inputs=TensorsCache(
                    OrderedDict(
                        hidden_states=TensorCache(channels_dim=None, reshape=None),
                        encoder_hidden_states=TensorCache(channels_dim=None, reshape=None),
                        image_rotary_emb=TensorCache(channels_dim=1, reshape=ReshapeFn()),
                    ),
                ),
                outputs=TensorCache(channels_dim=None, reshape=None),
            )
        else:
            return super()._init_cache(name, module)

    def iter_samples(self) -> tp.Generator[ModuleForwardInput, None, None]:
        dataloader = self.dataset.build_loader(
            batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=self.config.num_workers
        )
        for data in dataloader:
            yield ModuleForwardInput(args=data["input_args"], kwargs=data["input_kwargs"])

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
        kwargs = {k: v for k, v in kwargs.items()}  # noqa: C416
        if "res_hidden_states_tuple" in kwargs:
            kwargs["res_hidden_states_tuple"] = None
            # tree_map(lambda x: x.detach().cpu(), kwargs["res_hidden_states_tuple"])
        if "hidden_states" in kwargs:
            hidden_states = kwargs.pop("hidden_states")
            assert len(args) == 0, f"Invalid args: {args}"
        else:
            hidden_states = args[0]
        if isinstance(m, (FluxTransformerBlock, JointTransformerBlock)):
            if "encoder_hidden_states" in kwargs:
                encoder_hidden_states = kwargs.pop("encoder_hidden_states")
            else:
                encoder_hidden_states = args[1]
            return ModuleForwardInput(
                args=[
                    hidden_states.detach().cpu() if save_all else MISSING,
                    encoder_hidden_states.detach().cpu() if save_all else MISSING,
                ],
                kwargs=kwargs,
            )
        else:
            return ModuleForwardInput(
                args=[hidden_states.detach().cpu() if save_all else MISSING, *args[1:]], kwargs=kwargs
            )

    def _convert_layer_outputs(self, m: nn.Module, outputs: tp.Any) -> dict[str | int, tp.Any]:
        """Convert layer outputs to dictionary for updating the next layer inputs.

        Args:
            m (`nn.Module`):
                Layer.
            outputs (`Any`):
                Layer outputs.

        Returns:
            `dict[str | int, Any]`:
                Dictionary for updating the next layer inputs.
        """
        if isinstance(m, (FluxTransformerBlock, JointTransformerBlock)):
            assert isinstance(outputs, tuple) and len(outputs) == 2
            encoder_hidden_states, hidden_states = outputs
            return {0: hidden_states.detach().cpu(), 1: encoder_hidden_states.detach().cpu()}
        else:
            return super()._convert_layer_outputs(m, outputs)

    def iter_layer_activations(  # noqa: C901
        self,
        model: nn.Module | DiffusionModelStruct,
        *args,
        needs_inputs_fn: tp.Callable[[str, nn.Module], bool],
        needs_outputs_fn: tp.Callable[[str, nn.Module], bool] | None = None,
        action: CacheAction | None = None,
        skip_pre_modules: bool = True,
        skip_post_modules: bool = True,
        **kwargs,
    ) -> tp.Generator[
        tuple[
            str,
            tuple[
                DiffusionBlockStruct | nn.Module,
                dict[str, IOTensorsCache],
                dict[str, tp.Any],
            ],
        ],
        None,
        None,
    ]:
        """Iterate over model activations in layers.

        Args:
            model (`nn.Module`):
                Model.
            action (`CacheAction`):
                Action for caching activations.
            needs_inputs_fn (`Callable[[str, nn.Module], bool]` or `bool` or `None`, *optional*, defaults to `True`):
                Function for determining whether to cache inputs for a module given its name and itself.
            needs_outputs_fn (`Callable[[str, nn.Module], bool]` or `bool` or `None`, *optional*, defaults to `None`):
                Function for determining whether to cache outputs for a module given its name and itself.
            *args: Arguments for ``iter_samples``.
            **kwargs: Keyword arguments for ``iter_samples``.

        Yields:
            Generator[
                tuple[str, tuple[DiffusionBlockStruct | nn.Module, dict[str, IOTensorsCache], dict[str, tp.Any]]],
                None,
                None
            ]:
                Generator of tuple of
                    - layer name
                    - a tuple of
                        - layer itself
                        - inputs and outputs cache of each module in the layer
                        - layer input arguments
        """
        if not isinstance(model, DiffusionModelStruct):
            model_struct = DiffusionModelStruct.construct(model)
        else:
            model_struct = model
            model = model_struct.module
        assert isinstance(model_struct, DiffusionModelStruct)
        assert isinstance(model, nn.Module)
        action = DiffusionConcatCacheAction("cpu") if action is None else action
        layers, layer_structs, recomputes, use_prev_layer_outputs = model_struct.get_iter_layer_activations_args(
            skip_pre_modules=skip_pre_modules,
            skip_post_modules=skip_post_modules,
            **self.dataset[0]["input_kwargs"],
        )
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
            layer_kwargs = {k: v for k, v in layer_inputs[0].kwargs.items()}  # noqa: C416
            layer_kwargs.pop("hidden_states", None)
            layer_kwargs.pop("encoder_hidden_states", None)
            layer_kwargs.pop("temb", None)
            layer_kwargs.pop("image_rotary_emb", None)
            layer_struct = layer_structs[layer_idx]
            if isinstance(layer_struct, DiffusionBlockStruct):
                assert layer_struct.name == layer_name
                assert layer is layer_struct.module
                for transformer_block_struct in layer_struct.iter_transformer_block_structs():
                    for attn_struct in transformer_block_struct.iter_attention_structs():
                        if attn_struct.q_proj_name in layer_cache:
                            if not attn_struct.is_cross_attn():
                                cache = layer_cache[attn_struct.q_proj_name]
                                layer_cache[attn_struct.k_proj_name] = cache
                                layer_cache[attn_struct.v_proj_name] = cache
                        if attn_struct.add_k_proj_name in layer_cache:
                            assert not attn_struct.is_self_attn()
                            cache = layer_cache[attn_struct.add_k_proj_name]
                            layer_cache[attn_struct.add_v_proj_name] = cache
                            if attn_struct.is_joint_attn():
                                layer_cache[attn_struct.add_q_proj_name] = cache
                    ffn_struct = transformer_block_struct.ffn_struct
                    num_experts = ffn_struct.config.num_experts
                    if ffn_struct is not None and num_experts > 1:
                        for expert_idx in range(num_experts):
                            if ffn_struct.up_proj_names[expert_idx] in layer_cache:
                                cache = layer_cache[ffn_struct.up_proj_names[expert_idx]]
                                for up_proj_name in ffn_struct.up_proj_names[expert_idx::num_experts]:
                                    layer_cache[up_proj_name] = cache
                            if ffn_struct.down_proj_names[expert_idx] in layer_cache:
                                cache = layer_cache[ffn_struct.down_proj_names[expert_idx]]
                                for down_proj_name in ffn_struct.down_proj_names[expert_idx::num_experts]:
                                    layer_cache[down_proj_name] = cache
            yield layer_name, (layer_struct, layer_cache, layer_kwargs)
