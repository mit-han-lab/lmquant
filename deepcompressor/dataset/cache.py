# -*- coding: utf-8 -*-
"""Caching calibration dataset."""

import functools
import gc
import typing as tp
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import MISSING

import psutil
import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.hooks
from tqdm import tqdm

from ..data.cache import IOTensorsCache, ModuleForwardInput, TensorCache
from ..data.utils.reshape import ConvInputReshapeFn, ConvOutputReshapedFn, LinearReshapeFn
from ..utils import tools
from ..utils.common import tree_copy_with_ref, tree_map
from ..utils.hooks import EarlyStopException, EarlyStopHook, Hook
from .action import CacheAction

__all__ = ["BaseCalibCacheLoader"]


class BaseCalibCacheLoader(ABC):
    """Base class for caching calibration dataset."""

    dataset: torch.utils.data.Dataset
    batch_size: int

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size: int):
        """Initialize the dataset.

        Args:
            dataset (`torch.utils.data.Dataset`):
                Calibration dataset.
            batch_size (`int`):
                Batch size.
        """
        self.dataset = dataset
        self.batch_size = batch_size

    @property
    def num_samples(self) -> int:
        """Number of samples in the dataset."""
        return len(self.dataset)

    @abstractmethod
    def iter_samples(self, *args, **kwargs) -> tp.Generator[ModuleForwardInput, None, None]:
        """Iterate over model input samples."""
        ...

    def _init_cache(self, name: str, module: nn.Module) -> IOTensorsCache:
        """Initialize activation cache.

        Args:
            name (`str`):
                Module name.
            module (`nn.Module`):
                Module.

        Returns:
            `IOTensorsCache`:
                Tensors cache for inputs and outputs.
        """
        if isinstance(module, (nn.Linear,)):
            return IOTensorsCache(
                inputs=TensorCache(channels_dim=-1, reshape=LinearReshapeFn()),
                outputs=TensorCache(channels_dim=-1, reshape=LinearReshapeFn()),
            )
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            assert module.padding_mode == "zeros", f"Padding mode {module.padding_mode} is not supported"
            if isinstance(module.padding, str):
                if module.padding == "valid":
                    padding = (0,) * len(module.kernel_size)
                elif module.padding == "same":
                    padding = tuple(reversed(tuple(t for t in module._reversed_padding_repeated_twice[::2])))
            else:
                padding = tuple(module.padding)
            return IOTensorsCache(
                inputs=TensorCache(
                    channels_dim=1,
                    reshape=ConvInputReshapeFn(module.kernel_size, padding, module.stride, module.dilation),
                ),
                outputs=TensorCache(channels_dim=1, reshape=ConvOutputReshapedFn()),
            )
        else:
            raise NotImplementedError(f"Module {module.__class__.__name__} is not supported")

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
        return ModuleForwardInput(args=[x, *args[1:]], kwargs=kwargs)

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
        if not isinstance(outputs, torch.Tensor):
            outputs = outputs[0]
        assert isinstance(outputs, torch.Tensor), f"Invalid outputs type: {type(outputs)}"
        return {0: outputs.detach().cpu()}

    def _layer_forward_pre_hook(
        self,
        m: nn.Module,
        args: tuple[torch.Tensor, ...],
        kwargs: dict[str, tp.Any],
        cache: list[ModuleForwardInput],
        save_all: bool = False,
    ) -> None:
        inputs = self._convert_layer_inputs(m, args, kwargs, save_all=save_all)
        if len(cache) > 0:
            inputs.args = tree_copy_with_ref(inputs.args, cache[0].args)
            inputs.kwargs = tree_copy_with_ref(inputs.kwargs, cache[0].kwargs)
        else:
            inputs.args = tree_map(lambda x: x, inputs.args)
            inputs.kwargs = tree_map(lambda x: x, inputs.kwargs)
        cache.append(inputs)

    @torch.inference_mode()
    def _iter_layer_activations(  # noqa: C901
        self,
        model: nn.Module,
        *args,
        action: CacheAction,
        layers: tp.Sequence[nn.Module] | None = None,
        needs_inputs_fn: tp.Callable[[str, nn.Module], bool] | bool | None = True,
        needs_outputs_fn: tp.Callable[[str, nn.Module], bool] | bool | None = None,
        recomputes: list[bool] | None = None,
        use_prev_layer_outputs: list[bool] | None = None,
        early_stop_module: nn.Module | None = None,
        clear_after_yield: bool = True,
        **kwargs,
    ) -> tp.Generator[
        tuple[
            str,
            tuple[
                nn.Module,
                dict[str, IOTensorsCache],
                list[ModuleForwardInput],
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
            layers (`Sequence[nn.Module]` or `None`, *optional*, defaults to `None`):
                Layers to cache activations. If `None`, cache all layers.
            needs_inputs_fn (`Callable[[str, nn.Module], bool]` or `bool` or `None`, *optional*, defaults to `True`):
                Function for determining whether to cache inputs for a module given its name and itself.
            needs_outputs_fn (`Callable[[str, nn.Module], bool]` or `bool` or `None`, *optional*, defaults to `None`):
                Function for determining whether to cache outputs for a module given its name and itself.
            recomputes (`list[bool]` or `bool` or `None`, *optional*, defaults to `None`):
                Whether to recompute the activations for each layer.
            use_prev_layer_outputs (`list[bool]` or `bool` or `None`, *optional*, defaults to `None`):
                Whether to use the previous layer outputs as inputs for the current layer.
            early_stop_module (`nn.Module` or `None`, *optional*, defaults to `None`):
                Module for early stopping.
            clear_after_yield (`bool`, *optional*, defaults to `True`):
                Whether to clear the cache after yielding the activations.
            *args: Arguments for ``iter_samples``.
            **kwargs: Keyword arguments for ``iter_samples``.

        Yields:
            Generator[
                tuple[str, tuple[nn.Module, dict[str, IOTensorsCache], list[ModuleForwardInput]]],
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
        if needs_outputs_fn is None:
            needs_outputs_fn = lambda name, module: False  # noqa: E731
        elif isinstance(needs_outputs_fn, bool):
            if needs_outputs_fn:
                needs_outputs_fn = lambda name, module: True  # noqa: E731
            else:
                needs_outputs_fn = lambda name, module: False  # noqa: E731
        if needs_inputs_fn is None:
            needs_inputs_fn = lambda name, module: False  # noqa: E731
        elif isinstance(needs_inputs_fn, bool):
            if needs_inputs_fn:
                needs_inputs_fn = lambda name, module: True  # noqa: E731
            else:
                needs_inputs_fn = lambda name, module: False  # noqa: E731
        if layers is None:
            recomputes = [True]
            use_prev_layer_outputs = [False]
        else:
            assert isinstance(layers, (nn.Sequential, nn.ModuleList, list, tuple))
            if recomputes is None:
                recomputes = [False] * len(layers)
            elif isinstance(recomputes, bool):
                recomputes = [recomputes] * len(layers)
            if use_prev_layer_outputs is None:
                use_prev_layer_outputs = [True] * len(layers)
            elif isinstance(use_prev_layer_outputs, bool):
                use_prev_layer_outputs = [use_prev_layer_outputs] * len(layers)
            use_prev_layer_outputs[0] = False
            assert len(recomputes) == len(use_prev_layer_outputs) == len(layers)
        cache: dict[str, dict[str, IOTensorsCache]] = {}
        module_names: dict[str, list[str]] = {"": []}
        named_layers: OrderedDict[str, nn.Module] = {"": model}
        # region we first collect infomations for yield modules
        forward_cache: dict[str, list[ModuleForwardInput]] = {}
        info_hooks: list[Hook] = []
        forward_hooks: list[torch.utils.hooks.RemovableHandle] = []
        hook_args: dict[str, list[tuple[str, nn.Module, bool, bool]]] = {}
        layer_name = ""
        for module_name, module in model.named_modules():
            if layers is not None and module_name and module in layers:
                layer_name = module_name
                assert layer_name not in module_names
                named_layers[layer_name] = module
                module_names[layer_name] = []
                forward_cache[layer_name] = []
            if layers is None or (layer_name and module_name.startswith(layer_name)):
                # we only cache modules in the layer
                needs_inputs = needs_inputs_fn(module_name, module)
                needs_outputs = needs_outputs_fn(module_name, module)
                if needs_inputs or needs_outputs:
                    module_names[layer_name].append(module_name)
                    cache.setdefault(layer_name, {})[module_name] = self._init_cache(module_name, module)
                    hook_args.setdefault(layer_name, []).append((module_name, module, needs_inputs, needs_outputs))
                    info_hooks.extend(
                        action.register(
                            name=module_name,
                            module=module,
                            cache=cache[layer_name][module_name],
                            info_mode=True,
                            needs_inputs=needs_inputs,
                            needs_outputs=needs_outputs,
                        )
                    )
        if len(cache) == 0:
            return
        if layers is not None:
            module_names.pop("")
            named_layers.pop("")
            assert layer_name, "No layer in the given layers is found in the model"
            assert "" not in cache, "The model should not have empty layer name"
            ordered_named_layers: OrderedDict[str, nn.Module] = OrderedDict()
            for layer in layers:
                for name, module in named_layers.items():
                    if module is layer:
                        ordered_named_layers[name] = module
                        break
            assert len(ordered_named_layers) == len(named_layers)
            assert len(ordered_named_layers) == len(layers)
            named_layers = ordered_named_layers
            del ordered_named_layers
            for layer_idx, (layer_name, layer) in enumerate(named_layers.items()):
                forward_hooks.append(
                    layer.register_forward_pre_hook(
                        functools.partial(
                            self._layer_forward_pre_hook,
                            cache=forward_cache[layer_name],
                            save_all=not recomputes[layer_idx] and not use_prev_layer_outputs[layer_idx],
                        ),
                        with_kwargs=True,
                    )
                )
        else:
            assert len(named_layers) == 1 and "" in named_layers
            assert len(module_names) == 1 and "" in module_names
            assert len(cache) == 1 and "" in cache
        # endregion
        with tools.logging.redirect_tqdm():
            # region we then collect cache information by running the model with all samples
            if early_stop_module is not None:
                forward_hooks.append(early_stop_module.register_forward_hook(EarlyStopHook()))
            with torch.inference_mode():
                device = "cuda" if torch.cuda.is_available() else "cpu"
                tbar = tqdm(
                    desc="collecting acts info",
                    leave=False,
                    total=self.num_samples,
                    unit="samples",
                    dynamic_ncols=True,
                )
                num_samples = 0
                for sample in self.iter_samples(*args, **kwargs):
                    num_samples += self.batch_size
                    sample = sample.to(device=device)
                    try:
                        model(*sample.args, **sample.kwargs)
                    except EarlyStopException:
                        pass
                    tbar.update(self.batch_size)
                    tbar.set_postfix({"ram usage": psutil.virtual_memory().percent})
                    if psutil.virtual_memory().percent > 90:
                        raise RuntimeError("memory usage > 90%%, aborting")
            for layer_cache in cache.values():
                for module_cache in layer_cache.values():
                    module_cache.set_num_samples(num_samples)
            for hook in forward_hooks:
                hook.remove()
            for hook in info_hooks:
                hook.remove()
            del info_hooks, forward_hooks
            # endregion
            for layer_idx, (layer_name, layer) in enumerate(named_layers.items()):
                # region we first register hooks for caching activations
                layer_hooks: list[Hook] = []
                for module_name, module, needs_inputs, needs_outputs in hook_args[layer_name]:
                    layer_hooks.extend(
                        action.register(
                            name=module_name,
                            module=module,
                            cache=cache[layer_name][module_name],
                            info_mode=False,
                            needs_inputs=needs_inputs,
                            needs_outputs=needs_outputs,
                        )
                    )
                hook_args.pop(layer_name)
                # endregion
                if recomputes[layer_idx]:
                    if layers is None:
                        if early_stop_module is not None:
                            layer_hooks.append(EarlyStopHook().register(early_stop_module))
                    else:
                        layer_hooks.append(EarlyStopHook().register(layer))
                    tbar = tqdm(
                        desc=f"collecting acts in {layer_name}",
                        leave=False,
                        total=self.num_samples,
                        unit="samples",
                        dynamic_ncols=True,
                    )
                    for sample in self.iter_samples(*args, **kwargs):
                        sample = sample.to(device=device)
                        try:
                            model(*sample.args, **sample.kwargs)
                        except EarlyStopException:
                            pass
                        tbar.update(self.batch_size)
                        tbar.set_postfix({"ram usage": psutil.virtual_memory().percent})
                        if psutil.virtual_memory().percent > 90:
                            raise RuntimeError("memory usage > 90%%, aborting")
                        gc.collect()
                else:
                    # region we then forward the layer to collect activations
                    device = next(layer.parameters()).device
                    layer_outputs: list[tp.Any] = []
                    tbar = tqdm(
                        forward_cache[layer_name],
                        desc=f"collecting acts in {layer_name}",
                        leave=False,
                        unit="batches",
                        dynamic_ncols=True,
                    )
                    if not use_prev_layer_outputs[layer_idx]:
                        prev_layer_outputs: list[dict[str | int, tp.Any]] = [None] * len(tbar)
                    for i, inputs in enumerate(tbar):
                        inputs = inputs.update(prev_layer_outputs[i]).to(device=device)
                        outputs = layer(*inputs.args, **inputs.kwargs)
                        layer_outputs.append(self._convert_layer_outputs(layer, outputs))
                        tbar.set_postfix({"ram usage": psutil.virtual_memory().percent})
                        if psutil.virtual_memory().percent > 90:
                            raise RuntimeError("memory usage > 90%%, aborting")
                    prev_layer_outputs = layer_outputs
                    del inputs, outputs, layer_outputs
                    if (layer_idx == len(named_layers) - 1) or not use_prev_layer_outputs[layer_idx + 1]:
                        del prev_layer_outputs
                    # endregion
                for hook in layer_hooks:
                    hook.remove()
                del layer_hooks
                layer_inputs = forward_cache.pop(layer_name, [])
                if not recomputes[layer_idx] and not use_prev_layer_outputs[layer_idx]:
                    layer_inputs = [
                        self._convert_layer_inputs(layer, inputs.args, inputs.kwargs) for inputs in layer_inputs
                    ]
                gc.collect()
                torch.cuda.empty_cache()
                yield layer_name, (layer, cache[layer_name], layer_inputs)
                # region clear layer cache
                if clear_after_yield:
                    for module_cache in cache[layer_name].values():
                        module_cache.clear()
                cache.pop(layer_name)
                del layer_inputs
                gc.collect()
                torch.cuda.empty_cache()
                # endregion

    @abstractmethod
    def iter_layer_activations(  # noqa: C901
        self,
        model: nn.Module,
        *args,
        action: CacheAction,
        needs_inputs_fn: tp.Callable[[str, nn.Module], bool] | bool | None = True,
        needs_outputs_fn: tp.Callable[[str, nn.Module], bool] | bool | None = None,
        **kwargs,
    ) -> tp.Generator[
        tuple[
            str,
            tuple[
                nn.Module,
                dict[str, IOTensorsCache],
                list[ModuleForwardInput],
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
                tuple[str, tuple[nn.Module, dict[str, IOTensorsCache], list[ModuleForwardInput]]],
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
        ...
