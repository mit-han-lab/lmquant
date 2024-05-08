# -*- coding: utf-8 -*-
"""Caching calibration dataset."""

import functools
import gc
import typing as tp
from abc import ABC, abstractmethod

import psutil
import torch
import torch.nn as nn
import torch.utils.hooks
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from ..config import BaseCalibDatasetConfig
from ..transform import ConvTransformFn, LinearTransformFn
from .action import CacheAction
from .activation import ActivationCache, IOActivationsCache

__all__ = ["CalibrationCache"]


class CalibrationCache(ABC):
    """Base class for caching calibration dataset."""

    def __init__(self, config: BaseCalibDatasetConfig) -> None:
        """Initialize the cache.

        Args:
            config (BaseCalibrationConfig): Configuration for caching calibration dataset.
        """
        self.config = config
        self.cached_samples: list[torch.Tensor] = []

    @property
    def num_samples(self) -> int:
        """Number of samples."""
        return self.config.num_samples

    def reset(self) -> None:
        """Reset cache."""
        self.cached_samples = []

    @abstractmethod
    def _iter_samples(self, *args, **kwargs) -> tp.Generator[torch.Tensor, None, None]:
        """Iterate over model inputs."""
        ...

    def iter_samples(self, *args, needs_caching: bool, **kwargs) -> tp.Generator[torch.Tensor, None, None]:
        """Iterate over model input samples.

        Args:
            needs_caching (bool): Whether to cache input samples.

        Yields:
            Generator[torch.Tensor, None, None]: Generator of model input samples.
        """
        if needs_caching and len(self.cached_samples) > 0:
            for sample in self.cached_samples:
                yield sample
        else:
            for sample in self._iter_samples(*args, **kwargs):
                if needs_caching:
                    self.cached_samples.append(sample)
                yield sample

    def get_samples(self, *args, needs_caching: bool, **kwargs) -> list[torch.Tensor]:
        """Get model input samples.

        Args:
            needs_caching (bool): Whether to cache input samples.

        Returns:
            list[torch.Tensor]: List of model input samples.
        """
        if needs_caching:
            if len(self.cached_samples) == 0:
                self.cached_samples = list(self._iter_samples(*args, **kwargs))
            return self.cached_samples
        else:
            return list(self._iter_samples(*args, **kwargs))

    def _init_cache(self, m: nn.Module, /) -> IOActivationsCache:
        """Initialize input cache.

        Args:
            m (nn.Module): Module.

        Returns:
            IOActivationsCache: Cache for inputs and outputs.

        Raises:
            NotImplementedError: If the module is not supported.
        """
        if isinstance(m, (nn.Linear,)):
            return IOActivationsCache(
                inputs=ActivationCache(channels_dim=-1, transform=LinearTransformFn()),
                outputs=ActivationCache(channels_dim=-1, transform=LinearTransformFn()),
            )
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return IOActivationsCache(
                inputs=ActivationCache(
                    channels_dim=1, transform=ConvTransformFn(m.kernel_size, m.padding, m.stride, m.dilation)
                ),
                outputs=ActivationCache(channels_dim=1, transform=LinearTransformFn()),
            )
        else:
            raise NotImplementedError(f"Module {m.__class__.__name__} is not supported")

    def _pre_layer_args_hook(
        self,
        m: nn.Module,
        args: tuple[torch.Tensor, ...],
        args_cache: list[tuple[torch.Tensor]],
    ) -> None:
        assert all(isinstance(x, torch.Tensor) for x in args)
        args_cache.append(tuple(x.detach().cpu() for x in args))

    def _pre_layer_kwargs_hook(
        self,
        m: nn.Module,
        args: tuple[torch.Tensor, ...],
        kwargs: dict[str, tp.Any],
        kwargs_cache: dict[str, tp.Any],
    ) -> None:
        if kwargs_cache:
            assert len(kwargs_cache) == len(kwargs), "kwargs_cache should have the same length as kwargs"
            for k, v in kwargs.items():
                assert k in kwargs_cache, f"kwargs_cache should have the same keys as kwargs, but missing {k}"
                cached = kwargs_cache[k]
                if isinstance(v, torch.Tensor):
                    assert v.allclose(cached), f"kwargs_cache[{k}] should be the same as kwargs[{k}]"
                else:
                    assert v == cached, f"kwargs_cache[{k}] should be the same as kwargs[{k}]"
        else:
            for k, v in kwargs.items():
                kwargs_cache[k] = v

    def _iter_layer_activations(  # noqa: C901
        self,
        model: nn.Module,
        *args,
        action: CacheAction,
        layers: nn.Sequential | nn.ModuleList,
        needs_inputs_fn: tp.Callable[[str, nn.Module], bool],
        needs_outputs_fn: tp.Callable[[str, nn.Module], bool] | None = None,
        needs_samples_caching: bool = True,
        **kwargs,
    ) -> tp.Generator[
        tuple[
            str,
            tuple[
                nn.Module,
                dict[str, IOActivationsCache],
                dict[str, tp.Any],
            ],
        ],
        None,
        None,
    ]:
        """Iterate over model activations in layers.

        Args:
            model (nn.Module): Model.
            layers (nn.Sequential | nn.ModuleList): Layers to cache activations.
            action (CacheAction): Action for caching activations.
            needs_inputs_fn (Callable[[str, nn.Module], bool]): Function for determining whether to cache inputs
                for a module given its name and itself.
            needs_outputs_fn (Callable[[str, nn.Module], bool], optional): Function for determining whether to
                cache outputs for a module given its name and itself. Defaults to ``None``. If ``None``,
                ``False`` is always returned.
            needs_samples_caching (bool, optional): Whether to cache input samples. Defaults to ``True``.
            *args: Arguments for ``_iter_samples``.
            **kwargs: Keyword arguments for ``_iter_samples``.

        Yields:
            Generator[tuple[str, tuple[nn.Module, dict[str, IOActivationsCache], dict[str, Any]]],
                    None, None]: Generator of tuple of
                - layer name
                - a tuple of
                    - layer itself
                    - inputs and outputs cache of each module in the layer
                    - layer input keyword arguments
        """
        if needs_outputs_fn is None:
            needs_outputs_fn = lambda name, module: False  # noqa: E731
        assert isinstance(layers, (nn.Sequential, nn.ModuleList))

        caches: dict[str, IOActivationsCache] = {}
        named_layers: dict[str, nn.Module] = {"": model}
        layer_names: dict[str, list[str]] = {"": []}
        layer_name = ""
        # we first collect kwargs for yield modules
        cache_hook_args: list[tuple[str, str, nn.Module, bool, bool]] = []
        cache_info_hooks: list[torch.utils.hooks.RemovableHandle] = []
        layer_hooks: list[torch.utils.hooks.RemovableHandle] = []
        layer_args_cache: list[tuple[torch.Tensor]] = []
        layer_kwargs_caches: dict[str, dict[str, tp.Any]] = {}
        layer_hooks.append(
            layers[0].register_forward_pre_hook(
                functools.partial(self._pre_layer_args_hook, args_cache=layer_args_cache)
            )
        )
        for module_name, module in model.named_modules():
            if module_name and module in layers:
                assert module_name not in layer_names
                layer_name = module_name
                named_layers[layer_name] = module
                layer_names[layer_name] = []
                layer_kwargs_caches[layer_name] = {}
                layer_hooks.append(
                    module.register_forward_pre_hook(
                        functools.partial(self._pre_layer_kwargs_hook, kwargs_cache=layer_kwargs_caches[layer_name]),
                        with_kwargs=True,
                    )
                )
            if layer_name and module_name.startswith(layer_name):  # we only cache modules in the layer
                needs_in, needs_out = needs_inputs_fn(module_name, module), needs_outputs_fn(module_name, module)
                if needs_in or needs_out:
                    layer_names[layer_name].append(module_name)
                    caches[module_name] = self._init_cache(module)
                    cache_hook_args.append((layer_name, module_name, module, needs_in, needs_out))
                    cache_info_hooks.append(
                        module.register_forward_hook(
                            functools.partial(
                                action.info_hook,
                                name=module_name,
                                cache=caches[module_name],
                                needs_inputs_caching=needs_in,
                                needs_outputs_caching=needs_out,
                            ),
                            with_kwargs=True,
                        )
                    )
        assert layer_name, "No layer in the given layers is found in the model"
        named_layers.pop("")
        layer_names.pop("")
        for layer, module in zip(layers, named_layers.values()):
            assert layer is module, "yield modules must be the same as layers"
        with logging_redirect_tqdm():
            # region we first collect cache information by running the model with all samples
            with torch.inference_mode():
                device = "cuda" if torch.cuda.is_available() else "cpu"
                for sample in tqdm(
                    self.iter_samples(*args, needs_caching=needs_samples_caching, **kwargs),
                    desc="collecting calibration data information",
                    leave=False,
                    total=self.num_samples,
                ):
                    model(sample.to(device=device))
                    if psutil.virtual_memory().percent > 90:
                        raise RuntimeError("memory usage > 90%%, aborting")
            for hook in layer_hooks:
                hook.remove()
            for hook in cache_info_hooks:
                hook.remove()
            del cache_info_hooks, layer_hooks
            # endregion
            cache_hooks: dict[str, list[torch.utils.hooks.RemovableHandle]] = {}
            for layer_name, module_name, module, needs_in, needs_out in cache_hook_args:
                cache_hooks.setdefault(layer_name, []).append(
                    module.register_forward_hook(
                        functools.partial(
                            action.apply_hook,
                            name=module_name,
                            cache=caches[module_name],
                            needs_inputs_caching=needs_in,
                            needs_outputs_caching=needs_out,
                        ),
                        with_kwargs=True,
                    )
                )
            del cache_hook_args
            gc.collect()
            torch.cuda.empty_cache()
            with torch.inference_mode():
                for layer_name, layer in named_layers.items():
                    device = next(layer.parameters()).device
                    next_layer_args_cache: list[list[torch.Tensor]] = []
                    layer_kwargs = layer_kwargs_caches[layer_name]
                    for layer_args in tqdm(
                        layer_args_cache, desc=f"collecting calibration activations in {layer_name}", leave=False
                    ):
                        num_args = len(layer_args)
                        layer_args = [arg.to(device=device) for arg in layer_args]
                        outputs = layer(*layer_args, **layer_kwargs)
                        if not isinstance(outputs, (list, tuple)):
                            outputs = (outputs,)
                        assert num_args <= len(outputs)
                        next_layer_args_cache.append([output.detach().cpu() for output in outputs[:num_args]])
                    layer_args_cache = next_layer_args_cache
                    for hook in cache_hooks[layer_name]:
                        hook.remove()
                    del layer_args, outputs, next_layer_args_cache
                    # ! direct return caches, so that user can pre-forward next layer and get caches
                    # layer_caches = {module_name: caches[module_name] for module_name in layer_names[layer_name]}
                    yield layer_name, (layer, caches, layer_kwargs)
                    # region clear layer cache
                    del layer_kwargs
                    for module_name in layer_names[layer_name]:
                        caches.pop(module_name)
                    layer_kwargs_caches.pop(layer_name)
                    # endregion
                    gc.collect()
                    torch.cuda.empty_cache()

    @abstractmethod
    def iter_layer_activations(  # noqa: C901
        self,
        model: nn.Module,
        *args,
        action: CacheAction,
        needs_inputs: tp.Callable[[str, nn.Module], bool],
        needs_outputs: tp.Callable[[str, nn.Module], bool] | None = None,
        needs_samples_caching: bool = True,
        **kwargs,
    ) -> tp.Generator[
        tuple[
            str,
            tuple[
                nn.Module,
                dict[str, IOActivationsCache],
                dict[str, tp.Any],
            ],
        ],
        None,
        None,
    ]:
        """Iterate over model activations in layers.

        Args:
            model (nn.Module): Model.
            action (CacheAction): Action for caching activations.
            needs_inputs (Callable[[str, nn.Module], bool]): Function for determining whether to cache inputs
                for a module given its name and itself.
            needs_outputs (Callable[[str, nn.Module], bool], optional): Function for determining whether to
                cache outputs for a module given its name and itself. Defaults to ``None``. If ``None``,
                ``False`` is always returned.
            needs_samples_caching (bool, optional): Whether to cache input samples. Defaults to ``True``.
            *args: Arguments for ``_iter_samples``.
            **kwargs: Keyword arguments for ``_iter_samples``.

        Yields:
            Generator[tuple[str, tuple[nn.Module, dict[str, IOActivationsCache], dict[str, Any]]],
                    None, None]: Generator of tuple of
                - layer name
                - a tuple of
                    - layer itself
                    - inputs and outputs cache of each module in the layer
                    - layer input keyword arguments
        """
        ...

    def get_layer_activations(
        self,
        model: nn.Module,
        *args,
        action: CacheAction,
        needs_inputs: tp.Callable[[str, nn.Module], bool],
        needs_outputs: tp.Callable[[str, nn.Module], bool] | None = None,
        needs_samples_caching: bool = True,
        **kwargs,
    ) -> dict[str, IOActivationsCache]:
        """Get cached activations for a model.

        Args:
            model (nn.Module): Model.
            action (CacheAction): Action for caching activations.
            needs_inputs (Callable[[str, nn.Module], bool]): Function for determining whether to cache inputs
                for a module given its name and itself.
            needs_outputs (Callable[[str, nn.Module], bool], optional): Function for determining whether to
                cache outputs for a module given its name and itself. Defaults to ``None``. If ``None``,
                ``False`` is always returned.
            needs_samples_caching (bool, optional): Whether to cache input samples. Defaults to ``True``.
            *args: Arguments for ``_iter_samples``.
            **kwargs: Keyword arguments for ``_iter_samples``.

        Returns:
            dict[str, IOActivationsCache]: Dictionary of module names and their cached activations.
        """
        cache = {}
        for _, (_, layer_cache, _) in self.iter_layer_activations(
            model,
            *args,
            action=action,
            needs_inputs=needs_inputs,
            needs_outputs=needs_outputs,
            needs_samples_caching=needs_samples_caching,
            **kwargs,
        ):
            cache.update(layer_cache)
        return cache
