# -*- coding: utf-8 -*-
"""Actions for caching inputs and outputs."""

import typing as tp
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ..data.cache import IOTensorsCache, TensorsCache
from ..utils.hooks import BaseInputPackager, BaseOutputPackager, Hook, IOHook, KeyedInputPackager, KeyedOutputPackager

__all__ = ["CacheAction", "ConcatCacheAction"]


class CacheHook(IOHook):
    def __init__(
        self, name: str, module: nn.Module, action: "CacheAction", cache: TensorsCache, info_mode: bool, is_output: bool
    ):
        """Initialize the hook.

        Args:
            name (``str``):
                Module name.
            module (``nn.Module``):
                Module.
            action (``CacheAction``):
                Cache action.
            cache (``TensorsCache``):
                Cache.
            info_mode (``bool``):
                Whether to update cache information.
            is_output (``bool``):
                Whether the hook is an output hook.
        """
        super().__init__(
            pre=not is_output,
            post=is_output,
            input_packager=None if is_output else action.get_input_packager(name, module, cache),
            output_packager=action.get_output_packager(name, module, cache) if is_output else None,
        )
        self.name = name
        self.action = action
        self.cache = cache
        self.info_mode = info_mode

    def pre_forward(
        self,
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
    ) -> None:
        tensors = self.input_packager.unpack(module, input_args, input_kwargs)
        if self.info_mode:
            self.action.info(self.name, module, tensors, self.cache)
        assert len(tensors) == self.cache.num_tensors, f"Expected {self.cache.num_tensors} args, but got {len(tensors)}"
        if not self.info_mode:
            self.action.apply(self.name, module, tensors, self.cache)

    def post_forward(
        self,
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> None:
        tensors = self.output_packager.unpack(module, input_args, input_kwargs, output)
        if self.info_mode:
            self.action.info(self.name, module, tensors, self.cache)
        assert len(tensors) == self.cache.num_tensors, f"Expected {self.cache.num_tensors} args, but got {len(tensors)}"
        if not self.info_mode:
            self.action.apply(self.name, module, tensors, self.cache)


class CacheAction(ABC):
    """Actions for caching activations."""

    device: torch.device | None = None

    def __init__(self, device: torch.device | str | None = None) -> None:
        """Initialize the action.

        Args:
            device (`torch.device or `str` or `None, *optional*, defaults to `None`):
                Device for caching.
        """
        self.device = device

    @abstractmethod
    def apply(
        self,
        name: str,
        module: nn.Module,
        tensors: dict[int | str, torch.Tensor],
        cache: TensorsCache,
    ) -> None:
        """Cache activations.

        Args:
            name (`str`):
                Module name.
            module (`nn.Module`):
                Module.
            tensors (`dict[int or str, torch.Tensor]`):
                Tensors to cache.
            cache (`TensorsCache`):
                Cache.
        """
        ...

    @abstractmethod
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
            tensors (`dict[int or str, torch.Tensor]`):
                Tensors to cache.
            cache (`TensorsCache`):
                Cache.
        """
        ...

    def get_input_packager(self, name: str, module: nn.Module, cache: TensorsCache) -> BaseInputPackager:
        """Get input packager.

        Args:
            name (`str`):
                Module name.
            module (`nn.Module`):
                Module.
            cache (`TensorsCache`):
                Cache.

        Returns:
            `BaseInputPackager`:
                Input packager.
        """
        return KeyedInputPackager(module=module, index_or_keys=list(cache.keys()))

    def get_output_packager(self, name: str, module: nn.Module, cache: TensorsCache) -> BaseOutputPackager:
        """Get output packager.

        Args:
            name (`str`):
                Module name.
            module (`nn.Module`):
                Module.
            cache (`TensorsCache`):
                Cache.

        Returns:
            `BaseOutputPackager`:
                Output packager.
        """
        return KeyedOutputPackager(index_or_keys=list(cache.keys()))

    def register(
        self,
        name: str,
        module: nn.Module,
        cache: IOTensorsCache,
        info_mode: bool,
        needs_inputs: bool,
        needs_outputs: bool,
    ) -> list[Hook]:
        """Register hooks for caching activations.

        Args:
            name (`str`):
                Module name.
            module (`nn.Module`):
                Module.
            cache (`IOTensorsCache`):
                Cache.
            info_mode (`bool`):
                Whether to update cache information.
            needs_inputs (`bool`):
                Whether to cache inputs.
            needs_outputs (`bool`):
                Whether to cache outputs.

        Returns:
            `list[Hook]`:
                Cache hooks.
        """
        hooks = []
        if needs_inputs:
            assert cache.inputs is not None
            hooks.append(CacheHook(name, module, self, cache.inputs, info_mode, is_output=False).register(module))
        if needs_outputs:
            assert cache.outputs is not None
            hooks.append(CacheHook(name, module, self, cache.outputs, info_mode, is_output=True).register(module))
        return hooks


class ConcatCacheAction(CacheAction):
    """Action for concatenating cached activations for calibration."""

    def apply(
        self,
        name: str,
        module: nn.Module,
        tensors: dict[int | str, torch.Tensor],
        cache: TensorsCache,
    ) -> None:
        """Concatenate cached activations along the sample dimension.

        Args:
            name (`str`):
                Module name.
            module (`nn.Module`):
                Module.
            tensors (`dict[int or str, torch.Tensor]`):
                Tensors to cache.
            cache (`TensorsCache`):
                Cache.
        """
        for k, c in cache.tensors.items():
            x = tensors[k]
            shape, device = x.shape, self.device or x.device
            num_prev_cached = c.num_cached
            c.num_cached += shape[0]
            if num_prev_cached == 0:
                assert len(c.data) == 0
                c.data.append(torch.empty((c.num_total, *shape[1:]), dtype=x.dtype, device=device))
            c.data[0][num_prev_cached : c.num_cached].copy_(x)

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
            tensors (`dict[int or str, torch.Tensor]`):
                Tensors to cache.
            cache (`TensorsCache`):
                Cache.
        """
        for k, c in cache.tensors.items():
            x = tensors[k]
            c.num_total += x.shape[0]
            c.orig_device = x.device
