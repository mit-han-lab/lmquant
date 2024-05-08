# -*- coding: utf-8 -*-
"""Actions for caching inputs and outputs."""

import typing as tp
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.utils.hooks

from .activation import ActivationsCache, IOActivationsCache

__all__ = ["CacheAction", "ConcatCache", "AverageCache"]


class CacheAction(ABC):
    """Actions for caching inputs."""

    device: torch.device | None = None

    def __init__(self, device: torch.device | None = None) -> None:
        """Initialize the action.

        Args:
            device (torch.device, optional): Device for caching. Defaults to ``None``.
        """
        self.device = device

    def _unpack(
        self, name: str, module: nn.Module, args: tuple[torch.Tensor, ...], kwargs: dict[str, tp.Any] | None
    ) -> tuple[torch.Tensor, ...]:
        """Unpack inputs.

        Args:
            args (tuple[torch.Tensor, ...]): Inputs.
            kwargs (dict[str, tp.Any] | None): Keyword arguments.

        Returns:
            tuple[torch.Tensor, ...]: Unpacked inputs.
        """
        return args

    @abstractmethod
    def _apply(
        self,
        name: str,
        module: nn.Module,
        tensors: tuple[torch.Tensor, ...],
        cache: ActivationsCache,
    ) -> None:
        """Cache activations.

        Args:
            name (str): Module name.
            module (nn.Module): Module.
            tensors (tuple[torch.Tensor, ...]): Tensors to cache.
            cache (ActivationsCache): Cache.
        """
        ...

    @abstractmethod
    def _info(
        self,
        name: str,
        module: nn.Module,
        tensors: tuple[torch.Tensor, ...],
        cache: ActivationsCache,
    ) -> None:
        """Update cache information.

        Args:
            name (str): Module name.
            module (nn.Module): Module.
            tensors (tuple[torch.Tensor, ...]): Tensors to cache.
            cache (ActivationsCache): Cache.
        """
        ...

    def apply(
        self,
        name: str,
        module: nn.Module,
        args: tuple[torch.Tensor, ...],
        kwargs: dict[str, tp.Any] | None,
        cache: ActivationsCache,
    ) -> None:
        """Cache activations.

        Args:
            name (str): Module name.
            module (nn.Module): Module.
            args (tuple[torch.Tensor, ...]): arguments.
            kwargs (dict[str, tp.Any] | None): Keyword arguments.
            cache (ActivationsCache): Cache.
        """
        tensors = self._unpack(name, module, args, kwargs)
        assert len(tensors) == cache.num_sources, f"Expected {cache.num_sources} tensors, but got {len(tensors)}"
        self._apply(name, module, tensors, cache)

    def info(
        self,
        name: str,
        module: nn.Module,
        args: tuple[torch.Tensor, ...],
        kwargs: dict[str, tp.Any] | None,
        cache: ActivationsCache,
    ) -> None:
        """Update cache information.

        Args:
            name (str): Module name.
            module (nn.Module): Module.
            args (tuple[torch.Tensor, ...]): arguments.
            kwargs (dict[str, tp.Any] | None): Keyword arguments.
            cache (ActivationsCache): Cache.
        """
        tensors = self._unpack(name, module, args, kwargs)
        assert len(tensors) == cache.num_sources, f"Expected {cache.num_sources} args, but got {len(tensors)}"
        self._info(name, module, tensors, cache)

    def info_hook(
        self,
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
        output: torch.Tensor | tuple[torch.Tensor, ...],
        *,
        name: str,
        cache: IOActivationsCache,
        needs_inputs_caching: bool,
        needs_outputs_caching: bool,
    ) -> None:
        """Collect activation information for calibration.

        This is post-forward hook and should be registered with ``register_forward_hook``.

        Args:
            module (nn.Module): Module.
            input_args (tuple[torch.Tensor, ...]): Inputs to the module.
            input_kwargs (dict[str, tp.Any]): Keyword arguments for the module.
            output (torch.Tensor | tuple[torch.Tensor, ...]): Outputs of the module.
            name (str): Module name.
            cache (IOActivationsCache): Cache for inputs and outputs.
            needs_inputs_caching (bool): Whether to cache inputs.
            needs_outputs_caching (bool): Whether to cache outputs.
        """
        if needs_inputs_caching:
            assert isinstance(input_args, tuple)
            assert isinstance(input_kwargs, dict)
            self.info(name, module, input_args, input_kwargs, cache=cache.inputs)
        if needs_outputs_caching:
            if not isinstance(output, (tuple, list)):
                output = (output,)
            self.info(name, module, output, None, cache=cache.outputs)

    def apply_hook(
        self,
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
        output: torch.Tensor | tuple[torch.Tensor, ...],
        *,
        name: str,
        cache: IOActivationsCache,
        needs_inputs_caching: bool,
        needs_outputs_caching: bool,
    ) -> None:
        """Collect activation information for calibration.

        This is post-forward hook and should be registered with ``register_forward_hook``.

        Args:
            module (nn.Module): Module.
            inputs (tuple[torch.Tensor, ...]): Inputs to the module.
            outputs (torch.Tensor): Outputs of the module.
            name (str): Module name.
            cache (IOActivationsCache): Cache for inputs and outputs.
            needs_inputs_caching (bool): Whether to cache inputs.
            needs_outputs_caching (bool): Whether to cache outputs.
        """
        if needs_inputs_caching:
            assert isinstance(input_args, tuple)
            assert isinstance(input_kwargs, dict)
            self.apply(name, module, input_args, input_kwargs, cache=cache.inputs)
        if needs_outputs_caching:
            if not isinstance(output, (tuple, list)):
                output = (output,)
            self.apply(name, module, output, None, cache=cache.outputs)


class ConcatCache(CacheAction):
    """Action for concatenating cached activations."""

    def _apply(
        self,
        name: str,
        module: nn.Module,
        tensors: tuple[torch.Tensor, ...],
        cache: ActivationsCache,
    ) -> None:
        """Concatenate cached activations along the sample dimension.

        Args:
            name (str): Module name.
            module (nn.Module): Module.
            tensors (tuple[torch.Tensor, ...]): Tensors to cache.
            cache (ActivationsCache): Cache.
        """
        for x, c in zip(tensors, cache):
            x, shape, device = x.detach(), x.shape, self.device or x.device
            num_prev_cached = c.num_cached
            c.num_cached += shape[0]
            if num_prev_cached == 0:
                assert len(c.cached) == 0
                c.cached.append(torch.empty((c.num_total, *shape[1:]), dtype=x.dtype, device=device))
            c.cached[0][num_prev_cached : c.num_cached].copy_(x)
            if c.num_cached == c.num_total:
                avg_size = c.num_total // cache.num_samples
                c.num_cached = avg_size * cache.num_samples
                c.cached[0] = c.cached[0][: c.num_cached]  # .reshape(cache.num_samples, avg_size, *shape[1:])

    def _info(
        self,
        name: str,
        module: nn.Module,
        tensors: tuple[torch.Tensor, ...],
        cache: ActivationsCache,
    ) -> None:
        """Update cache information.

        Args:
            name (str): Module name.
            module (nn.Module): Module.
            tensors (tuple[torch.Tensor, ...] | torch.Tensor): Tensors to cache.
            cache (ActivationsCache): Cache.
        """
        cache.num_samples += 1
        for x, c in zip(tensors, cache):
            c.num_total += x.shape[0]
            c.orig_device = x.device


class AverageCache(CacheAction):
    """Action for averaging cached activations."""

    def _apply(
        self,
        name: str,
        module: nn.Module,
        tensors: tuple[torch.Tensor, ...],
        cache: ActivationsCache,
    ) -> None:
        """Average cached activations.

        Args:
            name (str): Module name.
            module (nn.Module): Module.
            tensors (tuple[torch.Tensor, ...] | torch.Tensor): Tensors.
            cache (ActivationsCache): Cache.
        """
        for x, c in zip(tensors, cache):
            x, shape, ndim, device = x.detach(), x.shape, x.ndim, self.device or x.device
            rshape = shape[c.channels_dim :]
            ldim = ndim - len(rshape)
            x = x.view(-1, *rshape).sum(dim=0).view(*([1] * ldim), *rshape).to(device)
            num_cached = c.num_cached
            c.num_cached += shape[: c.channels_dim].numel()
            if num_cached == 0:
                assert len(c.cached) == 0
                c.cached.append(x)
            else:
                c.cached[0].add_(x)
            if c.num_cached == c.num_total:
                c.cached[0].div_(c.num_total)

    def _info(
        self,
        name: str,
        module: nn.Module,
        tensors: tuple[torch.Tensor, ...],
        cache: ActivationsCache,
    ) -> None:
        """Update cache information.

        Args:
            name (str): Module name.
            module (nn.Module): Module.
            tensors (tuple[torch.Tensor, ...] | torch.Tensor): Tensors.
            cache (ActivationsCache): Cache.
        """
        cache.num_samples += 1
        for x, c in zip(tensors, cache):
            c.num_total += x.shape[: c.channels_dim].numel()
            c.orig_device = x.device
