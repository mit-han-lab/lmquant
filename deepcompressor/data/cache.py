# -*- coding: utf-8 -*-
"""Activation cache module."""

import math
import typing as tp
from collections import OrderedDict
from dataclasses import dataclass, field

import torch

from ..utils.common import tree_map
from .utils.reshape import ReshapeFn

__all__ = ["TensorCache", "TensorsCache", "IOTensorsCache"]


@dataclass
class ModuleForwardInput:
    """Module forward input."""

    args: list[tp.Any] = field(default_factory=list)
    kwargs: dict[str, tp.Any] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> "ModuleForwardInput":
        """Move input to device.

        Args:
            device (`torch.device` or `str`):
                Device.

        Returns:
            `ModuleForwardInput`:
                Module forward input.
        """
        return ModuleForwardInput(
            args=tree_map(lambda x: x.to(device=device), self.args),
            kwargs=tree_map(lambda x: x.to(device=device), self.kwargs),
        )

    def update(self, x: dict[str | int, tp.Any] | None = None) -> "ModuleForwardInput":
        """Return a new ModuleForwardInput with updated values.

        Args:
            x (`dict[str | int, tp.Any]` or `None`, *optional*, defaults to `None`):
                Values to update.

        Returns:
            `ModuleForwardInput`:
                Module forward input.
        """
        args, kwargs = tree_map(lambda x: x, self.args), tree_map(lambda x: x, self.kwargs)
        if x is not None:
            for k, v in x.items():
                if isinstance(k, int):
                    args[k] = v
                else:
                    kwargs[k] = v
        return ModuleForwardInput(args=args, kwargs=kwargs)


@dataclass
class TensorCache:
    """Tensor cache.

    Args:
        data (`list[torch.Tensor]`):
            Cached tensors.
        channels_dim (`int`, *optional*, defaults to `1`):
            Channels dimension.
        reshape (`ReshapeFn`, *optional*, defaults to `ReshapeFn()`):
            Function for reshaping inputs to 2-dimension used for GEMM.

        num_cached (`int`, *optional*, defaults to `0`):
            Number of cached tensors.
        num_total (`int`, *optional*, defaults to `0`):
            Number of total tensors.
        num_samples (`int`, *optional*, defaults to `0`):
            Number of samples.
        orig_device (`torch.device` or `str`, *optional*, defaults to `torch.device("cpu")`):
            Original device.
    """

    data: list[torch.Tensor] = field(default_factory=list)
    channels_dim: int = 1
    reshape: ReshapeFn = ReshapeFn()

    num_cached: int = 0
    num_total: int = 0
    num_samples: int = 0
    orig_device: torch.device | str = torch.device("cpu")

    def clear(self):
        """Clear cached tensors."""
        self.data.clear()
        self.num_cached = 0

    def get_factory_kwargs(self, **kwargs) -> dict[str, tp.Any]:
        """Get factory kwargs."""
        kwargs.setdefault("channels_dim", self.channels_dim)
        kwargs.setdefault("reshape", self.reshape)
        kwargs.setdefault("num_cached", self.num_cached)
        kwargs.setdefault("num_total", self.num_total)
        kwargs.setdefault("orig_device", self.orig_device)
        return kwargs

    def get_standardized_data(self, reshape: bool = False) -> list[torch.Tensor]:
        """Get standardized data, i.e., flatten dimensions before `channels_dim`.

        Args:
            reshape (`bool`, *optional*, defaults to `False`):
                Whether to apply reshape function.

        Returns:
            `list[torch.Tensor]`:
                Standardized data.
        """
        if reshape:
            return [self.reshape(x.view(-1, *x.shape[self.channels_dim :])) for x in self.data]
        else:
            return [x.view(-1, *x.shape[self.channels_dim :]) for x in self.data]

    def repartition(self, max_batch_size: int, max_size: int, standardize: bool, reshape: bool) -> "TensorCache":
        """Relocate data based on the maximum batch size and size.

        Args:
            max_batch_size (`int`):
                Maximum batch size.
            max_size (`int`):
                Maximum size.
            standardize (`bool`):
                Whether to standardize data, i.e., flatten dimensions before `channels_dim`.
            reshape (`bool`):
                Whether to apply reshape function.

        Returns:
            `TensorCache`:
                Tensor cache.
        """
        assert len(self.data) > 0, "No data to relocate."
        assert max_batch_size != 0, "max_batch_size must be non-zero."
        assert max_size != 0, "max_size must be non-zero."
        assert all(x.ndim == self.data[0].ndim for x in self.data), "All tensors must have the same #dims."
        assert all(x.shape == self.data[0].shape for x in self.data), "All tensors must have the same shape."
        data, dim, fn = self.data, self.channels_dim, self.reshape
        if standardize:
            data = [x.view(-1, *x.shape[dim:]) for x in self.data]
            dim = 1
            if reshape:
                data = [fn(x) for x in data]
                dim = -1
                fn = ReshapeFn()
        dim = dim % data[0].ndim
        orig_total = data[0].shape[0] * len(data)
        if max_batch_size > 0:
            batch_size = data[0].shape[0]
            if batch_size > max_batch_size:
                data = [
                    x[i * max_batch_size : (i + 1) * max_batch_size]
                    for x in data
                    for i in range(int(batch_size // max_batch_size))
                ]
            batch_size = data[0].shape[0]
            if max_size > 0 and batch_size * len(data) > max_size:
                assert max_size >= batch_size, "max_size must be greater than or equal to batch_size."
                data = data[:: int(len(data) // (max_size // batch_size))]
        else:
            assert max_size < 0, "max_size must be negative if max_batch_size is negative."
        used_total = data[0].shape[0] * len(data)
        ratio = used_total / orig_total
        return TensorCache(
            data,
            channels_dim=dim,
            reshape=fn,
            orig_device=self.orig_device,
            num_cached=int(math.ceil(ratio * self.num_cached)),
            num_total=int(math.ceil(ratio * self.num_total)),
            num_samples=int(math.ceil(ratio * self.num_samples)),
        )


class TensorsCache:
    """Tensors cache."""

    tensors: OrderedDict[str | int, TensorCache]

    def __init__(self, tensors: OrderedDict[str | int, TensorCache] | TensorCache) -> None:
        """Post initialization."""
        self.tensors = OrderedDict({0: tensors}) if isinstance(tensors, TensorCache) else tensors

    @property
    def num_tensors(self) -> int:
        """Get the number of tensor caches."""
        return len(self.tensors)

    def front(self) -> TensorCache:
        """Get the first tensor cache."""
        return next(iter(self.tensors.values()))

    def items(self) -> tp.ItemsView[str | int, TensorCache]:
        """Iterate over tensor caches."""
        return self.tensors.items()

    def keys(self) -> tp.KeysView[str | int]:
        """Get tensor cache keys."""
        return self.tensors.keys()

    def values(self) -> tp.ValuesView[TensorCache]:
        """Get tensor caches."""
        return self.tensors.values()

    def __getitem__(self, key: str | int) -> TensorCache:
        """Get tensor cache."""
        return self.tensors[key]

    def __iter__(self) -> tp.Iterator[TensorCache]:
        """Iterate over tensor caches."""
        return iter(self.tensors.values())

    def __len__(self) -> int:
        """Get the number of tensor caches."""
        return len(self.tensors)

    def clear(self):
        """Clear cached tensors."""
        for tensor in self.tensors.values():
            tensor.clear()

    def set_num_samples(self, num_samples: int):
        """Set the number of samples."""
        for tensor in self.tensors.values():
            tensor.num_samples = num_samples

    def extract(self, index: int, kwargs: dict[str, tp.Any]) -> ModuleForwardInput:
        """Extract data for binding to module forward.

        Args:
            index (`int`):
                Index.
            kwargs (`dict[str, tp.Any]`):
                Keyword arguments.

        Returns:
            `ModuleForwardInput`:
                Module forward input.
        """
        _args, _kwargs = [], {}
        _kwargs.update(kwargs)
        for key, tensor in self.tensors.items():
            if isinstance(key, int):
                assert len(_args) == key, f"Expected {key} args, but got {len(_args)}"
                _args.append(tensor.data[index].to(tensor.orig_device, non_blocking=True))
            else:
                _kwargs[key] = tensor.data[index].to(tensor.orig_device, non_blocking=True)
        return ModuleForwardInput(args=_args, kwargs=_kwargs)


class IOTensorsCache:
    """Input and output cache."""

    inputs: TensorsCache | None
    outputs: TensorsCache | None

    def __init__(
        self, inputs: TensorCache | TensorsCache | None = None, outputs: TensorCache | TensorsCache | None = None
    ):
        self.inputs = TensorsCache(inputs) if isinstance(inputs, TensorCache) else inputs
        self.outputs = TensorsCache(outputs) if isinstance(outputs, TensorCache) else outputs

    def clear(self):
        """Clear cached tensors."""
        if self.inputs is not None:
            self.inputs.clear()
        if self.outputs is not None:
            self.outputs.clear()

    def set_num_samples(self, num_samples: int):
        """Set the number of samples."""
        if self.inputs is not None:
            self.inputs.set_num_samples(num_samples)
        if self.outputs is not None:
            self.outputs.set_num_samples(num_samples)
