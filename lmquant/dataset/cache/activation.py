# -*- coding: utf-8 -*-
"""Activation cache module."""

import typing as tp
from dataclasses import dataclass, field

import torch
import torch.utils.hooks

from ..transform import TransformFn

__all__ = ["ActivationCache", "ActivationsCache", "IOActivationsCache"]


@dataclass
class ActivationCache:
    """Activation tensor cache.

    Args:
        cached (torch.Tensor): Cached tensors.
        channels_dim (int): Dimension of channels.
        transform (TransformFn): Transform function for transforming inputs to the desired shape used for GEMM.
    """

    cached: list[torch.Tensor] = field(default_factory=list)
    channels_dim: int = 1
    transform: TransformFn = TransformFn()

    num_cached: int = 0
    num_total: int = 0
    orig_device: torch.device = torch.device("cpu")

    def reorder(self, index: torch.Tensor) -> "ActivationCache":
        """Reorder the cached tensors.

        Args:
            index (torch.Tensor): Index for reordering.

        Returns:
            ActivationCache: Reordered tensor cache.
        """
        index = index.to(self.cached[0].device)
        return ActivationCache(
            cached=[x.index_select(dim=self.channels_dim, index=index.to(x.device)) for x in self.cached],
            channels_dim=self.channels_dim,
            transform=self.transform,
        )

    def clear_cached(self):
        """Clear cached tensors."""
        self.cached.clear()
        self.num_cached = 0


@dataclass
class ActivationsCache:
    """Activations cache."""

    sources: list[ActivationCache]
    num_samples: int = 0

    def __post_init__(self) -> None:
        """Post initialization."""
        if isinstance(self.sources, ActivationCache):
            self.sources = [self.sources]

    @property
    def num_sources(self) -> int:
        """Number of tensors."""
        return len(self.sources)

    def __getitem__(self, idx: int) -> ActivationCache:
        """Get tensor cache information.

        Args:
            idx (int): Index of tensor cache information.

        Returns:
            TensorCacheInfo: Tensor cache information.
        """
        return self.sources[idx]

    def __iter__(self) -> tp.Generator[ActivationCache, None, None]:
        """Iterate over tensor cache information.

        Yields:
            Generator[TensorCacheInfo, None, None]: Generator of tensor cache information.
        """
        for source in self.sources:
            yield source

    def clear_cached(self):
        """Clear cached tensors."""
        for source in self.sources:
            source.clear_cached()


@dataclass
class IOActivationsCache:
    """Input and output activations cache."""

    inputs: ActivationsCache = None
    outputs: ActivationsCache = None

    def __post_init__(self) -> None:
        """Post initialization."""
        if isinstance(self.inputs, ActivationCache):
            self.inputs = ActivationsCache(self.inputs)
        if isinstance(self.outputs, ActivationCache):
            self.outputs = ActivationsCache(self.outputs)

    def clear_cached(self):
        """Clear cached tensors."""
        if self.inputs is not None:
            self.inputs.clear_cached()
        if self.outputs is not None:
            self.outputs.clear_cached()
