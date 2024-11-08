# -*- coding: utf-8 -*-
"""Configuration for collecting calibration dataset for quantization."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from omniconfig import configclass
from torch.utils.data import DataLoader, Dataset

from .cache import BaseCalibCacheLoader

__all__ = ["BaseDataLoaderConfig"]


@configclass
@dataclass(kw_only=True)
class BaseDataLoaderConfig(ABC):
    """Configuration for dataset loader.

    Args:
        data (`str`):
            Dataset name.
        num_samples (`int`):
            Number of dataset samples.
        batch_size (`int`):
            Batch size when loading dataset.
    """

    data: str
    num_samples: int
    batch_size: int

    def generate_dirnames(self, *, prefix: str = "", **kwargs) -> list[str]:
        """Get the names of the configuration fields.

        Args:
            prefix (`str`, *optional*):
                Prefix for the names.

        Returns:
            `list[str]`:
                Names of the configuration.
        """
        name = f"{self.data}.{self.num_samples}"
        return [f"{prefix}.{name}" if prefix else name]

    @abstractmethod
    def build_dataset(self, *args, **kwargs) -> Dataset:
        """Build dataset."""
        ...

    @abstractmethod
    def build_loader(self, *args, **kwargs) -> DataLoader | BaseCalibCacheLoader:
        """Build data loader."""
        ...
