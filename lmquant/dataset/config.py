# -*- coding: utf-8 -*-
"""Configuration for collecting calibration dataset for quantization."""

import os
from dataclasses import dataclass, field

from omniconfig import configclass

__all__ = ["BaseCalibDatasetConfig"]


@configclass
@dataclass(kw_only=True)
class BaseCalibDatasetConfig:
    """Configuration for collecting calibration dataset for quantization.

    Args:
        data (str): Dataset name.
        num_samples (int): Number of samples to collect.
        cache_root (str): Root directory for caching.

    Attributes:
        data (str): Dataset name.
        num_samples (int): Number of samples to collect.
        cache_root (str): Root directory for caching the calibration results.
        cache_dirpath (str): Directory path for caching the calibration results.
    """

    data: str
    num_samples: int
    cache_root: str
    cache_dirpath: str = field(init=False)

    def __post_init__(self) -> None:
        self.cache_dirpath = os.path.join(self.cache_root, *self.generate_dirnames())

    def generate_dirnames(self) -> list[str]:
        """Get the names of the configuration fields."""
        return [f"{self.data}.{self.num_samples}"]
