# -*- coding: utf-8 -*-
"""Quantization calibrator configurations."""

import enum
from dataclasses import dataclass

from omniconfig import configclass

from ...utils.common import num2str

__all__ = [
    "SearchBasedCalibStrategy",
    "SearchBasedCalibGranularity",
    "SearchBasedCalibObjective",
    "SearchBasedCalibConfig",
]


class SearchBasedCalibStrategy(enum.Enum):
    """The strategy for search-based quantization calibration."""

    Manual = enum.auto()
    GridSearch = enum.auto()
    # RandomSearch = enum.auto()
    # Bayesian = enum.auto()
    # EvolutionaryAlgorithm = enum.auto()
    # EvolutionaryStrategy = enum.auto()


class SearchBasedCalibGranularity(enum.Enum):
    """The granularity for search-based quantization calibration."""

    Group = enum.auto()
    ChannelGroup = enum.auto()
    Layer = enum.auto()


class SearchBasedCalibObjective(enum.Enum):
    """The objective for search-based quantization calibration."""

    TensorError = enum.auto()
    """minimize the quantization error of the tensor."""
    ProductsError = enum.auto()
    """minimize the error of the the multiplication products."""
    OutputsError = enum.auto()
    """minimize the error of the outputs of the evaluation module."""


@configclass
@dataclass
class SearchBasedCalibConfig:
    """The base configuration for search-based quantization calibration.

    Args:
        degree (`int`, *optional*, default=`2`):
            The power degree for the quantization error. Defaults to `2`.
        objective (`SearchBasedCalibObjective`, *optional*, default=`SearchBasedCalibObjective.OutputsError`):
            The objective for quantization calibration.
        strategy (`SearchBasedCalibStrategy`, *optional*, default=`SearchBasedCalibStrategy.Manual`):
            The strategy for quantization calibration.
        granularity (`SearchBasedCalibGranularity`, *optional*, default=`SearchBasedCalibGranularity.Layer`):
            The granularity for quantization calibration.
        element_batch_size (`int`, *optional*, default=`-1`):
            The element batch size for calibration.
        sample_batch_size (`int`, *optional*, default=`-1`):
            The samples batch size for calibration.
        element_size (`int`, *optional*, default=`-1`):
            The calibration element size.
        sample_size (`int`, *optional*, default=`-1`):
            The calibration sample size.
        pre_reshape (`bool`, *optional*, default=`True`):
            Whether to enable reshaping the tensor before calibration.
        outputs_device (`str`, *optional*, default=`"cpu"`):
            The device to store the precomputed outputs of the module.
    """

    degree: int = 2
    objective: SearchBasedCalibObjective = SearchBasedCalibObjective.OutputsError
    strategy: SearchBasedCalibStrategy = SearchBasedCalibStrategy.Manual
    granularity: SearchBasedCalibGranularity = SearchBasedCalibGranularity.Layer
    element_batch_size: int = -1
    sample_batch_size: int = -1
    element_size: int = -1
    sample_size: int = -1
    pre_reshape: bool = True
    outputs_device: str = "cpu"

    def __post_init__(self) -> None:
        if self.outputs_device != "cpu":
            self.outputs_device = None
        if self.element_size != 0 or self.sample_size != 0:
            assert self.element_batch_size != 0, "element_batch_size must not be zero"
            assert self.sample_batch_size != 0, "sample_batch_size must not be zero"
            assert self.element_size != 0, "element_size must not be zero"
            assert self.sample_size != 0, "sample_size must not be zero"
        else:
            assert self.objective == SearchBasedCalibObjective.TensorError
        if self.objective == SearchBasedCalibObjective.TensorError:
            pass
        elif self.granularity == SearchBasedCalibGranularity.Layer:
            self.objective = SearchBasedCalibObjective.OutputsError
            self.element_batch_size = -1
            self.element_size = -1

    @property
    def needs_search(self) -> bool:
        """Whether the search is enabled."""
        return self.strategy != SearchBasedCalibStrategy.Manual

    def generate_dirnames(self, *, prefix: str = "", **kwargs) -> list[str]:
        """Generate the directory names of the configuration.

        Args:
            prefix (`str`, *optional*, default=`""`):
                The prefix of the directory.

        Returns:
            `list[str]`:
                The directory names.
        """
        name = f"{self.objective.name}.{self.strategy.name}.{self.granularity.name}.d{num2str(self.degree)}"
        name += f".e{num2str(self.element_size)}.s{num2str(self.sample_size)}"
        if prefix:
            name = f"{prefix}.{name}"
        return [name]
