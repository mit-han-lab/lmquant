# -*- coding: utf-8 -*-
"""Search-based quantization calibration configurations."""
import enum
from dataclasses import dataclass

from omniconfig import configclass

from .....utils import num2str
from .common import BaseQuantCalibConfig

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
class SearchBasedCalibConfig(BaseQuantCalibConfig):
    """The base configuration for search-based quantization calibration.

    Args:
        objective (SearchBasedCalibObjective): The objective for quantization calibration.
            Defaults to ``SearchBasedCalibObjective.OutputsError``.
        strategy (SearchBasedCalibStrategy): The strategy for quantization calibration.
            Defaults to ``SearchBasedCalibStrategy.Manual``.
        granularity (SearchBasedCalibGranularity): The granularity for quantization calibration.
            Defaults to ``SearchBasedCalibGranularity.Layer``.
        degree (int): The power degree for the quantization error. Defaults to ``2``.
        element_batch_size (int): The element batch size for calibration. Defaults to ``-1``.
        sample_batch_size (int): The samples batch size for calibration. Defaults to ``-1``.
        element_size (int): The calibration element size. Defaults to ``-1``.
        sample_size (int): The calibration sample size. Defaults to ``-1``.
        pre_reshape (bool): Whether to enable reshaping the tensor before calibration.
            Defaults to ``True``.
        outputs_device (str): The device to store the precomputed outputs of the module.
            Defaults to ``"cpu"``.
        allow_kernel_calib (bool): Whether to allow kernel calibration (e.g., GPTQ). Defaults to ``False``.
        skips (list[str]): The keys of the modules to skip. Defaults to ``[]``.
    """

    objective: SearchBasedCalibObjective = SearchBasedCalibObjective.OutputsError
    strategy: SearchBasedCalibStrategy = SearchBasedCalibStrategy.Manual
    granularity: SearchBasedCalibGranularity = SearchBasedCalibGranularity.Layer
    element_batch_size: int = -1
    sample_batch_size: int = -1
    element_size: int = -1
    sample_size: int = -1
    pre_reshape: bool = True
    outputs_device: str = "cpu"
    allow_kernel_calib: bool = False

    @property
    def needs_search(self) -> bool:
        """Whether the search is enabled."""
        return self.strategy != SearchBasedCalibStrategy.Manual

    def __post_init__(self) -> None:
        super().__post_init__()
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

    def _generate_dirnames(self) -> list[str]:
        """Generate the directory names of the configuration."""
        return []

    def generate_dirnames(self, prefix: str = "") -> list[str]:
        """Generate the directory names of the configuration.

        Returns:
            list[str]: The directory names. The last directory name is the modules to skip.
        """
        name = f"{self.objective.name}.{self.strategy.name}.{self.granularity.name}.d{num2str(self.degree)}"
        name += f".e{num2str(self.element_size)}.s{num2str(self.sample_size)}"
        if self.allow_kernel_calib:
            name += ".krnl"
        names = [name, *self._generate_dirnames(), f"skip.[{'+'.join(self.skips)}]"]
        if prefix:
            names = [f"{prefix}.{name}" for name in names]
        return names
