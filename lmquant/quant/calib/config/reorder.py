# -*- coding: utf-8 -*-
"""Channel reorder configuration."""

import enum
from dataclasses import dataclass, field

from omniconfig import configclass

from .base.search import (
    SearchBasedCalibConfig,
    SearchBasedCalibGranularity,
    SearchBasedCalibObjective,
    SearchBasedCalibStrategy,
)

__all__ = ["QuantChannelOrderCalibConfig", "QuantReorderConfig"]


@configclass
@dataclass
class QuantChannelOrderCalibConfig(SearchBasedCalibConfig):
    """Configuration for channel order calibration in group quantization.

    Args:
        strategy (SearchBasedCalibStrategy): The strategy for quantization calibration.
            Defaults to ``SearchBasedCalibStrategy.Manual``.
        degree (int): The power degree for the quantization error. Defaults to ``2``.
        channel_metric (ChannelMetricMode): The mode for computing the channel importance.
            Defaults to ``ChannelMetricMode.AbsNormalizedMean``.
        channel_index (ChannelIndexMode): The mode for ranking the channel importance.
            Defaults to ``ChannelIndexMode.Sequential``.
        dynamic (bool): Whether to enable dynamic channel order calibration. Defaults to ``False``.
        allow_kernel_calib (bool): Whether to allow kernel calibration (e.g., GPTQ). Defaults to ``False``.
        skips (list[str]): The keys of the modules to skip. Defaults to ``[]``.
    """

    class ChannelMetric(enum.Enum):
        """The mode for computing the channel importance."""

        InputsAbsMax = enum.auto()
        InputsAbsMean = enum.auto()
        InputsRootMeanSquare = enum.auto()
        WeightsAbsMax = enum.auto()
        WeightsAbsMean = enum.auto()
        WeightsRootMeanSquare = enum.auto()
        AbsMaxProduct = enum.auto()
        AbsMeanProduct = enum.auto()
        RootMeanSquareProduct = enum.auto()

    class ChannelIndex(enum.Enum):
        """The mode for ranking the channel importance."""

        Sequential = enum.auto()
        Transpose = enum.auto()

    objective: SearchBasedCalibObjective = field(init=False, default=SearchBasedCalibObjective.OutputsError)
    granularity: SearchBasedCalibGranularity = field(init=False, default=SearchBasedCalibGranularity.Layer)
    channel_metric: ChannelMetric = ChannelMetric.InputsAbsMax
    channel_index: ChannelIndex = ChannelIndex.Sequential
    dynamic: bool = False

    def __post_init__(self) -> None:
        if self.strategy == SearchBasedCalibStrategy.Manual:
            self.allow_kernel_calib = False
        else:
            self.strategy = SearchBasedCalibStrategy.GridSearch
        self.skips = sorted(set(self.skips or []))
        super().__post_init__()

    def __str__(self) -> str:
        if self.strategy == SearchBasedCalibStrategy.Manual:
            s = f"(metric={self.channel_metric}, rank={self.channel_index}, "
        else:
            s = f"(strategy={self.strategy.name}, "
        s += f"dynamic={self.dynamic}, skips={self.skips})"
        return s

    def _generate_dirnames(self) -> list[str]:
        """Generate the directory names of the configuration.

        Returns:
            list[str]: The directory names.
        """
        if self.strategy == SearchBasedCalibStrategy.Manual:
            name = f"{self.channel_metric.name}.{self.channel_index.name}"
        else:
            name = "search"
        if self.dynamic:
            name += ".dynamic"
        return [name]


@configclass
@dataclass
class QuantReorderConfig(QuantChannelOrderCalibConfig):
    """Configuration for channel reorder in group quantization.

    Args:
        strategy (SearchBasedCalibStrategy): The strategy for quantization calibration.
            Defaults to ``SearchBasedCalibStrategy.Manual``.
        degree (int): The power degree for the quantization error. Defaults to ``2``.
        channel_metric (ChannelMetricMode): The mode for computing the channel importance.
            Defaults to ``ChannelMetricMode.AbsNormalizedMean``.
        channel_index (ChannelIndexMode): The mode for ranking the channel importance.
            Defaults to ``ChannelIndexMode.Transpose``.
        dynamic (bool): Whether to enable dynamic channel order calibration. Defaults to ``False``.
        allow_kernel_calib (bool): Whether to allow kernel calibration (e.g., GPTQ). Defaults to ``False``.
        skips (list[str]): The keys of the modules to skip. Defaults to ``[]``.
    """

    pass
