# -*- coding: utf-8 -*-
"""Channel reorder configuration."""

import enum
from dataclasses import dataclass, field

from omniconfig import configclass

from ...utils.config import SkipBasedConfig
from .search import (
    SearchBasedCalibConfig,
    SearchBasedCalibGranularity,
    SearchBasedCalibObjective,
    SearchBasedCalibStrategy,
)

__all__ = ["ChannelOrderCalibConfig", "SkipBasedChannelOrderConfig"]


@configclass
@dataclass
class ChannelOrderCalibConfig(SearchBasedCalibConfig):
    """Configuration for channel order calibration in group quantization.

    Args:
        degree (`int`, *optional*, default=`2`):
            The power degree for the quantization error. Defaults to `2`.
        strategy (`SearchBasedCalibStrategy`, *optional*, default=`SearchBasedCalibStrategy.Manual`):
            The strategy for quantization calibration.
        sample_batch_size (`int`, *optional*, default=`-1`):
            The samples batch size for calibration.
        sample_size (`int`, *optional*, default=`-1`):
            The calibration sample size.
        allow_x_quant (`bool`, *optional*, default=`True`):
            Whether to allow input quantization during calibration.
        allow_w_quant (`bool`, *optional*, default=`True`):
            Whether to allow weight quantization during calibration.
        channel_metric (`ChannelMetricMode`, *optional*, default=`ChannelMetricMode.AbsNormalizedMean`):
            The mode for computing the channel importance.
        channel_index (`ChannelIndexMode`, *optional*, default=`ChannelIndexMode.Sequential`):
            The mode for ranking the channel importance.
        dynamic (`bool`, *optional*, default=`False`):
            Whether to enable dynamic channel reorder.
    """

    class ChannelMetric(enum.Enum):
        """The mode for computing the channel importance."""

        InputsAbsMax = "xMax"
        InputsAbsMean = "xAvg"
        InputsRootMeanSquare = "xRms"
        WeightsAbsMax = "wMax"
        WeightsAbsMean = "wAvg"
        WeightsRootMeanSquare = "wRms"
        AbsMaxProduct = "pMax"
        AbsMeanProduct = "pAvg"
        RootMeanSquareProduct = "pRms"

    class ChannelIndex(enum.Enum):
        """The mode for ranking the channel importance."""

        Sequential = "Seq"
        Transpose = "Trp"

    objective: SearchBasedCalibObjective = field(init=False, default=SearchBasedCalibObjective.OutputsError)
    granularity: SearchBasedCalibGranularity = field(init=False, default=SearchBasedCalibGranularity.Layer)
    element_batch_size: int = field(init=False, default=-1)
    element_size: int = field(init=False, default=-1)
    pre_reshape: bool = field(init=False, default=True)
    allow_x_quant: bool = True
    allow_w_quant: bool = True
    channel_metric: ChannelMetric = ChannelMetric.InputsAbsMax
    channel_index: ChannelIndex = ChannelIndex.Sequential
    dynamic: bool = False

    def __post_init__(self) -> None:
        if self.strategy != SearchBasedCalibStrategy.Manual:
            self.strategy = SearchBasedCalibStrategy.GridSearch
        super().__post_init__()

    def generate_dirnames(self, *, prefix: str = "", **kwargs) -> list[str]:
        """Generate the directory names of the configuration.

        Args:
            prefix (`str`, *optional*, default=`""`):
                The prefix of the directory.

        Returns:
            `list[str]`:
                The directory names.
        """
        names = super().generate_dirnames(**kwargs)
        if self.strategy == SearchBasedCalibStrategy.Manual:
            name = f"{self.channel_metric.name}.{self.channel_index.name}"
        else:
            name = "search"
        if self.dynamic:
            name += ".dynamic"
        names.append(name)
        disallows = []
        if not self.allow_x_quant:
            disallows.append("x")
        if not self.allow_w_quant:
            disallows.append("w")
        if disallows:
            names.append(f"disallow.[{'+'.join(disallows)}]")
        if prefix:
            names = [f"{prefix}.{name}" for name in names]
        return names


@configclass
@dataclass
class SkipBasedChannelOrderConfig(SkipBasedConfig, ChannelOrderCalibConfig):
    """Configuration for channel order calibration in group quantization.

    Args:
        degree (`int`, *optional*, default=`2`):
            The power degree for the quantization error. Defaults to `2`.
        strategy (`SearchBasedCalibStrategy`, *optional*, default=`SearchBasedCalibStrategy.Manual`):
            The strategy for quantization calibration.
        sample_batch_size (`int`, *optional*, default=`-1`):
            The samples batch size for calibration.
        sample_size (`int`, *optional*, default=`-1`):
            The calibration sample size.
        allow_x_quant (`bool`, *optional*, default=`True`):
            Whether to allow input quantization during calibration.
        allow_w_quant (`bool`, *optional*, default=`True`):
            Whether to allow weight quantization during calibration.
        channel_metric (`ChannelMetricMode`, *optional*, default=`ChannelMetricMode.AbsNormalizedMean`):
            The mode for computing the channel importance.
        channel_index (`ChannelIndexMode`, *optional*, default=`ChannelIndexMode.Sequential`):
            The mode for ranking the channel importance.
        dynamic (`bool`, *optional*, default=`False`):
            Whether to enable dynamic channel reorder.
        skips (`list[str]`, *optional*, default=`[]`):
            The keys of the modules to skip.
    """

    pass
