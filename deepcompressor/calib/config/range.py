# -*- coding: utf-8 -*-
"""Quantization dynamic range calibration configuration."""

from dataclasses import dataclass

from omniconfig import configclass

from ...utils.common import num2str
from ...utils.config import SkipBasedConfig
from .search import SearchBasedCalibConfig, SearchBasedCalibStrategy

__all__ = ["DynamicRangeCalibConfig", "SkipBasedDynamicRangeCalibConfig"]


@configclass
@dataclass
class DynamicRangeCalibConfig(SearchBasedCalibConfig):
    """Configuration for quantization dynamic range calibration.

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
        ratio (`float`, *optional*, default=`1.0`):
            The dynamic range ratio.
        max_shrink (`float`, *optional*, default=`0.2`):
            Maximum shrinkage ratio.
        max_expand (`float`, *optional*, default=`1.0`):
            Maximum expansion ratio.
        num_grids (`int`, *optional*, default=`80`):
            Number of grids for linear range search.
        allow_scale (`bool`, *optional*, default=`False`):
            Whether to allow range dynamic scaling.
    """

    ratio: float = 1.0
    max_shrink: float = 0.2
    max_expand: float = 1.0
    num_grids: int = 80
    allow_scale: bool = False

    def get_linear_ratios(self) -> list[float]:
        """Get the ratios for linear range search.

        Returns:
            `list[float]`:
                The dynamic range ratio candidates for linear range search.
        """
        num_grids, max_shrink, max_expand = self.num_grids, self.max_shrink, self.max_expand
        assert max_shrink < 1, "maximal shrinkage ratio must be less than 1"
        ratios = [1 - grid / num_grids * (1 - max_shrink) for grid in range(1, num_grids + 1)]
        if max_expand > 1:
            ratios += [1 + grid / num_grids * (max_expand - 1) for grid in range(1, num_grids + 1)]
        return ratios

    def get_ratios(self) -> list[list[float]]:
        """Get the ratios for linear range search.

        Returns:
            `list[list[float]]`:
                The dynamic range ratio candidates for linear range search.
        """
        if self.strategy == SearchBasedCalibStrategy.Manual:
            return [[self.ratio]]
        elif self.strategy == SearchBasedCalibStrategy.GridSearch:
            return [[1.0], self.get_linear_ratios()]
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")

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
            name = f"r.[{num2str(self.ratio)}]"
        elif self.strategy == SearchBasedCalibStrategy.GridSearch:
            name = f"r.[{num2str(self.max_shrink)}.{num2str(self.max_expand)}].g{self.num_grids}"
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")
        if self.allow_scale:
            name += ".scale"
        names.append(name)
        if prefix:
            names = [f"{prefix}.{name}" for name in names]
        return names


@configclass
@dataclass
class SkipBasedDynamicRangeCalibConfig(SkipBasedConfig, DynamicRangeCalibConfig):
    """Configuration for quantization dynamic range calibration.

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
        ratio (`float`, *optional*, default=`1.0`):
            The dynamic range ratio.
        max_shrink (`float`, *optional*, default=`0.2`):
            Maximum shrinkage ratio.
        max_expand (`float`, *optional*, default=`1.0`):
            Maximum expansion ratio.
        num_grids (`int`, *optional*, default=`80`):
            Number of grids for linear range search.
        allow_scale (`bool`, *optional*, default=`False`):
            Whether to allow range dynamic scaling.
        skips (`list[str]`, *optional*, default=`[]`):
            The keys of the modules to skip.
    """

    pass
