# -*- coding: utf-8 -*-
"""Quantization dynamic range calibration configuration."""

from dataclasses import dataclass

from omniconfig import configclass

from ....utils import num2str
from .base.search import SearchBasedCalibConfig, SearchBasedCalibStrategy

__all__ = ["DynamicRangeCalibConfig"]


@configclass
@dataclass
class DynamicRangeCalibConfig(SearchBasedCalibConfig):
    """Configuration for quantization dynamic range calibration.

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
        allow_kernel_calib (bool): Whether to allow kernel calibration (e.g., GPTQ) during dynamic range calibration.
            Defaults to ``False``.
        skips (list[str]): The keys of the modules to skip. Defaults to ``[]``.
        ratio (float): The dynamic range ratio. Defaults to ``1.0``.
        max_shrink (float): Maximum shrinkage ratio. Defaults to ``0.2``.
        max_expand (float): Maximum expansion ratio. Defaults to ``1.0``.
        num_grids (int): Number of grids. Defaults to ``80``.
    """

    ratio: float = 1.0
    max_shrink: float = 0.2
    max_expand: float = 1.0
    num_grids: int = 80

    def __str__(self) -> str:
        s = f"(objective={self.objective.name}, granularity={self.granularity.name}"
        if self.strategy == SearchBasedCalibStrategy.Manual:
            return s + f", range_ratio={self.ratio})"
        elif self.strategy == SearchBasedCalibStrategy.GridSearch:
            return s + f", shrink={self.max_shrink}, expand={self.max_expand}, num_grids={self.num_grids})"
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")

    def get_linear_ratios(self) -> list[float]:
        """Get the ratios for linear range search.

        Returns:
            list[float]: The scales.
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
            list[list[float]]: The scales.
        """
        if self.strategy == SearchBasedCalibStrategy.Manual:
            return [[self.ratio]]
        elif self.strategy == SearchBasedCalibStrategy.GridSearch:
            return [[1.0], self.get_linear_ratios()]
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")

    def _generate_dirnames(self) -> list[str]:
        """Generate the directory names of the configuration.

        Returns:
            list[str]: The directory names.
        """
        if self.strategy == SearchBasedCalibStrategy.Manual:
            name = f"r.[{num2str(self.ratio)}]"
        elif self.strategy == SearchBasedCalibStrategy.GridSearch:
            name = f"r.[{num2str(self.max_shrink)}.{num2str(self.max_expand)}].g{self.num_grids}"
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")
        return [name]
