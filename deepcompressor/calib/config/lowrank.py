# -*- coding: utf-8 -*-
"""Quantization SVD calibration configuration."""

from dataclasses import dataclass, field

from omniconfig import configclass

from ...quantizer.config import QuantLowRankConfig
from ...utils.common import num2str
from ...utils.config import SkipBasedConfig
from .search import SearchBasedCalibConfig, SearchBasedCalibGranularity, SearchBasedCalibStrategy

__all__ = ["QuantLowRankCalibConfig", "SkipBasedQuantLowRankCalibConfig"]


@configclass
@dataclass
class QuantLowRankCalibConfig(SearchBasedCalibConfig, QuantLowRankConfig):
    """Configuration for quantization low-rank branch calibration.

    Args:
        rank (`int`, *optional*, defaults to `32`):
            The rank of the low-rank branch.
        exclusive (`bool`, *optional*, defaults to `False`):
            Whether to use exclusive low-rank branch for each weight sharing the inputs.
        compensate (`bool`, *optional*, defaults to `False`):
            Whether the low-rank branch compensates the quantization error.
        degree (`int`, *optional*, default=`2`):
            The power degree for the quantization error. Defaults to `2`.
        objective (`SearchBasedCalibObjective`, *optional*, default=`SearchBasedCalibObjective.OutputsError`):
            The objective for quantization calibration.
        sample_batch_size (`int`, *optional*, default=`-1`):
            The samples batch size for calibration.
        sample_size (`int`, *optional*, default=`-1`):
            The calibration sample size.
        outputs_device (`str`, *optional*, default=`"cpu"`):
            The device to store the precomputed outputs of the module.
        num_iters (`int`, *optional*, default=`1`):
            The number of iterations.
        early_stop (`bool`, *optional*, default=`False`):
            Whether to stop the calibration early.
    """

    granularity: SearchBasedCalibGranularity = field(init=False, default=SearchBasedCalibGranularity.Layer)
    element_batch_size: int = field(init=False, default=-1)
    element_size: int = field(init=False, default=-1)
    pre_reshape: bool = field(init=False, default=True)
    num_iters: int = 1
    early_stop: bool = False

    def __post_init__(self):
        if self.strategy != SearchBasedCalibStrategy.Manual:
            self.strategy = SearchBasedCalibStrategy.GridSearch
        if self.compensate and self.num_iters <= 1:
            self.exclusive = True
        super().__post_init__()

    def generate_dirnames(self, *, prefix: str = "", **kwargs) -> list[str]:
        """Generate the directory names of the configuration.

        Returns:
            list[str]: The directory names.
        """
        names = super().generate_dirnames(**kwargs)
        name = f"i{num2str(self.num_iters)}.r{num2str(self.rank)}"
        if self.exclusive:
            name += ".exclusive"
        if self.compensate:
            name += ".compensate"
        if self.early_stop and self.num_iters > 1:
            name += ".earlystop"
        names.append(name)
        if prefix:
            names = [f"{prefix}.{name}" for name in names]
        return names


@configclass
@dataclass
class SkipBasedQuantLowRankCalibConfig(SkipBasedConfig, QuantLowRankCalibConfig):
    """Configuration for Quantization Low-Rank Branch calibration.

    Args:
        rank (`int`, *optional*, defaults to `32`):
            The rank of the low-rank branch.
        exclusive (`bool`, *optional*, defaults to `False`):
            Whether to use exclusive low-rank branch for each weight sharing the inputs.
        compensate (`bool`, *optional*, defaults to `False`):
            Whether the low-rank branch compensates the quantization error.
        degree (`int`, *optional*, default=`2`):
            The power degree for the quantization error. Defaults to `2`.
        objective (`SearchBasedCalibObjective`, *optional*, default=`SearchBasedCalibObjective.OutputsError`):
            The objective for quantization calibration.
        sample_batch_size (`int`, *optional*, default=`-1`):
            The samples batch size for calibration.
        sample_size (`int`, *optional*, default=`-1`):
            The calibration sample size.
        outputs_device (`str`, *optional*, default=`"cpu"`):
            The device to store the precomputed outputs of the module.
        num_iters (`int`, *optional*, default=`1`):
            The number of iterations.
        early_stop (`bool`, *optional*, default=`False`):
            Whether to stop the calibration early.
        skips (`list[str]`, *optional*, default=`[]`):
            The keys of the modules to skip.
    """

    pass
