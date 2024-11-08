# -*- coding: utf-8 -*-
"""Channel reordering module."""

import gc
import typing as tp
from dataclasses import _MISSING_TYPE, MISSING, dataclass

import torch
import torch.nn as nn

from ..data.cache import TensorsCache
from ..data.common import TensorType
from ..quantizer.processor import Quantizer
from ..utils import math, tools
from ..utils.hooks import BaseInputPackager, BaseOutputPackager, BaseTensorProcessor
from .config import (
    ChannelOrderCalibConfig,
    SearchBasedCalibGranularity,
    SearchBasedCalibObjective,
    SearchBasedCalibStrategy,
)
from .metric import ChannelMetric
from .search import SearchBasedCalibrator

__all__ = ["ChannelOrderCalibrator", "ChannelReorderer"]


@dataclass
class ChannelReorderer(BaseTensorProcessor):
    """Activation channel reordering processor."""

    index: torch.Tensor
    channels_dim: int
    # region hook-related attributes
    input_packager: BaseInputPackager | None = None
    output_packager: BaseOutputPackager | None = None
    # endregion

    def is_enabled(self) -> bool:
        return self.index is not None

    def get_input_packager(self) -> BaseInputPackager | None:
        return self.input_packager

    def get_output_packager(self) -> BaseOutputPackager | None:
        return self.output_packager

    def process(self, tensor: torch.Tensor) -> torch.Tensor:
        """Process the tensor.

        Args:
            tensor (torch.Tensor): The tensor to process.

        Returns:
            torch.Tensor: The processed tensor.
        """
        self.index = self.index.to(device=tensor.device)
        return tensor.index_select(dim=self.channels_dim, index=self.index)


def get_channel_index_from_rank(
    rank: torch.Tensor,
    num_channels: int,
    num_groups: int,
    index_mode: ChannelOrderCalibConfig.ChannelIndex,
) -> torch.Tensor:
    """Get the index from the rank.

    Args:
        rank (`torch.Tensor`):
            The rank of the channels.
        num_channels (`int`):
            The number of channels.
        num_groups (`int`):
            The number of groups.
        index_mode (`ChannelOrderCalibConfig.ChannelIndex`):
            The index mode.

    Returns:
        `torch.Tensor`:
            The index of the channels, i.e., the order of the channels.
    """
    if index_mode == ChannelOrderCalibConfig.ChannelIndex.Transpose:
        return rank.view(num_channels // num_groups, num_groups).t().reshape(-1)
    elif index_mode == ChannelOrderCalibConfig.ChannelIndex.Sequential:
        return rank
    else:
        raise ValueError(f"Unsupported index mode: {index_mode}")


def get_channel_metric(
    inputs: TensorsCache,
    weights: tp.Sequence[torch.Tensor],
    metric_mode: ChannelOrderCalibConfig.ChannelMetric,
    num_channels: int,
    num_heads: int = 1,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Get the metric value of the channels.

    Args:
        inputs (`TensorsCache`):
            The input activations.
        weights (`Sequence[torch.Tensor]`):
            The weight tensors.
        metric_mode (`ChannelOrderCalibConfig.ChannelMetric`):
            The channel metric mode.
        num_channels (`int`):
            The number of channels.
        num_heads (`int`, *optional*, defaults to `1`):
            The number of heads.
        device (`torch.device` or `str` or `None`, *optional*, defaults to `None`):
            The device of the metric value tensor.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            The data type of the metric value tensor.

    Returns:
        `torch.Tensor`:
            The metric value of the channels.
    """
    metric_name = metric_mode.name
    if metric_name.endswith("Product"):
        metric_name = metric_name[:-7]
        ipts_metric = get_channel_metric(
            inputs=inputs,
            weights=weights,
            metric_mode=ChannelOrderCalibConfig.ChannelMetric[f"Inputs{metric_name}"],
            num_channels=num_channels,
            num_heads=num_heads,
            device=device,
            dtype=dtype,
        )
        wgts_metric = get_channel_metric(
            inputs=inputs,
            weights=weights,
            metric_mode=ChannelOrderCalibConfig.ChannelMetric[f"Weights{metric_name}"],
            num_channels=num_channels,
            num_heads=num_heads,
            device=device,
            dtype=dtype,
        )
        return ipts_metric * wgts_metric
    else:
        if metric_name.startswith("Inputs"):
            assert inputs.num_tensors == 1, f"Only one input source is allowed, got {inputs.num_tensors}"
            metric_name, tensors = metric_name[6:], inputs.front().get_standardized_data(reshape=False)
        else:
            assert metric_name.startswith("Weights")
            metric_name, tensors = metric_name[7:], weights
        group_shape = [-1] * tensors[0].ndim
        group_shape[1] = num_channels // num_heads
        # convert metric name from camel case to snake case
        metric_name = "".join(["_" + c.lower() if c.isupper() else c for c in metric_name])
        metric_name = metric_name.lstrip("_")
        metric_fn = getattr(ChannelMetric, metric_name)
        return metric_fn(tensors, num_channels, group_shape, device=device, dtype=dtype).view(num_channels)


def update_channel_metric(
    metric: torch.Tensor | None,
    inputs: TensorsCache,
    weights: tp.Sequence[torch.Tensor],
    metric_mode: ChannelOrderCalibConfig.ChannelMetric,
    num_channels: int,
    num_heads: int = 1,
    device: torch.device | str = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Update the metric value of the channels.

    Args:
        metric (`torch.Tensor` or `None`):
            The metric value of the channels.
        inputs (`TensorsCache`):
            The input activations.
        weights (`Sequence[torch.Tensor]`):
            The weight tensors.
        metric_mode (`ChannelOrderCalibConfig.ChannelMetric`):
            The channel metric mode.
        num_channels (`int`):
            The number of channels.
        num_heads (`int`, *optional*, defaults to `1`):
            The number of heads.
        device (`torch.device` or `str`, *optional*, defaults to `None`):
            The device of the metric value tensor.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            The data type of the metric value tensor.

    Returns:
        `torch.Tensor`:
            The updated metric value of the channels.
    """
    _metric = get_channel_metric(
        inputs=inputs,
        weights=weights,
        metric_mode=metric_mode,
        num_channels=num_channels,
        num_heads=num_heads,
        device=device,
        dtype=dtype,
    )
    if metric is None:
        return _metric
    elif "Max" in metric_mode.name:
        return torch.maximum(metric, _metric)
    else:
        return metric.add_(_metric)


def init_channel_index_from_metric(
    metric: torch.Tensor,
    /,
    metric_mode: ChannelOrderCalibConfig.ChannelMetric,
    index_mode: ChannelOrderCalibConfig.ChannelIndex,
    group_size: int,
    num_heads: int = 1,
    num_head_repeats: int = 1,
) -> torch.Tensor:
    """Get the index of the channels.

    Args:
        metric (`torch.Tensor`):
            The metric value of the channels.
        metric_mode (`ChannelOrderCalibConfig.ChannelMetric`):
            The channel metric mode.
        index_mode (`ChannelOrderCalibConfig.ChannelIndex`):
            The index mode.
        group_size (`int`):
            The quantization group size.
        num_heads (`int`, *optional*, defaults to `1`):
            The number of heads.
        num_head_repeats (`int`, *optional*, defaults to `1`):
            The number of head repeats.

    Returns:
        `torch.Tensor`:
            The index of the channels.
    """
    num_channels = metric.numel()
    num_groups = num_channels // group_size
    if num_heads > 1:
        head_channels = num_channels // num_heads
        if num_head_repeats > 1:
            num_unique_heads = num_heads // num_head_repeats
            metric = metric.view(num_unique_heads, num_head_repeats, head_channels)
            metric = metric.amax(dim=1, keepdim=True) if "Max" in metric_mode.name else metric.sum(dim=1, keepdim=True)
            rank = metric.argsort(dim=-1).expand(num_unique_heads, num_head_repeats, -1).reshape(num_heads, -1)
        else:
            rank = metric.view(num_heads, head_channels).argsort(dim=-1)
        rank += torch.arange(0, num_channels, head_channels, dtype=torch.long, device=rank.device).view(num_heads, 1)
        index = torch.empty_like(rank)
        for head in range(num_heads):
            index[head] = get_channel_index_from_rank(
                rank[head],
                num_channels=head_channels,
                num_groups=max(num_groups // num_heads, 1),
                index_mode=index_mode,
            )
        return index.view(-1)
    else:
        rank = metric.argsort()
        return get_channel_index_from_rank(
            rank, num_channels=num_channels, num_groups=num_groups, index_mode=index_mode
        )


class ChannelOrderCalibrator(SearchBasedCalibrator[ChannelOrderCalibConfig, torch.Tensor]):
    """The calibrator for quantization channel reordering."""

    def __init__(
        self,
        config: ChannelOrderCalibConfig,
        weight_quantizer: Quantizer | None,
        input_quantizer: Quantizer | None,
        num_heads: int = 1,
        num_head_repeats: int = 1,
        develop_dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the calibrator.

        Args:
            config (`ChannelOrderCalibConfig`):
                The channel order calibration configuration.
            weight_quantizer (`Quantizer` or `None`):
                The quantizer for the weights.
            input_quantizer (`Quantizer` or `None`):
                The quantizer for the inputs.
            num_heads (`int`, *optional*, defaults to `1`):
                The number of heads.
            num_head_repeats (`int`, *optional*, defaults to `1`):
                The number of head repeats.
            develop_dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                The development data type.
        """
        super().__init__(
            tensor_type=TensorType.Weights,
            config=config,
            w_quantizer=weight_quantizer,
            x_quantizer=input_quantizer,
            y_quantizer=None,
            develop_dtype=develop_dtype,
        )
        assert self.config.objective == SearchBasedCalibObjective.OutputsError
        assert self.config.granularity == SearchBasedCalibGranularity.Layer
        if self.config.strategy == SearchBasedCalibStrategy.Manual:
            self.index_modes = [self.config.channel_index]
            self.metric_modes = [self.config.channel_metric]
        else:
            self.metric_modes = list(ChannelOrderCalibConfig.ChannelMetric.__members__.values())
            self.index_modes = list(ChannelOrderCalibConfig.ChannelIndex.__members__.values())
        self.num_index_modes, self.num_metric_modes = len(self.index_modes), len(self.metric_modes)
        self.num_heads = num_heads
        self.num_head_repeats = num_head_repeats
        self.metrics, self.channel_indexes = None, None

    @property
    def population_size(self) -> int:
        """Get the population size."""
        size = self.num_index_modes * self.num_metric_modes
        return (size + 1) if self.config.strategy != SearchBasedCalibStrategy.Manual else size

    @property
    def allows_x_quant_for_wgts(self) -> bool:
        """Whether the calibrator needs activation quantization when tensor_type is Weights."""
        return self.config.allow_x_quant

    @property
    def allows_w_quant_for_wgts(self) -> bool:
        """Whether the calibrator needs weight quantization when tensor_type is Weights."""
        return self.config.allow_w_quant

    def update_channel_metrics(self, weights: list[torch.Tensor | nn.Parameter], inputs: TensorsCache) -> None:
        """Update the metrics of the channels.

        Args:
            weights (list[torch.Tensor | nn.Parameter]): The weight tensors.
            inputs (TensorsCache): The input activations.
        """
        weights = [w.data for w in weights]
        if self.metrics is None:
            self.num_channels = weights[0].shape[1]
            self.device = weights[0].device
            self.metrics = [None] * len(self.metric_modes)
        for metric_id, metric_mode in enumerate(self.metric_modes):
            self.metrics[metric_id] = update_channel_metric(
                metric=self.metrics[metric_id],
                inputs=inputs,
                weights=weights,
                metric_mode=metric_mode,
                num_channels=self.num_channels,
                num_heads=self.num_heads,
                device=self.device,
                dtype=self.develop_dtype,
            )

    def init_channel_indexes(self) -> None:
        """Initialize the indexes."""
        if self.needs_x_quant:
            ipts_group_size = self.x_quantizer.config.smallest_group_shape[1]
        else:
            ipts_group_size = -1
        if ipts_group_size <= 0:
            ipts_group_size = self.num_channels
        if self.needs_w_quant:
            wgts_group_size = self.w_quantizer.config.smallest_group_shape[1]
        else:
            wgts_group_size = -1
        if wgts_group_size <= 0:
            wgts_group_size = self.num_channels
        group_size = min(ipts_group_size, wgts_group_size)
        self.channel_indexes = [None] + [
            init_channel_index_from_metric(
                metric,
                metric_mode=metric_mode,
                index_mode=index_mode,
                group_size=group_size,
                num_heads=self.num_heads,
                num_head_repeats=self.num_head_repeats,
            )
            for metric_mode, metric in zip(self.metric_modes, self.metrics, strict=True)
            for index_mode in self.index_modes
        ]
        self.arange = torch.arange(self.num_channels, dtype=torch.long, device=self.device)
        self.metrics = None
        gc.collect()
        torch.cuda.empty_cache()

    def _reset(self, x_wgts: list[torch.Tensor | nn.Parameter], x_acts: TensorsCache, **kwargs) -> None:
        """Reset the calibrator.

        Args:
            x_wgts (list[list[torch.Tensor | nn.Parameter]]): Weight tensors.
            x_acts (TensorsCache): Input activations.
        """
        if self.channel_indexes is None:
            self.update_channel_metrics(x_wgts, x_acts)
            self.init_channel_indexes()
        if self.config.strategy == SearchBasedCalibStrategy.Manual and self.channel_indexes[0] is None:
            self.channel_indexes = self.channel_indexes[1:]
        assert len(self.channel_indexes) == self.population_size
        self.baseline_errors, self.best_error, self.best_candidate_id = None, None, None
        self.error_stats_history = []

    def get_best(self) -> torch.Tensor:
        """Get the best candidate.

        Returns:
            torch.Tensor: The best candidate.
        """
        return self.channel_indexes[self.best_candidate_id]

    def _ask(self) -> torch.Tensor:
        """Ask for the next candidate.

        Returns:
            torch.Tensor: The next candidate.
        """
        channel_index = self.channel_indexes[self.candidate_id]
        channel_index_inverse = None
        if channel_index is not None:
            channel_index_inverse = torch.zeros_like(channel_index)
            channel_index_inverse[channel_index] = self.arange.to(device=channel_index.device)
        self.candidate_inverse = channel_index_inverse
        return channel_index

    def _tell(self, errors: list[tuple[torch.Tensor, ...]]) -> None:  # noqa: C901
        """Tell the error of the last candidate and update the best candidate.

        Args:
            errors (list[tuple[torch.Tensor, ...]]): The error of the last candidate.
        """
        errors = [tuple(math.root_(e.to(torch.float64), self.config.degree) for e in error) for error in errors]
        if self.baseline_errors is None:
            self.baseline_errors = errors
        error_stats = [0, 0, 0, 0, 0]
        for baseline_error, error in zip(self.baseline_errors, errors, strict=True):
            for be, e in zip(baseline_error, error, strict=True):
                _d = e.item() - be.item()
                if e > be:
                    error_stats[0] += 1
                if e < be:
                    error_stats[1] -= 1
                error_stats[2] += max(_d, 0)
                error_stats[3] += min(_d, 0)
                error_stats[4] += e.item()
        if self.best_error is None or error_stats < self.best_error:
            self.best_error = error_stats
            self.best_candidate_id = self.candidate_id
        if self.logger.level <= tools.logging.DEBUG:
            self.logger.debug(
                f"+ {self._get_metric_index_mode_str(self.candidate_id)} : {self._get_error_str(error_stats)}"
            )
            if self.is_last_candidate_in_iter():
                self.logger.debug(f"+ {self._get_metric_index_mode_str(self.best_candidate_id)} is the best candidate.")

    def _get_error_str(self, e: list[int | float]) -> str:
        return f"[{e[0]:+d}, {e[1]:+d}, {e[2]:>10.4f}, {e[3]:>10.4f}, {e[4]:>10.4f}]"

    def _get_metric_index_mode_str(self, candidate_id: int) -> str:
        if candidate_id == 0:
            if self.config.strategy == SearchBasedCalibStrategy.Manual:
                metric_mode, index_mode = self.metric_modes[0], self.index_modes[0]
            else:
                return f"{'baseline':>20}   {'':>10}"
        else:
            metric_id = (candidate_id - 1) % self.num_metric_modes
            index_id = (candidate_id - 1) // self.num_metric_modes
            metric_mode, index_mode = self.metric_modes[metric_id], self.index_modes[index_id]
        return f"{metric_mode.name:>20} - {index_mode.name:>10}"

    def _process_x_in_xw(self, x: torch.Tensor, channels_dim: int | _MISSING_TYPE = MISSING) -> torch.Tensor:
        if not self.needs_x_quant_for_wgts:
            return x
        if channels_dim is MISSING:
            channels_dim = self.x_quantizer.channels_dim
        if self.candidate is not None:
            x = x.index_select(dim=channels_dim, index=self.candidate.to(x.device))
        x = self.x_quantizer.quantize(x, channels_dim=channels_dim).data
        if self.candidate is not None:
            x = x.index_select(dim=channels_dim, index=self.candidate_inverse.to(x.device))
        return x

    def _process_w_in_xw(self, w: torch.Tensor) -> torch.Tensor:
        if not self.needs_w_quant_for_wgts:
            return w
        if self.candidate is not None:
            w = w.index_select(dim=1, index=self.candidate.to(w.device))
        w = self.w_quantizer.quantize(w.data, kernel=None, develop_dtype=self.develop_dtype).data
        if self.candidate is not None:
            w = w.index_select(dim=1, index=self.candidate_inverse.to(w.device))
        return w

    def _process_x_in_yx(self, x: torch.Tensor, channels_dim: int) -> torch.Tensor:
        raise RuntimeError("_process_x_in_yx should not be called in ChannelOrderCalibrator.")

    def _process_y_in_yx(self, x: torch.Tensor, channels_dim: int) -> torch.Tensor:
        raise RuntimeError("_process_y_in_yx should not be called in ChannelOrderCalibrator.")

    def _process_xw_in_yx(self, w: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("_process_xw_in_yx should not be called in ChannelOrderCalibrator.")

    def _process_yw_in_yx(self, w: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("_process_yw_in_yx should not be called in ChannelOrderCalibrator.")

    def _process_wgts_centric_mod(
        self,
        wgts: list[nn.Parameter],
        mods: list[nn.Module],
        *,
        reorder_wgts: list[tuple[nn.Parameter, int]],
        reorder_ipt_mods: list[tuple[nn.Module, int, BaseInputPackager | None]],
        reorder_opt_mods: list[tuple[nn.Module, int, BaseOutputPackager | None]],
        update_state_dict: bool = True,
        **kwargs,
    ) -> None:
        channels_index = self.candidate
        if update_state_dict:
            self._state_dict.extend([(w, w.data) for w, _ in reorder_wgts])
        if channels_index is not None:
            for w, d in reorder_wgts:
                w.data = w.data.index_select(dim=d, index=channels_index.to(w.device))
            for m, channels_dim, packager in reorder_ipt_mods:
                self._hooks.append(
                    ChannelReorderer(channels_index, channels_dim, input_packager=packager).as_hook().register(m)
                )
            for m, channels_dim, packager in reorder_opt_mods:
                self._hooks.append(
                    ChannelReorderer(channels_index, channels_dim, output_packager=packager)
                    .as_hook(is_output=True)
                    .register(m)
                )
        self._candidate_backup = channels_index
        self.candidate = None  # we have already reordered and thus do not need to reorder again in _process
        super()._process_wgts_centric_mod(wgts, mods, update_state_dict=False)

    def _recover_mod(self) -> None:
        super()._recover_mod()
        self.candidate = self._candidate_backup
        self._candidate_backup = None
