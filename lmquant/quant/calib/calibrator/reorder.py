# -*- coding: utf-8 -*-
"""Channel reordering module."""

import gc
import logging
import typing as tp

import torch
import torch.nn as nn
import torch.utils.hooks

from ....dataset.cache import ActivationsCache
from ....utils.math import root_
from ...data.metric import ChannelMetric
from ...quantizer.base import Quantizer
from ..config import (
    QuantChannelOrderCalibConfig,
    QuantTensorType,
    SearchBasedCalibGranularity,
    SearchBasedCalibObjective,
    SearchBasedCalibStrategy,
)
from .base import SearchBasedQuantCalibrator

__all__ = ["ChannelOrderCalibrator"]


def get_channel_index_from_rank(
    rank: torch.Tensor,
    num_channels: int,
    num_groups: int,
    index_mode: QuantChannelOrderCalibConfig.ChannelIndex,
) -> torch.Tensor:
    """Get the index from the rank.

    Args:
        rank (torch.Tensor): The rank of the channels.
        num_channels (int): The number of channels.
        num_groups (int): The number of groups.
        index_mode (QuantChannelOrderCalibConfig.ChannelIndex): The index mode.

    Returns:
        torch.Tensor: The index of the channels.
    """
    if index_mode == QuantChannelOrderCalibConfig.ChannelIndex.Transpose:
        return rank.view(num_channels // num_groups, num_groups).t().reshape(-1)
    elif index_mode == QuantChannelOrderCalibConfig.ChannelIndex.Sequential:
        return rank
    else:
        raise ValueError(f"Unsupported index mode: {index_mode}")


def get_channel_metric(
    ipts: ActivationsCache,
    wgts: list[torch.Tensor],
    metric_mode: QuantChannelOrderCalibConfig.ChannelMetric,
    num_channels: int,
    num_heads: int = 1,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Get the metric value of the channels.

    Args:
        ipts (ActivationsCache): The input activations.
        wgts (list[torch.Tensor]): The weight tensors.
        metric_mode (QuantChannelOrderCalibConfig.ChannelMetric): The metric mode.
        num_channels (int): The number of channels.
        num_heads (int, optional): The number of heads. Defaults to ``1``.
        device (torch.device, optional): The device to use. Defaults to ``None``.
        dtype (torch.dtype, optional): The data type to use. Defaults to ``torch.float32``.

    Returns:
        torch.Tensor: The metric value of the channels.
    """
    metric_name = metric_mode.name
    if metric_name.endswith("Product"):
        metric_name = metric_name[:-7]
        ipts_metric = get_channel_metric(
            ipts=ipts,
            wgts=wgts,
            metric_mode=QuantChannelOrderCalibConfig.ChannelMetric[f"Inputs{metric_name}"],
            num_channels=num_channels,
            num_heads=num_heads,
            device=device,
            dtype=dtype,
        )
        wgts_metric = get_channel_metric(
            ipts=ipts,
            wgts=wgts,
            metric_mode=QuantChannelOrderCalibConfig.ChannelMetric[f"Weights{metric_name}"],
            num_channels=num_channels,
            num_heads=num_heads,
            device=device,
            dtype=dtype,
        )
        return ipts_metric * wgts_metric
    else:
        if metric_name.startswith("Inputs"):
            assert ipts.num_sources == 1, f"Only one input source is allowed, got {ipts.num_sources}"
            metric_name = metric_name[6:]
            tensors = [x.view(-1, *x.shape[ipts[0].channels_dim :]) for x in ipts[0].cached]
        else:
            assert metric_name.startswith("Weights")
            metric_name = metric_name[7:]
            tensors = wgts
        group_shape = [-1] * tensors[0].ndim
        group_shape[1] = num_channels // num_heads
        # convert metric name from camel case to snake case
        metric_name = "".join(["_" + c.lower() if c.isupper() else c for c in metric_name])
        metric_name = metric_name.lstrip("_")
        metric_fn = getattr(ChannelMetric, metric_name)
        return metric_fn(tensors, num_channels, group_shape, device=device, dtype=dtype).view(num_channels)


def update_channel_metric(
    metric: torch.Tensor | None,
    ipts: ActivationsCache,
    wgts: list[torch.Tensor],
    metric_mode: QuantChannelOrderCalibConfig.ChannelMetric,
    num_channels: int,
    num_heads: int = 1,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Update the metric value of the channels.

    Args:
        metric (torch.Tensor | None): The metric value of the channels.
        ipts (ActivationsCache): The input activations.
        wgts (list[torch.Tensor]): The weight tensors.
        metric_mode (QuantChannelOrderCalibConfig.ChannelMetric): The metric mode.
        num_channels (int): The number of channels.
        num_heads (int, optional): The number of heads. Defaults to ``1``.
        device (torch.device, optional): The device to use. Defaults to ``None``.
        dtype (torch.dtype, optional): The data type to use. Defaults to ``torch.float32``.

    Returns:
        torch.Tensor: The updated metric value of the channels.
    """
    _metric = get_channel_metric(
        ipts=ipts,
        wgts=wgts,
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
    metric_mode: QuantChannelOrderCalibConfig.ChannelMetric,
    index_mode: QuantChannelOrderCalibConfig.ChannelIndex,
    group_size: int,
    num_heads: int = 1,
    num_head_repeats: int = 1,
) -> torch.Tensor:
    """Get the index of the channels.

    Args:
        metric (torch.Tensor, optional): The metric of the channels.
        metric_mode (QuantChannelOrderCalibConfig.ChannelMetric): The metric mode.
        index_mode (QuantChannelOrderCalibConfig.ChannelIndex): The index mode.
        group_size (int): The size of the group.
        num_heads (int, optional): The number of heads. Defaults to ``1``.
        num_head_repeats (int, optional): The number of head repeats. Defaults to ``1``.

    Returns:
        torch.Tensor: The index of the channels.
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


_UNPACK_INPUTS_FN = tp.Callable[[nn.Module, tuple[tp.Any, ...], dict[str, tp.Any]], torch.Tensor]
_REPACK_INPUTS_FN = tp.Callable[
    [torch.Tensor, nn.Module, tuple[tp.Any, ...], dict[str, tp.Any]], tuple[tuple[tp.Any, ...], dict[str, tp.Any]]
]
_UNPACK_OUTPUTS_FN = tp.Callable[[nn.Module, tuple[tp.Any, ...], dict[str, tp.Any], tp.Any], torch.Tensor]
_REPACK_OUTPUTS_FN = tp.Callable[[torch.Tensor, nn.Module, tuple[tp.Any, ...], dict[str, tp.Any], tp.Any], tp.Any]


class ChannelOrderCalibrator(SearchBasedQuantCalibrator[QuantChannelOrderCalibConfig, torch.Tensor]):
    """The calibrator for quantization channel reordering."""

    def __init__(
        self,
        calib_config: QuantChannelOrderCalibConfig,
        wgts_quantizer: Quantizer,
        ipts_quantizer: Quantizer,
        num_heads: int = 1,
        num_head_repeats: int = 1,
        allow_kernel_calib: bool = True,
        develop_dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the calibrator.

        Args:
            calib_config (QuantChannelOrderCalibConfig): The calibration configuration.
            wgts_quantizer (KernelQuantizer): The quantizer for the weights.
            ipts_quantizer (KernelQuantizer): The quantizer for the inputs.
            num_heads (int, optional): The number of heads. Defaults to ``1``.
            num_head_repeats (int, optional): The number of head repeats. Defaults to ``1``.
            allow_kernel_calib (bool, optional): Whether to allow kernel calibration. Defaults to ``True``.
            develop_dtype (torch.dtype, optional): The data type for development. Defaults to ``torch.float32``.
        """
        super().__init__(
            tensor_type=QuantTensorType.Weights,
            calib_config=calib_config,
            wgts_quantizer=wgts_quantizer,
            ipts_quantizer=ipts_quantizer,
            opts_quantizer=None,
            allow_kernel_calib=allow_kernel_calib,
            develop_dtype=develop_dtype,
        )
        assert self.calib_config.objective == SearchBasedCalibObjective.OutputsError
        assert self.calib_config.granularity == SearchBasedCalibGranularity.Layer
        if self.calib_config.strategy == SearchBasedCalibStrategy.Manual:
            self.index_modes = [self.calib_config.channel_index]
            self.metric_modes = [self.calib_config.channel_metric]
        else:
            self.metric_modes = [_ for _ in QuantChannelOrderCalibConfig.ChannelMetric.__members__.values()]
            self.index_modes = [_ for _ in QuantChannelOrderCalibConfig.ChannelIndex.__members__.values()]
        self.num_index_modes, self.num_metric_modes = len(self.index_modes), len(self.metric_modes)
        self.num_heads = num_heads
        self.num_head_repeats = num_head_repeats
        self.metrics, self.channel_indexes = None, None

    @property
    def population_size(self) -> int:
        """Get the population size."""
        size = self.num_index_modes * self.num_metric_modes
        return (size + 1) if self.calib_config.strategy != SearchBasedCalibStrategy.Manual else size

    @property
    def enabled_quant_ipts_for_wgts(self) -> bool:
        """Whether the calibrator needs activation quantization when tensor_type is Weights."""
        return True

    def update_channel_metrics(self, wgts: list[torch.Tensor | nn.Parameter], ipts: ActivationsCache) -> None:
        """Update the metrics of the channels.

        Args:
            wgts (list[torch.Tensor | nn.Parameter]): The weight tensors.
            ipts (ActivationsCache): The inputs.
        """
        wgts = [w.data for w in wgts]
        if self.metrics is None:
            self.num_channels = wgts[0].shape[1]
            self.device = wgts[0].device
            self.metrics = [None] * len(self.metric_modes)
        for metric_id, metric_mode in enumerate(self.metric_modes):
            self.metrics[metric_id] = update_channel_metric(
                metric=self.metrics[metric_id],
                ipts=ipts,
                wgts=wgts,
                metric_mode=metric_mode,
                num_channels=self.num_channels,
                num_heads=self.num_heads,
                device=self.device,
                dtype=self.develop_dtype,
            )

    def init_channel_indexes(self) -> None:
        """Initialize the indexes."""
        if (
            self.ipts_quantizer is not None
            and self.ipts_quantizer.config is not None
            and self.ipts_quantizer.config.dtype is not None
        ):
            ipts_group_size = self.ipts_quantizer.config.smallest_group_shape[1]
        else:
            ipts_group_size = -1
        if ipts_group_size <= 0:
            ipts_group_size = self.num_channels
        if (
            self.wgts_quantizer is not None
            and self.wgts_quantizer.config is not None
            and self.wgts_quantizer.config.dtype is not None
        ):
            wgts_group_size = self.wgts_quantizer.config.smallest_group_shape[1]
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
            for metric_mode, metric in zip(self.metric_modes, self.metrics)
            for index_mode in self.index_modes
        ]
        self.arange = torch.arange(self.num_channels, dtype=torch.long, device=self.device)
        self.metrics = None
        gc.collect()
        torch.cuda.empty_cache()

    def _reset(self, ipt_wgts: list[torch.Tensor | nn.Parameter], ipts: ActivationsCache, **kwargs) -> None:
        """Reset the calibrator.

        Args:
            wgts (list[list[torch.Tensor | nn.Parameter]]): Weight tensors.
            ipts (ActivationsCache): Input activations.
        """
        self._ipts_for_wgts_quant = ipts if self.allow_kernel_calib else None
        if self.channel_indexes is None:
            self.update_channel_metrics(ipt_wgts, ipts)
            self.init_channel_indexes()
        if self.calib_config.strategy == SearchBasedCalibStrategy.Manual and self.channel_indexes[0] is None:
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
        return self.channel_indexes[self.candidate_id]

    def _tell(self, errors: list[tuple[torch.Tensor, ...]]) -> None:  # noqa: C901
        """Tell the error of the last candidate and update the best candidate.

        Args:
            errors (list[tuple[torch.Tensor, ...]]): The error of the last candidate.
        """
        errors = [tuple(root_(e.to(torch.float64), self.calib_config.degree) for e in error) for error in errors]
        if self.baseline_errors is None:
            self.baseline_errors = errors
        error_stats = [0, 0, 0, 0, 0]
        for baseline_error, error in zip(self.baseline_errors, errors):
            for be, e in zip(baseline_error, error):
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
        if self.logger.level <= logging.DEBUG:
            self.logger.debug(
                f"+ {self._get_metric_index_mode_str(self.candidate_id)} : {self._get_error_str(error_stats)}"
            )
            if self.is_last_candidate_in_iter():
                self.logger.debug(f"+ {self._get_metric_index_mode_str(self.best_candidate_id)} is the best candidate.")

    def _get_error_str(self, e: list[int | float]) -> str:
        return f"[{e[0]:+d}, {e[1]:+d}, {e[2]:>10.4f}, {e[3]:>10.4f}, {e[4]:>10.4f}]"

    def _get_metric_index_mode_str(self, candidate_id: int) -> str:
        if candidate_id == 0:
            if self.calib_config.strategy == SearchBasedCalibStrategy.Manual:
                metric_mode, index_mode = self.metric_modes[0], self.index_modes[0]
            else:
                return f"{'baseline':>20}   {'':>10}"
        else:
            metric_id = (candidate_id - 1) % self.num_metric_modes
            index_id = (candidate_id - 1) // self.num_metric_modes
            metric_mode, index_mode = self.metric_modes[metric_id], self.index_modes[index_id]
        return f"{metric_mode.name:>20} - {index_mode.name:>10}"

    def _process_x_in_xw(self, x: torch.Tensor, channels_dim: int) -> torch.Tensor:
        if not self.needs_quant_ipts:
            return x
        return self.ipts_quantizer.quantize(x, channels_dim=channels_dim).data

    def _process_w_in_xw(self, w: torch.Tensor) -> torch.Tensor:
        if not self.needs_quant_wgts:
            return w
        return self.wgts_quantizer.quantize(
            w.data,
            kernel_config=self.kernel_config if self._ipts_for_wgts_quant is not None else None,
            inputs=self._ipts_for_wgts_quant,
            develop_dtype=self.develop_dtype,
        ).data

    def _process_w_in_yx(self, w: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("_process_w_in_yx should not be called in ChannelOrderCalibrator.")

    def _process_x_in_yx(self, x: torch.Tensor, channels_dim: int) -> torch.Tensor:
        raise RuntimeError("_process_x_in_yx should not be called in ChannelOrderCalibrator.")

    def _process_y_in_yx(self, x: torch.Tensor, channels_dim: int) -> torch.Tensor:
        raise RuntimeError("_process_y_in_yx should not be called in ChannelOrderCalibrator.")

    def _process_wgts_centric_mod(
        self,
        wgts: list[nn.Parameter],
        mods: list[nn.Module],
        *,
        reorder_wgts: list[tuple[nn.Parameter, int]],
        reorder_ipt_mods: list[tuple[nn.Module, int, _UNPACK_INPUTS_FN | None, _REPACK_OUTPUTS_FN | None]],
        reorder_opt_mods: list[tuple[nn.Module, int, _UNPACK_OUTPUTS_FN | None, _REPACK_OUTPUTS_FN | None]],
        update_state_dict: bool = True,
        **kwargs,
    ) -> None:
        channel_index = self.candidate
        channel_index_inverse = None
        if channel_index is not None:
            channel_index_inverse = torch.zeros_like(channel_index)
            channel_index_inverse[channel_index] = self.arange.to(device=channel_index.device)
        if update_state_dict:
            self._state_dict.extend([(w, w.data) for w in wgts])
            self._state_dict.extend([(w, w.data) for w, _ in reorder_wgts])
        if channel_index is not None:
            for w in wgts:
                w.data = w.data.index_select(dim=1, index=channel_index.to(w.device))
            for w, d in reorder_wgts:
                w.data = w.data.index_select(dim=d, index=channel_index.to(w.device))
            for m, channels_dim, unpack_fn, repack_fn in reorder_ipt_mods:
                self._hooks.append(
                    m.register_forward_pre_hook(
                        self._get_reorder_in_hook(channel_index, channels_dim, unpack_fn, repack_fn),
                        prepend=True,
                        with_kwargs=True,
                    )
                )
            for m, channels_dim, unpack_fn, repack_fn in reorder_opt_mods:
                self._hooks.append(
                    m.register_forward_hook(
                        self._get_reorder_out_hook(channel_index_inverse, channels_dim, unpack_fn, repack_fn),
                        prepend=True,
                        with_kwargs=True,
                    )
                )
        self._orig_ipts_for_wgts_quant = self._ipts_for_wgts_quant
        if self._ipts_for_wgts_quant is not None and self.allow_kernel_calib:
            assert self._ipts_for_wgts_quant.num_sources == 1, "Only one input source is allowed."
            if channel_index is not None:
                self._ipts_for_wgts_quant = ActivationsCache(
                    self._ipts_for_wgts_quant[0].reorder(channel_index), self._ipts_for_wgts_quant.num_samples
                )
        else:
            self._ipts_for_wgts_quant = None
        super()._process_wgts_centric_mod(wgts, mods, update_state_dict=False)
        self._ipts_for_wgts_quant = self._orig_ipts_for_wgts_quant

    @staticmethod
    def _get_reorder_in_hook(
        channel_index: torch.Tensor,
        channels_dim: int,
        unpack_fn: _UNPACK_INPUTS_FN | None = None,
        repack_fn: _REPACK_INPUTS_FN | None = None,
    ) -> tp.Callable[
        [nn.Module, tuple[torch.Tensor, ...], dict[str, tp.Any]], tuple[tuple[torch.Tensor, ...], dict[str, tp.Any]]
    ]:
        def hook(
            module: nn.Module, input_args: tuple[torch.Tensor, ...], input_kwargs: dict[str, tp.Any]
        ) -> tuple[torch.Tensor, ...]:
            if unpack_fn is None:
                assert isinstance(input_args, tuple)
                x = input_args[0]
            else:
                x = unpack_fn(module, input_args, input_kwargs)
            assert isinstance(x, torch.Tensor)
            x = x.index_select(dim=channels_dim, index=channel_index.to(x.device))
            if repack_fn is None:
                return (x, *input_args[1:]), input_kwargs
            else:
                return repack_fn(x, module, input_args, input_kwargs)

        return hook

    @staticmethod
    def _get_reorder_out_hook(
        channel_index,
        channels_dim,
        unpack_fn: _UNPACK_OUTPUTS_FN | None = None,
        repack_fn: _REPACK_OUTPUTS_FN | None = None,
    ) -> tp.Callable[[nn.Module, tuple[torch.Tensor, ...], dict[str, tp.Any], tp.Any], tp.Any]:
        def hook(
            module: nn.Module,
            input_args: tuple[torch.Tensor, ...],
            input_kwargs: dict[str, tp.Any],
            outputs: tp.Any,
        ) -> tp.Any:
            if unpack_fn is None:
                y = outputs[0] if not isinstance(outputs, torch.Tensor) else outputs
            else:
                y = unpack_fn(module, input_args, input_kwargs, outputs)
            assert isinstance(y, torch.Tensor)
            y = y.index_select(dim=channels_dim, index=channel_index.to(y.device))
            if repack_fn is None:
                return (y, *outputs[1:]) if not isinstance(outputs, torch.Tensor) else y
            else:
                return repack_fn(y, module, input_args, input_kwargs, outputs)

        return hook
