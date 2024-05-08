# -*- coding: utf-8 -*-
"""Smooth quantization module."""

import logging
import typing as tp

import torch
import torch.nn as nn

from ....dataset.cache import ActivationsCache
from ....utils import tools
from ....utils.math import root_
from ...data.metric import ChannelMetric
from ...quantizer.base import Quantizer
from ..config import QuantSmoothCalibConfig, QuantTensorType
from .base import SearchBasedQuantCalibrator

__all__ = [
    "get_smooth_scale",
    "get_smooth_range",
    "SmoothCalibrator",
    "SmoothLinearCalibrator",
    "SmoothAttentionCalibrator",
]


@torch.inference_mode()
def get_smooth_scale(*, ipts_range: torch.Tensor, wgts_range: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """Calculate the smooth scale for quantization.

    Args:
        ipts_range (torch.Tensor): Input range.
        wgts_range (torch.Tensor): Weight range.
        alpha (float): Smooth factor for input.
        beta (float): Smooth factor for weight.

    Returns:
        torch.Tensor: Smooth scale.
    """
    assert 0 <= alpha <= 1 and 0 <= beta <= 1, "The smooth factors should be in [0, 1]."
    if alpha > 0:
        scale = ipts_range.pow(alpha)
        if beta > 0:
            scale = scale.div_(wgts_range.pow(beta))
    else:
        scale = wgts_range.pow(-beta)
    scale[scale == 0] = 1
    assert not scale.isnan().any(), "The smooth scale contains NaN."
    assert not scale.isinf().any(), "The smooth scale contains Inf."
    return scale


@torch.inference_mode()
def get_smooth_range(
    tensors: list[torch.Tensor],
    /,
    *,
    group_shape: tuple[int, ...],
    range_mode: QuantSmoothCalibConfig.RangeMode,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Calculate the smooth range for input.

    Args:
        tensors (list[torch.Tensor]): Tensors to calculate the range.
        group_shape (tuple[int, ...]): Quantization group shape.
        range_mode (QuantSmoothCalibConfig.RangeMode): Smooth quantization range mode.
        device (torch.device, optional): Device. Defaults to ``None``.
        dtype (torch.dtype, optional): Data type. Defaults to ``torch.float32``.

    Returns:
        torch.Tensor: Smooth range.
    """
    # convert range mode name from camel case to snake case
    range_name = "".join(["_" + c.lower() if c.isupper() else c for c in range_mode.name]).lstrip("_")
    range_fn = getattr(ChannelMetric, range_name)
    r: torch.Tensor = range_fn(tensors, tensors[0].shape[1], group_shape, device=device, dtype=dtype)
    return r


class SmoothCalibrator(SearchBasedQuantCalibrator[QuantSmoothCalibConfig, torch.Tensor]):
    """The quantization smoothing calibrator."""

    def __init__(
        self,
        tensor_type: QuantTensorType,
        calib_config: QuantSmoothCalibConfig,
        wgts_quantizer: Quantizer | None,
        ipts_quantizer: Quantizer | None,
        opts_quantizer: Quantizer | None,
        num_heads: int = 1,
        num_head_repeats: int = 1,
        with_rope: bool = False,
        allow_kernel_calib: bool = True,
        develop_dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the calibrator.

        Args:
            tensor_type (QuantTensorType): The type of tensor to quantize. Choices are ``Weights`` and ``Outputs``.
            calib_config (QuantSmoothCalibConfig): The calibration configuration.
            wgts_quantizer (KernelQuantizer): The weight quantizer.
            ipts_quantizer (KernelQuantizer): The input quantizer.
            opts_quantizer (KernelQuantizer): The output quantizer.
            num_heads (int): The number of heads. Defaults to ``1``.
            num_head_repeats (int): The number of head repeats. Defaults to ``1``.
            with_rope (bool): Whether rotary position embedding is used. Defaults to ``False``.
            allow_kernel_calib (bool): Whether to allow kernel calibration. Defaults to ``True``.
            develop_dtype (torch.dtype): The development data type. Defaults to ``torch.float32``.
        """
        assert tensor_type in (QuantTensorType.Weights, QuantTensorType.Outputs)
        super().__init__(
            tensor_type=tensor_type,
            calib_config=calib_config,
            wgts_quantizer=wgts_quantizer,
            ipts_quantizer=ipts_quantizer,
            opts_quantizer=opts_quantizer,
            allow_kernel_calib=allow_kernel_calib,
            develop_dtype=develop_dtype,
        )
        self.num_heads = num_heads
        self.num_head_repeats = num_head_repeats
        self.with_rope = False if self.tensor_type == QuantTensorType.Weights else with_rope
        # region set group shapes of weights, inputs and outputs
        if self.needs_quant_wgts:
            w_group_shape = list(self.wgts_quantizer.config.largest_group_shape)
        else:
            w_group_shape = [1, None, -1]
        if self.needs_quant_ipts:
            x_group_shape = list(self.ipts_quantizer.config.largest_group_shape)
        else:
            x_group_shape = [1, None, -1]
        if self.needs_quant_opts:
            y_group_shape = list(self.opts_quantizer.config.largest_group_shape)
        else:
            y_group_shape = [1, None, -1]
        w_group_shape[1] = x_group_shape[1] if w_group_shape[1] is None else w_group_shape[1]
        if self.tensor_type == QuantTensorType.Weights:
            x_group_shape[1] = w_group_shape[1] if x_group_shape[1] is None else x_group_shape[1]
        else:
            x_group_shape[1] = y_group_shape[1] if x_group_shape[1] is None else x_group_shape[1]
        y_group_shape[1] = x_group_shape[1] if y_group_shape[1] is None else y_group_shape[1]
        self.w_group_shape, self.x_group_shape, self.y_group_shape = w_group_shape, x_group_shape, y_group_shape
        # endregion
        self.alpha_beta_pairs = self.calib_config.get_alpha_beta_pairs()
        self.num_alpha_beta_pairs = len(self.alpha_beta_pairs)
        self.num_ranges = len(self.range_modes)
        self.num_iters = 1

    @property
    def population_size(self) -> int:
        """Get the population size."""
        return self.num_alpha_beta_pairs * self.num_ranges

    @property
    def enabled_quant_wgts_for_ipts(self) -> bool:
        """Whether the calibrator needs weight quantization when tensor_type is not Weights."""
        return True

    @property
    def enabled_quant_ipts_for_wgts(self) -> bool:
        """Whether the calibrator needs activation quantization when tensor_type is Weights."""
        return True

    @property
    def enabled_quant_wgts_for_opts(self) -> bool:
        """Whether the calibrator needs weight quantization when tensor_type is not Outputs."""
        return False

    @property
    def range_modes(self) -> list[tuple[QuantSmoothCalibConfig.RangeMode, QuantSmoothCalibConfig.RangeMode]]:
        """Get the range modes."""
        return self.calib_config.ranges

    @property
    def x_range_modes(self) -> list[QuantSmoothCalibConfig.RangeMode]:
        """Get the input range modes."""
        return self.calib_config.x_ranges

    @property
    def w_range_modes(self) -> list[QuantSmoothCalibConfig.RangeMode]:
        """Get the weight range modes."""
        return self.calib_config.w_ranges

    @property
    def y_range_modes(self) -> list[QuantSmoothCalibConfig.RangeMode]:
        """Get the output range modes."""
        return self.calib_config.y_ranges

    def _reset(  # noqa: C901
        self,
        *,
        ipt_wgts: list[torch.Tensor | nn.Parameter],
        ipts: ActivationsCache,
        opt_wgts: list[torch.Tensor | nn.Parameter] = None,
        opts: ActivationsCache = None,
        **kwargs,
    ) -> None:
        """Reset the calibrator.

        Args:
            ipt_wgts (list[torch.Tensor | nn.Parameter]): The weights related to the input activations.
            ipts (ActivationsCache): The input activations.
            opt_wgts (list[torch.Tensor | nn.Parameter], optional): The weights related to the output activations.
                Defaults to ``None``.
            opts (ActivationsCache, optional): The output activations. Defaults to ``None``.
        """
        wgts_centric = self.tensor_type == QuantTensorType.Weights
        self.num_channels = ipt_wgts[0].shape[1] if wgts_centric else ipt_wgts[0].shape[0]
        device = ipt_wgts[0].device
        if self.num_heads > 1 and self.num_head_repeats > 1:
            self.num_unique_heads = self.num_heads // self.num_head_repeats
        else:
            self.num_unique_heads = 0
        # region get x ranges
        assert ipts.num_sources == 1, f"Only one input source is allowed, got {ipts.num_sources}"
        assert all(x.shape[ipts[0].channels_dim] == self.num_channels for x in ipts[0].cached)
        _ipts = [x.view(-1, *x.shape[ipts[0].channels_dim :]) for x in ipts[0].cached]
        x_ranges = {}
        for range_mode in self.x_range_modes:
            x_range = get_smooth_range(
                _ipts,
                group_shape=self.x_group_shape,
                range_mode=range_mode,
                device=device,
                dtype=self.develop_dtype,
            )
            if self.num_unique_heads > 0:
                x_range = x_range.view(self.num_unique_heads, self.num_head_repeats, -1)
                x_range = (x_range.amax if "Max" in range_mode.name else x_range.mean)(dim=1, keepdim=True)
                x_range = x_range.expand(self.num_unique_heads, self.num_head_repeats, -1).reshape(-1)
            if self.tensor_type == QuantTensorType.Outputs and self.with_rope:
                x_range = x_range.view(self.num_heads, 2, -1)
                x_range = (x_range.amax if "Max" in range_mode.name else x_range.mean)(dim=1, keepdim=True)
                x_range = x_range.expand(self.num_heads, 2, -1).reshape(-1)
            x_ranges[range_mode] = x_range
            if self.logger.level <= logging.DEBUG:
                self.logger.debug("+ ipts - %s", range_mode.name)
                self.logger.debug("+ ipts  = [min=%.4f, max=%.4f]", x_range.min().item(), x_range.max().item())
        del _ipts
        # endregion
        if wgts_centric:
            self._ipts_for_wgts_quant = ipts if self.allow_kernel_calib else None
            assert all(w.shape[1] == self.num_channels for w in ipt_wgts)
            _wgts = [w.data for w in ipt_wgts]
            w_ranges = {}
            for range_mode in self.w_range_modes:
                w_range = get_smooth_range(
                    _wgts,
                    group_shape=self.w_group_shape,
                    range_mode=range_mode,
                    dtype=self.develop_dtype,
                )
                if self.num_unique_heads > 0:
                    w_range = w_range.view(self.num_unique_heads, self.num_head_repeats, -1)
                    w_range = (w_range.amax if "Max" in range_mode.name else w_range.mean)(dim=1, keepdim=True)
                    w_range = w_range.expand(self.num_unique_heads, self.num_head_repeats, -1).reshape(-1)
                w_ranges[range_mode] = w_range
                if self.logger.level <= logging.DEBUG:
                    self.logger.debug("+ wgts - %s", range_mode.name)
                    self.logger.debug("+ wgts  = [min=%.4f, max=%.4f]", w_range.min().item(), w_range.max().item())
            self.ranges: list[tuple[torch.Tensor, torch.Tensor]] = [
                (x_ranges[x_range_mode], w_ranges[w_range_mode]) for x_range_mode, w_range_mode in self.range_modes
            ]
        else:
            assert opts.num_sources == 1, f"Only one output source is allowed, got {opts.num_sources}"
            self._ipts_for_wgts_quant = None
            if self.num_unique_heads > 0:
                num_opt_channels = self.num_channels // self.num_head_repeats
            else:
                num_opt_channels = self.num_channels
            assert all(w.shape[0] == self.num_channels for w in ipt_wgts)
            assert all(w.shape[0] == num_opt_channels for w in opt_wgts)
            assert all(x.shape[opts[0].channels_dim] == num_opt_channels for x in opts[0].cached)
            _opts = [x.view(-1, *x.shape[opts[0].channels_dim :]) for x in opts[0].cached]
            y_ranges = {}
            for range_mode in self.y_range_modes:
                y_range = get_smooth_range(
                    _opts,
                    group_shape=self.x_group_shape,
                    range_mode=range_mode,
                    device=device,
                    dtype=self.develop_dtype,
                )
                if self.num_unique_heads > 0:
                    y_range = y_range.view(self.num_unique_heads, 1, -1)
                    y_range = y_range.expand(self.num_unique_heads, self.num_head_repeats, -1).reshape(-1)
                if self.tensor_type == QuantTensorType.Outputs and self.with_rope:
                    y_range = y_range.view(self.num_heads, 2, -1)
                    y_range = (y_range.amax if "Max" in range_mode.name else y_range.mean)(dim=1, keepdim=True)
                    y_range = y_range.expand(self.num_heads, 2, -1).reshape(-1)
                y_ranges[range_mode] = y_range
                if self.logger.level <= logging.DEBUG:
                    self.logger.debug("+ opts - %s", range_mode.name)
                    self.logger.debug("+ opts  = [min=%.4f, max=%.4f]", y_range.min().item(), y_range.max().item())
            self.ranges: list[tuple[torch.Tensor, torch.Tensor]] = [
                (y_ranges[y_range_mode], x_ranges[x_range_mode]) for x_range_mode, y_range_mode in self.range_modes
            ]
        self.best_error: list[torch.Tensor] = None
        self.best_scale: torch.Tensor = None
        self.error_history: list[tuple[float, float]] = []

    def _split_candidate_id(self, candidate_id: int) -> tuple[int, int]:
        """Split the candidate id into alpha id, beta id and range id.

        Args:
            candidate_id (int): The candidate id.

        Returns:
            tuple[int, int]: The alpha_beta id and range id.
        """
        alpha_beta_id = candidate_id % self.num_alpha_beta_pairs
        range_id = candidate_id // self.num_alpha_beta_pairs
        return alpha_beta_id, range_id

    def get_best(self) -> torch.Tensor:
        """Get the best candidate.

        Returns:
            torch.Tensor: The best candidate.
        """
        return self.best_scale

    def _ask(self) -> torch.Tensor:
        """Ask for the next candidate.

        Returns:
            torch.Tensor: The next candidate.
        """
        alpha_beta_id, range_id = self._split_candidate_id(self.candidate_id)
        alpha, beta = self.alpha_beta_pairs[alpha_beta_id]
        x_range, w_range = self.ranges[range_id]
        if alpha == 0 and beta == 0:
            scale = torch.ones_like(w_range, dtype=self.develop_dtype)
        else:
            scale = get_smooth_scale(ipts_range=x_range, wgts_range=w_range, alpha=alpha, beta=beta)
        return scale

    def _tell(self, error: list[torch.Tensor]) -> None:  # noqa: C901
        """Tell the error of the last candidate and update the best candidate.

        Args:
            error (list[torch.Tensor]): The error of the last candidate.
        """
        numel = error[0].numel()
        assert all(e.numel() == numel for e in error)
        scale = self.candidate
        self.best_error, self.best_scale = self._update_best(
            best_error=self.best_error,
            best_scale=self.best_scale,
            error=error,
            scale=scale,
            numel=numel,
            num_channels=self.num_channels,
            num_heads=self.num_heads,
            num_head_repeats=self.num_head_repeats,
        )
        if self.logger.level <= logging.DEBUG:
            self.error_history.append(
                (
                    sum(root_(e.to(torch.float64).sum(), self.calib_config.degree).item() for e in error),
                    sum(root_(b.to(torch.float64).sum(), self.calib_config.degree).item() for b in self.best_error),
                )
            )
            if self.is_last_candidate_in_iter():
                logs: list[list[list[tuple]]] = [[] for _ in range(self.num_ranges)]
                for i in range(self.population_size):
                    c, r = self._split_candidate_id(i)
                    alpha, beta = self.alpha_beta_pairs[c]
                    if c % 5 == 0:
                        logs[r].append([])
                    logs[r][-1].append((alpha, beta, self.error_history[i][0], self.error_history[i][1]))
                for r in range(self.num_ranges):
                    self.logger.debug(
                        "  - x / w range = %s / %s", self.range_modes[r][0].name, self.range_modes[r][1].name
                    )
                    for log in logs[r]:
                        self.logger.debug(
                            "  - alpha       = [%s]",
                            ", ".join(f"{alpha:10.4f}" for alpha, beta, e, b in log),
                        )
                        self.logger.debug(
                            "  - beta        = [%s]",
                            ", ".join(f"{beta:10.4f}" for alpha, beta, e, b in log),
                        )
                        self.logger.debug(
                            "  - sum  error  = [%s]", ", ".join(f"{e:10.4f}" for alpha, beta, e, b in log)
                        )
                        self.logger.debug(
                            "  - best error  = [%s]",
                            ", ".join(f"{b:10.4f}" for alpha, beta, e, b in log),
                        )
                del logs
                self.error_history.clear()
                if self.is_last_iter():
                    scale = self.get_best()
                    tools.logging.Formatter.indent_dec()
                    self.logger.debug(
                        "  + error = %.4f",
                        sum(root_(b.to(torch.float64).sum(), self.calib_config.degree).item() for b in self.best_error),
                    )
                    self.logger.debug("  + scale = [min=%.4f, max=%.4f]", scale.min().item(), scale.max().item())
                    tools.logging.Formatter.indent_inc()

    def _reshape_scale(
        self, scale: torch.Tensor, tensor: torch.Tensor, channels_dim: int = 1, needs_reduction: bool = False
    ) -> torch.Tensor:
        if self.num_unique_heads > 0 and needs_reduction:
            scale = scale.view(self.num_unique_heads, self.num_head_repeats, -1)[:, 0, :].reshape(-1)
        shape = [1] * tensor.ndim
        shape[channels_dim] = -1
        return scale.view(shape)

    def _process_x_in_xw(self, x: torch.Tensor, channels_dim: int) -> torch.Tensor:
        if not self.needs_quant_ipts:
            return x
        shape, dtype = x.shape, x.dtype
        scale = self._reshape_scale(self.candidate, x, channels_dim)
        x = x.to(dtype=self.develop_dtype) if dtype != self.develop_dtype else x.clone()
        x = x.div_(scale)
        x = self.ipts_quantizer.quantize(
            x, channels_dim=channels_dim, default_dtype=dtype, develop_dtype=self.develop_dtype
        ).data
        x = x.mul_(scale).to(dtype=dtype)
        return x.view(shape)

    def _process_x_in_yx(self, x: torch.Tensor, channels_dim: int) -> torch.Tensor:
        if not self.needs_quant_ipts:
            return x
        shape, dtype = x.shape, x.dtype
        return self.ipts_quantizer.quantize(
            x,
            channels_dim=channels_dim,
            default_dtype=dtype,
            develop_dtype=self.develop_dtype,
        ).data.view(shape)

    def _process_w_in_xw(self, w: torch.Tensor) -> torch.Tensor:
        if not self.needs_quant_wgts:
            return w
        dtype = w.dtype
        scale = self._reshape_scale(self.candidate, w, channels_dim=1)
        w = w.to(dtype=self.develop_dtype) if dtype != self.develop_dtype else w.clone()
        w = self.wgts_quantizer.quantize(
            w.mul_(scale),
            kernel_config=self.kernel_config,
            inputs=self._ipts_for_wgts_quant,
            default_dtype=dtype,
            develop_dtype=self.develop_dtype,
        ).data
        w = w.div_(scale).to(dtype=dtype)
        return w

    def _process_y_in_yx(self, y: torch.Tensor, channels_dim: int) -> torch.Tensor:
        if not self.needs_quant_opts:
            return y
        shape, dtype = y.shape, y.dtype
        return self.opts_quantizer.quantize(
            y,
            default_dtype=dtype,
            develop_dtype=self.develop_dtype,
        ).data.view(shape)

    def _process_w_in_yx(self, w: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("_process_w_in_yx should not be called in SmoothCalibrator.")

    def _process_opts_centric_mod(
        self,
        *,
        ipt_wgts: list[nn.Parameter],
        opt_wgts: list[nn.Parameter],
        ipt_mods: list[nn.Module],
        opt_mods: list[nn.Module],
        update_state_dict: bool = True,
        **kwargs,
    ) -> None:
        for w in ipt_wgts:
            if update_state_dict:
                self._state_dict.append((w, w.data))
            scale = self._reshape_scale(self.candidate, w, channels_dim=0, needs_reduction=False)
            w.data = w.detach().data.to(dtype=self.develop_dtype).mul(scale).to(dtype=w.dtype)
        for w in opt_wgts:
            if update_state_dict:
                self._state_dict.append((w, w.data))
            scale = self._reshape_scale(self.candidate, w, channels_dim=0, needs_reduction=True)
            w.data = w.detach().data.to(dtype=self.develop_dtype).div(scale).to(dtype=w.dtype)
        super()._process_opts_centric_mod(
            ipt_wgts=ipt_wgts,
            opt_wgts=opt_wgts,
            ipt_mods=ipt_mods,
            opt_mods=opt_mods,
            update_state_dict=False,
            **kwargs,
        )

    @staticmethod
    def _update_best(
        *,
        best_error: list[torch.Tensor],
        best_scale: torch.Tensor,
        error: list[torch.Tensor],
        scale: torch.Tensor,
        numel: int,
        num_channels: int,
        num_heads: int,
        num_head_repeats: int,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        if best_error is None:
            return error, scale
        elif numel == 1:  # tensor wise quantization error
            if all(e <= b for b, e in zip(best_error, error)):
                return error, scale
            return best_error, best_scale
        else:  # channel group wise quantization error
            assert num_channels % numel == 0
            group_size, num_groups = num_channels // numel, numel
            needs_reduction = num_heads > 1 and num_head_repeats > 1
            if needs_reduction:
                num_head_channels = num_channels // num_heads
                num_unique_heads = num_heads // num_head_repeats
                if num_head_channels >= group_size:
                    assert num_head_channels % group_size == 0
                    num_groups_per_head = num_head_channels // group_size
                    num_repeats = num_head_repeats
                    num_unqiue_heads_per_group = 1
                else:
                    assert group_size % num_head_channels == 0
                    num_heads_per_group = group_size // num_head_channels
                    if num_heads_per_group < num_head_repeats:
                        assert num_head_repeats % num_heads_per_group == 0
                        num_groups_per_head = 1
                        num_repeats = num_head_repeats // num_heads_per_group
                        num_unqiue_heads_per_group = 1
                    else:
                        assert num_heads_per_group % num_head_repeats == 0
                        num_groups_per_head = 1
                        num_repeats = 1
                        num_unqiue_heads_per_group = num_heads_per_group // num_head_repeats
                num_uniques = num_unique_heads // num_unqiue_heads_per_group
            needs_reduction = needs_reduction and num_repeats > 1

            pos = torch.full((numel,), True, device=error[0][0].device)
            for b, e in zip(best_error, error):
                if needs_reduction:
                    b = b.view(num_uniques, num_repeats, num_groups_per_head).sum(dim=1, keepdim=True)
                    e = e.view(num_uniques, num_repeats, num_groups_per_head).sum(dim=1, keepdim=True)
                    pos = pos & (e < b).expand(num_uniques, num_repeats, num_groups_per_head).reshape_as(pos)
                else:
                    pos = pos & (e < b)
            for b, e in zip(best_error, error):
                b[pos] = e[pos]
            pos = pos.view(num_groups, 1).expand(num_groups, group_size)
            best_scale = best_scale.view(num_groups, group_size)
            best_scale[pos] = scale.view(num_groups, group_size)[pos]
            return best_error, best_scale


class SmoothLinearCalibrator(SmoothCalibrator):
    """The smooth quantization calibrator for linear module."""

    def __init__(
        self,
        calib_config: QuantSmoothCalibConfig,
        wgts_quantizer: Quantizer | None,
        ipts_quantizer: Quantizer | None,
        num_heads: int = 1,
        num_head_repeats: int = 1,
        allow_kernel_calib: bool = True,
        develop_dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the calibrator.

        Args:
            calib_config (QuantSmoothCalibConfig): The calibration configuration.
            wgts_quantizer (KernelQuantizer): The weight quantizer.
            ipts_quantizer (KernelQuantizer): The input quantizer.
            num_heads (int): The number of heads. Defaults to ``1``.
            num_head_repeats (int): The number of head repeats. Defaults to ``1``.
            allow_kernel_calib (bool): Whether kernel calibration is allowed. Defaults to ``True``.
            develop_dtype (torch.dtype): The development data type. Defaults to ``torch.float32``.
        """
        super().__init__(
            tensor_type=QuantTensorType.Weights,
            calib_config=calib_config,
            wgts_quantizer=wgts_quantizer,
            ipts_quantizer=ipts_quantizer,
            opts_quantizer=None,
            num_heads=num_heads,
            num_head_repeats=num_head_repeats,
            allow_kernel_calib=allow_kernel_calib,
            develop_dtype=develop_dtype,
        )


class SmoothAttentionCalibrator(SmoothCalibrator):
    """The smooth quantization calibrator for attention module."""

    def __init__(
        self,
        calib_config: QuantSmoothCalibConfig,
        q_quantizer: Quantizer | None,
        k_quantizer: Quantizer | None,
        num_heads: int = 1,
        num_head_repeats: int = 1,
        with_rope: bool = True,
        develop_dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the calibrator.

        Args:
            calib_config (QuantSmoothCalibConfig): The calibration configuration.
            q_quantizer (Quantizer): The query quantizer.
            k_quantizer (Quantizer): The key quantizer.
            num_heads (int): The number of heads. Defaults to ``1``.
            num_head_repeats (int): The number of head repeats. Defaults to ``1``.
            with_rope (bool): Whether rotary position embedding is used. Defaults to ``False``.
            develop_dtype (torch.dtype): The development data type. Defaults to ``torch.float32``.
        """
        super().__init__(
            tensor_type=QuantTensorType.Outputs,
            calib_config=calib_config,
            wgts_quantizer=None,
            ipts_quantizer=q_quantizer,
            opts_quantizer=k_quantizer,
            num_heads=num_heads,
            num_head_repeats=num_head_repeats,
            allow_kernel_calib=False,
            with_rope=with_rope,
            develop_dtype=develop_dtype,
        )

    def calibrate(
        self,
        q_wgt: nn.Parameter,
        k_wgt: nn.Parameter,
        qs: ActivationsCache,
        ks: ActivationsCache,
        q_mod: nn.Module,
        k_mod: nn.Module,
        eval_mod: nn.Module | None = None,
        eval_ipt: ActivationsCache | None = None,
        eval_kwargs: dict[str, tp.Any] | None = None,
    ) -> tp.Any:
        """Calibrate the quantization for attention.

        Args:
            q_wgt (nn.Parameter): Query projection weight.
            k_wgt (nn.Parameter): Key projection weight.
            qs (ActivationsCache): Query activations.
            ks (ActivationsCache): Key activations.
            q_mod (nn.Module): Query module (either proj_q for pre-rope or q_rotary_emb for post-rope)
            k_mod (nn.Module): Key module (either proj_k for pre-rope or k_rotary_emb for post-rope)
            eval_mod (nn.Module): Evaluation module.
            eval_ipt (ActivationsCache): Evaluation inputs.
            eval_kwargs (dict[str, tp.Any]): Evaluation keyword arguments.

        Returns:
            tp.Any: The evaluation result.
        """
        return super().calibrate(
            ipt_wgts=[q_wgt],
            opt_wgts=[k_wgt],
            ipts=qs,
            opts=ks,
            ipt_mods=[q_mod],
            opt_mods=[k_mod],
            eval_mod=eval_mod,
            eval_ipt=eval_ipt,
            eval_kwargs=eval_kwargs,
        )
