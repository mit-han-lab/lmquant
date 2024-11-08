# -*- coding: utf-8 -*-
"""Smooth quantization module."""

import gc
import typing as tp
from dataclasses import _MISSING_TYPE, MISSING, dataclass

import torch
import torch.nn as nn

from ..data.cache import TensorsCache
from ..data.common import TensorType
from ..quantizer.processor import Quantizer
from ..utils import math, tools
from ..utils.common import split_sequence
from ..utils.hooks import BaseInputPackager, BaseOutputPackager, BaseTensorProcessor
from .config import SearchBasedCalibObjective, SmoothCalibConfig, SmoothSpanMode
from .metric import ChannelMetric
from .search import SearchBasedCalibrator

__all__ = [
    "smooth_linear_modules",
    "smooth_attention",
    "convert_smooth_upscale_to_downscale",
    "ActivationSmoother",
    "get_smooth_scale",
    "get_smooth_span",
    "SmoothCalibrator",
    "SmoothLinearCalibrator",
    "SmoothAttentionCalibrator",
]


@dataclass
class ActivationSmoother(BaseTensorProcessor):
    """The quantization smoothing processor."""

    smooth_scale: torch.Tensor
    channels_dim: int
    upscale: bool = False
    develop_dtype: torch.dtype | None = None
    # region hook-related attributes
    input_packager: BaseInputPackager | None = None
    output_packager: BaseOutputPackager | None = None
    # endregion

    def is_enabled(self) -> bool:
        return self.smooth_scale is not None

    def get_input_packager(self) -> BaseInputPackager | None:
        return self.input_packager

    def get_output_packager(self) -> BaseOutputPackager | None:
        return self.output_packager

    def process(self, tensor: torch.Tensor) -> torch.Tensor:
        """Process the tensor.

        Args:
            tensor (`torch.Tensor`):
                The tensor to smooth.

        Returns:
            `torch.Tensor`:
                The smoothed tensor.
        """
        device, dtype = tensor.device, tensor.dtype
        if self.develop_dtype is None:
            self.develop_dtype = dtype
        self.smooth_scale = self.smooth_scale.to(device=device, dtype=self.develop_dtype)
        tensor = tensor.to(dtype=self.develop_dtype)
        smooth_scale_view_shape = [1] * tensor.ndim
        smooth_scale_view_shape[self.channels_dim] = -1
        smooth_scale = self.smooth_scale.view(smooth_scale_view_shape)
        if self.upscale:
            return tensor.mul(smooth_scale).to(dtype=dtype)
        else:
            return tensor.div(smooth_scale).to(dtype=dtype)


@torch.inference_mode()
def get_smooth_span(
    tensors: tp.Sequence[torch.Tensor],
    /,
    *,
    group_shape: tp.Sequence[int],
    span_mode: SmoothSpanMode,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Calculate the value span of tensors for calculating smoothing scale.

    Args:
        tensors (`Sequence[torch.Tensor]`):
            Tensors to calculate the span.
        group_shape (`Sequence[int]`):
            Quantization group shape.
        span_mode (`SmoothSpanMode`):
            The quantization smoothing span mode.
        device (`torch.device` or `str` or `None`, *optional*, defaults to `None`):
            Device to store the span.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Data type of the span.

    Returns:
        `torch.Tensor`:
            The span of the tensors for calculating smoothing scale.
    """
    # convert span mode name from camel case to snake case
    range_name = "".join(["_" + c.lower() if c.isupper() else c for c in span_mode.name]).lstrip("_")
    range_fn = getattr(ChannelMetric, range_name)
    r: torch.Tensor = range_fn(tensors, tensors[0].shape[1], group_shape, device=device, dtype=dtype)
    return r


@torch.inference_mode()
def get_smooth_scale(*, alpha_base: torch.Tensor, beta_base: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """Calculate the smoothing scale for quantization. Scale = alpha_base^alpha / beta_base^beta.

    Args:
        alpha_base (`torch.Tensor`):
            Base span for alpha.
        beta_base (`torch.Tensor`):
            Base span for beta.
        alpha (`float`):
            Alpha.
        beta (`float`):
            Beta.

    Returns:
        `torch.Tensor`:
            Smoothing scale.
    """
    assert 0 <= alpha <= 1 and 0 <= beta <= 1, "The smooth factors should be in [0, 1]."
    if alpha > 0:
        smooth_scale = alpha_base.pow(alpha)
        if beta > 0:
            smooth_scale = smooth_scale.div_(beta_base.pow(beta))
    else:
        smooth_scale = beta_base.pow(-beta)
    smooth_scale[smooth_scale == 0] = 1
    assert not smooth_scale.isnan().any(), "The smooth scale contains NaN."
    assert not smooth_scale.isinf().any(), "The smooth scale contains Inf."
    return smooth_scale


class SmoothCalibrator(SearchBasedCalibrator[SmoothCalibConfig, torch.Tensor]):
    """The quantization smoothing calibrator."""

    def __init__(
        self,
        tensor_type: TensorType,
        config: SmoothCalibConfig,
        w_quantizer: Quantizer | None,
        x_quantizer: Quantizer | None,
        y_quantizer: Quantizer | None,
        num_heads: int = 1,
        num_head_repeats: int = 1,
        with_rope: bool = False,
        develop_dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the calibrator.

        Args:
            tensor_type (`TensorType`):
                The type of tensor to quantize. Choices are ``Weights`` and ``Outputs``.
            config (`SmoothCalibConfig`):
                The quantization smoothing calibration configuration.
            w_quantizer (`Quantizer` or `None`):
                The w quantizer for x-w computation.
            x_quantizer (`Quantizer` or `None`):
                The x quantizer for x-w or y-x computation.
            y_quantizer (`Quantizer` or `None`):
                The y quantizer for y-x computation.
            num_heads (`int`, *optional*, defaults to ``1``):
                The number of heads.
            num_head_repeats (`int`, *optional*, defaults to ``1``):
                The number of head repeats.
            with_rope (`bool`, *optional*, defaults to ``False``):
                Whether rotary position embedding is used for y-x computation.
            develop_dtype (torch.dtype, *optional*, defaults to ``torch.float32``):
                The development data type.
        """
        assert tensor_type in (TensorType.Weights, TensorType.Outputs)
        super().__init__(
            tensor_type=tensor_type,
            config=config,
            w_quantizer=w_quantizer,
            x_quantizer=x_quantizer,
            y_quantizer=y_quantizer,
            develop_dtype=develop_dtype,
        )
        self.num_heads = num_heads
        self.num_head_repeats = num_head_repeats
        self.with_rope = self.tensor_type != TensorType.Weights and with_rope
        # region set group shapes of weights, inputs and outputs
        if self.needs_w_quant:
            w_group_shape = list(self.w_quantizer.config.largest_group_shape)
        else:
            w_group_shape = [1, None, -1]
        if self.needs_x_quant:
            x_group_shape = list(self.x_quantizer.config.largest_group_shape)
        else:
            x_group_shape = [1, None, -1]
        if self.needs_y_quant:
            y_group_shape = list(self.y_quantizer.config.largest_group_shape)
        else:
            y_group_shape = [1, None, -1]
        w_group_shape[1] = x_group_shape[1] if w_group_shape[1] is None else w_group_shape[1]
        if self.tensor_type == TensorType.Weights:
            x_group_shape[1] = w_group_shape[1] if x_group_shape[1] is None else x_group_shape[1]
        else:
            x_group_shape[1] = y_group_shape[1] if x_group_shape[1] is None else x_group_shape[1]
        y_group_shape[1] = x_group_shape[1] if y_group_shape[1] is None else y_group_shape[1]
        self.w_group_shape, self.x_group_shape, self.y_group_shape = w_group_shape, x_group_shape, y_group_shape
        # endregion
        self.alpha_beta_pairs = self.config.get_alpha_beta_pairs()
        self.num_iters = 1

    @property
    def population_size(self) -> int:
        """Get the population size."""
        return len(self.alpha_beta_pairs) * len(self.span_mode_pairs)

    @property
    def allows_x_quant_for_wgts(self) -> bool:
        """Whether the calibrator allows input quantization when tensor_type is Weights."""
        return self.config.allow_a_quant

    @property
    def allows_w_quant_for_wgts(self) -> bool:
        """Whether the calibrator needs weight quantization when tensor_type is Weights."""
        return self.config.allow_b_quant

    @property
    def allows_w_quant_for_ipts(self) -> bool:
        """Whether the calibrator allows weight quantization when tensor_type is Inputs."""
        return self.config.allow_b_quant

    @property
    def allows_x_quant_for_opts(self) -> bool:
        """Whether the calibrator allows x quantization when tensor_type is Outputs."""
        return self.config.allow_b_quant

    @property
    def allows_y_quant_for_opts(self) -> bool:
        """Whether the calibrator allows y quantization when tensor_type is Outputs."""
        return self.config.allow_a_quant

    @property
    def allows_w_quant_for_opts(self) -> bool:
        """Whether the calibrator allows weight quantization when tensor_type is Outputs."""
        return False

    @property
    def span_mode_pairs(self) -> list[tuple[SmoothSpanMode, SmoothSpanMode]]:
        """Get the span modes."""
        return self.config.spans

    @property
    def alpha_span_modes(self) -> list[SmoothSpanMode]:
        """Get the span modes for alpha."""
        return self.config.a_spans

    @property
    def beta_span_modes(self) -> list[SmoothSpanMode]:
        """Get the span modes for beta."""
        return self.config.b_spans

    def _reset(  # noqa: C901
        self,
        *,
        x_wgts: list[torch.Tensor | nn.Parameter],
        x_acts: TensorsCache,
        y_wgts: list[torch.Tensor | nn.Parameter] = None,
        y_acts: TensorsCache | None = None,
        **kwargs,
    ) -> None:
        """Reset the calibrator.

        Args:
            x_wgts (`list[torch.Tensor | nn.Parameter]`):
                The weights in x-w computation, or weights that generates x for y-x computation.
            x_acts (`TensorsCache`):
                The x activations. It should be x for x-w or y-x computation.
            y_wgts (`list[torch.Tensor | nn.Parameter]` or `None`, *optional*, defaults to `None`):
                The weights that generates y for y-x computation.
            y_acts (`TensorsCache` or `None`, *optional*, defaults to `None`):
                The y activations. It should be y for y-x computation.
        """
        wgts_centric = self.tensor_type == TensorType.Weights
        self.num_in_channels = x_wgts[0].shape[1] if wgts_centric else x_wgts[0].shape[0]
        device = x_wgts[0].device
        if self.num_heads > 1 and self.num_head_repeats > 1:
            self.num_unique_heads = self.num_heads // self.num_head_repeats
        else:
            self.num_unique_heads = 0
        # region get x spans
        assert (
            x_acts.num_tensors == 1
        ), f"Only one input is allowed, got {x_acts.num_tensors}=len({list(x_acts.keys())})"
        x_tensors = x_acts.front().get_standardized_data(reshape=False)
        assert all(x.shape[1] == self.num_in_channels for x in x_tensors)
        x_spans = {}
        for span_mode in self.alpha_span_modes if wgts_centric else self.beta_span_modes:
            x_span = get_smooth_span(
                x_tensors,
                group_shape=self.x_group_shape,
                span_mode=span_mode,
                device=device,
                dtype=self.develop_dtype,
            )
            if self.num_unique_heads > 0:
                x_span = x_span.view(self.num_unique_heads, self.num_head_repeats, -1)
                x_span = (x_span.amax if "Max" in span_mode.name else x_span.mean)(dim=1, keepdim=True)
                x_span = x_span.expand(self.num_unique_heads, self.num_head_repeats, -1).reshape(-1)
            if self.tensor_type == TensorType.Outputs and self.with_rope:
                x_span = x_span.view(self.num_heads, 2, -1)
                x_span = (x_span.amax if "Max" in span_mode.name else x_span.mean)(dim=1, keepdim=True)
                x_span = x_span.expand(self.num_heads, 2, -1).reshape(-1)
            x_spans[span_mode] = x_span
            if self.logger.level <= tools.logging.DEBUG:
                self.logger.debug("+ x - %s", span_mode.name)
                self.logger.debug("+ x  = [min=%.4f, max=%.4f]", x_span.min().item(), x_span.max().item())
        del x_tensors
        # endregion
        if wgts_centric:
            assert all(w.shape[1] == self.num_in_channels for w in x_wgts)
            w_tensors = [w.data for w in x_wgts]
            w_spans = {}
            for span_mode in self.beta_span_modes:
                w_span = get_smooth_span(
                    w_tensors,
                    group_shape=self.w_group_shape,
                    span_mode=span_mode,
                    dtype=self.develop_dtype,
                )
                if self.num_unique_heads > 0:
                    w_span = w_span.view(self.num_unique_heads, self.num_head_repeats, -1)
                    w_span = (w_span.amax if "Max" in span_mode.name else w_span.mean)(dim=1, keepdim=True)
                    w_span = w_span.expand(self.num_unique_heads, self.num_head_repeats, -1).reshape(-1)
                w_spans[span_mode] = w_span
                if self.logger.level <= tools.logging.DEBUG:
                    self.logger.debug("+ w - %s", span_mode.name)
                    self.logger.debug("+ w  = [min=%.4f, max=%.4f]", w_span.min().item(), w_span.max().item())
            self.span_pairs: list[tuple[torch.Tensor, torch.Tensor]] = [
                (x_spans[x_span_mode], w_spans[w_span_mode]) for x_span_mode, w_span_mode in self.span_mode_pairs
            ]
        else:
            assert y_acts.num_tensors == 1, f"Only one output source is allowed, got {y_acts.num_tensors}"
            if self.num_unique_heads > 0:
                num_out_channels = self.num_in_channels // self.num_head_repeats
            else:
                num_out_channels = self.num_in_channels
            assert all(w.shape[0] == self.num_in_channels for w in x_wgts)
            assert all(w.shape[0] == num_out_channels for w in y_wgts)
            y_tensors = y_acts.front().get_standardized_data(reshape=False)
            assert all(y.shape[1] == num_out_channels for y in y_tensors)
            y_spans = {}
            for span_mode in self.alpha_span_modes:
                y_span = get_smooth_span(
                    y_tensors,
                    group_shape=self.x_group_shape,
                    span_mode=span_mode,
                    device=device,
                    dtype=self.develop_dtype,
                )
                if self.num_unique_heads > 0:
                    y_span = y_span.view(self.num_unique_heads, 1, -1)
                    y_span = y_span.expand(self.num_unique_heads, self.num_head_repeats, -1).reshape(-1)
                if self.tensor_type == TensorType.Outputs and self.with_rope:
                    y_span = y_span.view(self.num_heads, 2, -1)
                    y_span = (y_span.amax if "Max" in span_mode.name else y_span.mean)(dim=1, keepdim=True)
                    y_span = y_span.expand(self.num_heads, 2, -1).reshape(-1)
                y_spans[span_mode] = y_span
                if self.logger.level <= tools.logging.DEBUG:
                    self.logger.debug("+ y - %s", span_mode.name)
                    self.logger.debug("+ y  = [min=%.4f, max=%.4f]", y_span.min().item(), y_span.max().item())
            self.span_pairs: list[tuple[torch.Tensor, torch.Tensor]] = [
                (y_spans[y_span_mode], x_spans[x_span_mode]) for y_span_mode, x_span_mode in self.span_mode_pairs
            ]
        self.best_error: list[torch.Tensor] = None
        self.best_scale: torch.Tensor = None
        self.error_history: list[tuple[float, float]] = []

    def _split_candidate_id(self, candidate_id: int) -> tuple[int, int]:
        """Split the candidate id into alpha_beta id and span_pair id.

        Args:
            candidate_id (`int`):
                The candidate id.

        Returns:
            `tuple[int, int]`:
                The alpha_beta id and span_mode id.
        """
        alpha_beta_id = candidate_id % len(self.alpha_beta_pairs)
        span_pair_id = candidate_id // len(self.alpha_beta_pairs)
        return alpha_beta_id, span_pair_id

    def get_best(self) -> torch.Tensor:
        """Get the best candidate.

        Returns:
            `torch.Tensor`:
                The best candidate.
        """
        return self.best_scale

    def _ask(self) -> torch.Tensor:
        """Ask for the next candidate.

        Returns:
            `torch.Tensor`:
                The next candidate.
        """
        alpha_beta_id, span_pair_id = self._split_candidate_id(self.candidate_id)
        alpha, beta = self.alpha_beta_pairs[alpha_beta_id]
        a_span, b_span = self.span_pairs[span_pair_id]
        if alpha == 0 and beta == 0:
            scale = torch.ones_like(a_span, dtype=self.develop_dtype)
        else:
            scale = get_smooth_scale(alpha_base=a_span, beta_base=b_span, alpha=alpha, beta=beta)
        return scale

    def _tell(self, error: list[torch.Tensor]) -> None:  # noqa: C901
        """Tell the error of the last candidate and update the best candidate.

        Args:
            error (`list[torch.Tensor]`):
                The error of the last candidate.
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
            num_channels=self.num_in_channels,
            num_heads=self.num_heads,
            num_head_repeats=self.num_head_repeats,
        )
        if self.logger.level <= tools.logging.DEBUG:
            self.error_history.append(
                (
                    sum(math.root_(e.to(torch.float64).sum(), self.config.degree).item() for e in error),
                    sum(math.root_(b.to(torch.float64).sum(), self.config.degree).item() for b in self.best_error),
                )
            )
            if self.is_last_candidate_in_iter():
                logs: list[list[list[tuple]]] = [[] for _ in range(len(self.span_mode_pairs))]
                for i in range(self.population_size):
                    c, r = self._split_candidate_id(i)
                    alpha, beta = self.alpha_beta_pairs[c]
                    if c % 5 == 0:
                        logs[r].append([])
                    logs[r][-1].append((alpha, beta, self.error_history[i][0], self.error_history[i][1]))
                for r in range(len(self.span_mode_pairs)):
                    self.logger.debug(
                        "  - x / w range = %s / %s", self.span_mode_pairs[r][0].name, self.span_mode_pairs[r][1].name
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
                        sum(math.root_(b.to(torch.float64).sum(), self.config.degree).item() for b in self.best_error),
                    )
                    self.logger.debug("  + scale = [min=%.4f, max=%.4f]", scale.min().item(), scale.max().item())
                    tools.logging.Formatter.indent_inc()

    def _reshape_scale(
        self, scale: torch.Tensor, tensor: torch.Tensor, channels_dim: int, needs_reduction: bool = False
    ) -> torch.Tensor:
        if self.num_unique_heads > 0 and needs_reduction:
            scale = scale.view(self.num_unique_heads, self.num_head_repeats, -1)[:, 0, :].reshape(-1)
        shape = [1] * tensor.ndim
        shape[channels_dim] = -1
        return scale.view(shape)

    def _process_x_in_xw(self, x: torch.Tensor, channels_dim: int | _MISSING_TYPE = MISSING) -> torch.Tensor:
        if not self.needs_x_quant_for_wgts:
            return x
        if channels_dim is MISSING:
            channels_dim = self.x_quantizer.channels_dim
        shape, dtype = x.shape, x.dtype
        scale = self._reshape_scale(self.candidate, x, channels_dim)
        x = x.to(dtype=self.develop_dtype) if dtype != self.develop_dtype else x.clone()
        x = x.div_(scale)
        x = self.x_quantizer.quantize(
            x, channels_dim=channels_dim, default_dtype=dtype, develop_dtype=self.develop_dtype
        ).data
        x = x.mul_(scale).to(dtype=dtype)
        return x.view(shape)

    def _process_w_in_xw(self, w: torch.Tensor) -> torch.Tensor:
        if not self.needs_w_quant_for_wgts:
            return w
        dtype = w.dtype
        channels_dim = 1 if self.w_quantizer.channels_dim is None else self.w_quantizer.channels_dim
        scale = self._reshape_scale(self.candidate, w, channels_dim=channels_dim)
        w = w.to(dtype=self.develop_dtype) if dtype != self.develop_dtype else w.clone()
        w = self.w_quantizer.quantize(
            w.mul_(scale), kernel=None, default_dtype=dtype, develop_dtype=self.develop_dtype
        ).data
        w = w.div_(scale).to(dtype=dtype)
        return w

    def _process_x_in_yx(self, x: torch.Tensor, channels_dim: int | _MISSING_TYPE = MISSING) -> torch.Tensor:
        if not self.needs_x_quant_for_opts:
            return x
        shape, dtype = x.shape, x.dtype
        if self.objective != SearchBasedCalibObjective.OutputsError:
            if channels_dim is MISSING:
                channels_dim = self.x_quantizer.channels_dim
            scale = self._reshape_scale(self.candidate, x, channels_dim, needs_reduction=False)
            x = x.to(dtype=self.develop_dtype) if dtype != self.develop_dtype else x.clone()
            x = x.mul_(scale)
        # ! `x` is already scaled during `_process_opts_centric_mod` by scaling `xw`
        x = self.x_quantizer.quantize(
            x,
            channels_dim=channels_dim,
            default_dtype=dtype,
            develop_dtype=self.develop_dtype,
        ).data
        if self.objective != SearchBasedCalibObjective.OutputsError:
            x = x.div_(scale).to(dtype=dtype)
        return x.view(shape)

    def _process_y_in_yx(self, y: torch.Tensor, channels_dim: int | _MISSING_TYPE = MISSING) -> torch.Tensor:
        if not self.needs_y_quant_for_opts:
            return y
        shape, dtype = y.shape, y.dtype
        if self.objective != SearchBasedCalibObjective.OutputsError:
            if channels_dim is MISSING:
                channels_dim = self.x_quantizer.channels_dim
            scale = self._reshape_scale(self.candidate, y, channels_dim, needs_reduction=True)
            y = y.to(dtype=self.develop_dtype) if dtype != self.develop_dtype else y.clone()
            y = y.div_(scale)
        # ! `y` is already scaled during `_process_opts_centric_mod` by scaling `yw`
        y = self.y_quantizer.quantize(
            y,
            channels_dim=channels_dim,
            default_dtype=dtype,
            develop_dtype=self.develop_dtype,
        ).data
        if self.objective != SearchBasedCalibObjective.OutputsError:
            y = y.mul_(scale).to(dtype=dtype)
        return y.view(shape)

    def _process_xw_in_yx(self, w: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("The method `_process_xw_in_yx` should not be called in SmoothCalibrator.")

    def _process_yw_in_yx(self, w: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("The method `_process_yw_in_yx` should not be called in SmoothCalibrator.")

    def _process_wgts_centric_mod(
        self,
        wgts: list[nn.Parameter],
        mods: list[nn.Module],
        update_state_dict: bool = True,
        splits: list[int] | None = None,
        **kwargs,
    ) -> None:
        if self.needs_w_quant_for_wgts and self.config.allow_low_rank and self.w_quantizer.is_enabled_low_rank():
            assert len(wgts) == len(mods)
            for wgt in wgts:
                if update_state_dict:
                    self._state_dict.append((wgt, wgt.data))
                dtype = wgt.dtype
                scale = self._reshape_scale(self.candidate, wgt.data, channels_dim=1)
                wgt.data = wgt.data.to(dtype=self.develop_dtype).mul(scale).to(dtype=dtype)
            input_packager = self.x_quantizer.get_input_packager() if self.needs_x_quant else None
            for mod in mods:
                self._hooks.append(
                    ActivationSmoother(
                        self.candidate,
                        self.x_quantizer.channels_dim,
                        develop_dtype=self.develop_dtype,
                        input_packager=input_packager,
                    )
                    .as_hook()
                    .register(mod)
                )
            if splits:
                wgts_splits: list[list[nn.Parameter]] = split_sequence(wgts, splits)
                mods_splits: list[list : nn.Module] = split_sequence(mods, splits)
            else:
                wgts_splits, mods_splits = [wgts], [mods]
            for wgts_split, mods_split in zip(wgts_splits, mods_splits, strict=True):
                for qwgt, lowr, wgt, mod in zip(
                    *self.w_quantizer.quantize_with_low_rank(wgts_split, kernel=None, develop_dtype=self.develop_dtype),
                    wgts_split,
                    mods_split,
                    strict=True,
                ):
                    wgt.data = qwgt.data
                    self._hooks.append(lowr.as_hook(input_packager=input_packager).register(mod))
                    if self.needs_x_quant_for_wgts:
                        self._hooks.append(self.x_quantizer.as_hook().register(mod))
        else:
            super()._process_wgts_centric_mod(wgts=wgts, mods=mods, update_state_dict=update_state_dict, **kwargs)

    def _process_opts_centric_mod(
        self,
        x_wgts: list[nn.Parameter],
        y_wgts: list[nn.Parameter],
        x_mods: list[nn.Module],
        y_mods: list[nn.Module],
        update_state_dict: bool = True,
        **kwargs,
    ) -> None:
        for w in x_wgts:
            if update_state_dict:
                self._state_dict.append((w, w.data))
            scale = self._reshape_scale(self.candidate, w, channels_dim=0, needs_reduction=False)
            w.data = w.detach().data.to(dtype=self.develop_dtype).mul(scale).to(dtype=w.dtype)
        for w in y_wgts:
            if update_state_dict:
                self._state_dict.append((w, w.data))
            scale = self._reshape_scale(self.candidate, w, channels_dim=0, needs_reduction=True)
            w.data = w.detach().data.to(dtype=self.develop_dtype).div(scale).to(dtype=w.dtype)
        super()._process_opts_centric_mod(
            x_wgts=x_wgts,
            y_wgts=y_wgts,
            x_mods=x_mods,
            y_mods=y_mods,
            update_state_dict=False,
            **kwargs,
        )

    @staticmethod
    def _update_best(
        *,
        best_error: list[torch.Tensor] | None,
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
            if all(e <= b for b, e in zip(best_error, error, strict=True)):
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
            for b, e in zip(best_error, error, strict=True):
                if needs_reduction:
                    b = b.view(num_uniques, num_repeats, num_groups_per_head).sum(dim=1, keepdim=True)
                    e = e.view(num_uniques, num_repeats, num_groups_per_head).sum(dim=1, keepdim=True)
                    pos = pos & (e < b).expand(num_uniques, num_repeats, num_groups_per_head).reshape_as(pos)
                else:
                    pos = pos & (e < b)
            for b, e in zip(best_error, error, strict=True):
                b[pos] = e[pos]
            pos = pos.view(num_groups, 1).expand(num_groups, group_size)
            best_scale = best_scale.view(num_groups, group_size)
            best_scale[pos] = scale.view(num_groups, group_size)[pos]
            return best_error, best_scale


class SmoothLinearCalibrator(SmoothCalibrator):
    """The smooth quantization calibrator for linear module."""

    def __init__(
        self,
        config: SmoothCalibConfig,
        weight_quantizer: Quantizer | None,
        input_quantizer: Quantizer | None,
        num_heads: int = 1,
        num_head_repeats: int = 1,
        develop_dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the calibrator.

        Args:
            config (`SmoothCalibConfig`):
                The quantization smoothing calibration configuration.
            weight_quantizer (`Quantizer` or `None`):
                The weight quantizer.
            input_quantizer (`Quantizer` or `None`):
                The input quantizer.
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
            num_heads=num_heads,
            num_head_repeats=num_head_repeats,
            develop_dtype=develop_dtype,
        )


class SmoothAttentionCalibrator(SmoothCalibrator):
    """The smooth quantization calibrator for attention module."""

    def __init__(
        self,
        config: SmoothCalibConfig,
        query_quantizer: Quantizer | None,
        key_quantizer: Quantizer | None,
        num_heads: int = 1,
        num_head_repeats: int = 1,
        with_rope: bool = True,
        develop_dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the calibrator.

        Args:
            config (`SmoothCalibConfig`):
                The quantization smoothing calibration configuration.
            query_quantizer (`Quantizer` or `None`):
                The query quantizer.
            key_quantizer (`Quantizer` or `None`):
                The key quantizer.
            num_heads (`int`, *optional*, defaults to `1`):
                The number of heads.
            num_head_repeats (`int`, *optional*, defaults to `1`):
                The number of head repeats.
            with_rope (`bool`, *optional*, defaults to `True`):
                Whether rotary position embedding is used.
            develop_dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                The development data type.
        """
        super().__init__(
            tensor_type=TensorType.Outputs,
            config=config,
            w_quantizer=None,
            x_quantizer=query_quantizer,
            y_quantizer=key_quantizer,
            num_heads=num_heads,
            num_head_repeats=num_head_repeats,
            with_rope=with_rope,
            develop_dtype=develop_dtype,
        )

    def calibrate(
        self,
        q_proj_weight: nn.Parameter,
        k_proj_weight: nn.Parameter,
        queries: TensorsCache,
        keys: TensorsCache,
        query_module: nn.Module,
        key_module: nn.Module,
        eval_module: nn.Module | None = None,
        eval_inputs: TensorsCache | None = None,
        eval_kwargs: dict[str, tp.Any] | None = None,
    ) -> tp.Any:
        """Calibrate the quantization for attention.

        Args:
            q_proj_weight (`nn.Parameter`):
                The query projection weight.
            k_proj_weight (`nn.Parameter`):
                The key projection weight.
            queries (`TensorsCache`):
                The query activations.
            keys (`TensorsCache`):
                The key activations.
            query_module (`nn.Module`):
                The module that generates the query activations,
                e.g., either `q_proj` for pre-rope or `q_rotary_emb` for post-rope.
            key_module (`nn.Module`):
                The module that generates the key activations,
                e.g., either `k_proj` for pre-rope or `k_rotary_emb` for post-rope.
            eval_module (`nn.Module`, *optional*):
                The evaluation module.
            eval_inputs (`TensorsCache`, *optional*):
                The evaluation inputs.
            eval_kwargs (`dict[str, tp.Any]`, *optional*):
                The evaluation keyword arguments.

        Returns:
            tp.Any: The evaluation result.
        """
        return super().calibrate(
            x_wgts=[q_proj_weight],
            y_wgts=[k_proj_weight],
            x_acts=queries,
            y_acts=keys,
            x_mods=[query_module],
            y_mods=[key_module],
            eval_module=eval_module,
            eval_inputs=eval_inputs,
            eval_kwargs=eval_kwargs,
        )


def smooth_upscale_param(param: nn.Parameter, scale: torch.Tensor, channels_dim: int = 1) -> None:
    """In-place smooth the parameter by upscaling.

    Args:
        param (`nn.Parameter`):
            The parameter to smooth.
        scale (`torch.Tensor`):
            The scale to upscale.
        channels_dim (`int`, *optional*, defaults to `1`):
            The dimension of channels.
    """
    dtype = param.dtype
    view_shape = [1] * param.ndim
    view_shape[channels_dim] = -1
    scale = scale.to(device=param.device).view(view_shape)
    param.data = param.data.to(dtype=scale.dtype).mul_(scale).to(dtype=dtype)
    assert not param.data.isnan().any(), "NaN found in param when smoothing"
    assert not param.data.isinf().any(), "Inf found in param when smoothing"


def smooth_downscale_param(param: nn.Parameter, scale: torch.Tensor, channels_dim: int = 0) -> None:
    """In-place smooth the parameter by downscaling.

    Args:
        param (`nn.Parameter`):
            The parameter to smooth.
        scale (`torch.Tensor`):
            The scale to downscale.
        channels_dim (`int`, *optional*, defaults to `0`):
            The dimension of channels.
    """
    dtype = param.dtype
    view_shape = [1] * param.ndim
    view_shape[channels_dim] = -1
    scale = scale.to(device=param.device).view(view_shape)
    param_data = param.data.to(dtype=scale.dtype)
    param_data.narrow(channels_dim, 0, scale.numel()).div_(scale)
    param.data = param_data.to(dtype=dtype)
    assert not param.data.isnan().any(), "NaN found in param when smoothing"
    assert not param.data.isinf().any(), "Inf found in param when smoothing"


def convert_smooth_upscale_to_downscale(
    scale: torch.Tensor, num_heads: int = 1, num_head_repeats: int = 1
) -> torch.Tensor:
    """Convert the upscale smooth scale to downscale smooth scale.

    Args:
        scale (`torch.Tensor`):
            The upscale smooth scale.
        num_heads (`int`, *optional*, defaults to `1`):
            The number of heads.
        num_head_repeats (`int`, *optional*, defaults to `1`):
            The number of head repeats.

    Returns:
        `torch.Tensor`:
            The downscale smooth scale.
    """
    if num_heads > 1 and num_head_repeats > 1:
        head_channels = scale.numel() // num_heads
        num_unique_heads = num_heads // num_head_repeats
        return scale.view(num_unique_heads, num_head_repeats, head_channels)[:, 0, :].reshape(-1)
    else:
        return scale


@torch.inference_mode()
def smooth_linear_modules(
    prevs: nn.Module | tp.Sequence[nn.Module] | None,
    modules: tp.Sequence[nn.Linear] | nn.Linear,
    *,
    scale: torch.Tensor | None,
    config: SmoothCalibConfig | None = None,
    weight_quantizer: Quantizer | None = None,
    input_quantizer: Quantizer | None = None,
    weights: list[nn.Parameter] | None = None,
    inputs: TensorsCache | None = None,
    eval_inputs: TensorsCache | None = None,
    eval_module: nn.Module = None,
    eval_kwargs: dict[str, tp.Any] = None,
    num_heads: int = 1,
    num_head_repeats: int = 1,
    splits: list[int] | None = None,
    extra_modules: list[nn.Linear] | None = None,
    develop_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Smooth two consecutive modules.

    Args:
        prevs (`nn.Module` or `list[nn.Module]`):
            The first module(s).
        modules (`nn.Linear` or `list[nn.Linear]`):
            The second module(s).
        scale (`torch.Tensor` or `None`, *optional*, defaults to `None`):
            The smooth quantization scale.
        config (`SmoothCalibConfig` or `None`, *optional*, defaults to `None`):
            The smooth quantization configuration.
        weight_quantizer (`Quantizer` or `None`, *optional*, defaults to `None`):
            The quantizer for weights.
        input_quantizer (`Quantizer` or `None`, *optional*, defaults to `None`):
            The quantizer for inputs.
        weights (`list[nn.Parameter]` or `None`, *optional*, defaults to `None`):
            The weights of the modules. If `None`, the weights of the modules will be used.
        inputs (`TensorsCache` or `None`, *optional*, defaults to `None`):
            The cache of the input activations.
        eval_inputs (`TensorsCache` or `None`, *optional*, defaults to `None`):
            The cache of the inputs corresponding to the `eval_module`.
        eval_module (`nn.Module`, *optional*, defaults to `None`):
            The module to evaluate the quantization error.
        eval_kwargs (`dict[str, tp.Any]`, *optional*, defaults to `None`):
            The keyword arguments for evaluation.
        num_heads (`int`, *optional*, defaults to `1`):
            The number of heads.
        num_head_repeats (`int`, *optional*, defaults to `1`):
            The number of head repeats.
        extra_modules (`list[nn.Module]` or `None`, *optional*, defaults to `None`):
            Extra modules to smooth.
        develop_dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            The development data type.

    Returns:
        `torch.Tensor`:
            The smooth quantization scale in CPU.
    """
    if not isinstance(modules, (list, tuple)):
        modules = [modules]
    extra_modules = [] if extra_modules is None else extra_modules
    if scale is None:
        assert inputs is not None or eval_inputs is not None, "inputs or eval_inputs must be provided"
        scale = SmoothLinearCalibrator(
            config=config,
            weight_quantizer=weight_quantizer,
            input_quantizer=input_quantizer,
            num_heads=num_heads,
            num_head_repeats=num_head_repeats,
            develop_dtype=develop_dtype,
        ).calibrate(
            x_wgts=[module.weight for module in modules] if weights is None else weights,
            x_acts=inputs,
            x_mods=modules,
            eval_inputs=eval_inputs,
            eval_module=eval_module,
            eval_kwargs=eval_kwargs,
            splits=splits,
        )
        gc.collect()
        torch.cuda.empty_cache()
    upscale = scale
    for module in modules + extra_modules:
        upscale = upscale.to(device=module.weight.device)
        smooth_upscale_param(module.weight, upscale, channels_dim=1)
    if prevs is not None:
        downscale = convert_smooth_upscale_to_downscale(upscale, num_heads=num_heads, num_head_repeats=num_head_repeats)
        if isinstance(prevs, nn.Module):
            prevs = [prevs]
        for module in prevs:
            downscale = downscale.to(device=module.weight.device)
            smooth_downscale_param(module.weight, downscale, channels_dim=0)
            if hasattr(module, "bias") and module.bias is not None:
                smooth_downscale_param(module.bias, downscale, channels_dim=0)
    return scale.to(device="cpu")


@torch.inference_mode()
def smooth_attention(
    *,
    q_proj: nn.Linear,
    k_proj: nn.Linear,
    scale: torch.Tensor | None,
    config: SmoothCalibConfig | None = None,
    query_quantizer: Quantizer | None = None,
    key_quantizer: Quantizer | None = None,
    queries: TensorsCache | None = None,
    keys: TensorsCache | None = None,
    attn_q: nn.Module | None = None,
    attn_k: nn.Module | None = None,
    eval_inputs: TensorsCache | None = None,
    eval_module: nn.Module = None,
    eval_kwargs: dict[str, tp.Any] = None,
    num_heads: int = 1,
    num_head_repeats: int = 1,
    with_rope: bool = True,
    develop_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Smooth attention.

    Args:
        q_proj (`nn.Linear`):
            The query projection module.
        k_proj (`nn.Linear`):
            The key projection module.
        scale (`torch.Tensor` or `None`, *optional*, defaults to `None`):
            The smooth quantization scale.
        config (`SmoothCalibConfig` or `None`, *optional*, defaults to `None`):
            The smooth quantization configuration.
        query_quantizer (`Quantizer` or `None`, *optional*, defaults to `None`):
            The quantizer for queries.
        key_quantizer (`Quantizer` or `None`, *optional*, defaults to `None`):
            The quantizer for keys.
        queries (`TensorsCache` or `None`, *optional*, defaults to `None`):
            The cache of the queries.
        keys (`TensorsCache` or `None`, *optional*, defaults to `None`):
            The cache of the keys.
        attn_q (`nn.Module` or `None`, *optional*, defaults to `None`):
            The module that generates the queries.
        attn_k (`nn.Module` or `None`, *optional*, defaults to `None`):
            The module that generates the keys.
        eval_inputs (`TensorsCache` or `None`, *optional*, defaults to `None`):
            The cache of the inputs corresponding to the evaluation module.
        eval_module (`nn.Module`, *optional*, defaults to `None`):
            The module to evaluate the quantization error.
        eval_kwargs (`dict[str, tp.Any]`, *optional*, defaults to `None`):
            The keyword arguments for evaluation.
        num_heads (`int`, *optional*, defaults to `1`):
            The number of heads.
        num_head_repeats (`int`, *optional*, defaults to `1`):
            The number of head repeats.
        with_rope (`bool`, *optional*, defaults to `True`):
            Whether quantization is applied after rotary position embedding.
        develop_dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            The development data type.

    Returns:
        `torch.Tensor`:
            The smooth quantization scale in CPU.
    """
    if scale is None:
        assert queries is not None and keys is not None and eval_inputs is not None
        assert attn_q is not None and attn_k is not None, "modules must be provided"
        scale = SmoothAttentionCalibrator(
            config=config,
            query_quantizer=query_quantizer,
            key_quantizer=key_quantizer,
            num_heads=num_heads,
            num_head_repeats=num_head_repeats,
            with_rope=with_rope,
            develop_dtype=develop_dtype,
        ).calibrate(
            q_proj_weight=q_proj.weight,
            k_proj_weight=k_proj.weight,
            queries=queries,
            keys=keys,
            query_module=attn_q,
            key_module=attn_k,
            eval_inputs=eval_inputs,
            eval_module=eval_module,
            eval_kwargs=eval_kwargs,
        )
        gc.collect()
        torch.cuda.empty_cache()
    upscale = scale.to(device=q_proj.weight.device)
    smooth_upscale_param(q_proj.weight, upscale, channels_dim=0)
    downscale = convert_smooth_upscale_to_downscale(upscale, num_heads=num_heads, num_head_repeats=num_head_repeats)
    smooth_downscale_param(k_proj.weight, downscale, channels_dim=0)
    return scale.to(device="cpu")
