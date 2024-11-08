# -*- coding: utf-8 -*-
"""Quantization dynamic range calibration."""

import gc
import typing as tp
from dataclasses import _MISSING_TYPE, MISSING

import torch
import torch.nn as nn

from ..data.cache import TensorsCache
from ..data.common import TensorType
from ..data.range import DynamicRange
from ..data.scale import QuantScale
from ..data.utils.shape import infer_view_shape
from ..quantizer.impl.info import QuantInfo
from ..quantizer.processor import Quantizer
from ..utils import math, tools
from .config import DynamicRangeCalibConfig, SearchBasedCalibGranularity
from .search import SearchBasedCalibrator

__all__ = ["DynamicRangeCalibrator", "calibrate_dynamic_range"]


class DynamicRangeCalibrator(SearchBasedCalibrator[DynamicRangeCalibConfig, DynamicRange]):
    """The quantization dynamic range calibrator."""

    def __init__(
        self,
        tensor_type: TensorType,
        config: DynamicRangeCalibConfig,
        static: bool,
        quantizer: Quantizer,
        pre_scale: torch.Tensor | None = None,
    ) -> None:
        """Initialize the calibrator.

        Args:
            tensor_type (`TensorType`):
                The tensor type.
            config (`DynamicRangeCalibConfig`):
                The dynamic range calibration configuration.
            static (`bool`):
                Whether the dynamic range is static, i.e., whether the quantization is static.
            quantizer (`Quantizer`):
                The quantizer.
            pre_scale (`torch.Tensor` or `None`):
                The joint scale tensor of the previous quantization steps.
        """
        super().__init__(
            tensor_type=tensor_type,
            config=config,
            w_quantizer=quantizer if tensor_type == TensorType.Weights else None,
            x_quantizer=quantizer if tensor_type == TensorType.Inputs else None,
            y_quantizer=quantizer if tensor_type == TensorType.Outputs else None,
            develop_dtype=quantizer.develop_dtype,
        )
        assert self.needs_quant, "The tensor should be quantized."
        self.static = static
        self.pre_scale = pre_scale
        self.ratios = self.config.get_ratios()
        self.num_iters = len(self.ratios)

    @property
    def population_size(self) -> int:
        """Return the population size of the current iteration."""
        return len(self.ratios[self.iter])

    def is_clamp_based(self) -> bool:
        """Return whether the calibration is clamp-based."""
        return self.static or not self.config.allow_scale

    def _reset(  # noqa: C901
        self,
        x_wgts: list[torch.Tensor | nn.Parameter],
        x_acts: TensorsCache | None,
        y_acts: TensorsCache | None,
        **kwargs,
    ) -> None:
        """Reset the calibrator.

        Args:
            x_wgts (`list[torch.Tensor | nn.Parameter]`):
                The weights in x-w computation.
            x_acts (`TensorsCache` or `None`):
                The x activations in x-w computation.
            y_acts (`TensorsCache` or `None`):
                The y activations in y-x computation.
        """
        self.base_range: DynamicRange = DynamicRange()
        self.best_range: DynamicRange = DynamicRange()
        self.best_error: torch.Tensor = None
        self.error_history: list[tuple[float, float]] = []
        self.device = None
        if self.tensor_type == TensorType.Weights:
            assert len(x_wgts) == 1, "The weight should be a single tensor."
            wgts = x_wgts[0].data
            assert isinstance(wgts, torch.Tensor), "The weight should be a tensor."
            tensors = [wgts]
            self.device = wgts.device
        elif self.tensor_type == TensorType.Inputs:
            assert x_acts is not None, "The input activations should be provided."
            assert x_acts.num_tensors == 1, f"Only one input is allowed, got {x_acts.num_tensors}"
            acts = x_acts.front()
            tensors = acts.get_standardized_data(reshape=False)
            self.device = acts.orig_device
        else:
            assert y_acts is not None, "The output activations should be provided."
            assert y_acts.num_tensors == 1, f"Only one output is allowed, got {y_acts.num_tensors}"
            acts = y_acts.front()
            tensors = acts.get_standardized_data(reshape=False)
            self.device = acts.orig_device
        shape = tensors[0].shape
        view_shape = infer_view_shape(
            shape,
            self.quantizer.config.largest_group_shape,
            skip_first_dim=self.tensor_type != TensorType.Weights,
        )
        # region get range scale shape
        self.pos_view_shape = torch.Size([1, 1, view_shape[2], *([1] * (len(view_shape) - 3))])
        self.range_shape = torch.Size([gs if i % 2 == 0 else 1 for i, gs in enumerate(view_shape)])
        if self.granularity == SearchBasedCalibGranularity.Layer:
            self.ratio_shape = self.error_shape = torch.Size((1,))
            self.ratio_view_shape = self.ratio_shape
        elif self.granularity == SearchBasedCalibGranularity.ChannelGroup:
            self.ratio_shape = self.error_shape = torch.Size((view_shape[2],))
            self.ratio_view_shape = self.pos_view_shape
        elif self.granularity == SearchBasedCalibGranularity.Group:
            self.ratio_shape = self.error_shape = torch.Size(view_shape[::2])
            self.ratio_view_shape = self.range_shape
        else:
            raise ValueError(f"Invalid granularity: {self.granularity}")
        assert self.ratio_shape.numel() == self.ratio_view_shape.numel()
        if self.pre_scale is not None:
            assert len(shape) * 2 == len(self.pre_scale.shape)
            self.pre_view_shape = infer_view_shape(shape, self.pre_scale.shape[1::2])
        else:
            self.pre_view_shape = torch.Size()
        # endregion
        if self.is_clamp_based():
            if self.pre_scale is not None:
                tensors = [self._preprocess_with_pre_scale(t) for t in tensors]
            tensors = [t.view(view_shape).to(dtype=self.develop_dtype) for t in tensors]
            self.base_range = DynamicRange.construct(
                tensors,
                zero_domain=self.quantizer.config.zero_domain,
                is_float_point=self.quantizer.config.quant_dtype.is_float_point,
            )
            gc.collect()
            torch.cuda.empty_cache()

    def get_best(self) -> DynamicRange:
        """Get the best candidate.

        Returns:
            `DynamicRange`:
                The best candidate.
        """
        if self.static:
            return DynamicRange(min=self.best_range.min, max=self.best_range.max)
        elif self.is_clamp_based():
            return DynamicRange(min=self.best_range.min, max=self.best_range.max, ratio=1.0)
        else:
            return DynamicRange(ratio=self.best_range.ratio.view(self.ratio_view_shape))

    def _ask(self) -> DynamicRange:
        """Ask for the next candidate.

        Returns:
            `DynamicRange`:
                The next candidate.
        """
        ratio = self.ratios[self.iter][self.candidate_id]
        if self.is_clamp_based():
            return self.base_range.scale(
                ratio=ratio,
                zero_domain=self.quantizer.config.zero_domain,
                is_float_point=self.quantizer.config.quant_dtype.is_float_point,
            )
        else:
            return DynamicRange(ratio=ratio)

    def _tell(self, error: list[torch.Tensor]) -> None:  # noqa: C901
        """Tell the error of the last candidate and update the best candidate.

        Args:
            errors (`list[torch.Tensor]`):
                The error of the last candidate.
        """
        assert len(error) == 1, "The error should only have one value."
        error = error[0]
        assert isinstance(error, torch.Tensor)
        assert error.shape == self.error_shape, f"Error shape {error.shape} != {self.error_shape}."
        assert isinstance(self.candidate, DynamicRange)
        candidate_ratio = self.ratios[self.iter][self.candidate_id]
        if self.best_error is None:
            self.best_error = error
            if self.is_clamp_based():
                self.best_range.min = self.candidate.min
                self.best_range.max = self.candidate.max
            self.best_range.ratio = torch.full(
                size=self.ratio_shape, fill_value=candidate_ratio, device=self.device, dtype=self.develop_dtype
            )
        elif error.numel() > 1:
            pos = error < self.best_error
            self.best_error[pos] = error[pos]
            if self.is_clamp_based():
                if self.error_shape.numel() != self.range_shape.numel():
                    pos = pos.view(self.pos_view_shape).expand(*self.range_shape)
                else:
                    pos = pos.view(self.range_shape)
                self.best_range.max[pos] = self.candidate.max[pos]
                if isinstance(self.candidate.min, torch.Tensor):
                    self.best_range.min[pos] = self.candidate.min[pos]
            self.best_range.ratio[pos.view(self.ratio_shape)] = candidate_ratio
        elif error < self.best_error:
            self.best_error = error
            if self.is_clamp_based():
                self.best_range.min = self.candidate.min
                self.best_range.max = self.candidate.max
            self.best_range.ratio.fill_(candidate_ratio)
        if self.logger.level <= tools.logging.DEBUG:
            self.error_history.append(
                (
                    math.root_(error.to(torch.float64).sum(), self.config.degree).item(),
                    math.root_(self.best_error.to(torch.float64).sum(), self.config.degree).item(),
                )
            )
            if self.is_last_candidate_in_iter():
                stype_id = self.iter
                ratios, population_size = self.ratios[stype_id], self.population_size
                for i in range(0, population_size, 5):
                    self.logger.debug(
                        "  - range ratio = [%s]",
                        ", ".join(f"{ratios[j]:10.4f}" for j in range(i, min(i + 5, population_size))),
                    )
                    self.logger.debug(
                        "    sum  error  = [%s]",
                        ", ".join(f"{self.error_history[j][0]:10.4f}" for j in range(i, min(i + 5, population_size))),
                    )
                    self.logger.debug(
                        "    best error  = [%s]",
                        ", ".join(f"{self.error_history[j][1]:10.4f}" for j in range(i, min(i + 5, population_size))),
                    )
                self.error_history.clear()
                if self.is_last_iter():
                    self.logger.debug(
                        "+ error = [%.4f]",
                        math.root_(self.best_error.to(torch.float64).sum(), self.config.degree).item(),
                    )

    def _preprocess_with_pre_scale(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(self.pre_view_shape)
        t = t.to(dtype=self.develop_dtype) if t.dtype != self.develop_dtype else t.clone()
        t = t.div_(self.pre_scale)
        if self.quantizer.range_bound is not None and self.quantizer.range_bound.is_set():
            t = t.clamp_(min=self.quantizer.range_bound.min, max=self.quantizer.range_bound.max)
        return t

    def _process_wxy(self, tensor: torch.Tensor, channels_dim: int | _MISSING_TYPE = MISSING) -> torch.Tensor:
        shape, dtype = tensor.shape, tensor.dtype
        if self.pre_scale is not None:
            tensor = self._preprocess_with_pre_scale(tensor).view(shape)
        tensor = self.quantizer.quantize(
            tensor,
            kernel=None,
            channels_dim=channels_dim,
            dynamic_range=self.candidate,
            default_dtype=dtype,
            develop_dtype=self.develop_dtype,
        ).data
        if self.pre_scale is not None:
            tensor = tensor.view(self.pre_view_shape).mul_(self.pre_scale).to(dtype)
        tensor = tensor.view(shape)
        return tensor

    def _process_x_in_xw(self, x: torch.Tensor, channels_dim: int | _MISSING_TYPE = MISSING) -> torch.Tensor:
        if self.tensor_type != TensorType.Inputs:
            return x
        return self._process_wxy(x, channels_dim)

    def _process_w_in_xw(self, w: torch.Tensor) -> torch.Tensor:
        if self.tensor_type != TensorType.Weights:
            return w
        return self._process_wxy(w, channels_dim=None)

    def _process_y_in_yx(self, y: torch.Tensor, channels_dim: int | _MISSING_TYPE = MISSING) -> torch.Tensor:
        if self.tensor_type != TensorType.Outputs:
            return y
        return self._process_wxy(y, channels_dim)

    def _process_x_in_yx(self, x: torch.Tensor, channels_dim: int | _MISSING_TYPE = MISSING) -> torch.Tensor:
        raise RuntimeError("_process_x_in_yx should not be called in DynamicRangeCalibrator.")

    def _process_xw_in_yx(self, w: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("_process_xw_in_yx should not be called in DynamicRangeCalibrator.")

    def _process_yw_in_yx(self, w: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("_process_yw_in_yx should not be called in DynamicRangeCalibrator.")


def calibrate_dynamic_range(
    tensor_type: TensorType,
    config: DynamicRangeCalibConfig | None,
    static: bool,
    quantizer: Quantizer,
    modules: tp.Sequence[nn.Module],
    activations: TensorsCache,
    weights: tp.Sequence[nn.Parameter] | None = None,
    eval_inputs: TensorsCache | None = None,
    eval_module: nn.Module | None = None,
    eval_kwargs: dict[str, tp.Any] | None = None,
    orig_weights: tp.Sequence[tuple[nn.Parameter, torch.Tensor]] | None = None,
    orig_activations: TensorsCache | None = None,
    orig_eval_inputs: TensorsCache | None = None,
) -> tp.Sequence[DynamicRange] | None:
    """Calibrate the dynamic range.

    Args:
        tensor_type (`TensorType`):
            The tensor type.
        config (`DynamicRangeCalibConfig`):
            The quantization dynamic range calibration configuration.
        static (`bool`):
            Whether the dynamic range is static.
        quantizer (`Quantizer`):
            The quantizer.
        modules (`Sequence[nn.Module]`):
            The modules to calibrate.
        activations (`TensorsCache`):
            The inputs cache if the tensor type is not outputs, or the outputs cache if the tensor type is outputs.
        weights (`Sequence[nn.Parameter]` or `None`, *optional*, defaults to `None`):
            The weights to calibrate.
            If not provided, the weights of the modules will be used.
        eval_inputs (`TensorsCache` or `None`, *optional*, defaults to `None`):
            The cache of the inputs for evaluation.
            If not provided, the `activations` cache will be used.
        eval_module (`nn.Module` or `None`, *optional*, defaults to `None`):
            The module to evaluate the quantization error.
            If not provided, the module to calibrate will be used.
        eval_kwargs (`dict[str, tp.Any]` or `None`, *optional*, defaults to `None`):
            The keyword arguments for evaluation.
        orig_weights (`Sequence[tuple[nn.Parameter, torch.Tensor]]` or `None`, *optional*, defaults to `None`):
            The original weights.
        orig_activations (`TensorsCache` or `None`, *optional*, defaults to `None`):
            The original activations.
        orig_eval_inputs (`TensorsCache` or `None`, *optional*, defaults to `None`):
            The original evaluation inputs.

    Returns:
        `Sequence[DynamicRange]` or `None`:
            The dynamic ranges of each quantization step.
    """
    if config is None or not quantizer.is_enabled():
        return None

    decomposed_config = quantizer.config.decompose()
    num_steps = decomposed_config.num_steps
    # region dynamic range without search
    if not config.needs_search and (not static or tensor_type == TensorType.Weights):
        if config.ratio != 1.0:
            dynamic_range = DynamicRange(ratio=config.ratio)
            return tuple([dynamic_range] + [None] * (num_steps - 1))
        else:
            return None
    # endregion
    # region prepare for search
    if weights is None:
        weights = [module.weight for module in modules if hasattr(module, "weight")]
    if tensor_type == TensorType.Weights:
        assert len(modules) == 1, "only one module is supported for weight quantization calibration"
        assert len(weights) == 1, "only one weight is supported for weight quantization calibration"
        if eval_module is None:
            eval_module = modules[0]
            if eval_inputs is None:
                eval_inputs = activations
        else:
            assert eval_inputs is not None, "eval_inputs is required when eval_module is provided"
    else:
        assert activations is not None, "activations is required for activation quantization calibration"
        assert activations.num_tensors == 1, "only one tensor is supported for activation quantization calibration"
    if tensor_type != TensorType.Outputs:
        x_wgts, x_acts, x_mods, orig_x_wgts, orig_x_acts = weights, activations, modules, orig_weights, orig_activations
        y_wgts, y_acts, y_mods, orig_y_wgts, orig_y_acts = [], None, None, None, None
    else:
        x_wgts, x_acts, x_mods, orig_x_wgts, orig_x_acts = [], None, None, None, None
        y_wgts, y_acts, y_mods, orig_y_wgts, orig_y_acts = weights, activations, modules, orig_weights, orig_activations
    # endregion
    if num_steps == 1:
        dynamic_range = DynamicRangeCalibrator(
            tensor_type=tensor_type,
            config=config,
            static=static,
            quantizer=quantizer,
        ).calibrate(
            x_wgts=x_wgts,
            y_wgts=y_wgts,
            x_acts=x_acts,
            y_acts=y_acts,
            eval_inputs=eval_inputs,
            eval_module=eval_module,
            eval_kwargs=eval_kwargs,
            x_mods=x_mods,
            y_mods=y_mods,
            orig_x_wgts=orig_x_wgts,
            orig_y_wgts=orig_y_wgts,
            orig_x_acts=orig_x_acts,
            orig_y_acts=orig_y_acts,
            orig_eval_inputs=orig_eval_inputs,
        )
        return (dynamic_range,)
    # region prepare for search with progressive quantization
    if tensor_type == TensorType.Weights:
        tensor = weights[0].detach().data
    else:
        assert activations.num_tensors == 1, "Only one tensor is supported for activation quantization"
        acts = activations.front()
        assert len(acts.data) == 0, "Only one tensor is supported for activation quantization"
        tensor = acts.data[0].detach().data
        if acts.channels_dim is not None:
            tensor = tensor.reshape(-1, *tensor.shape[acts.channels_dim :])
    develop_dtype = quantizer.develop_dtype
    default_scale_dtype = quantizer.default_dtype or tensor.dtype
    develop_tensor = tensor.to(dtype=develop_dtype) if tensor.dtype != develop_dtype else tensor.clone()
    del tensor
    # endregion
    info = QuantInfo.construct(
        decomposed_config,
        tensor_shape=develop_tensor.shape,
        default_dtype=default_scale_dtype,
        quant_range=quantizer.quant_range,
        range_bound=quantizer.range_bound,
    )
    dynamic_ranges = []
    quant_scale = QuantScale()
    for step, step_info in enumerate(info.steps):
        step_quantizer = Quantizer(
            config=step_info.to_config(),
            kernel=quantizer.kernel if step == num_steps - 1 else None,
            quant_range=step_info.quant_range,
            range_bound=step_info.range_bound,
            default_dtype=quantizer.default_dtype,
            develop_dtype=quantizer.develop_dtype,
        )
        step_dynamic_range = DynamicRangeCalibrator(
            tensor_type=tensor_type,
            config=config,
            static=static,
            quantizer=step_quantizer,
            pre_scale=quant_scale.data,
        ).calibrate(
            x_wgts=x_wgts,
            y_wgts=y_wgts,
            x_acts=x_acts,
            y_acts=y_acts,
            eval_inputs=eval_inputs,
            eval_module=eval_module,
            eval_kwargs=eval_kwargs,
            x_mods=x_mods,
            y_mods=y_mods,
            orig_x_wgts=orig_x_wgts,
            orig_y_wgts=orig_y_wgts,
        )
        dynamic_ranges.append(step_dynamic_range)
        step_scale, _ = step_info.scale.quantize(
            tensor=develop_tensor.view(step_info.tensor_shape),
            dynamic_range=step_dynamic_range,
        )
        quant_scale.append(step_scale)
        if num_steps > 2 and step < num_steps - 1:
            step_quant_range = step_info.tensor_quant_range
            develop_tensor = develop_tensor.view(step_info.tensor_view_shape).div_(step_scale.data)
            develop_tensor = develop_tensor.clamp_(min=step_quant_range.min, max=step_quant_range.max)
    return tuple(dynamic_ranges)
