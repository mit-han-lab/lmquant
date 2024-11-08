# -*- coding: utf-8 -*-
"""Search-based uantization calibrator module."""

import gc
import typing as tp
from abc import ABC, abstractmethod
from dataclasses import _MISSING_TYPE, MISSING

import psutil
import torch
import torch.nn as nn
import torch.utils.hooks

from ..data.cache import TensorCache, TensorsCache
from ..data.common import TensorType
from ..data.utils.reshape import ReshapeFn
from ..data.utils.shape import infer_view_shape
from ..quantizer.processor import Quantizer
from ..utils import tools
from ..utils.hooks import Hook
from .config import SearchBasedCalibConfig, SearchBasedCalibGranularity, SearchBasedCalibObjective

__all__ = ["SearchBasedCalibrator"]


def _reshape_w_for_wgts(w: torch.Tensor, w_view_shape: torch.Size) -> torch.Tensor:
    # (#g0, gs0, #g1, gs1, ...)
    w = w.view(w_view_shape)
    # (#g0, gs0, #g1, gs1, ...) -> (#g0, ..., gs1, ..., gs0)
    w = w.permute(*range(0, len(w_view_shape), 2), *range(3, len(w_view_shape), 2), 1)
    # (#g0, ..., gs0, gs1, ...) -> (#g0, ..., gs1 * gs2 * ..., gs0)
    return w.reshape(*w_view_shape[::2], -1, w_view_shape[1])


def _reshape_x_for_wgts(x: torch.Tensor, w_view_shape: torch.Size) -> torch.Tensor:
    # x is unfolded already
    num_samples = x.shape[0]
    # (1, n, #g1, gs1, ...)
    x = x.view(1, num_samples, *w_view_shape[2:])
    # (1, n, #g1, gs1, ...) -> (1, #g1, ..., n, gs1, ...)
    x = x.permute(*range(0, len(w_view_shape), 2), *range(1, len(w_view_shape), 2))
    return x.reshape(1, *w_view_shape[2::2], num_samples, -1)


def _reshape_x_for_ipts(x: torch.Tensor, x_view_shape: torch.Size) -> torch.Tensor:
    # x is original tensor without unfolding
    # (#g0, gs0, #g1, gs1, ...)
    x = x.view(x_view_shape)
    # (#g0, gs0, #g1, gs1, ...) -> (#g0, #g1, ..., gs0, gs2, ..., gs1)
    x = x.permute(*range(0, len(x_view_shape), 2), 1, *range(5, len(x_view_shape), 2), 3)
    # (#g0, #g1, ..., gs0, gs2, ..., gs1) -> (#g0, #g1, ..., gs0 * gs2 * ..., gs1)
    return x.reshape(*x_view_shape[::2], -1, x_view_shape[3])


def _reshape_w_for_ipts(w: torch.Tensor, x_view_shape: torch.Size) -> torch.Tensor:
    return w.transpose(0, 1).reshape(1, x_view_shape[2], *([1] * (w.ndim - 2)), x_view_shape[3], -1)


_CANDIDATE = tp.TypeVar("_CANDIDATE")
_CONFIG = tp.TypeVar("_CONFIG", bound=SearchBasedCalibConfig)


class SearchBasedCalibrator(ABC, tp.Generic[_CONFIG, _CANDIDATE]):
    """The base class for search-based calibration."""

    config: _CONFIG
    candidate: _CANDIDATE

    def __init__(
        self,
        tensor_type: TensorType,
        config: _CONFIG,
        w_quantizer: Quantizer | None,
        x_quantizer: Quantizer | None,
        y_quantizer: Quantizer | None,
        develop_dtype: torch.dtype,
    ) -> None:
        """Initialize the search-based calibrator.

        Args:
            tensor_type (`TensorType`):
                The tensor type.
            config (`_CONFIG`):
                The calibration configuration.
            w_quantizer (`Quantizer` or `None`):
                The w quantizer for x-w computation.
            x_quantizer (`Quantizer` or `None`):
                The x quantizer for x-w or y-x computation.
            y_quantizer (`Quantizer` or `None`):
                The y quantizer for y-x computation.
            develop_dtype (`torch.dtype`):
                The development data type.
        """
        self.tensor_type = tensor_type
        self.config = config
        self.objective = self.config.objective
        self.granularity = self.config.granularity
        self.opts_device = None
        self.develop_dtype = develop_dtype
        self.w_quantizer = w_quantizer
        self.x_quantizer = x_quantizer
        self.y_quantizer = y_quantizer
        self.needs_w_quant = self.w_quantizer is not None and self.w_quantizer.is_enabled()
        self.needs_x_quant = self.x_quantizer is not None and self.x_quantizer.is_enabled()
        self.needs_y_quant = self.y_quantizer is not None and self.y_quantizer.is_enabled()
        self.needs_x_quant_for_wgts = self.allows_x_quant_for_wgts and self.needs_x_quant
        self.needs_w_quant_for_wgts = self.allows_w_quant_for_wgts and self.needs_w_quant
        self.needs_x_quant_for_ipts = self.allows_x_quant_for_ipts and self.needs_x_quant
        self.needs_w_quant_for_ipts = self.allows_w_quant_for_ipts and self.needs_w_quant
        self.needs_x_quant_for_opts = self.allows_x_quant_for_opts and self.needs_x_quant
        self.needs_y_quant_for_opts = self.allows_y_quant_for_opts and self.needs_y_quant
        self.needs_w_quant_for_opts = self.allows_w_quant_for_opts and self.needs_w_quant
        if self.tensor_type == TensorType.Weights:
            self.quantizer = self.w_quantizer
            self.needs_quant = self.needs_w_quant
        elif self.tensor_type == TensorType.Inputs:
            self.quantizer = self.x_quantizer
            self.needs_quant = self.needs_x_quant
        elif self.tensor_type == TensorType.Outputs:
            self.quantizer = self.y_quantizer
            self.needs_quant = self.needs_y_quant
        else:
            raise ValueError(f"unknown tensor type: {self.tensor_type}")
        self.num_iters = getattr(self.config, "num_iters", 1)
        self.logger = tools.logging.getLogger(f"{__name__}.{self.__class__.__name__.replace('Agent', '')}")

    @property
    @abstractmethod
    def population_size(self) -> int:
        """Get the population size."""
        ...

    @property
    def allows_x_quant_for_wgts(self) -> bool:
        """Whether the calibrator allows input quantization when tensor_type is Weights."""
        return False

    @property
    def allows_w_quant_for_wgts(self) -> bool:
        """Whether the calibrator allows weight quantization when tensor_type is Weights."""
        return True

    @property
    def allows_x_quant_for_ipts(self) -> bool:
        """Whether the calibrator allows input quantization when tensor_type is Inputs."""
        return True

    @property
    def allows_w_quant_for_ipts(self) -> bool:
        """Whether the calibrator allows weight quantization when tensor_type is Inputs."""
        return False

    @property
    def allows_x_quant_for_opts(self) -> bool:
        """Whether the calibrator allows x quantization when tensor_type is Outputs."""
        return True

    @property
    def allows_y_quant_for_opts(self) -> bool:
        """Whether the calibrator allows y quantization when tensor_type is Outputs."""
        return True

    @property
    def allows_w_quant_for_opts(self) -> bool:
        """Whether the calibrator allows weight quantization when tensor_type is Outputs."""
        return False

    @property
    def needs_to_pre_reshape_x_for_wgts(self) -> bool:
        """Whether the calibrator needs to pre-reshape the inputs for weight quantization calibration."""
        return not self.needs_x_quant_for_wgts and self.config.pre_reshape

    @property
    def needs_to_pre_reshape_w_for_ipts(self) -> bool:
        """Whether the calibrator needs to pre-reshape the weights for input quantization calibration."""
        return not self.needs_w_quant_for_ipts and self.config.pre_reshape

    def _reset(self, **kwargs) -> None:
        pass

    def reset(self, **kwargs) -> None:
        """Reset the calibrator."""
        self.iter = 0
        self.candidate_id = 0
        self._reset(**kwargs)
        self._state_dict: list[tuple[nn.Parameter, torch.Tensor]] = []
        self._hooks: list[Hook | torch.utils.hooks.RemovableHandle] = []

    def is_done(self) -> bool:
        """Check if the calibration is done."""
        return self.iter >= self.num_iters

    def is_last_iter(self) -> bool:
        """Check if the current iteration is the last one."""
        return self.iter == self.num_iters - 1

    def is_last_candidate_in_iter(self) -> bool:
        """Check if the current candidate is the last one in the current iteration."""
        return self.candidate_id == self.population_size - 1

    @abstractmethod
    def get_best(self) -> _CANDIDATE:
        """Get the best candidate.

        Returns:
            `_CANDIDATE`:
                The best candidate.
        """
        ...

    @abstractmethod
    def _ask(self) -> _CANDIDATE:
        """Ask for the next candidate.

        Returns:
            `_CANDIDATE`:
                The next candidate.
        """
        ...

    @abstractmethod
    def _tell(self, error: list[torch.Tensor]) -> None:
        """Tell the error of the last candidate and update the best candidate.

        Args:
            error (`list[torch.Tensor]`):
                The error of the last candidate.
        """
        ...

    def ask(self) -> _CANDIDATE:
        """Ask for the next candidate.

        Returns:
            `_CANDIDATE`:
                The next candidate.
        """
        self.candidate = self._ask()
        return self.candidate

    def tell(self, error: list[torch.Tensor]) -> None:
        """Tell the error of the last candidate and update the best candidate.

        Args:
            error (`list[torch.Tensor]`):
                The error of the last candidate.
        """
        self._tell(error)
        self.candidate_id += 1
        if self.candidate_id >= self.population_size:
            self.iter += 1
            self.candidate_id = 0

    def _parse_ipts(self, ipts: TensorsCache | None, set_device: bool = False) -> TensorsCache | None:
        if set_device:
            self.opts_device = None
        elif ipts is None:
            return None
        if self.objective == SearchBasedCalibObjective.ProductsError:
            batch_size = self.config.element_batch_size
            calib_size = self.config.element_size
        elif self.objective == SearchBasedCalibObjective.OutputsError:
            batch_size = self.config.sample_batch_size
            calib_size = self.config.sample_size
        else:
            assert self.objective == SearchBasedCalibObjective.TensorError
            batch_size = -1
            calib_size = -1
        prev_size = len(ipts.front().data)
        parsed_ipts = TensorsCache(
            {
                key: ipt.repartition(
                    max_batch_size=batch_size,
                    max_size=calib_size,
                    standardize=self.objective == SearchBasedCalibObjective.ProductsError,
                    reshape=self.tensor_type == TensorType.Weights,
                )
                for key, ipt in ipts.items()
            }
        )
        curr_size = len(parsed_ipts.front().data)
        assert all(len(ipt.data) == curr_size for ipt in parsed_ipts.values())
        if set_device and prev_size != curr_size:
            self.opts_device = self.config.outputs_device
        return parsed_ipts

    def _parse_args(  # noqa: C901
        self,
        x_wgts: list[nn.Parameter] | None,
        y_wgts: list[nn.Parameter] | None,
        x_acts: TensorsCache | None,
        y_acts: TensorsCache | None,
        eval_inputs: TensorsCache | None,
        eval_module: nn.Module | None,
        x_mods: list[nn.Module] | None,
        y_mods: list[nn.Module] | None,
        orig_x_wgts: list[tuple[nn.Parameter, torch.Tensor]] | None,
        orig_y_wgts: list[tuple[nn.Parameter, torch.Tensor]] | None,
        orig_x_acts: TensorsCache | None,
        orig_y_acts: TensorsCache | None,
        orig_eval_inputs: TensorsCache | None,
    ) -> tuple[
        list[torch.Tensor | nn.Parameter] | None,  # x_wgts
        list[torch.Tensor | nn.Parameter] | None,  # y_wgts
        TensorsCache | None,  # x_acts
        TensorsCache | None,  # y_acts
        TensorsCache | None,  # eval_inputs
        nn.Module | None,  # eval_module
        list[nn.Module] | None,  # x_mods
        list[nn.Module] | None,  # y_mods
        list[tuple[nn.Parameter, torch.Tensor]] | None,  # orig_x_wgts
        list[tuple[nn.Parameter, torch.Tensor]] | None,  # orig_y_wgts
        TensorCache | None,  # orig_x_acts
        TensorCache | None,  # orig_y_acts
        TensorCache | None,  # orig_eval_inputs
    ]:
        # region Check the types of the arguments
        if x_wgts is not None:
            assert isinstance(x_wgts, (tuple, list)), "x_wgts should be a list"
            assert all(isinstance(w, nn.Parameter) for w in x_wgts), "wgts should be a list of nn.Parameter"
        if y_wgts is not None:
            assert isinstance(y_wgts, (tuple, list)), "y_wgts should be a list"
            assert all(isinstance(w, nn.Parameter) for w in y_wgts), "wgts should be a list of nn.Parameter"
        if x_acts is not None:
            assert isinstance(x_acts, TensorsCache), "x_acts should be a TensorsCache"
        if y_acts is not None:
            assert isinstance(y_acts, TensorsCache), "y_acts should be a TensorsCache"
        if eval_inputs is not None:
            assert isinstance(eval_inputs, TensorsCache), "eval_inputs should be a TensorsCache"
        if x_mods is not None:
            assert isinstance(x_mods, (tuple, list)), "x_mods should be a list"
        if y_mods is not None:
            assert isinstance(y_mods, (tuple, list)), "y_mods should be a list"
        if orig_x_wgts is not None:
            assert isinstance(orig_x_wgts, (tuple, list)), "orig_x_wgts should be a list"
            assert all(
                isinstance(p, nn.Parameter) and isinstance(w, torch.Tensor) for p, w in orig_x_wgts
            ), "orig_x_wgts should be a list of tuples of nn.Parameter and torch.Tensor"
            if x_wgts is not None:
                assert len(orig_x_wgts) >= len(x_wgts), "orig_wgts should have at least as mtp.Any elements as wgts"
                assert all(
                    p is w for (p, _), w in zip(orig_x_wgts, x_wgts, strict=True)
                ), "the parameters in orig_wgts should be in wgts in the same order"
        if orig_y_wgts is not None:
            assert isinstance(orig_y_wgts, (tuple, list)), "orig_y_wgts should be a list"
            assert all(
                isinstance(p, nn.Parameter) and isinstance(w, torch.Tensor) for p, w in orig_y_wgts
            ), "orig_y_wgts should be a list of tuples of nn.Parameter and torch.Tensor"
            if y_wgts is not None:
                assert len(orig_y_wgts) >= len(y_wgts), "orig_wgts should have at least as mtp.Any elements as wgts"
                assert all(
                    p is w for (p, _), w in zip(orig_y_wgts, y_wgts, strict=True)
                ), "the parameters in orig_wgts should be in wgts in the same order"
        if orig_x_acts is not None:
            assert isinstance(orig_x_acts, TensorsCache), "orig_x_acts should be a TensorsCache"
        if orig_y_acts is not None:
            assert isinstance(orig_y_acts, TensorsCache), "orig_y_acts should be a TensorsCache"
        if orig_eval_inputs is not None:
            assert isinstance(orig_eval_inputs, TensorsCache), "orig_eval_inputs should be a TensorsCache"
        # endregion
        self.objective = self.config.objective
        self.granularity = self.config.granularity
        if self.tensor_type == TensorType.Outputs:
            # ! currently only support OutputsError and Layer granularity for Outputs
            self.objective = SearchBasedCalibObjective.OutputsError
            self.granularity = SearchBasedCalibGranularity.Layer
        if self.objective == SearchBasedCalibObjective.TensorError:
            if x_wgts is not None:
                x_wgts = [w.detach().data for w in x_wgts]
            if y_wgts is not None:
                y_wgts = [w.detach().data for w in y_wgts]
            if self.tensor_type == TensorType.Weights:
                assert x_wgts is not None, "wgts should not be None when tensor_type is Weights"
            elif self.tensor_type == TensorType.Inputs:
                assert x_acts is not None, "mod_ipts should not be None when tensor_type is Inputs"
                eval_inputs, orig_eval_inputs = x_acts, orig_x_acts
            else:  # self.tensor_type == TensorType.Outputs
                assert y_acts is not None, "opts should not be None when tensor_type is Outputs"
                eval_inputs, orig_eval_inputs = y_acts, orig_y_acts
            eval_module = None
        elif self.objective == SearchBasedCalibObjective.ProductsError:
            assert self.tensor_type in (
                TensorType.Weights,
                TensorType.Inputs,
            ), "tensor_type should be Weights or Inputs when objective is ProductsError"
            assert x_wgts is not None, "wgts should not be None when objective is ProductsError"
            x_wgts = [w.detach().data for w in x_wgts]
            if y_wgts is not None:
                y_wgts = [w.detach().data for w in y_wgts]
            x_acts = x_acts or eval_inputs
            orig_x_acts = orig_x_acts or orig_eval_inputs
            assert x_acts is not None, "x_acts should not be None when objective is ProductsError"
            eval_inputs, orig_eval_inputs = x_acts, orig_x_acts
        elif self.objective == SearchBasedCalibObjective.OutputsError:
            assert eval_inputs is not None, "eval_inputs should not be None when objective is OutputsError"
            assert eval_module is not None, "eval_module should not be None when OutputsError"
            if (
                isinstance(eval_module, (nn.Linear, nn.Conv2d))
                and self.granularity.value < SearchBasedCalibGranularity.Layer.value
                and self.tensor_type != TensorType.Outputs
            ):
                self.objective = SearchBasedCalibObjective.ProductsError
                x_wgts = [w.detach().data for w in x_wgts]
                if y_wgts is not None:
                    y_wgts = [w.detach().data for w in y_wgts]
                x_acts = x_acts or eval_inputs
                orig_x_acts = orig_x_acts or orig_eval_inputs
                assert x_acts is not None, "x_acts should not be None when objective is ProductsError"
                eval_inputs, orig_eval_inputs = x_acts, orig_x_acts
            else:
                self.objective = SearchBasedCalibObjective.OutputsError
                self.granularity = SearchBasedCalibGranularity.Layer
        else:
            raise ValueError(f"unknown objective: {self.objective}")
        self.logger.debug(
            f"+ tensor_type: {self.tensor_type}, objective: {self.objective}, granularity: {self.granularity}"
        )
        return (
            x_wgts,
            y_wgts,
            x_acts,
            y_acts,
            self._parse_ipts(eval_inputs, set_device=True),
            eval_module,
            x_mods,
            y_mods,
            orig_x_wgts,
            orig_y_wgts,
            orig_x_acts,
            orig_y_acts,
            self._parse_ipts(orig_eval_inputs),
        )

    # region Reshape functions for computing products
    def _reshape_w_for_wgts_centric_partial_products(self, w: torch.Tensor, *, view_shape: torch.Size) -> torch.Tensor:
        return _reshape_w_for_wgts(w, view_shape)

    def _reshape_x_for_wgts_centric_partial_products(
        self, x: torch.Tensor, *, view_shape: torch.Size, fn: ReshapeFn
    ) -> torch.Tensor:
        return _reshape_x_for_wgts(fn(x), view_shape)

    def _reshape_w_for_ipts_centric_partial_products(self, w: torch.Tensor, *, view_shape: torch.Size) -> torch.Tensor:
        return _reshape_w_for_ipts(w, view_shape)

    def _reshape_x_for_ipts_centric_partial_products(
        self, x: torch.Tensor, *, view_shape: torch.Size, fn: ReshapeFn = None
    ) -> torch.Tensor:
        return _reshape_x_for_ipts(x, view_shape)

    def _reshape_w_for_full_products(self, w: torch.Tensor, *, view_shape: torch.Size = None) -> torch.Tensor:
        return w.view(w.shape[0], -1).T

    def _reshape_x_for_full_products(
        self, x: torch.Tensor, *, fn: ReshapeFn, view_shape: torch.Size = None
    ) -> torch.Tensor:
        return fn(x).view(x.shape[0], -1)

    # endregion

    @abstractmethod
    def _process_x_in_xw(self, x: torch.Tensor, channels_dim: int | _MISSING_TYPE = MISSING) -> torch.Tensor: ...

    @abstractmethod
    def _process_w_in_xw(self, w: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def _process_y_in_yx(self, y: torch.Tensor, channels_dim: int | _MISSING_TYPE = MISSING) -> torch.Tensor: ...

    @abstractmethod
    def _process_x_in_yx(self, x: torch.Tensor, channels_dim: int | _MISSING_TYPE = MISSING) -> torch.Tensor: ...

    @abstractmethod
    def _process_xw_in_yx(self, w: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def _process_yw_in_yx(self, w: torch.Tensor) -> torch.Tensor: ...

    def _recover_mod(self) -> None:
        for p, w in self._state_dict:
            p.data = w
        self._state_dict.clear()
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _process_wgts_centric_mod(
        self, wgts: list[nn.Parameter], mods: list[nn.Module], update_state_dict: bool = True, **kwargs
    ) -> None:
        if self.needs_w_quant_for_wgts:
            for w in wgts:
                if update_state_dict:
                    self._state_dict.append((w, w.data))
                w.data = self._process_w_in_xw(w.data)
        if self.needs_x_quant_for_wgts:
            self._hooks.append(self.x_quantizer.as_hook(func=self._process_x_in_xw, is_output=False).register(mods))

    def _process_ipts_centric_mod(
        self, wgts: list[nn.Parameter], mods: list[nn.Module], update_state_dict: bool = True, **kwargs
    ) -> None:
        if self.needs_w_quant_for_ipts:
            for w in wgts:
                if update_state_dict:
                    self._state_dict.append((w, w.data))
                w.data = self._process_w_in_xw(w.data)
        if self.needs_x_quant_for_ipts:
            self._hooks.append(self.x_quantizer.as_hook(self._process_x_in_xw, is_output=False).register(mods))

    def _process_opts_centric_mod(
        self,
        x_wgts: list[nn.Parameter],
        y_wgts: list[nn.Parameter],
        x_mods: list[nn.Module],
        y_mods: list[nn.Module],
        update_state_dict: bool = True,
        **kwargs,
    ) -> None:
        if self.needs_w_quant_for_opts:
            for w in x_wgts:
                if update_state_dict:
                    self._state_dict.append((w, w.data))
                w.data = self._process_xw_in_yx(w.detach().data)
            for w in y_wgts:
                if update_state_dict:
                    self._state_dict.append((w, w.data))
                w.data = self._process_yw_in_yx(w.detach().data)
        if self.needs_x_quant_for_opts:
            self._hooks.append(self.x_quantizer.as_hook(self._process_x_in_yx, is_output=True).register(x_mods))
        if self.needs_y_quant_for_opts:
            self._hooks.append(self.y_quantizer.as_hook(self._process_y_in_yx, is_output=True).register(y_mods))

    def calibrate(
        self,
        x_wgts: list[nn.Parameter] | None = None,
        y_wgts: list[nn.Parameter] | None = None,
        x_acts: TensorsCache | None = None,
        y_acts: TensorsCache | None = None,
        x_mods: list[nn.Module] | None = None,
        y_mods: list[nn.Module] | None = None,
        eval_inputs: TensorsCache | None = None,
        eval_module: nn.Module | None = None,
        eval_kwargs: dict[str, tp.Any] | None = None,
        orig_x_wgts: list[tuple[nn.Parameter, torch.Tensor]] | None = None,
        orig_y_wgts: list[tuple[nn.Parameter, torch.Tensor]] | None = None,
        orig_x_acts: TensorsCache | None = None,
        orig_y_acts: TensorsCache | None = None,
        orig_eval_inputs: TensorsCache | None = None,
        **kwargs,
    ) -> _CANDIDATE:
        """Calibrate the quantization parameters.

        Args:
            x_wgts (`list[nn.Parameter]` or `None`, *optional*, defaults to `None`):
                The weights in x-w computation, or weights that generates x for y-x computation.
            y_wgts (`list[nn.Parameter]` or `None`, *optional*, defaults to `None`):
                The weights that generates y for y-x computation.
            x_acts (`TensorsCache` or `None`, *optional*, defaults to `None`):
                The x activations. It should be x for x-w or y-x computation.
            y_acts (`TensorsCache` or `None`, *optional*, defaults to `None`):
                The y activations. It should be y for y-x computation.
            eval_inputs (`TensorsCache` or `None`, *optional*, defaults to `None`):
                The inputs of evaluation module `eval_module`.
            eval_module (`nn.Module` or `None`, *optional*, defaults to `None`):
                The module used for evaluation.
            x_mods (`list[nn.Module]` or `None`, *optional*, defaults to `None`):
                The modules for x activation quantization.
                It should be the modules that take in x for x-w computation,
                or the modules that generates x for y-x computation.
            y_mods (`list[nn.Module]` or `None`, *optional*, defaults to `None`):
                The modules for y activation quantization.
                It should be the modules that generates y for y-x computation.
            orig_x_wgts (`list[tuple[nn.Parameter, torch.Tensor]]` or `None`, *optional*, defaults to `None`):
                The original weights for `x_mods`.
            orig_y_wgts (`list[tuple[nn.Parameter, torch.Tensor]]` or `None`, *optional*, defaults to `None`):
                The original weights for `y_mods`.
            orig_x_acts (`TensorsCache` or `None`, *optional*, defaults to `None`):
                The original x activations `x_acts`.
            orig_y_acts (`TensorsCache` or `None`, *optional*, defaults to `None`):
                The original y activations `y_acts`.
            orig_eval_inputs (`TensorsCache` or `None`, *optional*, defaults to `None`):
                The original inputs of evaluation module `eval_inputs`.
            eval_kwargs (`dict[str, tp.Any]` or `None`, *optional*, defaults to `None`):
                The keyword arguments for evaluation module `eval_module`.

        Returns:
            `_CANDIDATE`:
                The best candidate.
        """
        tools.logging.Formatter.indent_inc()
        if self.w_quantizer is not None and self.w_quantizer.is_enabled():
            self.logger.debug(f"+ w: {self.w_quantizer.config.quant_dtype}")
        else:
            self.logger.debug("+ w: None")
        if self.x_quantizer is not None and self.x_quantizer.is_enabled():
            self.logger.debug(f"+ x: {self.x_quantizer.config.quant_dtype}")
        else:
            self.logger.debug("+ x: None")
        if self.y_quantizer is not None and self.y_quantizer.is_enabled():
            self.logger.debug(f"+ y: {self.y_quantizer.config.quant_dtype}")
        else:
            self.logger.debug("+ y: None")
        (
            x_wgts,
            y_wgts,
            x_acts,
            y_acts,
            eval_inputs,
            eval_module,
            x_mods,
            y_mods,
            orig_x_wgts,
            orig_y_wgts,
            orig_x_acts,
            orig_y_acts,
            orig_eval_inputs,
        ) = self._parse_args(
            x_wgts,
            y_wgts,
            x_acts,
            y_acts,
            eval_inputs,
            eval_module,
            x_mods,
            y_mods,
            orig_x_wgts,
            orig_y_wgts,
            orig_x_acts,
            orig_y_acts,
            orig_eval_inputs,
        )
        eval_kwargs = eval_kwargs or {}
        self.logger.debug(f"+ finished parsing calibration arguments, ram usage: {psutil.virtual_memory().percent}")
        self.reset(
            x_wgts=x_wgts,
            y_wgts=y_wgts,
            x_acts=x_acts,
            y_acts=y_acts,
            eval_inputs=eval_inputs,
            eval_module=eval_module,
            x_mods=x_mods,
            y_mods=y_mods,
            orig_x_wgts=orig_x_wgts,
            orig_y_wgts=orig_y_wgts,
            orig_x_acts=orig_x_acts,
            orig_y_acts=orig_y_acts,
            orig_eval_inputs=orig_eval_inputs,
            eval_kwargs=eval_kwargs,
            **kwargs,
        )
        self.logger.debug(f"+ finished reseting calibrator, ram usage: {psutil.virtual_memory().percent}")
        gc.collect()
        torch.cuda.empty_cache()
        if self.tensor_type == TensorType.Weights:
            result = self._calibrate_wgts(
                x_wgts, eval_inputs, eval_module, x_mods, orig_x_wgts, orig_eval_inputs, eval_kwargs, **kwargs
            )
        elif self.tensor_type == TensorType.Inputs:
            result = self._calibrate_ipts(
                x_wgts, eval_inputs, eval_module, x_mods, orig_x_wgts, orig_eval_inputs, eval_kwargs, **kwargs
            )
        else:
            result = self._calibrate_opts(
                x_wgts,
                y_wgts,
                eval_inputs,
                eval_module,
                x_mods,
                y_mods,
                orig_x_wgts,
                orig_y_wgts,
                orig_eval_inputs,
                eval_kwargs,
                **kwargs,
            )
        tools.logging.Formatter.indent_dec()
        return result

    def _calibrate_wgts(  # noqa: C901
        self,
        wgts: list[torch.Tensor | nn.Parameter],
        ipts: TensorsCache | None,
        eval_module: nn.Module | None,
        mods: list[nn.Module] | None,
        orig_wgts: list[tuple[nn.Parameter, torch.Tensor]] | None,
        orig_ipts: TensorsCache | None,
        eval_kwargs: dict[str, tp.Any],
        **kwargs,
    ) -> tp.Any:
        # region Step 1: Calculate the baseline
        if self.objective == SearchBasedCalibObjective.TensorError:
            if orig_wgts is None:
                orig_wgts = [(None, w.detach().data) for w in wgts]
            assert all(w.shape[1:] == wgts[0].shape[1:] for w in wgts)
            assert all(w.shape[1:] == wgts[0].shape[1:] for _, w in orig_wgts)
            orig_opts = None
            w_view_shapes = [infer_view_shape(w.shape, self.w_quantizer.config.largest_group_shape) for w in wgts]
        elif self.objective == SearchBasedCalibObjective.ProductsError:
            if orig_wgts is None:
                orig_wgts = [(None, w.detach().data) for w in wgts]
            assert len(orig_wgts) == len(wgts)
            assert all(w.shape[1:] == wgts[0].shape[1:] for w in wgts)
            assert all(w.shape[1:] == wgts[0].shape[1:] for _, w in orig_wgts)
            w_view_shapes = [infer_view_shape(w.shape, self.w_quantizer.config.largest_group_shape) for w in wgts]
            if self.granularity != SearchBasedCalibGranularity.Layer:
                _reshape_x = self._reshape_x_for_wgts_centric_partial_products
                _reshape_w = self._reshape_w_for_wgts_centric_partial_products
            else:
                _reshape_x = self._reshape_x_for_full_products
                _reshape_w = self._reshape_w_for_full_products
            assert isinstance(ipts, TensorsCache), "ipts should not be None for ProductsError"
            if orig_ipts is None:
                orig_ipts = ipts
            same_ipts = orig_ipts is ipts
            orig_ipts = TensorsCache(
                {
                    key: TensorCache(
                        [_reshape_x(x, view_shape=w_view_shapes[0], fn=ipt.reshape) for x in ipt.data],
                        **ipt.get_factory_kwargs(channels_dim=1, reshape=ReshapeFn()),
                    )
                    for key, ipt in orig_ipts.items()
                },
            )
            orig_opts: dict[tuple[int, ...], torch.Tensor] = {}
            for j, (_, w) in enumerate(orig_wgts):
                w = _reshape_w(w, view_shape=w_view_shapes[j])
                for s, ipt in enumerate(orig_ipts):
                    for i, x in enumerate(ipt.data):
                        x = x.to(device=w.device, non_blocking=True)
                        y = torch.matmul(x, w)
                        y = y.view(*y.shape[:-2], y.shape[-2] * y.shape[-1])
                        orig_opts[(i, s, j)] = y.to(device=self.opts_device or y.device, non_blocking=True)
            if self.needs_to_pre_reshape_x_for_wgts:
                if same_ipts:
                    ipts = orig_ipts
                else:
                    ipts = TensorsCache(
                        {
                            key: TensorCache(
                                [_reshape_x(x, view_shape=w_view_shapes[0], fn=ipt.reshape) for x in ipt.data],
                                **ipt.get_factory_kwargs(channels_dim=1, reshape=ReshapeFn()),
                            )
                            for key, ipt in ipts.items()
                        }
                    )
            del orig_wgts, orig_ipts, same_ipts
        elif self.objective == SearchBasedCalibObjective.OutputsError:
            w_view_shapes, _state_dict = [], []
            if orig_wgts is not None:
                _state_dict = [(p, p.data) for p, _ in orig_wgts]
                for p, w in orig_wgts:
                    p.data = w.to(device=p.data.device)
            if orig_ipts is None:
                orig_ipts = ipts
            assert isinstance(orig_ipts, TensorsCache), "orig_ipts should not be None for OutputsError"
            orig_opts: dict[tuple[int, ...], torch.Tensor] = {}
            for i in range(len(orig_ipts.front().data)):
                ipt = orig_ipts.extract(i, eval_kwargs)
                y = eval_module(*ipt.args, **ipt.kwargs)
                y = y[0] if not isinstance(y, torch.Tensor) else y
                assert isinstance(y, torch.Tensor), "eval_mod should return a tensor"
                orig_opts[(i,)] = y.to(device=self.opts_device or y.device, non_blocking=True)
                del ipt, y
            for p, s in _state_dict:
                p.data = s
            del orig_wgts, orig_ipts, _state_dict
        else:
            raise ValueError(f"Unknown objective {self.objective}")
        gc.collect()
        torch.cuda.empty_cache()
        self.logger.debug(f"+ finished calculating the original outputs, ram usage: {psutil.virtual_memory().percent}")
        # endregion
        while not self.is_done():
            self.ask()
            e: list[torch.Tensor] = []
            # region Step 2: Calculate the errors
            if self.objective == SearchBasedCalibObjective.TensorError:
                assert isinstance(orig_wgts, (tuple, list))
                for w, (_, orig_w), w_view_shape in zip(wgts, orig_wgts, w_view_shapes, strict=True):
                    e_w = self._process_w_in_xw(w).sub_(orig_w)
                    if self.granularity == SearchBasedCalibGranularity.Group:
                        e_w = e_w.view(w_view_shape).abs_().pow_(self.config.degree)
                        e_w = e_w.sum(dim=tuple(range(1, len(w_view_shape), 2))).view(w_view_shape[::2])
                    elif self.granularity == SearchBasedCalibGranularity.ChannelGroup:
                        e_w = e_w.view(*w_view_shape[:4], -1).abs_().pow_(self.config.degree)
                        e_w = e_w.sum(dim=(0, 1, 3, 4)).view(w_view_shape[2])
                    elif self.granularity == SearchBasedCalibGranularity.Layer:
                        e_w = e_w.abs_().pow_(self.config.degree).sum().view(-1)
                    else:
                        raise ValueError(f"Unknown granularity {self.granularity}")
                    e.append(e_w)
            elif self.objective == SearchBasedCalibObjective.ProductsError:
                e = [None] * len(wgts)
                for j, w in enumerate(wgts):
                    w = _reshape_w(self._process_w_in_xw(w), view_shape=w_view_shapes[j])
                    for s, ipt in enumerate(ipts):
                        for i, x in enumerate(ipt.data):
                            x = x.to(device=w.device, non_blocking=True)
                            if not self.needs_to_pre_reshape_x_for_wgts:
                                x = self._process_x_in_xw(x, channels_dim=ipt.channels_dim)
                                x = _reshape_x(x, view_shape=w_view_shapes[j], fn=ipt.reshape)
                            y = torch.matmul(x, w)
                            y = y.view(*y.shape[:-2], y.shape[-2] * y.shape[-1])
                            y = y.sub_(orig_opts[(i, s, j)].to(device=w.device, non_blocking=True))
                            if self.granularity == SearchBasedCalibGranularity.Group:
                                y = y.to(self.develop_dtype).pow_(self.config.degree).sum(dim=-1)
                            elif self.granularity == SearchBasedCalibGranularity.ChannelGroup:
                                y = y.view(y.shape[0], y.shape[1], -1)
                                y = y.to(self.develop_dtype).pow_(self.config.degree).sum(dim=(0, 2))
                            elif self.granularity == SearchBasedCalibGranularity.Layer:
                                y = y.to(self.develop_dtype).pow_(self.config.degree).sum().view(-1)
                            else:
                                raise ValueError(f"Unknown granularity {self.granularity}")
                            if e[j] is None:
                                e[j] = y
                            else:
                                e[j].add_(y)
            elif self.objective == SearchBasedCalibObjective.OutputsError:
                self._process_wgts_centric_mod(wgts=wgts, mods=mods, **kwargs)
                e = [None]
                for i in range(len(ipts.front().data)):
                    ipt = ipts.extract(i, eval_kwargs)
                    y = eval_module(*ipt.args, **ipt.kwargs)
                    y = y[0] if not isinstance(y, torch.Tensor) else y
                    assert isinstance(y, torch.Tensor), "eval_mod should return a tensor"
                    y = (y - orig_opts[(i,)].to(device=y.device, non_blocking=True)).to(self.develop_dtype)
                    y = y.pow_(self.config.degree).sum().view(-1)
                    if e[0] is None:
                        e[0] = y
                    else:
                        e[0].add_(y)
                    del ipt, y
                self._recover_mod()
            else:
                raise ValueError(f"Unknown objective {self.objective}")
            # endregion
            self.tell(e)
        return self.get_best()

    def _calibrate_ipts(  # noqa: C901
        self,
        wgts: list[torch.Tensor | nn.Parameter],
        ipts: TensorsCache,
        eval_module: nn.Module | None,
        mods: list[nn.Module] | None,
        orig_wgts: list[tuple[nn.Parameter, torch.Tensor]] | None,
        orig_ipts: TensorsCache | None,
        eval_kwargs: dict[str, tp.Any],
        **kwargs,
    ) -> tp.Any:
        if orig_ipts is None:
            orig_ipts = ipts
        assert ipts.num_tensors == orig_ipts.num_tensors
        assert all(
            x.shape == orig_x.shape
            for ipt, orig_ipt in zip(ipts, orig_ipts, strict=True)
            for x, orig_x in zip(ipt.data, orig_ipt.data, strict=True)
        )
        # region Step 1: Calculate the outputs
        if self.objective == SearchBasedCalibObjective.TensorError:
            assert all(x.shape == ipt.data[0].shape for ipt in ipts for x in ipt.data)
            orig_opts = None
            x_view_shapes = [
                infer_view_shape(
                    ipt.data[0].view(-1, *ipt.data[0].shape[ipt.channels_dim :]).shape,
                    self.x_quantizer.config.largest_group_shape,
                    skip_first_dim=True,
                )
                for ipt in ipts
            ]
            del orig_wgts
        elif self.objective == SearchBasedCalibObjective.ProductsError:
            assert all(ipt.channels_dim == 1 for ipt in ipts)
            assert all(ipt.channels_dim == 1 for ipt in orig_ipts)
            assert all(x.shape[1:] == ipts.front().data[0].shape[1:] for ipt in ipts for x in ipt.data)
            if orig_wgts is None:
                orig_wgts = [(None, w.detach().data) for w in wgts]
            assert len(orig_wgts) == len(wgts)
            if self.granularity != SearchBasedCalibGranularity.Layer:
                _reshape_x = self._reshape_x_for_ipts_centric_partial_products
                _reshape_w = self._reshape_w_for_ipts_centric_partial_products
            else:
                _reshape_x = self._reshape_x_for_full_products
                _reshape_w = self._reshape_w_for_full_products
            x_view_shapes = [
                infer_view_shape(ipt.data[0].shape, self.x_quantizer.config.largest_group_shape, skip_first_dim=True)
                for ipt in ipts
            ]
            orig_opts: dict[tuple[int, ...], torch.Tensor] = {}
            for j, (_, w) in enumerate(orig_wgts):
                w = _reshape_w(w, view_shape=x_view_shapes[0])
                for s, ipt in enumerate(orig_ipts):
                    for i, x in enumerate(ipt.data):
                        x = x.to(device=w.device, non_blocking=True)
                        x = _reshape_x(x, view_shape=x_view_shapes[s], fn=ipt.reshape)
                        y = torch.matmul(x, w)
                        y = y.view(*y.shape[:-2], y.shape[-2] * y.shape[-1])
                        orig_opts[(i, s, j)] = y.to(device=self.opts_device or y.device, non_blocking=True)
            if self.needs_to_pre_reshape_w_for_ipts:
                for j, w in enumerate(wgts):
                    wgts[j] = _reshape_w(w, view_shape=x_view_shapes[0])
            del orig_wgts, orig_ipts
        elif self.objective == SearchBasedCalibObjective.OutputsError:
            x_view_shapes, _state_dict = [], []
            if orig_wgts is not None:
                _state_dict = [(p, p.data) for p, _ in orig_wgts]
                for p, w in orig_wgts:
                    p.data = w.to(device=p.data.device)
            orig_opts: dict[tuple[int, ...], torch.Tensor] = {}
            for i in range(len(orig_ipts.front().data)):
                ipt = orig_ipts.extract(i, eval_kwargs)
                y = eval_module(*ipt.args, **ipt.kwargs)
                y = y[0] if not isinstance(y, torch.Tensor) else y
                assert isinstance(y, torch.Tensor), "eval_mod should return a tensor"
                orig_opts[(i,)] = y.to(device=self.opts_device or y.device, non_blocking=True)
                del ipt, y
            for p, s in _state_dict:
                p.data = s
            del orig_wgts, orig_ipts, _state_dict
        else:
            raise ValueError(f"Unknown objective {self.objective}")
        gc.collect()
        torch.cuda.empty_cache()
        # endregion
        while not self.is_done():
            self.ask()
            e: list[torch.Tensor] = []
            # region Step 2: Calculate the outputs errors
            if self.objective == SearchBasedCalibObjective.TensorError:
                e = [None] * len(ipts)
                for s, (ipt, x_view_shape) in enumerate(zip(ipts, x_view_shapes, strict=True)):
                    for x in ipt.data:
                        e_x = self._process_x_in_xw(x, channels_dim=ipt.channels_dim).sub_(x)
                        if self.granularity == SearchBasedCalibGranularity.Group:
                            e_x = e_x.view(x_view_shape).abs_().pow_(self.config.degree)
                            e_x = e_x.sum(dim=tuple(range(1, len(x_view_shape), 2)))
                        if self.granularity == SearchBasedCalibGranularity.ChannelGroup:
                            e_x = e_x.view(*x_view_shape[:4], -1).abs_().pow_(self.config.degree)
                            e_x = e_x.sum(dim=(0, 1, 3, 4)).view(x_view_shape[2])
                        elif self.granularity == SearchBasedCalibGranularity.Layer:
                            e_x = e_x.abs_().pow_(self.config.degree).sum().view(-1)
                        else:
                            raise ValueError(f"Unknown granularity {self.granularity}")
                        if e[s] is None:
                            e[s] = e_x
                        else:
                            e[s].add_(e_x)
            elif self.objective == SearchBasedCalibObjective.ProductsError:
                e = [None] * len(ipts)
                for j, w in enumerate(wgts):
                    if not self.needs_to_pre_reshape_w_for_ipts:
                        w = self._process_w_in_xw(w)
                        w = _reshape_w(w, view_shape=x_view_shapes[0])
                    for s, ipt in enumerate(ipts):
                        for i, x in enumerate(ipt.data):
                            x = x.to(device=w.device, non_blocking=True)
                            x = self._process_x_in_xw(x, channels_dim=ipt.channels_dim)
                            x = _reshape_x(x, view_shape=x_view_shapes[s], fn=ipt.reshape)
                            y = torch.matmul(x, w)
                            y = y.view(*y.shape[:-2], y.shape[-2] * y.shape[-1])
                            y = y.sub_(orig_opts[(i, s, j)].to(device=w.device, non_blocking=True))
                            if self.granularity == SearchBasedCalibGranularity.Group:
                                y = y.to(self.develop_dtype).pow_(self.config.degree).sum(dim=-1)
                            elif self.granularity == SearchBasedCalibGranularity.ChannelGroup:
                                y = y.view(y.shape[0], y.shape[1], -1)
                                y = y.to(self.develop_dtype).pow_(self.config.degree).sum(dim=(0, 2))
                            elif self.granularity == SearchBasedCalibGranularity.Layer:
                                y = y.to(self.develop_dtype).pow_(self.config.degree).sum().view(-1)
                            else:
                                raise ValueError(f"Unknown granularity {self.granularity}")
                            if e[s] is None:
                                e[s] = y
                            else:
                                e[s].add_(y)
            elif self.objective == SearchBasedCalibObjective.OutputsError:
                self._process_ipts_centric_mod(wgts=wgts, mods=mods, **kwargs)
                e = [None]
                for i in range(len(ipts.front().data)):
                    ipt = ipts.extract(i, eval_kwargs)
                    y = eval_module(*ipt.args, **ipt.kwargs)
                    y = y[0] if not isinstance(y, torch.Tensor) else y
                    assert isinstance(y, torch.Tensor), "eval_mod should return a tensor"
                    y = (y - orig_opts[(i,)].to(device=y.device, non_blocking=True)).to(self.develop_dtype)
                    y = y.pow_(self.config.degree).sum().view(-1)
                    if e[0] is None:
                        e[0] = y
                    else:
                        e[0].add_(y)
                    del ipt, y
                self._recover_mod()
            else:
                raise ValueError(f"Unknown objective {self.objective}")
            # endregion
            self.tell(e)
        return self.get_best()

    def _calibrate_opts(  # noqa: C901
        self,
        x_wgts: list[torch.Tensor | nn.Parameter],
        y_wgts: list[torch.Tensor | nn.Parameter],
        eval_inputs: TensorsCache | None,
        eval_module: nn.Module | None,
        x_mods: list[nn.Module] | None,
        y_mods: list[nn.Module] | None,
        orig_x_wgts: list[tuple[nn.Parameter, torch.Tensor]] | None,
        orig_y_wgts: list[tuple[nn.Parameter, torch.Tensor]] | None,
        orig_eval_inputs: TensorsCache | None,
        eval_kwargs: dict[str, tp.Any],
        **kwargs,
    ) -> tp.Any:
        # region Step 1: Calculate the outputs
        if self.objective == SearchBasedCalibObjective.OutputsError:
            assert eval_inputs is not None, "eval_inputs should not be None when objective is OutputsError"
            if orig_eval_inputs is None:
                orig_eval_inputs = eval_inputs
            assert eval_inputs.num_tensors == orig_eval_inputs.num_tensors
            assert all(
                x.shape == orig_x.shape
                for key, ipt in eval_inputs.items()
                for x, orig_x in zip(ipt.data, orig_eval_inputs[key].data, strict=True)
            )
            _x_state_dict, _y_state_dict = [], []
            if orig_x_wgts is not None:
                _x_state_dict = [(p, p.data) for p, _ in orig_x_wgts]
                for p, w in orig_x_wgts:
                    p.data = w.to(device=p.data.device)
            if orig_y_wgts is not None:
                _y_state_dict = [(p, p.data) for p, _ in orig_y_wgts]
                for p, w in orig_y_wgts:
                    p.data = w.to(device=p.data.device)
            orig_opts: dict[tuple[int, ...], torch.Tensor] = {}
            for i in range(len(orig_eval_inputs.front().data)):
                ipt = orig_eval_inputs.extract(i, eval_kwargs)
                y = eval_module(*ipt.args, **ipt.kwargs)
                y = y[0] if not isinstance(y, torch.Tensor) else y
                assert isinstance(y, torch.Tensor), "eval_mod should return a tensor"
                orig_opts[(i,)] = y.to(device=self.opts_device or y.device, non_blocking=True)
                del ipt, y
            for p, s in _x_state_dict:
                p.data = s
            for p, s in _y_state_dict:
                p.data = s
            del orig_x_wgts, orig_y_wgts, orig_eval_inputs, _x_state_dict, _y_state_dict
        else:
            raise ValueError(f"Unknown objective {self.objective}")
        gc.collect()
        torch.cuda.empty_cache()
        # endregion
        while not self.is_done():
            self.ask()
            e: list[torch.Tensor] = []
            # region Step 2: Calculate the outputs errors
            if self.objective == SearchBasedCalibObjective.OutputsError:
                self._process_opts_centric_mod(
                    x_wgts=x_wgts,
                    y_wgts=y_wgts,
                    x_mods=x_mods,
                    y_mods=y_mods,
                    **kwargs,
                )
                e = [None]
                for i in range(len(eval_inputs.front().data)):
                    ipt = eval_inputs.extract(i, eval_kwargs)
                    y = eval_module(*ipt.args, **ipt.kwargs)
                    y = y[0] if not isinstance(y, torch.Tensor) else y
                    assert isinstance(y, torch.Tensor), "eval_mod should return a tensor"
                    y = (y - orig_opts[(i,)].to(device=y.device, non_blocking=True)).to(self.develop_dtype)
                    y = y.pow_(self.config.degree).sum().view(-1)
                    if e[0] is None:
                        e[0] = y
                    else:
                        e[0].add_(y)
                    del ipt, y
                self._recover_mod()
            else:
                raise ValueError(f"Unknown objective {self.objective}")
            # endregion
            self.tell(e)
        return self.get_best()
