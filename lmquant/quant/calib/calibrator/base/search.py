# -*- coding: utf-8 -*-
"""Search-based uantization calibrator module."""
import gc
import logging
import math
import typing as tp
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.utils.hooks

from .....dataset import ActivationCache, ActivationsCache
from .....dataset.transform import TransformFn
from .....utils import tools
from ....data.utils.shape import infer_view_shape
from ....quantizer.base import Quantizer
from ...config import QuantTensorType, SearchBasedCalibConfig, SearchBasedCalibGranularity, SearchBasedCalibObjective

__all__ = ["SearchBasedQuantCalibrator"]


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
_CALIB_CONFIG = tp.TypeVar("_CALIB_CONFIG", bound=SearchBasedCalibConfig)


class SearchBasedQuantCalibrator(ABC, tp.Generic[_CALIB_CONFIG, _CANDIDATE]):
    """The base class for search-based quantization calibration."""

    def __init__(
        self,
        tensor_type: QuantTensorType,
        calib_config: _CALIB_CONFIG,
        wgts_quantizer: Quantizer | None,
        ipts_quantizer: Quantizer | None,
        opts_quantizer: Quantizer | None,
        allow_kernel_calib: bool,
        develop_dtype: torch.dtype,
    ) -> None:
        """Initialize the search-based calibrator.

        Args:
            tensor_type (QuantTensorType): The tensor type.
            calib_config (_CALIB_CONFIG): The calibration configuration.
            wgts_quantizer (KernelQuantizer | None): The weight quantizer.
            ipts_quantizer (KernelQuantizer | None): The input quantizer.
            opts_quantizer (KernelQuantizer | None): The output quantizer.
            needs_quant_wgts (bool): Whether weight quantization is needed.
            needs_quant_ipts (bool): Whether input quantization is needed.
            needs_quant_opts (bool): Whether output quantization is needed.
            allow_kernel_calib (bool): Whether to allow kernel calibration.
            develop_dtype (torch.dtype): The development data type.
        """
        self.tensor_type = tensor_type
        self.calib_config = calib_config
        self.objective = self.calib_config.objective
        self.granularity = self.calib_config.granularity
        self.opts_device = None
        self.develop_dtype = develop_dtype
        self.wgts_quantizer = wgts_quantizer
        self.ipts_quantizer = ipts_quantizer
        self.opts_quantizer = opts_quantizer
        self.needs_quant_wgts = self.wgts_quantizer is not None and self.wgts_quantizer.enabled
        self.needs_quant_ipts = self.ipts_quantizer is not None and self.ipts_quantizer.enabled
        self.needs_quant_opts = self.opts_quantizer is not None and self.opts_quantizer.enabled
        self.needs_quant_ipts_for_wgts = self.enabled_quant_ipts_for_wgts and self.needs_quant_ipts
        self.needs_quant_wgts_for_ipts = self.enabled_quant_wgts_for_ipts and self.needs_quant_wgts
        self.needs_quant_wgts_for_opts = self.enabled_quant_wgts_for_opts and self.needs_quant_wgts
        self.kernel_config = None
        if self.tensor_type == QuantTensorType.Weights:
            self.tnsr_quantizer = self.wgts_quantizer
            self.needs_quant_tnsr = self.needs_quant_wgts
            if allow_kernel_calib and self.calib_config.allow_kernel_calib:
                if self.wgts_quantizer is not None and self.wgts_quantizer.enabled:
                    if self.wgts_quantizer.kernel_config is not None and self.wgts_quantizer.kernel_config.enabled:
                        self.kernel_config = self.wgts_quantizer.kernel_config
        elif self.tensor_type == QuantTensorType.Inputs:
            self.tnsr_quantizer = self.ipts_quantizer
            self.needs_quant_tnsr = self.needs_quant_ipts
        elif self.tensor_type == QuantTensorType.Outputs:
            self.tnsr_quantizer = self.opts_quantizer
            self.needs_quant_tnsr = self.needs_quant_opts
        else:
            raise ValueError(f"unknown tensor type: {self.tensor_type}")
        self.num_iters = getattr(self.calib_config, "num_iters", 1)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__.replace('Agent', '')}")

    @property
    @abstractmethod
    def population_size(self) -> int:
        """Get the population size."""
        ...

    @property
    def enabled_quant_ipts_for_wgts(self) -> bool:
        """Whether the calibrator needs activation quantization when tensor_type is Weights."""
        return False

    @property
    def enabled_quant_wgts_for_ipts(self) -> bool:
        """Whether the calibrator needs weight quantization when tensor_type is not Weights."""
        return False

    @property
    def enabled_quant_wgts_for_opts(self) -> bool:
        """Whether the calibrator needs weight quantization when tensor_type is Outputs."""
        return False

    @property
    def enabled_pre_reshape_ipts_for_wgts(self) -> bool:
        """Whether the calibrator can pre-reshape the inputs for weight quantization calibration."""
        return not self.needs_quant_ipts_for_wgts and self.calib_config.pre_reshape

    @property
    def enabled_pre_reshape_wgts_for_acts(self) -> bool:
        """Whether the calibrator can pre-reshape the weights for activation quantization calibration."""
        return not self.needs_quant_wgts_for_ipts and self.calib_config.pre_reshape

    @property
    def allow_kernel_calib(self) -> bool:
        """Whether to allow kernel calibration."""
        return self.kernel_config is not None and self.kernel_config.enabled

    def _reset(self, **kwargs) -> None:
        pass

    def reset(self, **kwargs) -> None:
        """Reset the calibrator."""
        self.iter = 0
        self.candidate_id = 0
        tools.logging.Formatter.indent_inc()
        self._reset(**kwargs)
        tools.logging.Formatter.indent_dec()
        self._state_dict: list[tuple[nn.Parameter, torch.Tensor]] = []
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

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
            _CANDIDATE: The best candidate.
        """
        ...

    @abstractmethod
    def _ask(self) -> _CANDIDATE:
        """Ask for the next candidate.

        Returns:
            _CANDIDATE: The next candidate.
        """
        ...

    @abstractmethod
    def _tell(self, error: list[torch.Tensor]) -> None:
        """Tell the error of the last candidate and update the best candidate.

        Args:
            error (list[torch.Tensor]): The error of the last candidate.
        """
        ...

    def ask(self) -> _CANDIDATE:
        """Ask for the next candidate.

        Returns:
            _CANDIDATE: The next candidate.
        """
        tools.logging.Formatter.indent_inc()
        self.candidate = self._ask()
        tools.logging.Formatter.indent_dec()
        return self.candidate

    def tell(self, error: list[torch.Tensor]) -> None:
        """Tell the error of the last candidate and update the best candidate.

        Args:
            error (list[torch.Tensor]): The error of the last candidate.
        """
        tools.logging.Formatter.indent_inc()
        self._tell(error)
        tools.logging.Formatter.indent_dec()
        self.candidate_id += 1
        if self.candidate_id >= self.population_size:
            self.iter += 1
            self.candidate_id = 0

    def _parse_acts(self, acts: ActivationsCache | None, set_device: bool = False) -> ActivationsCache:
        if set_device:
            self.opts_device = None
        if self.objective == SearchBasedCalibObjective.ProductsError:
            batch_size = self.calib_config.element_batch_size
            calib_size = self.calib_config.element_size
        elif self.objective == SearchBasedCalibObjective.OutputsError:
            batch_size = self.calib_config.sample_batch_size
            calib_size = self.calib_config.sample_size
        else:
            assert self.objective == SearchBasedCalibObjective.TensorError
            batch_size = -1
            calib_size = -1
        assert acts is not None, "activations should not be None when calib_size is not 0"
        parsed_acts: list[ActivationCache] = []
        num_calib_samples: int = None
        for act in acts:
            xs, dim, fn, device = act.cached, act.channels_dim, act.transform, act.orig_device
            assert isinstance(xs, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in xs)
            assert all(x.ndim == xs[0].ndim for x in xs), "all x should have the same number of dimensions"
            assert all(x.shape == xs[0].shape for x in xs), "all x should have the same shape"
            if self.objective == SearchBasedCalibObjective.ProductsError:
                xs = [x.view(-1, *x.shape[dim:]) for x in xs]
                dim = 1
            if dim < 0:
                dim = xs[0].ndim + dim
            orig_total = xs[0].shape[0] * len(xs)
            if batch_size > 0:
                if xs[0].shape[0] > batch_size:
                    num_sub_batches = int(xs[0].shape[0] // batch_size)
                    assert num_sub_batches > 0, "batch_size is too large"
                    xs = [x[i * batch_size : (i + 1) * batch_size] for x in xs for i in range(num_sub_batches)]
                    if set_device:
                        self.opts_device = self.calib_config.outputs_device
                batch_size = xs[0].shape[0]
                num_batches = len(xs)
                if calib_size > 0 and num_batches * batch_size > calib_size:
                    num_calib_batches = int(calib_size // batch_size)
                    assert (
                        num_calib_batches > 0
                    ), f"calib_size ({calib_size}) is too small for batch_size ({batch_size})"
                    xs = xs[:: int(num_batches / num_calib_batches)]
            used_total = xs[0].shape[0] * len(xs)
            if num_calib_samples is None:
                num_calib_samples = int(math.ceil(used_total / (orig_total / acts.num_samples)))
            else:
                assert num_calib_samples == int(
                    math.ceil(used_total / (orig_total / acts.num_samples))
                ), "all activations should have the same number of samples"
            parsed_acts.append(ActivationCache(xs, channels_dim=dim, transform=fn, orig_device=device))
        assert all(len(act.cached) == len(parsed_acts[0].cached) for act in parsed_acts)
        assert num_calib_samples is not None and num_calib_samples > 0
        return ActivationsCache(parsed_acts, num_samples=num_calib_samples)

    def _parse_args(  # noqa: C901
        self,
        ipt_wgts: list[nn.Parameter] | None,
        opt_wgts: list[nn.Parameter] | None,
        ipts: ActivationsCache | None,
        opts: ActivationsCache | None,
        eval_ipt: ActivationsCache | None,
        eval_mod: nn.Module | None,
        ipt_mods: list[nn.Module] | None,
        opt_mods: list[nn.Module] | None,
        orig_ipt_wgts: list[tuple[nn.Parameter, torch.Tensor]] | None,
        orig_opt_wgts: list[tuple[nn.Parameter, torch.Tensor]] | None,
    ) -> tuple[
        list[torch.Tensor | nn.Parameter] | None,  # ipt_wgts
        list[torch.Tensor | nn.Parameter] | None,  # opt_wgts
        ActivationsCache | None,  # ipts
        ActivationsCache | None,  # opts
        ActivationsCache | None,  # eval_ipt
        nn.Module | None,  # eval_mod
        list[nn.Module] | None,  # ipt_mods
        list[nn.Module] | None,  # opt_mods
        list[tuple[nn.Parameter, torch.Tensor]] | None,  # orig_ipt_wgts
        list[tuple[nn.Parameter, torch.Tensor]] | None,  # orig_opt_wgts
    ]:
        if ipt_wgts is not None:
            assert isinstance(ipt_wgts, (tuple, list)), "wgts should be a list"
        if opt_wgts is not None:
            assert isinstance(opt_wgts, (tuple, list)), "wgts should be a list"
        if ipts is not None:
            assert isinstance(ipts, ActivationsCache), "ipts should be a ActivationsCache"
        if opts is not None:
            assert isinstance(opts, ActivationsCache), "opts should be a ActivationsCache"
        if eval_ipt is not None:
            assert isinstance(eval_ipt, ActivationsCache), "eval_ipts should be a ActivationsCache"
        if ipt_mods is not None:
            assert isinstance(ipt_mods, (tuple, list)), "ipts_mods should be a list"
        if opt_mods is not None:
            assert isinstance(opt_mods, (tuple, list)), "opts_mods should be a list"
        if orig_ipt_wgts is not None:
            assert isinstance(orig_ipt_wgts, (tuple, list)), "orig_wgts should be a list"
            assert all(
                isinstance(p, nn.Parameter) and isinstance(w, torch.Tensor) for p, w in orig_ipt_wgts
            ), "orig_wgts should be a list of tuples of nn.Parameter and torch.Tensor"
        if orig_opt_wgts is not None:
            assert isinstance(orig_opt_wgts, (tuple, list)), "orig_wgts should be a list"
            assert all(
                isinstance(p, nn.Parameter) and isinstance(w, torch.Tensor) for p, w in orig_opt_wgts
            ), "orig_wgts should be a list of tuples of nn.Parameter and torch.Tensor"

        self.objective = self.calib_config.objective
        self.granularity = self.calib_config.granularity
        if self.tensor_type == QuantTensorType.Outputs:
            self.objective = SearchBasedCalibObjective.OutputsError
            self.granularity = SearchBasedCalibGranularity.Layer
        if self.objective == SearchBasedCalibObjective.TensorError:
            if ipt_wgts is not None:
                ipt_wgts = [w.detach().data for w in ipt_wgts]
            if opt_wgts is not None:
                opt_wgts = [w.detach().data for w in opt_wgts]
            if self.tensor_type == QuantTensorType.Weights:
                assert ipt_wgts is not None, "wgts should not be None when tensor_type is Weights"
            elif self.tensor_type == QuantTensorType.Inputs:
                assert ipts is not None, "mod_ipts should not be None when tensor_type is Inputs"
                eval_ipt = ipts
            else:
                assert opts is not None, "opts should not be None when tensor_type is Outputs"
                eval_ipt = opts
            eval_mod, ipt_mods, opt_mods = None, None, None
        elif self.objective == SearchBasedCalibObjective.ProductsError:
            assert self.tensor_type in (
                QuantTensorType.Weights,
                QuantTensorType.Inputs,
            ), "tensor_type should be Weights or Inputs when objective is ProductsError"
            assert ipt_wgts is not None, "wgts should not be None when objective is ProductsError"
            if orig_ipt_wgts is not None:
                assert len(orig_ipt_wgts) >= len(ipt_wgts), "orig_wgts should have at least as mtp.Any elements as wgts"
                assert all(
                    p is w for (p, _), w in zip(orig_ipt_wgts, ipt_wgts)
                ), "the parameters in orig_wgts should be in wgts in the same order"
            ipt_wgts = [w.detach().data for w in ipt_wgts]
            eval_ipt = ipts or eval_ipt
            assert eval_ipt is not None, "ipts should not be None when objective is ProductsError"
            ipts = eval_ipt
            opt_wgts, eval_mod, ipt_mods, opt_mods, orig_opt_wgts = None, None, None, None, None
        elif self.objective == SearchBasedCalibObjective.OutputsError:
            assert eval_ipt is not None, "eval_ipts should not be None when objective is OutputsError"
            assert eval_mod is not None, "eval_mod should not be None when OutputsError"
            if self.tensor_type == QuantTensorType.Outputs:
                if opt_wgts is not None:
                    assert all(isinstance(w, nn.Parameter) for w in opt_wgts)
            else:
                if ipt_wgts is not None:
                    assert all(isinstance(w, nn.Parameter) for w in ipt_wgts)
            if orig_ipt_wgts is not None:
                assert len(orig_ipt_wgts) >= len(ipt_wgts)
                assert all(p is w for (p, _), w in zip(orig_ipt_wgts, ipt_wgts))
            if orig_opt_wgts is not None:
                assert len(orig_opt_wgts) >= len(opt_wgts)
                assert all(p is w for (p, _), w in zip(orig_opt_wgts, opt_wgts))
            if (
                isinstance(eval_mod, nn.Linear)
                and self.granularity.value < SearchBasedCalibGranularity.Layer.value
                and self.tensor_type != QuantTensorType.Outputs
            ):
                self.objective = SearchBasedCalibObjective.ProductsError
                ipt_wgts = [w.detach().data for w in ipt_wgts]
                ipts = eval_ipt
                opt_wgts, eval_mod, ipt_mods, opt_mods, orig_opt_wgts = None, None, None, None, None
            else:
                self.objective = SearchBasedCalibObjective.OutputsError
                self.granularity = SearchBasedCalibGranularity.Layer
        else:
            raise ValueError(f"unknown objective: {self.objective}")
        return (
            ipt_wgts,
            opt_wgts,
            ipts,
            opts,
            self._parse_acts(eval_ipt, set_device=True),
            eval_mod,
            ipt_mods,
            opt_mods,
            orig_ipt_wgts,
            orig_opt_wgts,
        )

    # region Reshape functions for computing products
    def _reshape_w_for_wgts_centric_partial_products(self, w: torch.Tensor, *, view_shape: torch.Size) -> torch.Tensor:
        return _reshape_w_for_wgts(w, view_shape)

    def _reshape_x_for_wgts_centric_partial_products(
        self, x: torch.Tensor, *, view_shape: torch.Size, fn: TransformFn
    ) -> torch.Tensor:
        return _reshape_x_for_wgts(fn(x), view_shape)

    def _reshape_w_for_ipts_centric_partial_products(self, w: torch.Tensor, *, view_shape: torch.Size) -> torch.Tensor:
        return _reshape_w_for_ipts(w, view_shape)

    def _reshape_x_for_ipts_centric_partial_products(
        self, x: torch.Tensor, *, view_shape: torch.Size, fn: TransformFn = None
    ) -> torch.Tensor:
        return _reshape_x_for_ipts(x, view_shape)

    def _reshape_w_for_full_products(self, w: torch.Tensor, *, view_shape: torch.Size = None) -> torch.Tensor:
        return w.view(w.shape[0], -1).T

    def _reshape_x_for_full_products(
        self, x: torch.Tensor, *, fn: TransformFn, view_shape: torch.Size = None
    ) -> torch.Tensor:
        return fn(x).view(x.shape[0], -1)

    # endregion

    @abstractmethod
    def _process_w_in_xw(self, w: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def _process_x_in_xw(self, x: torch.Tensor, channels_dim: int) -> torch.Tensor: ...

    @abstractmethod
    def _process_w_in_yx(self, w: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def _process_x_in_yx(self, x: torch.Tensor, channels_dim: int) -> torch.Tensor: ...

    @abstractmethod
    def _process_y_in_yx(self, y: torch.Tensor, channels_dim: int) -> torch.Tensor: ...

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
        if self.needs_quant_wgts:
            for w in wgts:
                if update_state_dict:
                    self._state_dict.append((w, w.data))
                w.data = self._process_w_in_xw(w.data)
        if self.needs_quant_ipts_for_wgts:
            for m in mods:
                self._hooks.append(self.ipts_quantizer.quantize_module_inputs(m, quantize_fn=self._process_x_in_xw))

    def _process_ipts_centric_mod(
        self, wgts: list[nn.Parameter], mods: list[nn.Module], update_state_dict: bool = True, **kwargs
    ) -> None:
        if self.needs_quant_wgts_for_ipts:
            for w in wgts:
                if update_state_dict:
                    self._state_dict.append((w, w.data))
                w.data = self._process_w_in_xw(w.data)
        if self.needs_quant_ipts:
            for m in mods:
                self._hooks.append(self.ipts_quantizer.quantize_module_inputs(m, quantize_fn=self._process_x_in_xw))

    def _process_opts_centric_mod(
        self,
        ipt_wgts: list[nn.Parameter],
        opt_wgts: list[nn.Parameter],
        ipt_mods: list[nn.Module],
        opt_mods: list[nn.Module],
        update_state_dict: bool = True,
        **kwargs,
    ) -> None:
        if self.needs_quant_wgts_for_opts:
            for w in ipt_wgts:
                if update_state_dict:
                    self._state_dict.append((w, w.data))
                w.data = self._process_w_in_yx(w.detach().data)
            for w in opt_wgts:
                if update_state_dict:
                    self._state_dict.append((w, w.data))
                w.data = self._process_w_in_yx(w.detach().data)
        if self.needs_quant_ipts:
            for m in ipt_mods:
                self._hooks.append(self.ipts_quantizer.quantize_module_outputs(m, quantize_fn=self._process_x_in_yx))
        if self.needs_quant_opts:
            for m in opt_mods:
                self._hooks.append(self.opts_quantizer.quantize_module_outputs(m, quantize_fn=self._process_y_in_yx))

    def calibrate(
        self,
        ipt_wgts: list[nn.Parameter] | None = None,
        opt_wgts: list[nn.Parameter] | None = None,
        ipts: ActivationsCache | None = None,
        opts: ActivationsCache | None = None,
        eval_mod: nn.Module | None = None,
        eval_ipt: ActivationsCache | None = None,
        ipt_mods: list[nn.Module] | None = None,
        opt_mods: list[nn.Module] | None = None,
        orig_ipt_wgts: list[tuple[nn.Parameter, torch.Tensor]] | None = None,
        orig_opt_wgts: list[tuple[nn.Parameter, torch.Tensor]] | None = None,
        eval_kwargs: dict[str, tp.Any] | None = None,
        **kwargs,
    ) -> _CANDIDATE:
        """Calibrate the quantization parameters.

        Args:
            ipt_wgts (list[nn.Parameter] | None): The weights for input modules.
            opt_wgts (list[nn.Parameter] | None): The weights for output modules.
            ipts (ActivationsCache | None): The inputs corresponding to the weights.
            opts (ActivationsCache | None): The outputs corresponding to the weights.
            eval_ipt (ActivationsCache | None): The inputs for evaluation.
            eval_mod (nn.Module | None): The evaluation module.
            ipt_mods (list[nn.Module] | None): The input modules for quantization.
            opt_mods (list[nn.Module] | None): The output modules for quantization.
            orig_ipt_wgts (list[tuple[nn.Parameter, torch.Tensor]] | None): The original weights for input modules.
            orig_opt_wgts (list[tuple[nn.Parameter, torch.Tensor]] | None): The original weights for output modules.
            eval_kwargs (dict[str, tp.Any] | None): The keyword arguments for evaluation.

        Returns:
            _CANDIDATE: The best candidate.
        """
        ipt_wgts, opt_wgts, ipts, opts, eval_ipt, eval_mod, ipt_mods, opt_mods, orig_ipt_wgts, orig_opt_wgts = (
            self._parse_args(
                ipt_wgts, opt_wgts, ipts, opts, eval_ipt, eval_mod, ipt_mods, opt_mods, orig_ipt_wgts, orig_opt_wgts
            )
        )
        eval_kwargs = eval_kwargs or {}
        self.reset(
            ipt_wgts=ipt_wgts,
            opt_wgts=opt_wgts,
            ipts=ipts,
            opts=opts,
            eval_ipt=eval_ipt,
            eval_mod=eval_mod,
            ipt_mods=ipt_mods,
            opt_mods=opt_mods,
            orig_ipt_wgts=orig_ipt_wgts,
            orig_opt_wgts=orig_opt_wgts,
            eval_kwargs=eval_kwargs,
            **kwargs,
        )
        gc.collect()
        torch.cuda.empty_cache()
        if self.tensor_type == QuantTensorType.Weights:
            return self._calibrate_wgts(ipt_wgts, eval_ipt, eval_mod, ipt_mods, orig_ipt_wgts, eval_kwargs, **kwargs)
        elif self.tensor_type == QuantTensorType.Inputs:
            return self._calibrate_ipts(ipt_wgts, eval_ipt, eval_mod, ipt_mods, orig_ipt_wgts, eval_kwargs, **kwargs)
        else:
            return self._calibrate_opts(
                ipt_wgts,
                opt_wgts,
                eval_ipt,
                eval_mod,
                ipt_mods,
                opt_mods,
                orig_ipt_wgts,
                orig_opt_wgts,
                eval_kwargs,
                **kwargs,
            )

    def _calibrate_wgts(  # noqa: C901
        self,
        wgts: list[torch.Tensor | nn.Parameter],
        /,
        ipts: ActivationsCache | None,
        eval_mod: nn.Module | None,
        ipt_mods: list[nn.Module] | None,
        orig_wgts: list[tuple[nn.Parameter, torch.Tensor]] | None,
        eval_kwargs: dict[str, tp.Any],
        **kwargs,
    ) -> tp.Any:
        num_gpus = torch.cuda.device_count()
        devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]
        devices = devices + devices

        def get_device(y_device: torch.device, i: int) -> torch.device:
            i = i % num_gpus
            if i == 0 or self.opts_device is None:
                return self.opts_device or y_device
            else:
                return devices[y_device.index + i]

        # region Step 1: Calculate the outputs
        if self.objective == SearchBasedCalibObjective.TensorError:
            assert all(w.shape[1:] == wgts[0].shape[1:] for w in wgts)
            opts = None
            w_view_shapes = [infer_view_shape(w.shape, self.wgts_quantizer.config.largest_group_shape) for w in wgts]
            del orig_wgts
        elif self.objective == SearchBasedCalibObjective.ProductsError:
            assert isinstance(ipts, ActivationsCache), "ipts should not be None when objective is ProductsError"
            assert ipts.num_sources == 1, f"Only one input source is allowed, got {ipts.num_sources}"
            if orig_wgts is None:
                orig_wgts = [(None, w.detach().data) for w in wgts]
            assert len(orig_wgts) == len(wgts)
            assert all(w.shape[1:] == wgts[0].shape[1:] for w in wgts)
            assert all(w.shape[1:] == wgts[0].shape[1:] for _, w in orig_wgts)
            if self.granularity != SearchBasedCalibGranularity.Layer:
                _reshape_x = self._reshape_x_for_wgts_centric_partial_products
                _reshape_w = self._reshape_w_for_wgts_centric_partial_products
            else:
                _reshape_x = self._reshape_x_for_full_products
                _reshape_w = self._reshape_w_for_full_products
            w_view_shapes = [infer_view_shape(w.shape, self.wgts_quantizer.config.largest_group_shape) for w in wgts]
            if self.enabled_pre_reshape_ipts_for_wgts:
                ipts = ActivationsCache(
                    ActivationCache(
                        [_reshape_x(x, view_shape=w_view_shapes[0], fn=ipts[0].transform) for x in ipts[0].cached],
                        channels_dim=1,
                        transform=TransformFn(),
                        orig_device=ipts[0].orig_device,
                    ),
                    num_samples=ipts.num_samples,
                )
            num_x, num_w = len(ipts[0].cached), len(wgts)
            opts: list[torch.Tensor] = [None] * num_w * num_x
            if num_w <= num_x:
                for j, (_, w) in enumerate(orig_wgts):
                    w = _reshape_w(w, view_shape=w_view_shapes[j])
                    for i, x in enumerate(ipts[0].cached):
                        k = i * num_w + j
                        x = x.to(device=w.device, non_blocking=True)
                        if not self.enabled_pre_reshape_ipts_for_wgts:
                            x = _reshape_x(x, view_shape=w_view_shapes[0], fn=ipts[0].transform)
                        y = torch.matmul(x, w)
                        y = y.view(*y.shape[:-2], y.shape[-2] * y.shape[-1])
                        opts[k] = y.to(device=get_device(y.device, k), non_blocking=True)
            else:
                for i, x in enumerate(ipts[0].cached):
                    if not self.enabled_pre_reshape_ipts_for_wgts:
                        x = _reshape_x(x, view_shape=w_view_shapes[0], fn=ipts[0].transform)
                    for j, (_, w) in enumerate(orig_wgts):
                        k = i * num_w + j
                        w = _reshape_w(w, view_shape=w_view_shapes[j])
                        x = x.to(device=w.device, non_blocking=True)
                        y = torch.matmul(x, w)
                        y = y.view(*y.shape[:-2], y.shape[-2] * y.shape[-1])
                        opts[k] = y.to(device=get_device(y.device, k), non_blocking=True)
            del orig_wgts
        elif self.objective == SearchBasedCalibObjective.OutputsError:
            w_view_shapes = []
            if orig_wgts is not None:
                _state_dict = [p.data for p, _ in orig_wgts]
                for p, w in orig_wgts:
                    p.data = w.to(device=p.data.device)
            opts: list[torch.Tensor] = []
            for i in range(len(ipts.sources[0].cached)):
                y = eval_mod(
                    *[ipt.cached[i].to(device=ipt.orig_device, non_blocking=True) for ipt in ipts],
                    **eval_kwargs,
                )
                if not isinstance(y, torch.Tensor):
                    y = y[0]
                opts.append(y.to(device=self.opts_device or y.device, non_blocking=True))
            if orig_wgts is not None:
                for (p, _), s in zip(orig_wgts, _state_dict):
                    p.data = s
                del _state_dict
            del orig_wgts
        else:
            raise ValueError(f"Unknown objective {self.objective}")
        gc.collect()
        torch.cuda.empty_cache()
        # endregion
        while not self.is_done():
            self.ask()
            e: list[torch.Tensor] = []
            # region Step 2: Calculate the errors
            if self.objective == SearchBasedCalibObjective.TensorError:
                for i, w in enumerate(wgts):
                    view_shape = w_view_shapes[i]
                    e_w = self._process_w_in_xw(w).sub_(w.data)
                    if self.granularity == SearchBasedCalibGranularity.Group:
                        e_w = e_w.view(view_shape).abs_().pow_(self.calib_config.degree)
                        e_w = e_w.sum(dim=tuple(range(1, len(view_shape), 2))).view(view_shape[::2])
                    elif self.granularity == SearchBasedCalibGranularity.ChannelGroup:
                        e_w = e_w.view(*view_shape[:4], -1).abs_().pow_(self.calib_config.degree)
                        e_w = e_w.sum(dim=(0, 1, 3, 4)).view(view_shape[2])
                    elif self.granularity == SearchBasedCalibGranularity.Layer:
                        e_w = e_w.abs_().pow_(self.calib_config.degree).sum().view(-1)
                    else:
                        raise ValueError(f"Unknown granularity {self.granularity}")
                    e.append(e_w)
            elif self.objective == SearchBasedCalibObjective.ProductsError:
                num_x, num_w = len(ipts[0].cached), len(wgts)
                e = [None] * num_w
                if num_w <= num_x:
                    for j, w in enumerate(wgts):
                        w = self._process_w_in_xw(w)
                        w = _reshape_w(w, view_shape=w_view_shapes[j])
                        for i, x in enumerate(ipts[0].cached):
                            x = x.to(device=w.device, non_blocking=True)
                            if not self.enabled_pre_reshape_ipts_for_wgts:
                                x = self._process_x_in_xw(x, channels_dim=ipts[0].channels_dim)
                                x = _reshape_x(x, view_shape=w_view_shapes[0], fn=ipts[0].transform)
                            y = torch.matmul(x, w)
                            y = y.view(*y.shape[:-2], y.shape[-2] * y.shape[-1])
                            y = y.sub_(opts[i * num_w + j].to(device=w.device, non_blocking=True))
                            if self.granularity == SearchBasedCalibGranularity.Group:
                                y = y.to(self.develop_dtype).pow_(self.calib_config.degree).sum(dim=-1)
                            elif self.granularity == SearchBasedCalibGranularity.ChannelGroup:
                                y = y.view(y.shape[0], y.shape[1], -1)
                                y = y.to(self.develop_dtype).pow_(self.calib_config.degree).sum(dim=(0, 2))
                            elif self.granularity == SearchBasedCalibGranularity.Layer:
                                y = y.to(self.develop_dtype).pow_(self.calib_config.degree).sum().view(-1)
                            else:
                                raise ValueError(f"Unknown granularity {self.granularity}")
                            if e[j] is None:
                                e[j] = y
                            else:
                                e[j].add_(y)
                else:
                    for i, x in enumerate(ipts[0].cached):
                        if not self.enabled_pre_reshape_ipts_for_wgts:
                            x = self._process_x_in_xw(x, channels_dim=ipts[0].channels_dim)
                            x = _reshape_x(x, view_shape=w_view_shapes[0], fn=ipts[0].transform)
                        for j, w in enumerate(wgts):
                            w = self._process_w_in_xw(w)
                            w = _reshape_w(w, view_shape=w_view_shapes[j])
                            x = x.to(device=w.device, non_blocking=True)
                            y = torch.matmul(x, w)
                            y = y.view(*y.shape[:-2], y.shape[-2] * y.shape[-1])
                            y = y.sub_(opts[i * num_w + j].to(device=w.device, non_blocking=True))
                            if self.granularity == SearchBasedCalibGranularity.Group:
                                y = y.to(self.develop_dtype).pow_(self.calib_config.degree).sum(dim=-1)
                            elif self.granularity == SearchBasedCalibGranularity.ChannelGroup:
                                y = y.view(y.shape[0], y.shape[1], -1)
                                y = y.to(self.develop_dtype).pow_(self.calib_config.degree).sum(dim=(0, 2))
                            elif self.granularity == SearchBasedCalibGranularity.Layer:
                                y = y.to(self.develop_dtype).pow_(self.calib_config.degree).sum().view(-1)
                            else:
                                raise ValueError(f"Unknown granularity {self.granularity}")
                            if e[j] is None:
                                e[j] = y
                            else:
                                e[j].add_(y)
            elif self.objective == SearchBasedCalibObjective.OutputsError:
                self._process_wgts_centric_mod(wgts=wgts, mods=ipt_mods, **kwargs)
                e = [None]
                for i in range(len(ipts[0].cached)):
                    y = eval_mod(
                        *[ipt.cached[i].to(device=ipt.orig_device, non_blocking=True) for ipt in ipts],
                        **eval_kwargs,
                    )
                    if not isinstance(y, torch.Tensor):
                        y = y[0]
                    y = (
                        (y - opts[i].to(device=y.device, non_blocking=True))
                        .to(self.develop_dtype)
                        .pow_(self.calib_config.degree)
                        .sum()
                        .view(-1)
                    )
                    if e[0] is None:
                        e[0] = y
                    else:
                        e[0].add_(y)
                self._recover_mod()
            else:
                raise ValueError(f"Unknown objective {self.objective}")
            # endregion
            self.tell(e)
        return self.get_best()

    def _calibrate_ipts(  # noqa: C901
        self,
        wgts: list[torch.Tensor | nn.Parameter],
        /,
        ipts: ActivationsCache | None,
        eval_mod: nn.Module | None,
        ipt_mods: list[nn.Module] | None,
        orig_wgts: list[tuple[nn.Parameter, torch.Tensor]] | None,
        eval_kwargs: dict[str, tp.Any],
        **kwargs,
    ) -> tp.Any:
        num_gpus = torch.cuda.device_count()
        devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]
        devices = devices + devices

        def get_device(y_device: torch.device, i: int) -> torch.device:
            i = i % num_gpus
            if i == 0:
                return self.opts_device or y_device
            else:
                return devices[y_device.index + i]

        # region Step 1: Calculate the outputs
        if self.objective == SearchBasedCalibObjective.TensorError:
            assert ipts.num_sources == 1, f"Only one input source is allowed, got {ipts.num_sources}"
            assert all(x.shape == ipts[0].cached[0].shape for x in ipts[0].cached)
            opts = None
            x_view_shape = infer_view_shape(
                ipts[0].cached[0].view(-1, *ipts[0].cached[0].shape[ipts[0].channels_dim :]).shape,
                self.ipts_config.largest_group_shape,
                skip_first_dim=True,
            )
            del orig_wgts
        elif self.objective == SearchBasedCalibObjective.ProductsError:
            assert ipts.num_sources == 1, f"Only one input source is allowed, got {ipts.num_sources}"
            assert ipts[0].channels_dim == 1
            assert all(x.shape == ipts[0].cached[0].shape for x in ipts[0].cached)
            if orig_wgts is None:
                orig_wgts = [(None, w.detach().data) for w in wgts]
            assert len(orig_wgts) == len(wgts)
            if self.granularity != SearchBasedCalibGranularity.Layer:
                _reshape_x = self._reshape_x_for_ipts_centric_partial_products
                _reshape_w = self._reshape_w_for_ipts_centric_partial_products
            else:
                _reshape_x = self._reshape_x_for_full_products
                _reshape_w = self._reshape_w_for_full_products
            x_view_shape = infer_view_shape(
                ipts[0].cached[0].view(-1, *ipts[0].cached[0].shape[ipts[0].channels_dim :]).shape,
                self.ipts_config.largest_group_shape,
                skip_first_dim=True,
            )
            num_x, num_w = len(ipts[0].cached), len(wgts)
            opts: list[torch.Tensor] = [None] * num_w * num_x
            if num_w <= num_x:
                for j, (_, w) in enumerate(orig_wgts):
                    w = _reshape_w(w, view_shape=x_view_shape)
                    for i, x in enumerate(ipts[0].cached):
                        k = i * num_w + j
                        x = ipts[0][i].to(device=w.device, non_blocking=True)
                        x = _reshape_x(x, view_shape=x_view_shape, fn=ipts[0].transform)
                        y = torch.matmul(x, w)
                        y = y.view(*y.shape[:-2], y.shape[-2] * y.shape[-1])
                        opts[k] = y.to(device=get_device(y.device, k), non_blocking=True)
            else:
                for i, x in enumerate(ipts[0].cached):
                    x = _reshape_x(x, view_shape=x_view_shape, fn=ipts[0].transform)
                    for j, (_, w) in enumerate(orig_wgts):
                        k = i * num_w + j
                        w = _reshape_w(w, view_shape=x_view_shape)
                        x = x.to(device=w.device, non_blocking=True)
                        y = torch.matmul(x, w)
                        y = y.view(*y.shape[:-2], y.shape[-2] * y.shape[-1])
                        opts[k] = y.to(device=get_device(y.device, k), non_blocking=True)
            if self.enabled_pre_reshape_wgts_for_acts:
                for j, w in enumerate(wgts):
                    wgts[j] = _reshape_w(w, view_shape=x_view_shape)
            del orig_wgts
        elif self.objective == SearchBasedCalibObjective.OutputsError:
            x_view_shape = torch.Size()
            if orig_wgts is not None:
                _state_dict = [p.data for p, _ in orig_wgts]
                for p, w in orig_wgts:
                    p.data = w.to(device=p.data.device)
            opts: list[torch.Tensor] = []
            for i in range(len(ipts[0].cached)):
                y = eval_mod(
                    *[ipt.cached[i].to(device=ipt.orig_device, non_blocking=True) for ipt in ipts],
                    **eval_kwargs,
                )
                if not isinstance(y, torch.Tensor):
                    y = y[0]
                opts.append(y.to(device=self.opts_device or y.device, non_blocking=True))
            if orig_wgts is not None:
                for (p, _), s in zip(orig_wgts, _state_dict):
                    p.data = s
                del _state_dict
            del orig_wgts
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
                dim = ipts[0].channels_dim
                for i, x in enumerate(ipts[0].cached):
                    e_x = self._process_x_in_xw(x, channels_dim=dim).sub_(x)
                    if self.granularity == SearchBasedCalibGranularity.Group:
                        e_x = e_x.view(x_view_shape).abs_().pow_(self.calib_config.degree)
                        e_x = e_x.sum(dim=tuple(range(1, len(x_view_shape), 2)))
                    if self.granularity == SearchBasedCalibGranularity.ChannelGroup:
                        e_x = e_x.view(*x_view_shape[:4], -1).abs_().pow_(self.calib_config.degree)
                        e_x = e_x.sum(dim=(0, 1, 3, 4)).view(x_view_shape[2])
                    elif self.granularity == SearchBasedCalibGranularity.Layer:
                        e_x = e_x.abs_().pow_(self.calib_config.degree).sum().view(-1)
                    else:
                        raise ValueError(f"Unknown granularity {self.granularity}")
                    e.append(e_x)
            elif self.objective == SearchBasedCalibObjective.ProductsError:
                num_x, num_w = len(ipts[0].cached), len(wgts)
                e = [None] * num_w
                if num_w <= num_x:
                    for j, w in enumerate(wgts):
                        if not self.enabled_pre_reshape_wgts_for_acts:
                            w = self._process_w_in_xw(w)
                            w = _reshape_w(w, view_shape=x_view_shape)
                        for i, x in enumerate(ipts[0].cached):
                            x = x.to(device=w.device, non_blocking=True)
                            x = self._process_x_in_xw(x, channels_dim=ipts[0].channels_dim)
                            x = _reshape_x(x, view_shape=x_view_shape, fn=ipts[0].transform)
                            y = torch.matmul(x, w)
                            y = y.view(*y.shape[:-2], y.shape[-2] * y.shape[-1])
                            y = y.sub_(opts[i * num_w + j].to(device=w.device, non_blocking=True))
                            if self.granularity == SearchBasedCalibGranularity.Group:
                                y = y.to(self.develop_dtype).pow_(self.calib_config.degree).sum(dim=-1)
                            elif self.granularity == SearchBasedCalibGranularity.ChannelGroup:
                                y = y.view(y.shape[0], y.shape[1], -1)
                                y = y.to(self.develop_dtype).pow_(self.calib_config.degree).sum(dim=(0, 2))
                            elif self.granularity == SearchBasedCalibGranularity.Layer:
                                y = y.to(self.develop_dtype).pow_(self.calib_config.degree).sum().view(-1)
                            else:
                                raise ValueError(f"Unknown granularity {self.granularity}")
                            if e[j] is None:
                                e[j] = y
                            else:
                                e[j].add_(y)
                else:
                    for i, x in enumerate(ipts[0].cached):
                        x = self._process_x_in_xw(x, channels_dim=ipts[0].channels_dim)
                        x = _reshape_x(x, view_shape=x_view_shape, fn=ipts[0].transform)
                        for j, w in enumerate(wgts):
                            if not self.enabled_pre_reshape_wgts_for_acts:
                                w = self._process_w_in_xw(w)
                                w = _reshape_w(w, view_shape=x_view_shape)
                            x = x.to(device=w.device, non_blocking=True)
                            y = torch.matmul(x, w)
                            y = y.view(*y.shape[:-2], y.shape[-2] * y.shape[-1])
                            y = y.sub_(opts[i * num_w + j].to(device=w.device, non_blocking=True))
                            if self.granularity == SearchBasedCalibGranularity.Group:
                                y = y.to(self.develop_dtype).pow_(self.calib_config.degree).sum(dim=-1)
                            elif self.granularity == SearchBasedCalibGranularity.ChannelGroup:
                                y = y.view(y.shape[0], y.shape[1], -1)
                                y = y.to(self.develop_dtype).pow_(self.calib_config.degree).sum(dim=(0, 2))
                            elif self.granularity == SearchBasedCalibGranularity.Layer:
                                y = y.to(self.develop_dtype).pow_(self.calib_config.degree).sum().view(-1)
                            else:
                                raise ValueError(f"Unknown granularity {self.granularity}")
                            if e[j] is None:
                                e[j] = y
                            else:
                                e[j].add_(y)
            elif self.objective == SearchBasedCalibObjective.OutputsError:
                self._process_ipts_centric_mod(wgts=wgts, mods=ipt_mods, **kwargs)
                e = [None]
                for i in range(len(ipts[0].cached)):
                    y = eval_mod(
                        *[ipt.cached[i].to(device=ipt.orig_device, non_blocking=True) for ipt in ipts],
                        **eval_kwargs,
                    )
                    if not isinstance(y, torch.Tensor):
                        y = y[0]
                    y = (
                        (y - opts[i].to(device=y.device, non_blocking=True))
                        .to(self.develop_dtype)
                        .pow_(self.calib_config.degree)
                        .sum()
                        .view(-1)
                    )
                    if e[0] is None:
                        e[0] = y
                    else:
                        e[0].add_(y)
                self._recover_mod()
            else:
                raise ValueError(f"Unknown objective {self.objective}")
            # endregion
            self.tell(e)
        return self.get_best()

    def _calibrate_opts(  # noqa: C901
        self,
        ipt_wgts: list[torch.Tensor | nn.Parameter],
        opt_wgts: list[torch.Tensor | nn.Parameter],
        eval_ipt: ActivationsCache | None,
        eval_mod: nn.Module | None,
        ipt_mods: list[nn.Module] | None,
        opt_mods: list[nn.Module] | None,
        orig_ipt_wgts: list[tuple[nn.Parameter, torch.Tensor]] | None,
        orig_opt_wgts: list[tuple[nn.Parameter, torch.Tensor]] | None,
        eval_kwargs: dict[str, tp.Any],
        **kwargs,
    ) -> tp.Any:
        # region Step 1: Calculate the outputs
        if self.objective == SearchBasedCalibObjective.OutputsError:
            assert eval_ipt is not None, "eval_ipt should not be None when objective is OutputsError"
            if orig_ipt_wgts is not None:
                _state_dict_ipt = [p.data for p, _ in orig_ipt_wgts]
                for p, w in orig_ipt_wgts:
                    p.data = w.to(device=p.data.device)
            if orig_opt_wgts is not None:
                _state_dict_opt = [p.data for p, _ in orig_opt_wgts]
                for p, w in orig_opt_wgts:
                    p.data = w.to(device=p.data.device)
            opts: list[torch.Tensor] = []
            for i in range(len(eval_ipt[0].cached)):
                y = eval_mod(
                    *[ipt.cached[i].to(device=ipt.orig_device, non_blocking=True) for ipt in eval_ipt],
                    **eval_kwargs,
                )
                if not isinstance(y, torch.Tensor):
                    y = y[0]
                opts.append(y.to(device=self.opts_device or y.device, non_blocking=True))
            if orig_ipt_wgts is not None:
                for (p, _), s in zip(orig_ipt_wgts, _state_dict_ipt):
                    p.data = s
                del _state_dict_ipt
            if orig_opt_wgts is not None:
                for (p, _), s in zip(orig_opt_wgts, _state_dict_opt):
                    p.data = s
                del _state_dict_opt
            del orig_ipt_wgts, orig_opt_wgts
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
                    ipt_wgts=ipt_wgts,
                    opt_wgts=opt_wgts,
                    ipt_mods=ipt_mods,
                    opt_mods=opt_mods,
                    **kwargs,
                )
                e = [None]
                for i in range(len(eval_ipt[0].cached)):
                    y = eval_mod(
                        *[ipt.cached[i].to(device=ipt.orig_device, non_blocking=True) for ipt in eval_ipt],
                        **eval_kwargs,
                    )
                    if not isinstance(y, torch.Tensor):
                        y = y[0]
                    y = (
                        (y - opts[i].to(device=y.device, non_blocking=True))
                        .to(self.develop_dtype)
                        .pow_(self.calib_config.degree)
                        .sum()
                        .view(-1)
                    )
                    if e[0] is None:
                        e[0] = y
                    else:
                        e[0].add_(y)
                self._recover_mod()
            else:
                raise ValueError(f"Unknown objective {self.objective}")
            # endregion
            self.tell(e)
        return self.get_best()
