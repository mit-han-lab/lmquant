# -*- coding: utf-8 -*-
"""Quantization dynamic range calibration module."""

import gc
import logging

import torch
import torch.nn as nn

from ....dataset import ActivationsCache
from ....utils.math import root_
from ...data.range import DynamicRange
from ...data.utils.shape import infer_view_shape
from ...quantizer.base import Quantizer
from ..config import DynamicRangeCalibConfig, QuantTensorType, SearchBasedCalibGranularity
from .base import SearchBasedQuantCalibrator

__all__ = ["DynamicRangeCalibrator"]


class DynamicRangeCalibrator(SearchBasedQuantCalibrator[DynamicRangeCalibConfig, DynamicRange]):
    """The quantization dynamic range calibrator."""

    def __init__(
        self,
        tensor_type: QuantTensorType,
        calib_config: DynamicRangeCalibConfig,
        static: bool,
        quantizer: Quantizer,
        pre_scale: torch.Tensor | None = None,
        allow_kernel_calib: bool = True,
    ) -> None:
        """Initialize the calibrator.

        Args:
            tensor_type (QuantTensorType): The tensor type.
            calib_config (DynamicRangeCalibConfig): The calibration configuration.
            static (bool): Whether the dynamic range is static.
            quantizer (Quantizer): The quantizer.
            pre_scale (torch.Tensor | None): The pre-scale value.
            allow_kernel_calib (bool): Whether to allow kernel calibration.
        """
        super().__init__(
            tensor_type=tensor_type,
            calib_config=calib_config,
            wgts_quantizer=quantizer if tensor_type == QuantTensorType.Weights else None,
            ipts_quantizer=quantizer if tensor_type == QuantTensorType.Inputs else None,
            opts_quantizer=quantizer if tensor_type == QuantTensorType.Outputs else None,
            allow_kernel_calib=allow_kernel_calib,
            develop_dtype=quantizer.develop_dtype,
        )
        assert self.needs_quant_tnsr, "The tensor should be quantized."
        self.static = static
        self.pre_scale = pre_scale
        self.ratios = self.calib_config.get_ratios()
        self.num_iters = len(self.ratios)

    @property
    def population_size(self) -> int:
        """Return the population size of the current iteration."""
        return len(self.ratios[self.iter])

    def _reset(  # noqa: C901
        self,
        ipt_wgts: list[torch.Tensor | nn.Parameter],
        ipts: ActivationsCache,
        opts: ActivationsCache,
        **kwargs,
    ) -> None:
        """Reset the calibrator."""
        self.base_range: DynamicRange = DynamicRange()
        self.best_range: DynamicRange = DynamicRange()
        self.best_error: torch.Tensor = None
        self.error_history: list[tuple[float, float]] = []
        self._ipts_for_wgts_quant = None
        self.device = None
        if self.tensor_type == QuantTensorType.Weights:
            assert len(ipt_wgts) == 1, "The weight should be a single tensor."
            w = ipt_wgts[0].data
            assert isinstance(w, torch.Tensor), "The weight should be a tensor."
            tensors = [w]
            self._ipts_for_wgts_quant = ipts if self.allow_kernel_calib else None
            self.device = w.device
        elif self.tensor_type == QuantTensorType.Inputs:
            assert ipts.num_sources == 1, f"Only one input source is allowed, got {ipts.num_sources}"
            tensors = [x.view(-1, *x.shape[ipts[0].channels_dim :]) for x in ipts[0].cached]
            self.device = ipts[0].orig_device
        else:
            assert opts.num_sources == 1, f"Only one output source is allowed, got {opts.num_sources}"
            tensors = [x.view(-1, *x.shape[opts[0].channels_dim :]) for x in opts[0].cached]
            self.device = opts[0].orig_device
        shape = tensors[0].shape
        view_shape = infer_view_shape(
            shape,
            self.tnsr_quantizer.config.largest_group_shape,
            skip_first_dim=self.tensor_type != QuantTensorType.Weights,
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
        if self.static:
            if self.pre_scale is not None:
                tensors = [self._preprocess_with_pre_scale(t) for t in tensors]
            tensors = [t.view(view_shape).to(dtype=self.develop_dtype) for t in tensors]
            self.base_range = DynamicRange.build(
                tensors,
                has_zero_point=self.tnsr_quantizer.config.dtype.has_zero_point,
                is_float=self.tnsr_quantizer.config.dtype.is_float,
            )
            gc.collect()
            torch.cuda.empty_cache()

    def get_best(self) -> DynamicRange:
        """Get the best candidate.

        Returns:
            DynamicRange: The best candidate.
        """
        if not self.static:
            self.best_range.ratio = self.best_range.ratio.view(self.ratio_view_shape)
        return self.best_range

    def _ask(self) -> DynamicRange:
        """Ask for the next candidate.

        Returns:
            DynamicRange: The next candidate.
        """
        ratio = self.ratios[self.iter][self.candidate_id]
        if self.static:
            return self.base_range.scale(
                ratio=ratio,
                has_zero_point=self.tnsr_quantizer.config.dtype.has_zero_point,
                is_float=self.tnsr_quantizer.config.dtype.is_float,
            )
        else:
            return DynamicRange(ratio=ratio)

    def _tell(self, error: list[torch.Tensor]) -> None:  # noqa: C901
        """Tell the error of the last candidate and update the best candidate.

        Args:
            errors (list[torch.Tensor]): The error of the last candidate.
        """
        assert len(error) == 1, "The error should only have one value."
        error = error[0]
        assert isinstance(error, torch.Tensor)
        assert error.shape == self.error_shape, f"Error shape {error.shape} != {self.error_shape}."
        assert isinstance(self.candidate, DynamicRange)
        # candidate_range, candidate_scale = self.candidate
        if self.best_error is None:
            self.best_error = error
            if self.static:
                self.best_range.min = self.candidate.min
                self.best_range.max = self.candidate.max
            else:
                self.best_range.ratio = torch.full(
                    size=self.ratio_shape, fill_value=self.candidate.ratio, device=self.device, dtype=self.develop_dtype
                )
        elif error.numel() > 1:
            pos = error < self.best_error
            self.best_error[pos] = error[pos]
            if self.static:
                if self.error_shape.numel() != self.range_shape.numel():
                    pos = pos.view(self.pos_view_shape).expand(*self.range_shape)
                else:
                    pos = pos.view(self.range_shape)
                self.best_range.max[pos] = self.candidate.max[pos]
                if isinstance(self.candidate.min, torch.Tensor):
                    self.best_range.min[pos] = self.candidate.min[pos]
            else:
                self.best_range.ratio[pos.view(self.ratio_shape)] = self.candidate.ratio
        elif error < self.best_error:
            self.best_error = error
            if self.static:
                self.best_range.min = self.candidate.min
                self.best_range.max = self.candidate.max
            else:
                assert isinstance(self.candidate.ratio, float)
                self.best_range.ratio.fill_(self.candidate.ratio)
        if self.logger.level <= logging.DEBUG:
            self.error_history.append(
                (
                    root_(error.to(torch.float64).sum(), self.calib_config.degree).item(),
                    root_(self.best_error.to(torch.float64).sum(), self.calib_config.degree).item(),
                )
            )
            if self.is_last_candidate_in_iter():
                stype_id = self.iter
                scales, population_size = self.ratios[stype_id], self.population_size
                for i in range(0, population_size, 5):
                    self.logger.debug(
                        "  - range scale = [%s]",
                        ", ".join(f"{scales[j]:10.4f}" for j in range(i, min(i + 5, population_size))),
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
                        root_(self.best_error.to(torch.float64).sum(), self.calib_config.degree).item(),
                    )

    def _preprocess_with_pre_scale(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(self.pre_view_shape)
        t = t.to(dtype=self.develop_dtype) if t.dtype != self.develop_dtype else t.clone()
        t = t.div_(self.pre_scale)
        if self.tnsr_quantizer.range_bound is not None and self.tnsr_quantizer.range_bound.is_set():
            t = t.clamp_(min=self.tnsr_quantizer.range_bound.min, max=self.tnsr_quantizer.range_bound.max)
        return t

    def _process_wxy(self, tensor: torch.Tensor, channels_dim: int) -> torch.Tensor:
        shape, dtype = tensor.shape, tensor.dtype
        if self.pre_scale is not None:
            tensor = self._preprocess_with_pre_scale(tensor).view(shape)
        tensor = self.tnsr_quantizer.quantize(
            tensor,
            kernel_config=self.kernel_config,
            channels_dim=channels_dim,
            dynamic_range=self.candidate,
            inputs=self._ipts_for_wgts_quant if self.allow_kernel_calib else None,
            default_dtype=dtype,
            develop_dtype=self.develop_dtype,
        ).data
        if self.pre_scale is not None:
            tensor = tensor.view(self.pre_view_shape).mul_(self.pre_scale).to(dtype)
        tensor = tensor.view(shape)
        return tensor

    def _process_x_in_xw(self, x: torch.Tensor, channels_dim: int, **kwargs) -> torch.Tensor:
        if self.tensor_type != QuantTensorType.Inputs:
            return x
        return self._process_wxy(x, channels_dim)

    def _process_y_in_yx(self, y: torch.Tensor, channels_dim: int, **kwargs) -> torch.Tensor:
        if self.tensor_type != QuantTensorType.Outputs:
            return y
        return self._process_wxy(y, channels_dim)

    def _process_x_in_yx(self, x: torch.Tensor, channels_dim: int, **kwargs) -> torch.Tensor:
        raise RuntimeError("_process_x_in_yx should not be called in DynamicRangeCalibrator.")

    def _process_w_in_xw(self, w: torch.Tensor) -> torch.Tensor:
        if self.tensor_type != QuantTensorType.Weights:
            return w
        return self._process_wxy(w, channels_dim=None)

    def _process_w_in_yx(self, w: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("_process_w_in_yx should not be called in DynamicRangeCalibrator.")
