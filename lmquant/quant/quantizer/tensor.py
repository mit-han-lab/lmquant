# -*- coding: utf-8 -*-
"""Tensor Quantizer module."""

import typing as tp
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ...dataset.cache import ActivationsCache
from ..calib.calibrator.range import DynamicRangeCalibrator
from ..calib.config import QuantTensorType
from ..data.range import DynamicRange, QuantRange
from .base import Quantizer
from .config import QuantizerKernelConfig, TensorQuantizerConfig

__all__ = ["TensorQuantizer"]


@dataclass
class TensorQuantizer(Quantizer):
    """Range-based quantizer class.

    Args:
        key (str): The key of the quantizer. Defaults to ``""``.
        config (TensorQuantizerConfig): The quantization configuration. Defaults to ``None``.
        channels_dim (int | None, optional): The dimension of channels in activations. Defaults to ``None``.
        dynamic_range (DynamicRange | tuple[DynamicRange, ...], optional): The dynamic range. Defaults to ``None``.
        quant_range (QuantRange | None, optional): The quantization range. Defaults to ``None``.
        range_bound (RangeBound | None, optional): The range bound. Defaults to ``None``.
        default_dtype (torch.dtype | None, optional): The default dtype. Defaults to ``None``.
        develop_dtype (torch.dtype, optional): The develop dtype. Defaults to ``torch.float32``.
    """

    config: TensorQuantizerConfig = None
    kernel_config: QuantizerKernelConfig | None = field(init=False, default=None)
    tensor_type: QuantTensorType = QuantTensorType.Weights

    def __post_init__(self) -> None:
        self.kernel_config = self.config.calib_kernel

    def calibrate_dynamic_range(
        self,
        modules: list[nn.Module],
        activations: ActivationsCache,
        weights: list[nn.Parameter] = None,
        eval_inputs: ActivationsCache | None = None,
        eval_module: nn.Module | None = None,
        eval_kwargs: dict[str, tp.Any] | None = None,
        orig_weights: list[tuple[nn.Parameter, torch.Tensor]] | None = None,
    ) -> DynamicRange | tuple[DynamicRange, ...]:
        """Calibrate the dynamic range.

        Args:
            modules (list[nn.Module]): The modules to calibrate.
            weights (list[nn.Parameter]): The weights to calibrate. If not provided (``None``), the weights
                of the modules will be used.
            activations (ActivationsCache): The inputs cache if the tensor type is not outputs, or the outputs
                cache if the tensor type is outputs.
            eval_inputs (ActivationsCache, optional): The cache of the inputs for evaluation.
                If not provided, the ``activations`` cache will be used. Defaults to ``None``.
            eval_module (nn.Module, optional): The module to evaluate the quantization error.
                If not provided, the module to calibrate will be used. Defaults to ``None``.
            eval_kwargs (dict[str, tp.Any], optional): The keyword arguments for evaluation. Defaults to ``None``.
            orig_weights (list[tuple[nn.Parameter, torch.Tensor]], optional): The original weights.
                Defaults to ``None``.
        """
        if self.config is None or self.config.dtype is None:
            self.dynamic_range = DynamicRange()
            return self.dynamic_range
        if not self.config.enabled_calib_range or not self.config.calib_range.enabled_for(key=self.key):
            self.dynamic_range = DynamicRange()
            return self.dynamic_range
        if not self.config.calib_range.needs_search and not self.config.static:
            if self.config.calib_range.ratio != 1.0:
                dynamic_range = DynamicRange(ratio=self.config.calib_range.ratio)
                if self.config.compute_dtype is None:
                    self.dynamic_range = dynamic_range
                else:
                    self.dynamic_range = (dynamic_range, dynamic_range)
                return self.dynamic_range
            else:
                self.dynamic_range = None
                return self.dynamic_range
        if weights is None:
            weights = [module.weight for module in modules if hasattr(module, "weight")]
        if self.tensor_type == QuantTensorType.Weights:
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
            assert activations.num_sources == 1, "only one source is supported for activation quantization calibration"
        if self.tensor_type != QuantTensorType.Outputs:
            ipt_wgts, ipts, ipt_mods, orig_ipt_wgts = weights, activations, modules, orig_weights
            opt_wgts, opts, opt_mods, orig_opt_wgts = [], None, None, None
        else:
            ipt_wgts, ipts, ipt_mods, orig_ipt_wgts = [], None, None, None
            opt_wgts, opts, opt_mods, orig_opt_wgts = weights, activations, modules, orig_weights
        if self.config.compute_dtype is None:
            self.dynamic_range = DynamicRangeCalibrator(
                tensor_type=self.tensor_type,
                calib_config=self.config.calib_range,
                static=self.config.static,
                quantizer=self,
            ).calibrate(
                ipt_wgts=ipt_wgts,
                opt_wgts=opt_wgts,
                ipts=ipts,
                opts=opts,
                eval_ipt=eval_inputs,
                eval_mod=eval_module,
                eval_kwargs=eval_kwargs,
                ipt_mods=ipt_mods,
                opt_mods=opt_mods,
                orig_ipt_wgts=orig_ipt_wgts,
                orig_opt_wgts=orig_opt_wgts,
            )
            return self.dynamic_range
        if self.tensor_type == QuantTensorType.Weights:
            tensor = weights[0].detach().data
        else:
            assert len(activations[0].cached) == 0, "Only one tensor is supported for activation quantization"
            tensor = activations[0].cached[0].detach().data
        if self.config.saturate_compute_dtype:
            compute_quant_range = QuantRange.build(self.config.compute_dtype)
        else:
            compute_quant_range = QuantRange.build_protective(self.config.compute_dtype, self.config.dtype)
        compute_quantizer = Quantizer(
            config=self.config.get_compute_level_config(),
            kernel_config=None,
            quant_range=compute_quant_range,
            default_dtype=self.default_dtype,
            develop_dtype=self.develop_dtype,
        )
        compute_dynamic_range = DynamicRangeCalibrator(
            tensor_type=self.tensor_type,
            calib_config=self.config.calib_range,
            static=self.config.static,
            quantizer=compute_quantizer,
        ).calibrate(
            ipt_wgts=ipt_wgts,
            opt_wgts=opt_wgts,
            ipts=ipts,
            opts=opts,
            eval_ipt=eval_inputs,
            eval_mod=eval_module,
            eval_kwargs=eval_kwargs,
            ipt_mods=ipt_mods,
            opt_mods=opt_mods,
            orig_ipt_wgts=orig_ipt_wgts,
            orig_opt_wgts=orig_opt_wgts,
        )
        result = compute_quantizer.quantize(
            tensor, dynamic_range=compute_dynamic_range, return_with_dequant=False, return_with_quant=True
        )
        result._dequantized = None
        result._quantized = None
        torch.cuda.empty_cache()
        store_quantizer = Quantizer(
            config=self.config.get_store_level_config(),
            kernel_config=self.kernel_config,
            range_bound=compute_quant_range,
            default_dtype=self.default_dtype,
            develop_dtype=self.develop_dtype,
        )
        store_dynamic_range = DynamicRangeCalibrator(
            tensor_type=self.tensor_type,
            calib_config=self.config.calib_range,
            static=self.config.static,
            quantizer=store_quantizer,
            pre_scale=result.scale.data,
        ).calibrate(
            ipt_wgts=ipt_wgts,
            opt_wgts=opt_wgts,
            ipts=ipts,
            opts=opts,
            eval_ipt=eval_inputs,
            eval_mod=eval_module,
            eval_kwargs=eval_kwargs,
            ipt_mods=ipt_mods,
            opt_mods=opt_mods,
            orig_ipt_wgts=orig_ipt_wgts,
            orig_opt_wgts=orig_opt_wgts,
        )
        self.dynamic_range = (compute_dynamic_range, store_dynamic_range)
        return self.dynamic_range

    def state_dict(self, device: torch.device = torch.device("cpu")) -> dict[str, torch.Tensor | float | None]:
        """Get the state dictionary.

        Args:
            device (torch.device, optional): The device. Defaults to ``torch.device("cpu")``.

        Returns:
            dict[str, torch.Tensor | float | None]: The state dictionary.
        """
        if isinstance(self.dynamic_range, DynamicRange):
            dynamic_range = (self.dynamic_range,)
        else:
            dynamic_range = self.dynamic_range
        results: dict[str, torch.Tensor | float | None] = {}
        if dynamic_range is None:
            results["num_dynamic_range"] = 0
        else:
            results["num_dynamic_range"] = len(dynamic_range)
            for i, dr in enumerate(dynamic_range):
                if dr is None:
                    results[f"dynamic_range.{i}"] = None
                else:
                    assert isinstance(dr, DynamicRange), f"Invalid dynamic range: {dr}"
                    for k, v in dr.to_dict().items():
                        results[f"dynamic_range.{i}.{k}"] = v.to(device=device) if isinstance(v, torch.Tensor) else v
        return results

    def load_state_dict(
        self, state_dict: dict[str, torch.Tensor | float | None], device: torch.device = torch.device("cpu")
    ) -> None:
        """Load the state dictionary.

        Args:
            state_dict (dict[str, torch.Tensor | float | None]): The state dictionary.
            device (torch.device, optional): The device. Defaults to ``torch.device("cpu")``.
        """
        num_dynamic_range = state_dict["num_dynamic_range"]
        if num_dynamic_range == 0:
            dynamic_range = None
        else:
            dynamic_range_dict: list[dict[str, torch.Tensor | float | None]] = [{} for _ in range(num_dynamic_range)]
            for k, v in state_dict.items():
                if k.startswith("dynamic_range."):
                    ks = k.split(".")
                    if len(ks) == 3:
                        i, k = int(ks[1]), ks[2]
                        dynamic_range_dict[i][k] = v.to(device=device) if isinstance(v, torch.Tensor) else v
                    else:
                        assert len(ks) == 2, f"Invalid key: {k}"
                        dynamic_range_dict[i] = None
            dynamic_range = tuple(dr if dr is None else DynamicRange.from_dict(dr) for dr in dynamic_range_dict)
            if len(dynamic_range) == 1:
                dynamic_range = dynamic_range[0]
        self.dynamic_range = dynamic_range
