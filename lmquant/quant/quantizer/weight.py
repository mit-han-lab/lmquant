# -*- coding: utf-8 -*-
"""Weight Quantizer module."""

import typing as tp
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ...dataset.cache import ActivationsCache
from ..calib.config import QuantTensorType
from ..data.range import DynamicRange
from .config import WeightQuantizerConfig
from .tensor import TensorQuantizer

__all__ = ["WeightQuantizer"]


@dataclass
class WeightQuantizer(TensorQuantizer):
    """Weight Quantizer class.

    Args:
        key (str): The key of the quantizer. Defaults to ``""``.
        config (WeightQuantizerConfig): The quantization configuration. Defaults to ``None``.
        dynamic_range (DynamicRange | tuple[DynamicRange, ...], optional): The dynamic range. Defaults to ``None``.
        default_dtype (torch.dtype | None, optional): The default dtype. Defaults to ``None``.
        develop_dtype (torch.dtype, optional): The develop dtype. Defaults to ``torch.float32``.
    """

    config: WeightQuantizerConfig = None
    channels_dim: int | None = field(init=False, default=None)
    tensor_type: QuantTensorType = field(init=False, default=QuantTensorType.Weights)

    def calibrate_dynamic_range(
        self,
        module: nn.Module,
        inputs: ActivationsCache,
        eval_inputs: ActivationsCache | None = None,
        eval_module: nn.Module | None = None,
        eval_kwargs: dict[str, tp.Any] | None = None,
        weight: nn.Parameter | None = None,
    ) -> DynamicRange | tuple[DynamicRange, ...]:
        """Calibrate the dynamic range.

        Args:
            module (nn.Module): The module to calibrate.
            inputs (ActivationsCache): The input activations cache.
            eval_inputs (ActivationsCache, optional): The cache of the input activations for evaluation.
                If not provided, the ``inputs`` cache will be used. Defaults to ``None``.
            eval_module (nn.Module, optional): The module to evaluate the quantization error.
                If not provided, the module to calibrate will be used. Defaults to ``None``.
            eval_kwargs (dict[str, tp.Any], optional): The keyword arguments for evaluation.
                Defaults to ``None``.
            key (str, optional): The key of the module. Defaults to ``""``.
            weight (nn.Parameter, optional): The weight tensor. If not provided, the weight tensor
                of the module will be used. Defaults to ``None``.
        """
        return super().calibrate_dynamic_range(
            modules=[module],
            weights=[weight] if weight is not None else None,
            activations=inputs,
            eval_inputs=eval_inputs,
            eval_module=eval_module,
            eval_kwargs=eval_kwargs,
        )
