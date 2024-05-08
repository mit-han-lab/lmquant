# -*- coding: utf-8 -*-
"""Activation Quantizer module."""

from dataclasses import dataclass

from ..calib.config import QuantTensorType
from .config import ActivationQuantizerConfig
from .tensor import TensorQuantizer

__all__ = ["ActivationQuantizer"]


@dataclass
class ActivationQuantizer(TensorQuantizer):
    """Activation Quantizer class.

    Args:
        key (str): The key of the quantizer. Defaults to ``""``.
        config (ActivationQuantizerConfig): The quantization configuration. Defaults to ``None``.
        tensor_type (QuantTensorType, optional): The type of the tensor to calibrate.
            Defaults to ``QuantTensorType.Inputs``.
        channels_dim (int): The dimension of channels in activations.
        dynamic_range (DynamicRange | tuple[DynamicRange, ...], optional): The dynamic range. Defaults to ``None``.
        default_dtype (torch.dtype | None, optional): The default dtype. Defaults to ``None``.
        develop_dtype (torch.dtype, optional): The develop dtype. Defaults to ``torch.float32``.
    """

    config: ActivationQuantizerConfig = None
    tensor_type: QuantTensorType = QuantTensorType.Inputs

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.tensor_type != QuantTensorType.Weights, "The tensor type cannot be weights."
        assert isinstance(self.channels_dim, int), "The channels dimension must be provided."
