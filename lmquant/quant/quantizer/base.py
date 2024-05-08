# -*- coding: utf-8 -*-
"""Quantization kernels."""

import typing as tp
from dataclasses import MISSING, dataclass

import torch
import torch.nn as nn
import torch.utils.hooks

from ...dataset import ActivationsCache
from ..data.range import DynamicRange, QuantRange, RangeBound
from ..data.tensor import QuantTensor
from ..functional.config import QuantKernelConfig
from ..functional.quantize import quantize
from .config import QuantizerConfig, QuantizerKernelConfig

__all__ = ["Quantizer"]


@dataclass
class Quantizer:
    """Tensor Kernel Quantizer class.

    Args:
        key (str): The key of the quantizer. Defaults to ``""``.
        config (QuantizerConfig): The quantizer configuration. Defaults to ``None``.
        kernel_config (QuantizerKernelConfig | None, optional): The quantizer kernel configuration.
            Defaults to ``None``.
        channels_dim (int | None, optional): The dimension of channels in activations. Defaults to ``None``.
        scale (torch.Tensor | tuple[torch.Tensor, torch.Tensor], optional): The scale tensor. Defaults to ``None``.
        zero (torch.Tensor, optional): The zero point tensor. Defaults to ``None``.
        dynamic_range (DynamicRange | tuple[DynamicRange, ...], optional): The dynamic range. Defaults to ``None``.
        quant_range (QuantRange | None, optional): The quantization range. Defaults to ``None``.
        range_bound (RangeBound | None, optional): The range bound. Defaults to ``None``.
        round_delta (torch.Tensor | None, optional): The round alpha. Defaults to ``None``.
        default_dtype (torch.dtype | None, optional): The default dtype. Defaults to ``None``.
        develop_dtype (torch.dtype, optional): The develop dtype. Defaults to ``torch.float32``.
    """

    config: QuantizerConfig = None
    kernel_config: QuantizerKernelConfig | None = None
    channels_dim: int | None = None
    scale: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = None
    zero: torch.Tensor | None = None
    dynamic_range: DynamicRange | tuple[DynamicRange, ...] = None
    quant_range: QuantRange | None = None
    range_bound: RangeBound | None = None
    round_delta: torch.Tensor | None = None
    default_dtype: torch.dtype | None = None
    develop_dtype: torch.dtype = torch.float32
    key: str = ""

    @property
    def enabled(self) -> bool:
        """Whether the quantizer is enabled."""
        return self.config is not None and self.config.enabled_for(key=self.key)

    def quantize(
        self,
        tensor: torch.Tensor,
        *,
        inputs: ActivationsCache = None,
        return_with_dequant: bool = True,
        return_with_quant: bool = False,
        kernel_config: QuantizerKernelConfig | QuantKernelConfig | None = MISSING,
        channels_dim: int | None = MISSING,
        scale: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = MISSING,
        zero: torch.Tensor = MISSING,
        dynamic_range: DynamicRange | tuple[DynamicRange, DynamicRange] = MISSING,
        quant_range: QuantRange = MISSING,
        range_bound: RangeBound = MISSING,
        round_delta: torch.Tensor = MISSING,
        default_dtype: torch.dtype | None = MISSING,
        develop_dtype: torch.dtype = MISSING,
    ) -> QuantTensor:
        """Quantize a tensor.

        Args:
            tensor (torch.Tensor): The input tensor.
            kernel_config (QuantizerKernelConfig | QuantKernelConfig | None, optional):
                The quantization kernel config. Defaults to ``MISSING``.
            channels_dim (int, optional): The dimension of channels in activations. Defaults to ``MISSING``.
            scale (torch.Tensor | tuple[torch.Tensor, torch.Tensor], optional): The scale tensor.
                Defaults to ``MISSING``.
            zero (torch.Tensor, optional): The zero tensor. Defaults to ``MISSING``.
            dynamic_range (DynamicRange | tuple[DynamicRange, DynamicRange], optional): The dynamic range.
                Defaults to ``MISSING``.
            quant_range (QuantRange, optional): The quantization range. Defaults to ``None``.
            range_bound (RangeBound, optional): The dynamic range bound. Defaults to ``None``.
            inputs (ActivationsCache, optional): The input activations cache. Defaults to ``None``.
            return_with_dequant (bool, optional): Whether to return with dequantized tensor. Defaults to ``True``.
            return_with_quant (bool, optional): Whether to return with quantized tensor. Defaults to ``False``.
            default_dtype (torch.dtype | None, optional): The default dtype. Defaults to ``MISSING``.
            develop_dtype (torch.dtype, optional): The develop dtype. Defaults to ``MISSING``.

        Returns:
            QuantTensor: The quantized tensor.
        """
        if not self.enabled:
            return QuantTensor(tensor, tensor, view_shape=tensor.shape)
        channels_dim = self.channels_dim if channels_dim is MISSING else channels_dim
        scale = self.scale if scale is MISSING else scale
        zero = self.zero if zero is MISSING else zero
        dynamic_range = self.dynamic_range if dynamic_range is MISSING else dynamic_range
        quant_range = self.quant_range if quant_range is MISSING else quant_range
        range_bound = self.range_bound if range_bound is MISSING else range_bound
        round_delta = self.round_delta if round_delta is MISSING else round_delta
        default_dtype = self.default_dtype if default_dtype is MISSING else default_dtype
        develop_dtype = self.develop_dtype if develop_dtype is MISSING else develop_dtype
        if kernel_config is MISSING:
            kernel_config = self.kernel_config.specialize_for(self.key) if self.kernel_config is not None else None
        elif isinstance(kernel_config, QuantizerKernelConfig):
            kernel_config = kernel_config.specialize_for(self.key)
        assert isinstance(kernel_config, (QuantKernelConfig, type(None)))
        return quantize(
            tensor,
            self.config,
            kernel_config=kernel_config,
            channels_dim=channels_dim,
            scale=scale,
            zero=zero,
            dynamic_range=dynamic_range,
            quant_range=quant_range,
            range_bound=range_bound,
            round_delta=round_delta,
            inputs=inputs,
            return_with_dequant=return_with_dequant,
            return_with_quant=return_with_quant,
            default_dtype=default_dtype,
            develop_dtype=develop_dtype,
        )

    def unpack_inputs_in_inputs_hook(
        self, module: nn.Module, input_args: tuple[torch.Tensor, ...], input_kwargs: dict[str, tp.Any]
    ) -> torch.Tensor:
        """Unpack inputs in inputs hook.

        Args:
            module (nn.Module): Module.
            input_args (tuple[torch.Tensor, ...]): Input arguments.
            input_kwargs (dict[str, tp.Any]): Input keyword arguments.


        Returns:
            torch.Tensor: The unpacked input tensor.
        """
        return input_args[0]

    def unpack_outputs_in_outputs_hook(
        self,
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Unpack outputs in outputs hook.

        Args:
            module (nn.Module): Module.
            input_args (tuple[torch.Tensor, ...]): Input arguments.
            input_kwargs (dict[str, tp.Any]): Input keyword arguments.
            output (torch.Tensor): Output tensor.

        Returns:
            torch.Tensor: The unpacked output tensor.
        """
        return output

    def repack_inputs_in_inputs_hook(
        self,
        tensor: torch.Tensor,
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, tp.Any]]:
        """Repack inputs in inputs hook.

        Args:
            tensor (torch.Tensor): The input tensor.
            module (nn.Module): Module.
            input_args (tuple[torch.Tensor, ...]): Input arguments.
            input_kwargs (dict[str, tp.Any]): Input keyword arguments.

        Returns:
            tuple[tuple[torch.Tensor, ...], dict[str, tp.Any]]: The repacked input arguments and keyword arguments.
        """
        return (tensor, *input_args[1:]), input_kwargs

    def repack_outputs_in_outputs_hook(
        self,
        tensor: torch.Tensor,
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
        output: tp.Any,
    ) -> tp.Any:
        """Repack outputs in outputs hook.

        Args:
            tensor (torch.Tensor): The output tensor.
            module (nn.Module): Module.
            input_args (tuple[torch.Tensor, ...]): Input arguments.
            input_kwargs (dict[str, tp.Any]): Input keyword arguments.
            output (tp.Any): Output.

        Returns:
            tp.Any: The repacked output.
        """
        return tensor

    def quantize_module_inputs(
        self, module: nn.Module, quantize_fn: tp.Callable[[torch.Tensor, int], torch.Tensor] | None = None
    ) -> torch.utils.hooks.RemovableHandle:
        """Quantize module input activations.

        Args:
            module (nn.Module): Module to quantize.
            quantize_fn (tp.Callable[[torch.Tensor, int], torch.Tensor], optional): Function to quantize the inputs,
                which takes the tensor and channels dimension as input. Defaults to ``None``.

        Returns:
            torch.utils.hooks.RemovableHandle: The hook handle.
        """

        def hook(
            module: nn.Module,
            input_args: tuple[torch.Tensor, ...],
            input_kwargs: dict[str, tp.Any],
        ):
            if not self.enabled:
                return input_args, input_kwargs
            x = self.unpack_inputs_in_inputs_hook(module, input_args, input_kwargs)
            assert isinstance(x, torch.Tensor), "Input tensor must be a torch.Tensor."
            if quantize_fn is not None:
                x = quantize_fn(x, self.channels_dim)
            else:
                x = self.quantize(x).data
            return self.repack_inputs_in_inputs_hook(x, module, input_args, input_kwargs)

        return module.register_forward_pre_hook(hook, with_kwargs=True)

    def quantize_module_outputs(
        self, module: nn.Module, quantize_fn: tp.Callable[[torch.Tensor, int], torch.Tensor] | None = None
    ) -> torch.utils.hooks.RemovableHandle:
        """Quantize module output activations.

        Args:
            module (nn.Module): Module to quantize.
            quantize_fn (tp.Callable[[torch.Tensor, int], torch.Tensor], optional): Function to quantize the outputs,
                which takes the tensor and channels dimension as input. Defaults to ``None``.

        Returns:
            torch.utils.hooks.RemovableHandle: The hook handle.
        """

        def hook(
            module: nn.Module,
            input_args: tuple[torch.Tensor, ...],
            input_kwargs: dict[str, tp.Any],
            output: torch.Tensor,
        ):
            if not self.enabled:
                return output
            x = self.unpack_outputs_in_outputs_hook(module, input_args, input_kwargs, output)
            assert isinstance(x, torch.Tensor), "Output tensor must be a torch.Tensor."
            if quantize_fn is not None:
                x = quantize_fn(x, self.channels_dim)
            else:
                x = self.quantize(x).data
            return self.repack_outputs_in_outputs_hook(x, module, input_args, input_kwargs, output)

        return module.register_forward_hook(hook, with_kwargs=True)
