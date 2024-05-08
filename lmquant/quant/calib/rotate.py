# -*- coding: utf-8 -*-
"""Rotation Quantization module."""
import typing as tp

import torch
import torch.nn as nn

from ...utils.math import get_hadamard_matrices, hardmard_transform, random_hadamard_matrix

__all__ = [
    "rotate_in_channels",
    "rotate_out_channels",
    "hadamard_in_channels",
    "get_rotation_matrix",
    "transform_rms_norm_and_linear",
    "transform_layer_norm_to_rms_norm",
    "transform_norm_and_linear",
]


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Llm
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm)."""

    def __init__(self, hidden_size: int, eps=1e-6) -> None:
        """Initialize RMSNorm."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm normalization to hidden states."""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def rotate_in_channels(weight: nn.Parameter, /, *, rotation: torch.Tensor) -> None:
    """Rotate the input channels of a weight matrix."""
    dtype = weight.dtype
    weight.data = torch.matmul(weight.data.to(dtype=torch.float64), rotation.to(weight.device)).to(dtype=dtype)


def rotate_out_channels(weight: nn.Parameter, /, *, rotation: torch.Tensor, bias: nn.Parameter | None = None) -> None:
    """Rotate the output channels of a weight matrix."""
    dtype = weight.dtype
    weight.data = torch.matmul(rotation.T.to(weight.device), weight.data.to(dtype=torch.float64)).to(dtype=dtype)
    if bias is not None:
        bias.data = torch.matmul(rotation.T.to(weight.device), bias.data.to(dtype=torch.float64)).to(dtype=dtype)


def hadamard_in_channels(modules: tp.Iterable[nn.Module], hook_only: bool = False) -> None:
    """Apply Hadamard quantization to the input channels of the modules."""
    in_channels = None
    for module in modules:
        if isinstance(module, nn.Linear):
            device, dtype = module.weight.device, module.weight.dtype
            if in_channels is None:
                in_channels = module.in_features
                hadamard_1, hadamard_K, K = get_hadamard_matrices(in_channels)
                hadamard_1_float = hadamard_1.to(dtype=torch.float32).mul_(
                    1.0 / torch.tensor(in_channels, dtype=torch.float32).sqrt()
                )
                hadamard_K_float = hadamard_K.to(dtype=torch.float32).clone()
                hadamard_1 = hadamard_1_float.to(dtype=hadamard_K.dtype).clone()
            else:
                assert in_channels == module.in_features
            if not hook_only:
                hadamard_1_float = hadamard_1_float.to(device=device)
                hadamard_K_float = hadamard_K_float.to(device=device)
                module.weight.data = hardmard_transform(
                    module.weight.data.float(), hadamard_1_float, hadamard_K_float, K, scaled=True
                ).to(device=device, dtype=dtype)
            hadamard_1 = hadamard_1.to(device=device, dtype=dtype)
            hadamard_K = hadamard_K.to(device=device, dtype=dtype)
            module.hadamard_1 = hadamard_1
            module.hadamard_K = hadamard_K
            module.K = K
            module.register_forward_pre_hook(
                lambda module, input: hardmard_transform(
                    input[0], module.hadamard_1, module.hadamard_K, module.K, scaled=True
                ),
                prepend=True,
            )
        else:
            raise NotImplementedError(f"Module {module} not supported!")


def get_rotation_matrix(num_channels: int, random: bool = True) -> torch.Tensor:
    """Get a random rotation matrix for the given number of channels."""
    if random:
        return random_hadamard_matrix(num_channels)
    else:
        hadamard_1, hadamard_K, K = get_hadamard_matrices(num_channels)
        hadamard_1 = hadamard_1.to(dtype=torch.float64)
        hadamard_K = hadamard_K.to(dtype=torch.float64)
        if K == 1:
            rotation = hadamard_1
        else:
            rotation = torch.kron(hadamard_1, hadamard_K)
        return rotation.mul_(1.0 / torch.tensor(num_channels, dtype=torch.float64).sqrt())


def transform_rms_norm_and_linear(norm: nn.LayerNorm, next_modules: tp.Iterable[nn.Linear]) -> None:
    """Fuse the weight multiplication of rms norm into the next adjacent linear modules.

    Args:
        norm (nn.LayerNorm): normalization module.
        linear_modules (tp.Iterable[nn.Linear]): Linear modules.
    """
    ln_w = norm.weight.data.to(dtype=torch.float64)
    norm.weight.data = torch.ones_like(norm.weight.data)
    if hasattr(norm, "bias") and norm.bias is not None:
        ln_b = norm.bias.data.to(dtype=torch.float64)
        norm.bias = None
    else:
        ln_b = None
    for linear in next_modules:
        dtype = linear.weight.dtype
        fc_w = linear.weight.data.to(dtype=torch.float64)
        linear.weight.data = (fc_w * ln_w).to(dtype=dtype)
        if ln_b is not None:
            if linear.bias is None:
                linear.bias = nn.Parameter(torch.zeros(linear.out_features, dtype=dtype, device=linear.weight.device))
            linear.bias.data = (linear.bias.data.to(dtype=torch.float64) + torch.matmul(fc_w, ln_b)).to(dtype=dtype)


def transform_layer_norm_to_rms_norm(
    parent: nn.Module,
    norm_name: str,
    prev_modules: tp.Iterable[nn.Linear],
    prev_out_channels_dims: int | tp.Iterable[int] = 0,
) -> None:
    """Transform LayerNorm to RMSNorm.

    Args:
        parent (nn.Module): Parent module.
        norm_name (str): Name of the normalization module.
        prev_modules (tp.Iterable[nn.Linear]): Previous adjacent linear modules.
        prev_out_channels_dims (int | tp.Iterable[int], optional): Output channels dimension of the previous modules.
            Defaults to ``0``.
    """
    norm = getattr(parent, norm_name)
    assert isinstance(norm, nn.LayerNorm)
    assert len(norm.normalized_shape) == 1, f"LayerNorm's #dims must be 1, got {len(norm.normalized_shape)}"
    assert norm.bias is None, "LayerNorm's bias must be None"
    # region move substract mean to the previous linear modules
    assert len(prev_modules) > 0, "No previous modules found"
    if isinstance(prev_out_channels_dims, int):
        prev_out_channels_dims = [prev_out_channels_dims] * len(prev_modules)
    for module, dim in zip(prev_modules, prev_out_channels_dims):
        if isinstance(module, nn.LayerNorm):
            module.bias = None
        else:
            if isinstance(module, nn.Linear):
                assert dim == 0, "Linear module's output channels dimension is 0"
            elif isinstance(module, nn.Embedding):
                assert dim == 1, "Embedding module's output channels dimension is 1"
            dtype = module.weight.dtype
            W = module.weight.data.to(dtype=torch.float64)
            module.weight.data = W.sub_(W.mean(dim=dim, keepdim=True)).to(dtype=dtype)
            if hasattr(module, "bias") and module.bias is not None:
                B = module.bias.data.to(dtype=torch.float64)
                module.bias.data = B.sub_(B.mean()).to(dtype=dtype)
    # endregion
    # region replace LayerNorm with RMSNorm
    rms = RMSNorm(hidden_size=norm.normalized_shape[0], eps=norm.eps)
    rms.weight.data = norm.weight.data
    setattr(parent, norm_name, rms)
    # endregion


def transform_norm_and_linear(
    parent: nn.Module,
    norm_name: str,
    next_modules: tp.Iterable[nn.Linear],
    prev_modules: tp.Iterable[nn.Linear] | None = None,
    prev_out_channels_dims: int | tp.Iterable[int] = 0,
):
    """Transform the normalization module and the next adjacent linear modules.

    Args:
        parent (nn.Module): Parent module.
        norm_name (str): Name of the normalization module.
        next_modules (tp.Iterable[nn.Linear]): Next adjacent linear modules.
        prev_modules (tp.Iterable[nn.Linear]): Previous adjacent linear modules.
        prev_out_channels_dims (int | tp.Iterable[int], optional): Output channels dimension of the previous modules.
            Defaults to ``0``.
    """
    norm = getattr(parent, norm_name)
    transform_rms_norm_and_linear(norm, next_modules)
    if isinstance(norm, nn.LayerNorm):
        transform_layer_norm_to_rms_norm(parent, norm_name, prev_modules, prev_out_channels_dims)
