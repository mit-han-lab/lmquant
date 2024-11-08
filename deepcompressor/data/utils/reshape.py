# -*- coding: utf-8 -*-
"""Type hints used in deepcompressor."""

import torch
import torch.nn.functional as F

__all__ = [
    "ReshapeFn",
    "LinearReshapeFn",
    "ConvInputReshapeFn",
    "ConvOutputReshapedFn",
    "AttentionInputReshapeFn",
]


class ReshapeFn:
    """Reshape function."""

    def __call__(self, x: torch.Tensor, /, ic_last: bool = True) -> torch.Tensor:
        """Reshape input tensor to the desired shape used for GEMM.

        Args:
            x (`torch.Tensor`):
                Input tensor.
            ic_last (`bool`, *optional*, defaults to `True`):
                Whether input channel is the last dimension.

        Returns:
            `torch.Tensor`:
                Reshaped tensor.
        """
        return x


class LinearReshapeFn(ReshapeFn):
    """Inputs reshape function for linear layers."""

    def __call__(self, x: torch.Tensor, /, ic_last: bool = True) -> torch.Tensor:
        """Reshape input tensor to the desired 2D shape used for GEMM.

        Args:
            x (`torch.Tensor`):
                Input tensor.
            ic_last (`bool`, *optional*, defaults to `True`):
                Whether input channel is the last dimension.

        Returns:
            `torch.Tensor`:
                Reshaped tensor.
        """
        return x.view(-1, x.shape[-1]).permute(int(not ic_last), int(ic_last))


class ConvInputReshapeFn(ReshapeFn):
    """Inputs reshape function for convolutional layers."""

    def __init__(
        self, kernel_size: tuple[int, ...], padding: tuple[int, ...], stride: tuple[int, ...], dilation: tuple[int, ...]
    ) -> None:
        """Initialize the reshape function.

        Args:
            kernel_size (`tuple[int, ...]`):
                Kernel size.
            padding (`tuple[int, ...]`):
                Padding.
            stride (`tuple[int, ...]`):
                Stride.
            dilation (`tuple[int, ...]`):
                Dilation.
        """
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

    def __call__(self, x: torch.Tensor, /, ic_last: bool = True) -> torch.Tensor:
        """Reshape input tensor to the desired 2D shape used for GEMM.

        Args:
            x (`torch.Tensor`):
                Input tensor.
            ic_last (`bool`, *optional*, defaults to `True`):
                Whether input channel is the last dimension.

        Returns:
            `torch.Tensor`:
                Reshaped tensor.
        """
        x = F.unfold(
            x,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
            dilation=self.dilation,
        )
        ic = x.shape[1]
        if ic_last:
            return x.permute(0, 2, 1).reshape(-1, ic)
        else:
            return x.permute(1, 0, 2).reshape(ic, -1)


class ConvOutputReshapedFn(ReshapeFn):
    """Outputs reshape function for convolutional layers."""

    def __call__(self, x: torch.Tensor, /, ic_last: bool = True) -> torch.Tensor:
        """Reshape output tensor to the desired shape.

        Args:
            x (`torch.Tensor`):
                Input tensor.
            ic_last (`bool`, *optional*, defaults to `True`):
                Whether input channel is the last dimension.

        Returns:
            `torch.Tensor`:
                Reshaped tensor.
        """
        ic = x.shape[1]
        x = x.view(x.shape[0], ic, -1)
        if ic_last:
            return x.permute(0, 2, 1).reshape(-1, ic)
        else:
            return x.permute(1, 0, 2).reshape(ic, -1)


class AttentionInputReshapeFn(ReshapeFn):
    """Inputs reshape function for attention layer."""

    def __init__(self, channels_dim: int) -> None:
        """Initialize the reshape function.

        Args:
            channels_dim (`int`):
                The dimension of the channels.
        """
        self.channels_dim = channels_dim

    def __call__(self, x: torch.Tensor, /, ic_last: bool = True) -> torch.Tensor:
        """Reshape input tensor to the desired 2D shape used for GEMM.

        Args:
            x (`torch.Tensor`):
                Input tensor.
            ic_last (`bool`, *optional*, defaults to `True`):
                Whether input channel is the last dimension.

        Returns:
            `torch.Tensor`:
                Reshaped tensor.
        """
        num_channels = x.shape[self.channels_dim]
        shape_before = x.shape[: self.channels_dim]
        shape_after = x.shape[self.channels_dim + 1 :]
        x = x.view(shape_before.numel(), num_channels, shape_after.numel())
        if ic_last:
            return x.permute(0, 2, 1).reshape(-1, num_channels)
        else:
            return x.permute(1, 0, 2).reshape(num_channels, -1)
