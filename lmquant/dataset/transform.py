# -*- coding: utf-8 -*-
"""Type hints used in lmquant."""

import torch
import torch.nn.functional as F

__all__ = ["TransformFn", "LinearTransformFn", "ConvTransformFn"]


class TransformFn:
    """Transform function."""

    def __call__(self, x: torch.Tensor, /, ic_last: bool = True) -> torch.Tensor:
        """Transform input tensor to the desired shape used for GEMM.

        Args:
            x (torch.Tensor): Input tensor.
            ic_last (bool): Whether input channel is the last dimension.
                Defaults to ``True``.

        Returns:
            torch.Tensor: Transformed tensor.
        """
        return x


class LinearTransformFn(TransformFn):
    """Inputs transform function for linear layers."""

    def __call__(self, x: torch.Tensor, /, ic_last: bool = True) -> torch.Tensor:
        """Transform input tensor to the desired 2D shape used for GEMM.

        Args:
            x (torch.Tensor): Input tensor.
            ic_last (bool): Whether input channel is the last dimension.
                Defaults to ``True``.

        Returns:
            torch.Tensor: Transformed tensor.
        """
        return x.view(-1, x.shape[-1]).permute(int(not ic_last), int(ic_last))


class ConvTransformFn(TransformFn):
    """Inputs transform function for convolutional layers."""

    def __init__(
        self,
        kernel_size: tuple[int, ...],
        padding: str | tuple[int, ...],
        stride: tuple[int, ...],
        dilation: tuple[int, ...],
    ) -> None:
        """Initialize the transform function.

        Args:
            kernel_size (tuple[int, ...]): Kernel size.
            padding (str | tuple[int, ...]): Padding mode or padding size.
            stride (tuple[int, ...]): Stride size.
            dilation (tuple[int, ...]): Dilation size.
        """
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

    def __call__(self, x: torch.Tensor, /, ic_last: bool = True) -> torch.Tensor:
        """Transform input tensor to the desired 2D shape used for GEMM.

        Args:
            x (torch.Tensor): Input tensor.
            ic_last (bool): Whether input channel is the last dimension.
                Defaults to ``True``.

        Returns:
            torch.Tensor: Transformed tensor.
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
