# -*- coding: utf-8 -*-
"""Channel-wise metric calculation module."""

import typing as tp

import torch

from .utils.shape import infer_view_shape

__all__ = ["ChannelMetric"]


class ChannelMetric:
    """Channel-wise metric."""

    @staticmethod
    def _normalize(
        tensor: torch.Tensor,
        num_channels: int,
        group_shape: tp.Sequence[int],
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        tensor = tensor.to(dtype=dtype)
        shape, ndim = tensor.shape, tensor.ndim
        view_shape = infer_view_shape(tensor.shape, group_shape)
        # (d0, d1, d2, ...) -> (#g0, gs0, #g1, gs1, #g2, gs2, ...)
        tensor = tensor.view(view_shape)
        tensor_max = tensor.abs().amax(dim=list(range(1, ndim * 2, 2)), keepdim=True)
        tensor_max[tensor_max == 0] = 1
        tensor = tensor / tensor_max
        return tensor.view(shape)

    @staticmethod
    def _abs_max(
        tensor: torch.Tensor,
        num_channels: int,
        group_shape: tp.Sequence[int],
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            tensor.abs().view(tensor.shape[0], num_channels, -1).amax(dim=(0, 2)).view(-1).to(dtype=dtype),
            None,
        )

    @staticmethod
    def _abs_sum(
        tensor: torch.Tensor,
        num_channels: int,
        group_shape: tp.Sequence[int],
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tensor = tensor.view(tensor.shape[0], num_channels, -1)
        cnt = tensor.shape[0] * tensor.shape[2]
        return tensor.abs().to(dtype=dtype).sum(dim=(0, 2)).view(-1), cnt

    @staticmethod
    def _abs_normalize_sum(
        tensor: torch.Tensor,
        num_channels: int,
        group_shape: tp.Sequence[int],
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return ChannelMetric._abs_sum(
            ChannelMetric._normalize(tensor, num_channels, group_shape, dtype),
            num_channels,
            group_shape,
            dtype,
        )

    @staticmethod
    def _square_sum(
        tensor: torch.Tensor,
        num_channels: int,
        group_shape: tp.Sequence[int],
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tensor = tensor.view(tensor.shape[0], num_channels, -1)
        cnt = tensor.shape[0] * tensor.shape[2]
        return tensor.to(dtype=dtype).pow(2).sum(dim=(0, 2)).view(-1), cnt

    @staticmethod
    def _maximum(
        fn: tp.Callable[[torch.Tensor, int, tp.Sequence[int], torch.dtype], tuple[torch.Tensor, torch.Tensor]],
        tensors: tp.Iterable[torch.Tensor] | torch.Tensor,
        num_channels: int,
        group_shape: tp.Sequence[int],
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(tensors, torch.Tensor):
            device = device or tensors.device
            return fn(tensors.to(device=device), num_channels, group_shape, dtype, **kwargs)
        else:
            assert isinstance(tensors, (list, tuple))
            rst_0, rst_1 = ChannelMetric._maximum(fn, tensors[0], num_channels, group_shape, device, dtype, **kwargs)
            for tensor in tensors[1:]:
                _rst_0, _rst_1 = ChannelMetric._maximum(fn, tensor, num_channels, group_shape, device, dtype, **kwargs)
                rst_0 = torch.maximum(rst_0, _rst_0.to(device=rst_0.device))
                if rst_1 is not None:
                    if isinstance(rst_1, torch.Tensor):
                        rst_1 = torch.maximum(rst_1, _rst_1.to(device=rst_1.device))
                    else:
                        rst_1 = max(rst_1, _rst_1)
            return rst_0, rst_1

    @staticmethod
    def _sum(
        fn: tp.Callable[[torch.Tensor, int, tp.Sequence[int], torch.dtype], tuple[torch.Tensor, torch.Tensor]],
        tensors: tp.Iterable[torch.Tensor] | torch.Tensor,
        num_channels: int,
        group_shape: tp.Sequence[int],
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(tensors, torch.Tensor):
            device = device or tensors.device
            rst_0, rst_1 = fn(tensors.to(device), num_channels, group_shape, dtype, **kwargs)
            return rst_0, 1 if rst_1 is None else rst_1
        else:
            assert isinstance(tensors, (list, tuple))
            rst_0, rst_1 = ChannelMetric._sum(fn, tensors[0], num_channels, group_shape, device, dtype, **kwargs)
            for tensor in tensors[1:]:
                _rst_0, _rst_1 = ChannelMetric._sum(fn, tensor, num_channels, group_shape, device, dtype, **kwargs)
                rst_0 += _rst_0.to(device=rst_0.device)
                if isinstance(rst_1, torch.Tensor):
                    rst_1 += _rst_1.to(device=rst_1.device)
                else:
                    rst_1 += _rst_1
            return rst_0, rst_1

    @staticmethod
    def abs_max(
        tensors: tp.Iterable[torch.Tensor] | torch.Tensor,
        num_channels: int,
        group_shape: tp.Sequence[int],
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Get the absolute maximum of the tensors, where R[i] = AbsMax(T[i, :])."""
        return ChannelMetric._maximum(
            ChannelMetric._abs_max, tensors, num_channels, group_shape, device=device, dtype=dtype
        )[0]

    @staticmethod
    def abs_mean(
        tensors: tp.Iterable[torch.Tensor] | torch.Tensor,
        num_channels: int,
        group_shape: tp.Sequence[int],
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Get the absolute mean of the tensors, where R[i] = AbsMean(T[i, :])."""
        rst, cnt = ChannelMetric._sum(
            ChannelMetric._abs_sum, tensors, num_channels, group_shape, device=device, dtype=dtype
        )
        return rst.div_(cnt)

    @staticmethod
    def abs_normalize_mean(
        tensor: torch.Tensor,
        num_channels: int,
        group_shape: tp.Sequence[int],
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Get the absolute group normalized mean of the tensors, where R[i] = Mean(U[i, :])
        and U[i,j] = Abs(T[i, j]) / AbsMax(T[:, j]))."""
        rst, cnt = ChannelMetric._sum(
            ChannelMetric._abs_normalize_sum, tensor, num_channels, group_shape, device, dtype=dtype
        )
        return rst.div_(cnt)

    @staticmethod
    def root_mean_square(
        tensors: tp.Iterable[torch.Tensor] | torch.Tensor,
        num_channels: int,
        group_shape: tp.Sequence[int],
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Get the root mean square of the tensors, where R[i] = Root(Mean(T[i, :]^2))."""
        rst, cnt = ChannelMetric._sum(
            ChannelMetric._square_sum, tensors, num_channels, group_shape, device=device, dtype=dtype
        )
        return rst.div_(cnt).sqrt_()
