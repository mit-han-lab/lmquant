# -*- coding: utf-8 -*-
"""Utility functions for shape calulation in quantization."""

import typing as tp

import torch

from ..dtype import QuantDataType

__all__ = ["infer_group_shape_name", "format_group_configs", "infer_group_shapes", "infer_view_shape", "infer_shape"]


def infer_group_shape_name(group_shape: tuple[int]) -> str:
    """Get the name of the group shape.

    Args:
        group_shape (tuple[int]): The group shape.

    Returns:
        str: The name of the group shape.
    """
    if all(gs <= 0 for gs in group_shape[2:]):
        if group_shape[1] <= 0:
            if group_shape[0] <= 0:
                return "tsnr"
            elif group_shape[0] == 1:
                return "gchn"
            else:
                return f"t{group_shape[0]}gchn"
        else:
            if group_shape[0] <= 0:
                return f"tsg{group_shape[1]}"
            elif group_shape[0] == 1:
                return f"g{group_shape[1]}"
            else:
                return f"t{group_shape[0]}g{group_shape[1]}"
    elif all(gs == 1 for gs in group_shape[2:]):
        if group_shape[1] <= 0:
            if group_shape[0] <= 0:
                return "tspx"
            elif group_shape[0] == 1:
                return "vchn"
            else:
                return f"t{group_shape[0]}vchn"
        else:
            if group_shape[0] <= 0:
                return f"tsv{group_shape[1]}"
            elif group_shape[0] == 1:
                return f"v{group_shape[1]}"
            else:
                return f"t{group_shape[0]}v{group_shape[1]}"
    return f"{'x'.join(str(gs) if gs >= 1 else '_' for gs in group_shape)}"


def format_group_configs(
    *,
    group_shapes: tp.Iterable[tp.Iterable[int]] | tp.Iterable[int],
    group_scale_dtypes: tp.Iterable[torch.dtype | QuantDataType | None] | torch.dtype | QuantDataType | None,
) -> tuple[tuple[tuple[int, ...]], tuple[torch.dtype | QuantDataType | None]]:
    """Format the group shape and scale dtype.

    Args:
        group_shapes (tp.Iterable[tp.Iterable[int]] | tp.Iterable[int]): The group shapes.
        group_scale_dtypes (tp.Iterable[torch.dtype | QuantDataType | None] | torch.dtype | QuantDataType | None):
            The scale dtypes.

    Returns:
        tuple[tuple[tuple[int, ...]], tuple[torch.dtype | QuantDataType | None]]: The formatted group shapes
            and scale dtypes.
    """
    assert isinstance(group_shapes, tp.Iterable), "group_shapes must be a list or tuple"
    _group_shapes = []
    for group_shape in group_shapes:
        if isinstance(group_shape, tp.Iterable):
            n = len(group_shape)
            group_shape = tuple(map(int, group_shape))
            assert n >= 2, "the group shape must have at least two dimensions"
            assert all(gs >= -1 for gs in group_shape), "the group shape must be larger than -1"
            _group_shapes.append(tuple(group_shape) if n >= 3 else (*group_shape, -1))
    _scale_dtypes = tuple(group_scale_dtypes) if isinstance(group_scale_dtypes, tp.Iterable) else (group_scale_dtypes,)
    _scale_dtypes = tuple(
        dtype if isinstance(dtype, (torch.dtype, QuantDataType, type(None))) else QuantDataType.from_str(dtype)
        for dtype in _scale_dtypes
    )
    assert len(_group_shapes) > 0, "group_sizes must be a non-empty list"
    assert len(_group_shapes) == len(_scale_dtypes), (
        f"group_shapes and group_scale_dtypes must have the same length, "
        f"got {_group_shapes}(len={len(_group_shapes)}) and {_scale_dtypes}(len={len(_scale_dtypes)})"
    )
    exp_scale = True
    for dtype in reversed(_scale_dtypes):
        if isinstance(dtype, QuantDataType) and dtype.is_exp:
            if not exp_scale:
                raise ValueError("The exponential scale must be the last scale")
        else:
            exp_scale = False
    assert all(isinstance(dtype, QuantDataType) for dtype in _scale_dtypes[1:])
    return tuple(_group_shapes), _scale_dtypes


def infer_group_shapes(group_shapes: tuple[tuple[int, ...]], shape: torch.Size) -> list[torch.Size]:
    """Infer the group shapes using group shape config on the given tensor shape.

    Args:
        shape (torch.Size): The shape of the tensor.

    Returns:
        List[torch.Size]: The group shapes.
    """
    assert isinstance(shape, torch.Size), f"shape must be torch.Size, got {shape} ({type(shape)})"
    assert len(shape) >= 2, f"shape must have at least 2 dimensions, got {shape} ({len(shape)} < 2)"
    _group_shapes: list[torch.Size] = []
    _prev_group_shape = shape
    for level, group_shape in enumerate(group_shapes):
        m = len(group_shape) - 1
        _group_shape = []
        for i, ts in enumerate(shape):
            gs = group_shape[min(i, m)]
            if gs <= 0:
                gs = ts
            ps = _prev_group_shape[i]
            if gs > ps:
                gs = ps  # the group shape must be less than or equal to the previous group shape
            assert ps % gs == 0, (
                f"the level {level} group size ({gs}) must be divisible by "
                f"the previous group size ({_prev_group_shape}[{i}])"
            )
            _group_shape.append(gs)
        _group_shapes.append(torch.Size(_group_shape))
        _prev_group_shape = _group_shape
    return _group_shapes


def infer_view_shape(
    tensor_shape: torch.Size,
    /,
    group_shape: tp.Sequence[int],
    skip_first_dim: bool = False,
) -> torch.Size:
    """Infer the view shape from the tensor shape and the group shape.

    Args:
        tensor_shape (torch.Size): The tensor shape.
        group_shape (tp.Sequence[int]): The group shape.
        skip_first_dim (bool, optional): Whether to skip the first dimension.
            Defaults to ``False``.

    Returns:
        torch.Size: The view shape of (#g0, gs0, #g1, gs1, #g2, gs2, ...)
    """
    m, view_shape = len(group_shape) - 1, []
    for i, ts in enumerate(tensor_shape):
        gs = group_shape[min(i, m)]
        gs = ts if gs <= 0 else gs
        view_shape.append(ts // gs)
        view_shape.append(gs)
    if skip_first_dim:
        view_shape[0], view_shape[1] = 1, tensor_shape[0]
    return torch.Size(view_shape)


def infer_scale_view_shapes(
    group_shapes: tuple[tuple[int, ...]] | list[torch.Size], shape: torch.Size
) -> list[torch.Size]:
    """Infer the view shapes of quantization scale for the given tensor shape.

    Args:
        group_shapes (tuple[tuple[int, ...]]): The group shapes.
        shape (torch.Size): The shape of the tensor to be quantized.

    Returns:
        list[torch.Size]: list of view shapes of the scale tensor for each quantization group.
    """
    if not isinstance(group_shapes[0], torch.Size):
        group_shapes = infer_group_shapes(group_shapes=group_shapes, shape=shape)
    assert all(isinstance(gs, torch.Size) for gs in group_shapes), "group_shapes must be a list of torch.Size"
    min_group_shape = group_shapes[-1]
    s_view_shapes = []
    for group_shape in group_shapes:
        s_view_shape = []
        for ts, gs, mgs in zip(shape, group_shape, min_group_shape):
            num_groups, num_reduct = ts // gs, gs // mgs
            s_view_shape.append(num_groups)
            s_view_shape.append(num_reduct)
        s_view_shapes.append(torch.Size(s_view_shape))
    return s_view_shapes


def infer_shape(view_shape: torch.Size) -> torch.Size:
    """Infer the shape from the view shape.

    Args:
        view_shape (torch.Size): The view shape.

    Returns:
        torch.Size: The shape of the tensor.
    """
    return torch.Size(view_shape[i] * view_shape[i + 1] for i in range(0, len(view_shape), 2))
