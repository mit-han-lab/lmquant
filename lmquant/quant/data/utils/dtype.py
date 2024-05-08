# -*- coding: utf-8 -*-
"""Utility functions for dtype in quantization."""

import torch

from ..dtype import QuantDataType

__all__ = ["infer_dtype_bits", "infer_dtype_name"]


def infer_dtype_bits(dtype: torch.dtype | QuantDataType) -> int:
    """Get the number of bits of a torch.dtype or QuantDataType.

    Args:
        dtype (torch.dtype | QuantDataType): The dtype to get the number of bits of.

    Returns:
        int: The number of bits.

    Raises:
        ValueError: If the dtype is unknown.
    """
    if isinstance(dtype, QuantDataType):
        return dtype.total_bits
    else:
        if dtype == torch.float32:
            return 32
        elif dtype == torch.float16:
            return 16
        elif dtype == torch.float64:
            return 64
        elif dtype == torch.int32:
            return 32
        elif dtype == torch.int16:
            return 16
        elif dtype == torch.int8:
            return 8
        elif dtype == torch.uint8:
            return 8
        else:
            raise ValueError(f"Unknown dtype {dtype}")


def infer_dtype_name(dtype: torch.dtype | QuantDataType) -> str:
    """Get the string representation of a torch.dtype or QuantDataType.

    Args:
        dtype (torch.dtype | QuantDataType): The dtype to get the string representation of.

    Returns:
        str: The string representation.

    Raises:
        ValueError: If the dtype is unknown.
    """
    if isinstance(dtype, QuantDataType):
        return str(dtype)
    elif isinstance(dtype, torch.dtype):
        if dtype == torch.float16:
            return "fp16"
        elif dtype == torch.float32:
            return "fp32"
        elif dtype == torch.float64:
            return "fp64"
        else:
            return str(dtype).split(".")[-1]
    else:
        raise ValueError(f"Unknown dtype {dtype}")
