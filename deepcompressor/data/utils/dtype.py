# -*- coding: utf-8 -*-
"""Utility functions for dtype in quantization."""

import torch

from ..dtype import QuantDataType

__all__ = ["infer_dtype_bits", "infer_dtype_name", "eval_dtype"]


def infer_dtype_bits(dtype: torch.dtype | QuantDataType) -> int:
    """Get the number of bits of a torch.dtype or QuantDataType.

    Args:
        dtype (`torch.dtype` or `QuantDataType`):
            The dtype to get the number of bits of.

    Returns:
        `int`:
            The number of bits.
    """
    if isinstance(dtype, QuantDataType):
        return dtype.total_bits
    else:
        if dtype == torch.float32:
            return 32
        elif dtype == torch.float16 or dtype == torch.bfloat16:
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
        dtype (`torch.dtype` | `QuantDataType`):
            The dtype to get the string representation of.

    Returns:
        `str`:
            The string representation.
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
        elif dtype == torch.bfloat16:
            return "bf16"
        else:
            return str(dtype).split(".")[-1]
    else:
        raise ValueError(f"Unknown dtype {dtype}")


def eval_dtype(  # noqa: C901
    s: str | torch.dtype | QuantDataType | None, with_quant_dtype: bool = True, with_none: bool = True
) -> torch.dtype | QuantDataType | None:
    if isinstance(s, torch.dtype):
        return s
    if isinstance(s, QuantDataType):
        if with_quant_dtype:
            return s
        else:
            raise ValueError(f"Unknown dtype {s}")
    if s is None:
        if with_none:
            return None
        else:
            raise ValueError(f"Unknown dtype {s}")
    assert isinstance(s, str), f"Unknown dtype {s}"
    s = s.lower()
    if s in ("torch.float64", "float64", "fp64", "f64", "double"):
        return torch.float64
    elif s in ("torch.float32", "float32", "fp32", "f32", "single", "float"):
        return torch.float32
    elif s in ("torch.float16", "float16", "fp16", "f16", "half"):
        return torch.float16
    elif s in ("torch.bfloat16", "bfloat16", "bf16", "b16", "brain"):
        return torch.bfloat16
    elif s in ("torch.int64", "int64", "i64", "long"):
        return torch.int64
    elif s in ("torch.int32", "int32", "i32", "int"):
        return torch.int32
    elif s in ("torch.int16", "int16", "i16", "short"):
        return torch.int16
    elif s in ("torch.int8", "int8", "i8", "byte"):
        return torch.int8
    elif s in ("torch.uint8", "uint8", "u8", "ubyte"):
        return torch.uint8
    else:
        if with_none and s in ("", "none", "null", "nil"):
            return None
        if with_quant_dtype:
            return QuantDataType.from_str(s)
        raise ValueError(f"Unknown dtype {s}")
