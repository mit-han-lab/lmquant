# -*- coding: utf-8 -*-
"""Common uantization data."""

import enum

__all__ = ["TensorType"]


class TensorType(enum.Enum):
    """The tensor type."""

    Weights = enum.auto()
    Inputs = enum.auto()
    Outputs = enum.auto()
