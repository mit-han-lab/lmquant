# -*- coding: utf-8 -*-
"""Zero-point for quantization."""

import enum

__all__ = ["ZeroPointDomain"]


class ZeroPointDomain(enum.Enum):
    """Zero-point domain."""

    PreScale = enum.auto()
    PostScale = enum.auto()
