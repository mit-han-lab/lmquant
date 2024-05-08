# -*- coding: utf-8 -*-
"""Quantized tensor module."""

import torch

from .scale import QuantScale

__all__ = ["QuantTensor"]


class QuantTensor:
    """Quantized tensor."""

    _dequantized: torch.Tensor | None
    _quantized: torch.Tensor | None
    scale: QuantScale | None
    zero: torch.Tensor | float | None
    view_shape: torch.Size | None

    def __init__(
        self,
        dequantized: torch.Tensor = None,
        quantized: torch.Tensor = None,
        scale: QuantScale | None = None,
        zero: torch.Tensor | float | None = None,
        view_shape: torch.Size | None = None,
    ):
        """Initialize the quantized tensor."""
        assert (
            dequantized is not None or quantized is not None
        ), "Either the dequantized or quantized tensor must be provided."
        self.view_shape = view_shape
        self._dequantized = dequantized
        self._quantized = quantized
        self.scale = scale
        self.zero = zero

    @property
    def data(self) -> torch.Tensor:
        """Get the dequantized tensor."""
        return self._dequantized

    @property
    def qdata(self) -> torch.Tensor:
        """Get the quantized tensor."""
        return self._quantized
