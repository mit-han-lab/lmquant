# -*- coding: utf-8 -*-
"""Quantization scale module."""

from dataclasses import dataclass, field

import torch

from .utils.scale import join_scale_tensor

__all__ = ["QuantScale"]


@dataclass
class QuantScale:

    data: torch.Tensor = None
    _scales: list[torch.Tensor] = field(default_factory=list)
    _n: int = 0

    @property
    def num_levels(self) -> int:
        """Get the number of levels."""
        return self._n

    def __post_init__(self):
        if self._n > 0:
            assert len(self._scales) == self._n
            self.data = None
            for scale in self._scales:
                self.data = QuantScale.join(self.data, scale)
            self.data[self.data == 0] = 1
        else:
            self._n = 0
            self._scales = []
            assert self.data is None or isinstance(self.data, torch.Tensor)

    def get_level_scale(self, level: int = 0) -> torch.Tensor:
        """Get the scale tensor at the specified level."""
        return self.data if self._n <= 0 else self._scales[level]

    def append(self, scale: torch.Tensor) -> "QuantScale":
        """Append a scale tensor."""
        self.data = join_scale_tensor(self.data, scale)
        self._scales.append(scale)
        self._n += 1
        return self

    def extend(self, scale: "QuantScale") -> "QuantScale":
        """Extend the scale tensor."""
        self.data = join_scale_tensor(self.data, scale.data)
        self._scales.extend(scale._scales)
        self._n += scale._n
        return self
