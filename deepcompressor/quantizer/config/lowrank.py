# -*- coding: utf-8 -*-

from dataclasses import dataclass

from omniconfig import configclass

from ...utils.common import num2str
from ...utils.config import EnableConfig

__all__ = ["QuantLowRankConfig"]


@configclass
@dataclass
class QuantLowRankConfig(EnableConfig):
    """Quantization low-rank branch configuration.

    Args:
        rank (`int`, *optional*, defaults to `32`):
            The rank of the low-rank branch.
        exclusive (`bool`, *optional*, defaults to `False`):
            Whether to use exclusive low-rank branch for each weight sharing the inputs.
        compensate (`bool`, *optional*, defaults to `False`):
            Whether the low-rank branch compensates the quantization error.
    """

    rank: int = 32
    exclusive: bool = False
    compensate: bool = False

    def is_enabled(self) -> bool:
        return self.rank != 0

    def generate_dirnames(self, *, prefix: str = "", **kwargs) -> list[str]:
        """Generate the directory names of the configuration.

        Returns:
            list[str]: The directory names.
        """
        if not self.is_enabled():
            return []
        name = f"r{num2str(self.rank)}"
        if self.exclusive:
            name += ".exclusive"
        if self.compensate:
            name += ".compensate"
        return [f"{prefix}.{name}" if prefix else name]
