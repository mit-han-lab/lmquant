# -*- coding: utf-8 -*-
"""Quantization kernel configurations."""

import enum
import typing as tp
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import omniconfig
from omniconfig import configclass

from ....utils import num2str

__all__ = ["QuantKernelType", "QuantKernelConfig", "QuantGPTQConfig"]


class QuantKernelType(enum.Enum):
    RTN = enum.auto()
    GPTQ = enum.auto()
    # reserved for future possible quantization kernels


@configclass
@dataclass
class QuantKernelConfig(ABC):
    """Quantization kernel configuration.

    Args:
        includes (list[str]): The module keys to include. Defaults to ``[]``.
    """

    includes: list[str] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        """Whether the kernel is enabled."""
        return bool(self.includes)

    @property
    @abstractmethod
    def kernel(self) -> QuantKernelType:
        """The quantization kernel type."""
        ...

    def __post_init__(self) -> None:
        self.includes = sorted(set(self.includes or []))

    def enabled_for(self, key: str) -> bool:
        """Whether the kernel is enabled for the module key.

        Args:
            key (str): The key.

        Returns:
            bool: Whether the kernel is needed.
        """
        return key in self.includes

    def __str__(self) -> str:
        return f"(kernel={self.kernel.name}, skips={self.skips})"

    @abstractmethod
    def _generate_dirnames(self) -> list[str]:
        """Generate the directory names of the configuration."""
        ...

    def generate_dirnames(self, prefix: str = "") -> list[str]:
        """Generate the directory names of the configuration.

        Returns:
            list[str]: The directory names. The last directory name is the modules to include.
        """
        kernel_name = self.kernel.name.lower()
        names = []
        if self.includes:
            for name in self._generate_dirnames():
                names.append(f"{kernel_name}.{name}")
            names.append("{}.include.[{}]".format(kernel_name, "+".join(self.includes)))
        if prefix and names:
            names = [f"{prefix}.{name}" for name in names]
        return names

    @classmethod
    def update_get_arguments(
        cls: type["QuantKernelConfig"],
        *,
        overwrites: dict[str, tp.Callable[[omniconfig.Arguments], None] | None] | None = None,
        defaults: dict[str, tp.Any] | None = None,
    ) -> tuple[dict[str, tp.Any], dict[str, tp.Any]]:
        """Update the arguments settings for the quantization configuration."""
        overwrites = overwrites or {}
        defaults = defaults or {}

        collect_fn = omniconfig.ADD_PREFIX_BOOL_FIELDS("include", **defaults)

        def add_includes_argument(parser):
            collect_fn(parser)
            parser.add_argument("--includes", nargs="+", default=[], help="The keys of the modules to include.")

        overwrites.setdefault("includes", add_includes_argument)
        return overwrites, defaults

    @classmethod
    def update_from_dict(
        cls: type["QuantKernelConfig"], *, parsed_args: dict[str, tp.Any], overwrites: dict[str, tp.Any]
    ) -> tuple[dict[str, tp.Any], dict[str, tp.Any]]:
        """Update the arguments settings for the quantization configuration."""
        parsed_args.setdefault("includes", []).extend(omniconfig.COLLECT_PREFIX_BOOL_FIELDS(parsed_args, "include"))
        return parsed_args, overwrites


@configclass
@dataclass
class QuantGPTQConfig(QuantKernelConfig):
    """Configuration for GPTQ quantization.

    Args:
        damp_percentage (float): The percentage of damping. Defaults to ``0.01``.
        block_size (int): The block size. Defaults to ``128``.
        num_inv_tries (int): The number of tries for the inverse. Defaults to ``40``.
        hessian_block_size (int): The block size when calculing the Hessian. Defaults to ``-1``.
        includes (list[str]): The module keys to include. Defaults to ``[]``.
    """

    damp_percentage: float = 0.01
    block_size: int = 128
    num_inv_tries: int = 200
    hessian_block_size: int = -1

    @property
    def kernel(self) -> QuantKernelType:
        return QuantKernelType.GPTQ

    def __str__(self) -> str:
        s = super().__str__()
        return f"{s[:-1]}, damp_percentage={self.damp_percentage}, block_size={self.block_size})"

    def _generate_dirnames(self) -> list[str]:
        """Generate the directory names of the configuration.

        Returns:
            list[str]: The directory names.
        """
        return [f"d{num2str(self.damp_percentage)}.b{num2str(self.block_size)}"]
