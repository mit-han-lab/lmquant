# -*- coding: utf-8 -*-
"""Quantizatizer kernel configurations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields

import torch
from omniconfig import configclass

from ...data.dtype import QuantDataType
from ...data.range import QuantRange
from ...data.zero import ZeroPointDomain
from ...utils.config import EnableConfig, IncludeBasedConfig, KeyEnableConfig

__all__ = ["BaseQuantKernel", "BaseQuantKernelConfig", "BaseKeyEnableQuantKernelConfig"]


class BaseQuantKernel(ABC):
    """Quantization kernel."""

    @abstractmethod
    def quantize(
        self,
        tensor: torch.Tensor,
        *,
        view_shape: torch.Size,
        quant_dtype: QuantDataType,
        zero_domain: ZeroPointDomain | None,
        scale: torch.Tensor,
        zero: torch.Tensor,
        quant_range: QuantRange | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Quantize the tensor.

        Args:
            tensor (`torch.Tensor`):
                The tensor to quantize.
            view_shape (`torch.Size`):
                The view shape when quantizing the tensor.
            quant_dtype (`QuantDataType`):
                The quantization data type.
            zero_domain (`ZeroPointDomain` or `None`):
                The zero point domain.
            scale (`torch.Tensor`):
                The scale tensor.
            zero (`torch.Tensor`):
                The zero point tensor.
            quant_range (`QuantRange` or `None`, *optional*, defaults to `None`):
                The quantization range.
            **kwargs: Other keyword arguments for the quantization kernel.

        Returns:
            `torch.Tensor`:
                The quantized tensor in the shape of ``view_shape``.
        """
        ...


class BaseQuantKernelConfig(ABC):
    """Base quantization kernel configuration."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the quantization kernel."""
        ...

    @abstractmethod
    def build(self) -> BaseQuantKernel:
        """Build the quantization kernel."""
        ...

    @abstractmethod
    def generate_dirnames(self, *, prefix: str = "", **kwargs) -> list[str]:
        """Generate the directory names of the configuration.

        Args:
            prefix (`str`, *optional*, defaults to `""`):
                The prefix for the directory names.

        Returns:
            `list[str]`:
                The directory names.
        """
        ...


@configclass
@dataclass
class BaseKeyEnableQuantKernelConfig(KeyEnableConfig, EnableConfig):
    """Configuration for quantization kernel."""

    _names: list[str] = field(init=False, repr=False, compare=False, default_factory=list)
    _kernels: dict[str, BaseQuantKernelConfig | None] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        self.organize()

    def is_enabled(self) -> bool:
        return bool(self._kernels)

    def is_enabled_for(self, key: str) -> bool:
        return key in self._kernels

    def specialize_for(self, key: str) -> BaseQuantKernelConfig | None:
        """Get the kernel configuration for the module key.

        Args:
            key (`str`):
                The key.

        Returns:
            `QuantKernelConfig` or `None`:
                The kernel configuration for the key.
        """
        return self._kernels.get(key, None)

    def generate_dirnames(self, *, prefix: str = "", **kwargs) -> list[str]:
        """Generate the directory names of the configuration.

        Args:
            prefix (`str`, *optional*, defaults to `""`):
                The prefix for the directory names.

        Returns:
            `list[str]`:
                The directory names.
        """
        names = []
        if self.is_enabled():
            for name in self._names:
                config: IncludeBasedConfig = getattr(self, name)
                if config is not None and config.is_enabled():
                    names.extend(config.generate_dirnames(prefix=prefix, **kwargs))
        return names

    def organize(self) -> None:
        """Organize the configuration."""
        self._kernels.clear()
        for _field in fields(self):
            name = _field.name
            if name.startswith("_"):
                continue
            self._names.append(name)
            config = getattr(self, name)
            if config is not None:
                assert isinstance(
                    config, IncludeBasedConfig
                ), f"Field '{name}' must be an instance of IncludeBasedConfig."
                assert isinstance(
                    config, BaseQuantKernelConfig
                ), f"Field '{name}' must be an instance of BaseQuantKernelConfig."
                if config.is_enabled():
                    for key in config.includes:
                        assert (
                            key not in self._kernels
                        ), f"Key '{key}' is already included in other kernel configurations."
                        self._kernels[key] = config
                else:
                    setattr(self, name, None)
                    continue
