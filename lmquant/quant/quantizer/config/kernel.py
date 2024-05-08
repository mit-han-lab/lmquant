# -*- coding: utf-8 -*-
"""Quantizatizer kernel configurations."""

from dataclasses import dataclass, field

from omniconfig import configclass

from ...functional.config.kernel import QuantGPTQConfig, QuantKernelConfig, QuantKernelType

__all__ = ["QuantizerKernelConfig"]


@configclass
@dataclass
class QuantizerKernelConfig:
    """Configuration for quantization kernel.

    Args:
        gptq (QuantGPTQConfig): The GPTQ configuration. Defaults to ``None``.
    """

    _kernels: dict[str, QuantKernelConfig | None] = field(init=False, repr=False, compare=False, default_factory=dict)
    # for every quantization kernel (except RTN), add the corresponding configuration here
    gptq: QuantGPTQConfig | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._kernels)

    def __post_init__(self) -> None:
        for kernel in QuantKernelType:
            if kernel == QuantKernelType.RTN:
                continue
            config: QuantKernelConfig = getattr(self, kernel.name.lower())
            if config is not None:
                if not config.enabled:
                    setattr(self, kernel.name.lower(), None)
                    continue
                for key in config.includes:
                    assert key not in self._kernels, f"Key '{key}' is already included in other kernel configurations."
                    self._kernels[key] = config

    def enabled_for(self, key: str) -> bool:
        """Whether the kernel calibration is enabled for the module key.

        Args:
            key (str): The key.

        Returns:
            bool: Whether the kernel calibration is needed.
        """
        return key in self._kernels

    def specialize_for(self, key: str) -> QuantKernelConfig | None:
        """Get the kernel configuration for the module key.

        Args:
            key (str): The key.

        Returns:
            QuantKernelConfig | None: The kernel configuration.
        """
        return self._kernels.get(key, None)

    def __str__(self) -> str:
        s = "("
        for kernel in QuantKernelType:
            if kernel == QuantKernelType.RTN:
                continue
            config: QuantKernelConfig = getattr(self, kernel.name.lower())
            if config is not None:
                s += f"{kernel.name}={config}, "
        return s[:-2] + ")"

    def generate_dirnames(self, prefix: str = "") -> list[str]:
        """Generate the directory names of the configuration.

        Returns:
            list[str]: The directory names.
        """
        names = []
        if self.enabled:
            for kernel in QuantKernelType:
                if kernel == QuantKernelType.RTN:
                    continue
                config: QuantKernelConfig = getattr(self, kernel.name.lower())
                if config is not None and config.enabled:
                    names.extend(config.generate_dirnames(prefix=prefix))
        return names
