# -*- coding: utf-8 -*-
"""Quantizatizer config."""

from dataclasses import dataclass, field

from omniconfig import configclass

from ...calib.config import DynamicRangeCalibConfig
from .base import QuantizerConfig
from .kernel import QuantizerKernelConfig

__all__ = ["TensorQuantizerConfig", "WeightQuantizerConfig", "ActivationQuantizerConfig"]


@configclass
@dataclass
class TensorQuantizerConfig(QuantizerConfig):
    """Quantization configuration.

    Args:
        static (bool): Whether to use static quantization. Defaults to ``False``.
        dtype (QuantDataType): The quantization data type. Defaults to ``None``.
        group_shapes (list[list[int]] | list[int]): The shapes for per-group quantization.
            Defaults to ``((-1, -1, -1),)``.
        group_scale_dtypes (list[torch.dtype | QuantDataType | None] | torch.dtype | QuantDataType | None): The
            quantization scale data type for per-group quantization. Defaults to ``(None,)``.
        compute_dtype (QuantDataType | None): The quantization data type for compute. Defaults to ``None``.
        compute_group_level (int): The group level for compute. Defaults to ``-1``.
        saturate_compute_dtype (bool): Whether to saturate the compute dtype. Defaults to ``False``.
        calib_kernel (QuantizerKernelConfig | None): The quantizatizer kernel configuration. Defaults to ``None``.
        calib_range (DynamicRangeCalibConfig | None): The quantizatizer dynamic range calibration configuration.
            Defaults to ``None``.
    """

    static: bool = False
    calib_kernel: QuantizerKernelConfig | None = None
    calib_range: DynamicRangeCalibConfig | None = None

    @property
    def enabled_calib_kernel(self) -> bool:
        """Whether quantization kernel calibration is enabled."""
        return self.calib_kernel is not None and self.calib_kernel.enabled

    @property
    def enabled_calib_range(self) -> bool:
        """Whether quantization dynamic range calibration is enabled."""
        return self.calib_range is not None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dtype is None:
            self.static = False
            self.calib_kernel = None
            self.calib_range = None
        if not self.enabled_calib_kernel:
            self.calib_kernel = None
        if self.static and self.enabled_calib_range is None:
            self.enabled_calib_range = DynamicRangeCalibConfig()

    def __str__(self) -> str:
        return (
            super().__str__()[:-1]
            + f", static={self.static}, kernel={self.enabled_calib_kernel}, dynamic_range={self.enabled_calib_range})"
        )

    def generate_calib_kernel_dirnames(self, prefix: str = "") -> list[str]:
        """Generate the directory names for quantization kernel calibration.

        Args:
            prefix (str, optional): The prefix for the directory names. Defaults to ``""``.

        Returns:
            list[str]: The directory names.
        """
        if self.enabled and self.enabled_calib_kernel:
            return self.calib_kernel.generate_dirnames(prefix=prefix)
        return []

    def generate_calib_range_dirnames(self, prefix: str = "") -> list[str]:
        """Generate the directory names for quantization dynamic range calibration.

        Args:
            prefix (str, optional): The prefix for the directory names. Defaults to ``""``.

        Returns:
            list[str]: The directory names.
        """
        if self.enabled and self.enabled_calib_range:
            names = self.calib_range.generate_dirnames(prefix=prefix)
            if self.static:
                names[1] += ".static"
            return names
        return []

    def generate_calib_name(self) -> str:
        """Generate the name for quantization calibration.

        Returns:
            str: The name.
        """
        name = ""
        if self.static:
            name += ".static"
        if self.enabled_calib_kernel:
            name += ".kernel"
        if self.enabled_calib_range and (self.calib_range.needs_search or self.calib_range.ratio != 1):
            name += ".orange"
        return name[1:] if name else ""


@configclass
@dataclass
class WeightQuantizerConfig(TensorQuantizerConfig):
    """Weights quantization configuration.

    Args:
        static (bool): Whether to use static quantization. Defaults to ``False``.
        dtype (QuantDataType): The quantization data type. Defaults to ``None``.
        group_shapes (list[list[int]] | list[int]): The shapes for per-group quantization.
            Defaults to ``((-1, -1, -1),)``.
        group_scale_dtypes (list[torch.dtype | QuantDataType | None] | torch.dtype | QuantDataType | None): The
            quantization scale data type for per-group quantization. Defaults to ``(None,)``.
        compute_dtype (QuantDataType | None): The quantization data type for compute. Defaults to ``None``.
        compute_group_level (int): The group level for compute. Defaults to ``-1``.
        saturate_compute_dtype (bool): Whether to saturate the compute dtype. Defaults to ``False``.
        calib_kernel (QuantizerKernelConfig | None): The quantizatizer kernel configuration. Defaults to ``None``.
        calib_range (DynamicRangeCalibConfig | None): The quantizatizer dynamic range calibration configuration.
            Defaults to ``None``.
    """

    static: bool = field(default=True, init=False)
    calib_range: DynamicRangeCalibConfig | None = field(default=DynamicRangeCalibConfig)


@configclass
@dataclass
class ActivationQuantizerConfig(TensorQuantizerConfig):
    """Activation quantization configuration.

    Args:
        static (bool): Whether to use static quantization. Defaults to ``False``.
        dtype (QuantDataType): The quantization data type. Defaults to ``None``.
        group_shapes (list[list[int]] | list[int]): The shapes for per-group quantization.
            Defaults to ``((-1, -1, -1),)``.
        group_scale_dtypes (list[torch.dtype | QuantDataType | None] | torch.dtype | QuantDataType | None): The
            quantization scale data type for per-group quantization. Defaults to ``(None,)``.
        compute_dtype (QuantDataType | None): The quantization data type for compute. Defaults to ``None``.
        compute_group_level (int): The group level for compute. Defaults to ``-1``.
        saturate_compute_dtype (bool): Whether to saturate the compute dtype. Defaults to ``False``.
        calib_range (DynamicRangeCalibConfig | None): The quantizatizer dynamic range calibration configuration.
            Defaults to ``None``.
    """

    calib_kernel: QuantizerKernelConfig | None = field(init=False, default=None)
