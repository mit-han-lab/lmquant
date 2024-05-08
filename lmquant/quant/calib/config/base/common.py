# -*- coding: utf-8 -*-
"""Quantization tensor types."""
import enum
import typing as tp
from dataclasses import dataclass, field

import omniconfig
from omniconfig import configclass

__all__ = ["QuantTensorType", "BaseQuantCalibConfig"]


class QuantTensorType(enum.Enum):
    """The quantization tensor type."""

    Weights = enum.auto()
    Inputs = enum.auto()
    Outputs = enum.auto()


@configclass
@dataclass
class BaseQuantCalibConfig:
    """The base configuration for quantization calibration.

    Args:
        degree (int): The power degree for the quantization error. Defaults to ``2``.
        skips (list[str]): The keys of the modules to skip. Defaults to ``[]``.
    """

    degree: int = 2
    skips: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.skips = sorted(set(self.skips or []))

    def enabled_for(self, key: str) -> bool:
        """Whether the calibration is enabled for the module key.

        Args:
            key (str): The key.

        Returns:
            bool: Whether the calibration is needed.
        """
        return key not in self.skips

    @classmethod
    def update_get_arguments(
        cls: type["BaseQuantCalibConfig"],
        *,
        overwrites: dict[str, tp.Callable[[omniconfig.Arguments], None] | None] | None = None,
        defaults: dict[str, tp.Any] | None = None,
    ) -> tuple[dict[str, tp.Any], dict[str, tp.Any]]:
        """Update the arguments settings for the quantization configuration."""
        overwrites = overwrites or {}
        defaults = defaults or {}

        collect_fn = omniconfig.ADD_PREFIX_BOOL_FIELDS("skip", **defaults)

        def add_skips_argument(parser):
            collect_fn(parser)
            parser.add_argument("--skips", nargs="+", default=[], help="The keys of the modules to skip.")

        overwrites.setdefault("skips", add_skips_argument)
        return overwrites, defaults

    @classmethod
    def update_from_dict(
        cls: type["BaseQuantCalibConfig"], *, parsed_args: dict[str, tp.Any], overwrites: dict[str, tp.Any]
    ) -> tuple[dict[str, tp.Any], dict[str, tp.Any]]:
        """Update the arguments settings for the quantization configuration."""
        parsed_args.setdefault("skips", []).extend(omniconfig.COLLECT_PREFIX_BOOL_FIELDS(parsed_args, "skip"))
        return parsed_args, overwrites
