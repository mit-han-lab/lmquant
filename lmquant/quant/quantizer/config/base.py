# -*- coding: utf-8 -*-
"""Quantizer config."""

import typing as tp
from dataclasses import dataclass, field

import omniconfig
import torch
from omniconfig import configclass

from ...functional.config import QuantConfig

__all__ = ["QuantConfig", "QuantizerConfig"]


@configclass
@dataclass
class QuantizerConfig(QuantConfig):
    """Quantizer configuration.

    Args:
        dtype (QuantDataType): The quantization data type. Defaults to ``None``.
        group_shapes (list[list[int]] | list[int]): The shapes for per-group quantization.
            Defaults to ``((-1, -1, -1),)``.
        group_scale_dtypes (list[torch.dtype | QuantDataType | None] | torch.dtype | QuantDataType | None): The
            quantization scale data type for per-group quantization. Defaults to ``(None,)``.
        compute_dtype (QuantDataType | None): The quantization data type for compute. Defaults to ``None``.
        compute_group_level (int): The group level for compute. Defaults to ``-1``.
        saturate_compute_dtype (bool): Whether to saturate the compute dtype. Defaults to ``False``.
        skips (list[str]): The keys of the modules to skip. Defaults to ``[]``.
    """

    skips: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.skips = sorted(set(self.skips or []))

    def enabled_for(self, key: str) -> bool:
        """Whether the quantization configuration is enabled for the given key.

        Args:
            key (str): The key of the module.

        Returns:
            bool: Whether the quantization configuration is enabled for the given key.
        """
        return self.dtype is not None and key not in self.skips

    def __str__(self) -> str:
        s = super().__str__()
        return s[:-1] + f", skips={self.skips})"

    def generate_dirnames(
        self,
        shape: torch.Size = torch.Size((4096, 4096, 16, 16)),
        default_dtype: torch.dtype = torch.float16,
        prefix: str = "",
    ) -> list[str]:
        """Generate the directory names of the quantization configuration.

        Args:
            shape (torch.Size, optional): The shape of the input tensor.
                Defaults to ``torch.Size((4096, 4096, 16, 16))``.
            default_dtype (torch.dtype, optional): The dtype of the input tensor.
                Defaults to ``torch.float16``.

        Returns:
            list[str]: The directory names of the quantization configuration.
                - The number of effective bits.
                - The name of the quantization data type.
                - The name of the group shapes.
                - The name of the modules to skip.
        """
        names = [
            *super().generate_dirnames(shape=shape, default_dtype=default_dtype),
            "skip.[{}]".format("+".join(self.skips)),
        ]
        if prefix:
            names = [f"{prefix}.{name}" for name in names]
        return names

    @classmethod
    def update_get_arguments(
        cls: type["QuantizerConfig"],
        *,
        overwrites: dict[str, tp.Callable[[omniconfig.Arguments], None] | None] | None = None,
        defaults: dict[str, tp.Any] | None = None,
    ) -> tuple[dict[str, tp.Callable[[omniconfig.Arguments], None] | None], dict[str, tp.Any]]:
        """Get the arguments for the quantization configuration."""
        overwrites, defaults = super().update_get_arguments(overwrites=overwrites, defaults=defaults)
        collect_fn = omniconfig.ADD_PREFIX_BOOL_FIELDS("skip", **defaults)

        def add_skips_argument(parser):
            collect_fn(parser)
            parser.add_argument("--skips", nargs="+", default=[], help="The keys of the modules to skip.")

        overwrites.setdefault("skips", add_skips_argument)
        return overwrites, defaults

    @classmethod
    def update_from_dict(
        cls: type["QuantizerConfig"], *, parsed_args: dict[str, tp.Any], overwrites: dict[str, tp.Any]
    ) -> tuple[dict[str, tp.Any], dict[str, tp.Any]]:
        """Update the arguments settings for the quantization configuration."""
        parsed_args.setdefault("skips", []).extend(omniconfig.COLLECT_PREFIX_BOOL_FIELDS(parsed_args, "skip"))
        return parsed_args, overwrites
