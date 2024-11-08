# -*- coding: utf-8 -*-

import typing as tp
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import omniconfig
from omniconfig import configclass

__all__ = ["EnableConfig", "KeyEnableConfig", "SkipBasedConfig", "IncludeBasedConfig"]


class EnableConfig(ABC):
    @abstractmethod
    def is_enabled(self) -> bool:
        """Whether the configuration is enabled."""
        return True


class KeyEnableConfig(ABC):
    @abstractmethod
    def is_enabled_for(self, key: str) -> bool:
        """Whether the configuration is enabled for the given key."""
        return True


@configclass
@dataclass
class SkipBasedConfig(KeyEnableConfig, EnableConfig):
    """Skip-based configration.

    Args:
        skips (`list[str]`, *optional*, defaults to `[]`):
            The keys of the modules to skip.
    """

    skips: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        __post_init__ = getattr(super(), "__post_init__", None)
        if __post_init__:
            __post_init__()
        self.skips = sorted(set(self.skips or []))

    def is_enabled(self) -> bool:
        """Whether the configuration is enabled."""
        return super().is_enabled()

    def is_enabled_for(self, key: str) -> bool:
        """Whether the configuration is enabled for the given key.

        Args:
            key (`str`):
                The key.

        Returns:
            `bool`:
                Whether the configuration is enabled for the given key.
        """
        return self.is_enabled() and key not in self.skips

    def generate_dirnames(self, *, prefix: str = "", **kwargs) -> list[str]:
        """Generate the directory names of the configuration.

        Args:
            prefix (`str`, *optional*, defaults to `""`):
                The prefix of the directory names.

        Returns:
            `list[str]`:
                The directory names of the configuration.
        """
        names = [*super().generate_dirnames(**kwargs), "skip.[{}]".format("+".join(self.skips))]  # type: ignore
        if prefix:
            names = [f"{prefix}.{name}" for name in names]
        return names

    @classmethod
    def update_get_arguments(
        cls: type["SkipBasedConfig"],
        *,
        overwrites: dict[str, tp.Callable[[omniconfig.Arguments], None] | None] | None = None,
        defaults: dict[str, tp.Any] | None = None,
    ) -> tuple[dict[str, tp.Callable[[omniconfig.Arguments], None] | None], dict[str, tp.Any]]:
        """Get the arguments for the quantization configuration."""

        update_get_arguments = getattr(super(), "update_get_arguments", None)
        if update_get_arguments:
            overwrites, defaults = update_get_arguments(overwrites=overwrites, defaults=defaults)
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
        cls: type["SkipBasedConfig"], *, parsed_args: dict[str, tp.Any], overwrites: dict[str, tp.Any]
    ) -> tuple[dict[str, tp.Any], dict[str, tp.Any]]:
        """Update the arguments settings for the quantization configuration."""

        update_from_dict = getattr(super(), "update_from_dict", None)
        if update_from_dict:
            parsed_args, overwrites = update_from_dict(parsed_args=parsed_args, overwrites=overwrites)

        parsed_args.setdefault("skips", []).extend(omniconfig.COLLECT_PREFIX_BOOL_FIELDS(parsed_args, "skip"))
        return parsed_args, overwrites


@configclass
@dataclass
class IncludeBasedConfig(KeyEnableConfig, EnableConfig):
    """Include-based configuration.

    Args:
        includes (`list[str]`, *optional*, defaults to `[]`):
            The keys of the modules to include.
    """

    includes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        __post_init__ = getattr(super(), "__post_init__", None)
        if __post_init__:
            __post_init__()
        self.includes = sorted(set(self.includes or []))

    def is_enabled(self) -> bool:
        """Whether the kernel is enabled."""
        return super().is_enabled() and bool(self.includes)

    def is_enabled_for(self, key: str) -> bool:
        """Whether the config is enabled for the module key.

        Args:
            key (`str`):
                The key.

        Returns:
            `bool`:
                Whether the config is needed.
        """
        return self.is_enabled() and key in self.includes

    def generate_dirnames(self, *, prefix: str = "", **kwargs) -> list[str]:
        """Generate the directory names of the configuration.

        Args:
            prefix (`str`, *optional*, defaults to `""`):
                The prefix of the directory names.

        Returns:
            `list[str]`:
                The directory names. The last directory name is the modules to include.
        """
        names = []
        if self.includes:
            names = super().generate_dirnames(**kwargs)  # type: ignore
            names.append("include.[{}]".format("+".join(self.includes)))
        if prefix:
            names = [f"{prefix}.{name}" for name in names]
        return names

    @classmethod
    def update_get_arguments(
        cls: type["IncludeBasedConfig"],
        *,
        overwrites: dict[str, tp.Callable[[omniconfig.Arguments], None] | None] | None = None,
        defaults: dict[str, tp.Any] | None = None,
    ) -> tuple[dict[str, tp.Any], dict[str, tp.Any]]:
        """Update the arguments settings for the quantization configuration."""

        update_get_arguments = getattr(super(), "update_get_arguments", None)
        if update_get_arguments:
            overwrites, defaults = update_get_arguments(overwrites=overwrites, defaults=defaults)
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
        cls: type["IncludeBasedConfig"], *, parsed_args: dict[str, tp.Any], overwrites: dict[str, tp.Any]
    ) -> tuple[dict[str, tp.Any], dict[str, tp.Any]]:
        """Update the arguments settings for the quantization configuration."""

        update_from_dict = getattr(super(), "update_from_dict", None)
        if update_from_dict:
            parsed_args, overwrites = update_from_dict(parsed_args=parsed_args, overwrites=overwrites)

        parsed_args.setdefault("includes", []).extend(omniconfig.COLLECT_PREFIX_BOOL_FIELDS(parsed_args, "include"))
        return parsed_args, overwrites
