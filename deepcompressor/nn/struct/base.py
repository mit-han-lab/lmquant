# -*- coding: utf-8 -*-
"""Utility functions for Module Struct."""

import types
import typing as tp
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch.nn as nn

from ...utils.common import join_name

__all__ = ["BaseModuleStruct"]


@dataclass(kw_only=True)
class BaseModuleStruct(ABC):
    _factories: tp.ClassVar[
        dict[
            type[nn.Module],
            tp.Callable[[nn.Module, tp.Optional["BaseModuleStruct"], str, str, str, int], tp.Self],
        ]
    ]

    module: nn.Module = field(repr=False, kw_only=False)
    """The nn.Module instance."""
    parent: tp.Optional["BaseModuleStruct"] = field(repr=False, default=None)
    """The parent module struct that contains this module struct."""
    fname: str = field(default="")
    """The field name in the parent module struct."""
    idx: int = field(default=0)
    """The index of this module struct if it is in a list of the parent module struct."""
    rname: str
    """The relative name of this module from the parent module."""
    name: str = field(init=False, repr=False)
    """The absolute name of this module from the root module."""
    rkey: str
    """The relative key of this module from the parent module."""
    key: str = field(init=False, repr=False)
    """The absolute key of this module from the root module."""

    def __post_init__(self) -> None:
        if self.parent is None:
            assert self.idx == 0, f"idx must be 0 if parent is None, got {self.idx}"
            assert not self.fname, f"field name must be empty if parent is None, got {self.fname}"
            assert not self.rname, f"relative name must be empty if parent is None, got {self.rname}"
            assert not self.rkey, f"relative key must be empty if parent is None, got {self.rkey}"
            self.name = self.rname
            self.key = self.rkey
        else:
            assert self.fname, f"field name must not be empty if parent is not None, got {self.fname}"
            self.name = join_name(self.parent.name, self.rname)
            self.key = join_name(self.parent.key, self.rkey, sep="_")
            if hasattr(self.parent, f"{self.fname}_names"):
                assert self.name == getattr(self.parent, f"{self.fname}_names")[self.idx]
            else:
                assert self.idx == 0, f"idx must be 0 if parent is not None and {self.fname}_names not found"
                assert self.name == getattr(self.parent, f"{self.fname}_name")

    def __call__(self, *args: tp.Any, **kwds: tp.Any) -> tp.Any:
        return self.module(*args, **kwds)

    @abstractmethod
    def named_key_modules(self) -> tp.Generator[tp.Tuple[str, str, nn.Module, "BaseModuleStruct", str], None, None]:
        """Yield (module_key, module_name, module, parent_struct, field_name) tuple."""
        ...

    @classmethod
    def get_default_keys(cls) -> list[str]:
        """Get the default keys."""
        return []

    @classmethod
    def register_factory(
        cls,
        module_types: type[nn.Module] | tuple[type[nn.Module], ...],
        /,
        factory: tp.Callable[[nn.Module, tp.Optional["BaseModuleStruct"], str, str, str, int], tp.Self],
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a factory that constructs a module struct from a module.

        Args:
            module_types (`type[nn.Module]` or `tuple[type[nn.Module], ...]`):
                The module type(s).
            factory (`Callable[[nn.Module, BaseModuleStruct, str, str, str, int], BaseModuleStruct]`):
                The factory function.
            overwrite (`bool`, *optional*, defaults to `False`):
                Whether to overwrite the existing factory for the module type(s).
        """

        def unpack(module_types):
            if isinstance(module_types, tp._UnionGenericAlias) or isinstance(module_types, types.UnionType):
                args = []
                for arg in module_types.__args__:
                    args.extend(unpack(arg))
                return args
            elif isinstance(module_types, tuple):
                args = []
                for arg in module_types:
                    args.extend(unpack(arg))
                return args
            return [module_types]

        module_types = unpack(module_types)
        for module_type in module_types:
            # assert issubclass(module_type, nn.Module), f"{module_type} is not a subclass of nn.Module"
            if not hasattr(cls, "_factories"):
                cls._factories = {}
            if not overwrite:
                assert module_type not in cls._factories, f"factory for {module_type} already exists"
            cls._factories[module_type] = factory

    @classmethod
    def construct(
        cls,
        module: nn.Module,
        /,
        parent: tp.Optional["BaseModuleStruct"] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> tp.Self:
        """Construct a module struct from a module.

        Args:
            module (`nn.Module`):
                The module instance.
            parent (`BaseModuleStruct` or `None`, *optional*, defaults to `None):
                The parent module struct that contains this module struct.
            rname (`str`, *optional*, defaults to `""`):
                The relative name of this module from the parent module.
            rkey (`str`, *optional*, defaults to `""`):
                The relative key of this module from the parent module.
            idx (`int`, *optional*, defaults to `0`):
                The index of this module struct if it is in a list of the parent module struct.

        Returns:
            `Self`:
                The module struct.
        """
        factory = cls._factories[type(module)]
        return factory(module, parent=parent, fname=fname, rname=rname, rkey=rkey, idx=idx, **kwargs)
