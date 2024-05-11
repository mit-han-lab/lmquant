# -*- coding: utf-8 -*-
"""Utility functions for Module Struct."""

import typing as tp
from dataclasses import dataclass, field

import torch.nn as nn

__all__ = ["join_name", "ModuleStruct"]


def join_name(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


@dataclass
class ModuleStruct:
    module: nn.Module = field(repr=False)
    parent: tp.Optional["ModuleStruct"]
    idx: int = field(init=False, repr=False)
    name: str
    full_name: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not hasattr(self, "idx"):
            self.idx = 0
        if self.parent is None:
            assert self.idx == 0, "idx must be 0 if parent is None"
            assert not self.name, "name must be empty if parent is None"
            self.full_name = self.name
        else:
            assert self.name, "name must not be empty if parent is not None"
            if self.idx != 0:
                assert self.name.endswith(f".{self.idx}"), "name must end with the index"
            self.full_name = join_name(self.parent.full_name, self.name)
