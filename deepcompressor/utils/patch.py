# -*- coding: utf-8 -*-
"""Monkey-patching utilities."""

import copy
import functools
import types
import typing

import torch.nn as nn

__all__ = ["copy_func", "get_module_parents_map"]


def copy_func(f: types.FunctionType, globals: dict[str, typing.Any] | None = None):
    """Copied from https://stackoverflow.com/a/13503277/2988730 (@unutbu)
    and https://github.com/spcl/QuaRot/blob/main/fake_quant/monkeypatch.py.

    Copy a function.

    Args:
        f (`types.FunctionType`):
            Function to be copied.
        globals (`dict[str, typing.Any]` or `None`, *optional*, defaults to `None`):
            Globals.

    Returns:
        `types.FunctionType`:
            Copied function.
    """
    if globals is None:
        globals = f.__globals__
    g = types.FunctionType(f.__code__, globals, name=f.__name__, argdefs=f.__defaults__, closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__module__ = f.__module__
    g.__kwdefaults__ = copy.copy(f.__kwdefaults__)  # type: ignore
    return g


def get_module_parents_map(
    module: nn.Module, name: str = "", parents_map: dict[nn.Module, list[tuple[str, nn.Module, str]]] | None = None
) -> dict[nn.Module, list[tuple[str, nn.Module, str]]]:
    """Get module parents map.

    Args:
        module (`nn.Module`):
            Module.
        name (`str`, *optional*, defaults to `""`):
            Name.
        parents_map (`dict[nn.Module, list[tuple[str, nn.Module, str]]]`, *optional*, defaults to `None`):
            Parents map.

    Returns:
        `dict[nn.Module, list[tuple[str, nn.Module, str]]]`:
            Module parents map. The key is the child module and the value is a list of tuples.
            Each tuple contains the name of the parent module, the parent module,
            and the child module name in the parent module.
    """
    if parents_map is None:
        parents_map = {}
    for child_name, child_module in module._modules.items():
        if child_module is None:
            continue
        parents_map.setdefault(child_module, []).append((name, module, child_name))
        get_module_parents_map(child_module, f"{name}.{child_name}" if name else child_name, parents_map)
    return parents_map
