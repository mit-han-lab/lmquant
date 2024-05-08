# -*- coding: utf-8 -*-
"""Monkey-patching utilities."""

import copy
import functools
import types

__all__ = ["copy_func"]


def copy_func(f, globals=None):
    """Copied from https://stackoverflow.com/a/13503277/2988730 (@unutbu)
    and https://github.com/spcl/QuaRot/blob/main/fake_quant/monkeypatch.py."""
    if globals is None:
        globals = f.__globals__
    g = types.FunctionType(f.__code__, globals, name=f.__name__, argdefs=f.__defaults__, closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__module__ = f.__module__
    g.__kwdefaults__ = copy.copy(f.__kwdefaults__)
    return g
