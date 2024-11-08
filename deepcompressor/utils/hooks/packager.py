# -*- coding: utf-8 -*-
"""Packagers for input and output tensors in hooks."""

import functools
import inspect
import typing as tp
from abc import ABC, abstractmethod

import torch
import torch.ao.quantization
import torch.nn as nn
import torch.utils.hooks

__all__ = [
    "BaseInputPackager",
    "SimpleInputPackager",
    "KeyedInputPackager",
    "BaseOutputPackager",
    "SimpleOutputPackager",
    "KeyedOutputPackager",
]


class BaseInputPackager(ABC):
    """Base class for input packagers."""

    @abstractmethod
    def unpack(
        self, module: nn.Module, input_args: tuple[torch.Tensor, ...], input_kwargs: dict[str, tp.Any]
    ) -> dict[int | str, torch.Tensor]:
        """Unpack inputs in inputs packager.

        Args:
            module (`nn.Module`):
                Module.
            input_args (`tuple[torch.Tensor, ...]`):
                Input arguments.
            input_kwargs (`dict[str, tp.Any]`):
                Input keyword arguments.

        Returns:
            `dict[int | str, torch.Tensor]`:
                The unpacked input tensors.
        """
        ...

    @abstractmethod
    def repack(
        self,
        tensors: dict[int | str, torch.Tensor],
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, tp.Any]]:
        """Repack inputs in inputs packager.

        Args:
            tensors (`dict[int | str, torch.Tensor]`):
                The input tensors.
            module (`nn.Module`):
                Module.
            input_args (`tuple[torch.Tensor, ...]`):
                Input arguments.
            input_kwargs (`dict[str, tp.Any]`):
                Input keyword arguments.

        Returns:
            `tuple[tuple[torch.Tensor, ...], dict[str, tp.Any]]`:
                The repacked input arguments and keyword arguments.
        """
        ...


class SimpleInputPackager(BaseInputPackager):
    def unpack(
        self, module: nn.Module, input_args: tuple[torch.Tensor, ...], input_kwargs: dict[str, tp.Any]
    ) -> dict[int | str, torch.Tensor]:
        return {0: input_args[0]}

    def repack(
        self,
        tensors: dict[int | str, torch.Tensor],
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, tp.Any]]:
        return (tensors[0], *input_args[1:]), input_kwargs


class KeyedInputPackager(BaseInputPackager):
    def __init__(self, module: nn.Module, index_or_keys: list[int | str]):
        forward_name = "forward"
        if isinstance(module.forward, functools.partial):
            if hasattr(module, "_deepcompressor_orig_forward"):
                forward_name = "_deepcompressor_orig_forward"
            else:
                # this module has been wrapped in `accelerate` package
                assert hasattr(module, "_old_forward")
                assert module._old_forward is module.forward.__wrapped__  # type: ignore
                forward_name = "_old_forward"
        signature = inspect.signature(getattr(module, forward_name))
        args, kwargs = [], []
        for key, param in signature.parameters.items():
            if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                args.append(key)
            elif param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                args.append(key)
                kwargs.append(key)
            elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                kwargs.append(key)
        self.index_key_pairs: list[tuple[int | None, str | None]] = []
        for index_or_key in index_or_keys:
            if isinstance(index_or_key, int):
                index = index_or_key
                if index >= len(args) or args[index] not in kwargs:
                    self.index_key_pairs.append((index, None))
                else:
                    self.index_key_pairs.append((index, args[index]))
            else:
                key = index_or_key
                if key in args:
                    self.index_key_pairs.append((args.index(key), key))
                else:
                    self.index_key_pairs.append((None, key))
        self.index_or_keys = index_or_keys

    def unpack(
        self, module: nn.Module, input_args: tuple[torch.Tensor, ...], input_kwargs: dict[str, tp.Any]
    ) -> dict[int | str, torch.Tensor]:
        tensors = {}
        for index_or_key, (index, key) in zip(self.index_or_keys, self.index_key_pairs, strict=True):
            if index is not None and index < len(input_args):
                tensors[index_or_key] = input_args[index]
            else:
                assert key is not None
                tensors[index_or_key] = input_kwargs.get(key, None)
        return tensors

    def repack(
        self,
        tensors: dict[int | str, torch.Tensor],
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, tp.Any]]:
        _args, _kwargs = list(input_args), dict(input_kwargs)
        for index_or_key, (index, key) in zip(self.index_or_keys, self.index_key_pairs, strict=True):
            if index is not None and index < len(_args):
                _args[index] = tensors[index_or_key]
            else:
                assert key is not None
                _kwargs[key] = tensors[index_or_key]
        return tuple(_args), _kwargs


class BaseOutputPackager(ABC):
    """Base class for output packagers."""

    @abstractmethod
    def unpack(
        self,
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
        output: tp.Any,
    ) -> dict[int | str, torch.Tensor]:
        """Unpack outputs in outputs packager.

        Args:
            module (`nn.Module`):
                Module.
            input_args (`tuple[torch.Tensor, ...]`):
                Input arguments.
            input_kwargs (`dict[str, tp.Any]`):
                Input keyword arguments.
            output (`Any`):
                Output.

        Returns:
            `dict[int | str, torch.Tensor]`:
                The unpacked output tensors.
        """
        ...

    @abstractmethod
    def repack(
        self,
        tensors: dict[int | str, torch.Tensor],
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
        output: tp.Any,
    ) -> tp.Any:
        """Repack outputs in outputs packager.

        Args:
            tensors (`dict[int | str, torch.Tensor]`):
                The output tensors.
            module (`nn.Module`):
                Module.
            input_args (`tuple[torch.Tensor, ...]`):
                Input arguments.
            input_kwargs (`dict[str, tp.Any]`):
                Input keyword arguments.
            output (`Any`):
                Output.

        Returns:
            `Any`:
                The repacked output.
        """
        ...


class SimpleOutputPackager(BaseOutputPackager):
    def unpack(
        self,
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
        output: tp.Any,
    ) -> dict[int | str, torch.Tensor]:
        if not isinstance(output, torch.Tensor):
            output = output[0]
        return {0: output}

    def repack(
        self,
        tensors: dict[int | str, torch.Tensor],
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
        output: tp.Any,
    ) -> tp.Any:
        if isinstance(output, torch.Tensor):
            return tensors[0]
        else:
            return (tensors[0], *output[1:])


class KeyedOutputPackager(BaseOutputPackager):
    def __init__(self, index_or_keys: list[int | str]):
        self.index_or_keys = index_or_keys

    def unpack(
        self,
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
        output: tp.Any,
    ) -> dict[int | str, torch.Tensor]:
        tensors = {}
        if isinstance(output, (tuple, list)):
            for index_or_key in self.index_or_keys:
                assert isinstance(index_or_key, int) and index_or_key < len(output)
                tensors[index_or_key] = output[index_or_key]
        elif isinstance(output, dict):
            for index_or_key in self.index_or_keys:
                assert isinstance(index_or_key, str) and index_or_key in output
                tensors[index_or_key] = output[index_or_key]
        else:
            assert isinstance(output, torch.Tensor)
            assert len(self.index_or_keys) == 1
            assert self.index_or_keys[0] == 0
            tensors[0] = output
        return tensors

    def repack(
        self,
        tensors: dict[int | str, torch.Tensor],
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
        output: tp.Any,
    ) -> tp.Any:
        if isinstance(output, (tuple, list)):
            _output = list(output)
            for index_or_key in self.index_or_keys:
                assert isinstance(index_or_key, int) and index_or_key < len(_output)
                _output[index_or_key] = tensors[index_or_key]
            return tuple(_output)
        elif isinstance(output, dict):
            _output = dict(output)
            for index_or_key in self.index_or_keys:
                assert isinstance(index_or_key, str) and index_or_key in _output
                _output[index_or_key] = tensors[index_or_key]
            return _output
        else:
            assert isinstance(output, torch.Tensor)
            assert len(self.index_or_keys) == 1
            assert self.index_or_keys[0] == 0
            return tensors[0]
