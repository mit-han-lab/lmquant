# -*- coding: utf-8 -*-
"""Tensor processor."""

import abc
import typing as tp

import torch
import torch.ao.quantization
import torch.nn as nn
import torch.utils.hooks

from .hook import IOHook
from .packager import BaseInputPackager, BaseOutputPackager

__all__ = ["BaseTensorProcessor", "ProcessHook"]


class BaseTensorProcessor(abc.ABC):
    @abc.abstractmethod
    def is_enabled(self) -> bool: ...

    @abc.abstractmethod
    def get_input_packager(self) -> BaseInputPackager | None: ...

    @abc.abstractmethod
    def get_output_packager(self) -> BaseOutputPackager | None: ...

    @abc.abstractmethod
    def process(self, tensor: torch.Tensor) -> torch.Tensor: ...

    def as_hook(
        self, func: tp.Callable[[torch.Tensor], torch.Tensor] | None = None, *, is_output: bool = False
    ) -> "ProcessHook":
        """Convert the processor to a hook.

        Args:
            func (`Callable[[torch.Tensor], torch.Tensor]` or `None`, *optional*, defaults to `None`):
                Function to process the tensors.
            is_output (`bool`, *optional*, defaults to `False`):
                Whether to process the output tensors.

        Returns:
            `ProcessHook`:
                The hook for processing the tensor.
        """
        return ProcessHook(self, func, is_output=is_output)


class ProcessHook(IOHook):
    def __init__(
        self,
        processor: BaseTensorProcessor,
        func: tp.Callable[[torch.Tensor], torch.Tensor] | None = None,
        is_output: bool = False,
    ):
        super().__init__(
            pre=not is_output,
            post=is_output,
            input_packager=processor.get_input_packager(),
            output_packager=processor.get_output_packager(),
        )
        self.processor = processor
        self.func = func

    def process(self, tensors: dict[int | str, torch.Tensor]) -> dict[int | str, torch.Tensor]:
        for k, x in tensors.items():
            assert isinstance(x, torch.Tensor)
            if self.func is not None:
                tensors[k] = self.func(x)
            else:
                tensors[k] = self.processor.process(x)
        return tensors

    def pre_forward(
        self, module: nn.Module, input_args: tuple[torch.Tensor, ...], input_kwargs: dict[str, tp.Any]
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, tp.Any]]:
        if not self.processor.is_enabled():
            return input_args, input_kwargs
        return self.input_packager.repack(
            self.process(self.input_packager.unpack(module, input_args, input_kwargs)), module, input_args, input_kwargs
        )

    def post_forward(
        self,
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
        output: tuple[torch.Tensor, ...],
    ) -> tp.Any:
        if not self.processor.is_enabled():
            return output
        return self.output_packager.repack(
            self.process(self.output_packager.unpack(module, input_args, input_kwargs, output)),
            module,
            input_args,
            input_kwargs,
            output,
        )
