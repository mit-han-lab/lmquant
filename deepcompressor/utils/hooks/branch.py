# -*- coding: utf-8 -*-
"""Branch hook module."""

import typing as tp

import torch
import torch.nn as nn

from .hook import IOHook
from .packager import BaseInputPackager, BaseOutputPackager

__all__ = ["AccumBranchHook"]


class AccumBranchHook(IOHook):
    branch: nn.Module | None

    def __init__(
        self,
        branch: nn.Module | None,
        input_packager: BaseInputPackager | None = None,
        output_packager: BaseOutputPackager | None = None,
    ):
        super().__init__(pre=True, post=True, input_packager=input_packager, output_packager=output_packager)
        self.branch = branch
        self.tensor = None

    def pre_forward(
        self, module: nn.Module, input_args: tuple[torch.Tensor, ...], input_kwargs: dict[str, tp.Any]
    ) -> None:
        """Pre-forward function.

        Args:
            module (nn.Module): Module.
            input_args (tuple[torch.Tensor, ...]): Input arguments.
            input_kwargs (dict[str, tp.Any]): Input keyword arguments.
        """
        tensors = self.input_packager.unpack(module, input_args, input_kwargs)
        assert len(tensors) == 1, "BranchHook only supports single input tensor"
        self.tensor = next(iter(tensors.values()))
        return None

    def post_forward(
        self,
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
        output: tuple[torch.Tensor, ...],
    ) -> tp.Any:
        """Post-forward function.

        Args:
            module (nn.Module): Module.
            input_args (tuple[torch.Tensor, ...]): Input arguments.
            input_kwargs (dict[str, tp.Any]): Input keyword arguments.
            output (tuple[torch.Tensor, ...]): Output.
        """
        output_tensors = self.output_packager.unpack(module, input_args, input_kwargs, output)
        assert len(output_tensors) == 1, "LoRAHook only supports single output tensor"
        output_key, output_tensor = next(iter(output_tensors.items()))
        if self.branch is not None:
            output_tensor = output_tensor + self.branch(self.tensor)
        self.tensor = None
        return self.output_packager.repack({output_key: output_tensor}, module, input_args, input_kwargs, output)
