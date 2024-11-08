# -*- coding: utf-8 -*-
"""nn.Module Hook."""

import typing as tp
from collections import defaultdict

import torch
import torch.ao.quantization
import torch.nn as nn
import torch.utils.hooks

from .packager import BaseInputPackager, BaseOutputPackager, SimpleInputPackager, SimpleOutputPackager

__all__ = ["Hook", "EarlyStopException", "EarlyStopHook", "IOHook"]


class Hook:
    """Base class for hook."""

    handles: dict[nn.Module, list[torch.utils.hooks.RemovableHandle]]
    pre: bool
    post: bool
    activated: bool

    def __init__(self, *, pre: bool, post: bool) -> None:
        """Initialize the hook.

        Args:
            pre (`bool`):
                Whether the hook should be called before the forward pass.
            post (`bool`):
                Whether the hook should be called after the forward pass.

        Raises:
            AssertionError:
                If both `pre` and `post` are `False`.
        """
        self.handles = defaultdict(list)
        self.pre = pre
        self.post = post
        self.activated = True
        assert self.pre or self.post, "At least one of pre and post must be True."

    def is_in_hook(self) -> bool:
        """Whether the hook is an in-hook."""
        return self.pre and not self.post

    def is_out_hook(self) -> bool:
        """Whether the hook is an out-hook."""
        return not self.pre and self.post

    def is_inout_hook(self) -> bool:
        """Whether the hook is an in-out-hook."""
        return self.pre and self.post

    def activate(self) -> tp.Self:
        """Activate the hook."""
        self.activated = True
        return self

    def deactivate(self) -> tp.Self:
        """Deactivate the hook."""
        self.activated = False
        return self

    def pre_forward(
        self, module: nn.Module, input_args: tuple[torch.Tensor, ...], input_kwargs: dict[str, tp.Any]
    ) -> tp.Any:
        """Pre-forward function.

        Args:
            module (`nn.Module`):
                Module to process.
            input_args (`tuple[torch.Tensor, ...]`):
                Input arguments.
            input_kwargs (`dict[str, tp.Any]`):
                Input keyword arguments.
        """
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
            module (`nn.Module`):
                Module to process.
            input_args (`tuple[torch.Tensor, ...]`):
                Input arguments.
            input_kwargs (`dict[str, tp.Any]`):
                Input keyword arguments.
            output (`tuple[torch.Tensor, ...]`):
                Output.
        """
        return None

    def __call__(self, *args, **kwargs) -> tp.Any:
        if not self.activated:
            return None
        n = len(args) + len(kwargs)
        if n == 3:
            return self.pre_forward(*args, **kwargs)
        elif n == 4:
            return self.post_forward(*args, **kwargs)
        else:
            raise ValueError(f"Invalid number of arguments: {n}")

    def register(
        self,
        module: nn.Module | tp.Iterable[nn.Module],
        prepend: bool | tuple[bool, bool] = False,
        always_call: bool = False,
    ) -> tp.Self:
        """Register the hook to the module(s).

        Args:
            module (`nn.Module` or `Iterable[nn.Module]`):
                The module(s).
            prepend (`bool` or `tuple[bool, bool]`, *optional*, defaults to `False`):
                Whether to prepend the hook.
                If a tuple, the first element is for pre-hook and the second element is for post-hook.
            always_call (`bool`, *optional*, defaults to `False`):
                Whether to always call the hook. This is only used for post-hooks.
        """
        if isinstance(module, nn.Module):
            module = [module]
        prepends = (prepend, prepend) if isinstance(prepend, bool) else prepend
        if self.pre:
            for mod in module:
                self.handles[mod].append(mod.register_forward_pre_hook(self, prepend=prepends[0], with_kwargs=True))
        if self.post:
            for mod in module:
                self.handles[mod].append(
                    mod.register_forward_hook(self, prepend=prepends[1], with_kwargs=True, always_call=always_call)
                )
        return self

    def remove(self, module: nn.Module | tp.Iterable[nn.Module] | None = None) -> tp.Self:
        """Remove the hook from the module(s).

        Args:
            module (`nn.Module` or `Iterable[nn.Module]`, *optional*, defaults to `None`):
                The module(s) to remove the hook from. If `None`, remove the hook from all modules.
        """
        if module is None:
            for handles in self.handles.values():
                for handle in handles:
                    handle.remove()
                handles.clear()
            self.handles.clear()
            return self
        if isinstance(module, nn.Module):
            module = [module]
        for mod in module:
            handles = self.handles.pop(mod, [])
            for handle in handles:
                handle.remove()
            handles.clear()
        return self


class EarlyStopException(Exception):
    """Early stop exception."""

    pass


class EarlyStopHook(Hook):
    def __init__(self):
        super().__init__(pre=False, post=True)

    def pre_forward(self, *args, **kwargs) -> None:
        raise EarlyStopException()


class IOHook(Hook):
    """Base class for IO hooks."""

    input_packager: BaseInputPackager
    """Input packager, used to unpack and repack the input arguments."""
    output_packager: BaseOutputPackager
    """Output packager, used to unpack and repack the output."""

    def __init__(
        self,
        *,
        pre: bool,
        post: bool,
        input_packager: BaseInputPackager | None = None,
        output_packager: BaseOutputPackager | None = None,
    ):
        """Initialize the IO hook.

        Args:
            pre (`bool`):
                Whether the hook should be called before the forward pass.
            post (`bool`):
                Whether the hook should be called after the forward pass.
            input_packager (`BaseInputPackager`, *optional*, defaults to `None`):
                Input packager, used to unpack and repack the input arguments.
            output_packager (`BaseOutputPackager`, *optional*, defaults to `None`):
                Output packager, used to unpack and repack the output.
        """
        super().__init__(pre=pre, post=post)
        if pre:
            self.input_packager = input_packager or SimpleInputPackager()
            assert isinstance(self.input_packager, BaseInputPackager)
        else:
            self.input_packager = None
        if post:
            self.output_packager = output_packager or SimpleOutputPackager()
            assert isinstance(self.output_packager, BaseOutputPackager)
        else:
            self.output_packager = None
