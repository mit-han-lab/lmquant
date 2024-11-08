# -*- coding: utf-8 -*-
"""Quantization scale module."""

import typing as tp

import torch

__all__ = ["QuantScale"]


class QuantScale:
    data: torch.Tensor
    _children: list["QuantScale"]
    _leaves: list[torch.Tensor]

    def __init__(self):
        self.data, self._children, self._leaves = None, [], []  # type: ignore

    @property
    def num_children(self) -> int:
        """Get the number of children."""
        return len(self._children)

    @property
    def num_leaves(self) -> int:
        """Get the number of leaves."""
        return len(self._leaves)

    def is_quantized(self) -> bool:
        """Check if the scale is quantized."""
        return self.data is not None and bool(self._leaves or all(child.is_quantized() for child in self._children))

    def get_child(self, index: int) -> "QuantScale":
        """Get a child scale."""
        return self._children[index]

    def append(self, scale: tp.Union[torch.Tensor, "QuantScale"]) -> "QuantScale":
        """Append a scale."""
        if isinstance(scale, torch.Tensor):
            assert not self._children, "Cannot append a tensor scale to a non-leaf QuantScale."
            self.data = _join_scale_tensor(self.data, scale)
            self._leaves.append(scale)
        elif isinstance(scale, QuantScale):
            assert not self._leaves, "Cannot append a non-leaf QuantScale to a leaf QuantScale."
            self.data = _join_scale_tensor(self.data, scale.data)
            self._children.append(scale)
        else:
            raise TypeError(f"Unsupported scale type: {type(scale)}")
        return self

    def extend(self, scale: "QuantScale") -> "QuantScale":
        """Extend with another QuantScale."""
        self.data = _join_scale_tensor(self.data, scale.data)
        if scale._children:
            assert not self._leaves, "Cannot extend a leaf QuantScale with a non-leaf QuantScale."
            self._children.extend(scale._children)
        elif scale._leaves:
            assert not scale._children, "Cannot extend a non-leaf QuantScale with a leaf QuantScale."
            self._leaves.extend(scale._leaves)
        return self

    def join(self, scale: "QuantScale") -> "QuantScale":
        """Return a new QuantScale by joining with another QuantScale."""
        return QuantScale().append(self).append(scale)

    def remove_zero(self) -> "QuantScale":
        """Remove zero scales."""
        self.data[self.data == 0] = 1
        return self

    def state_dict(
        self,
        param_name: str,
        device: torch.device | str = "cpu",
        flatten: bool = True,
        level_base: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Get the state dictionary."""
        if self._children:
            state_dict = {}
            for i, child in enumerate(self._children):
                child_param_name = param_name if flatten else f"{param_name}.{i}"
                child_level_base = len(state_dict) if flatten else 0
                child_state_dict = child.state_dict(child_param_name, device, flatten, child_level_base)
                state_dict.update(child_state_dict)
            return state_dict
        else:
            return {f"{param_name}.{level_base + i}": leaf.to(device) for i, leaf in enumerate(self._leaves)}


def _join_scale_tensor(global_scale: torch.Tensor | None, local_scale: torch.Tensor) -> torch.Tensor:
    """Multiply the local scale tensor by the global scale tensor.

    Args:
        global_scale (`torch.Tensor` or `None`):
            Global scale tensor.
        local_scale (`torch.Tensor`):
            Local scale tensor.

    Returns:
        `torch.Tensor`:
            The compounded scale tensor.
    """
    # global_scale: (#gs_g0, 1, #gs_g1, 1, #gs_g2, 1, ...)
    # local_scale:  (#ss_g0, 1, #ss_g1, 1, #ss_g2, 1, ...) -> (#gs_g0, rs0, #gs_g1, rs1, #gs_g2, rs2, ...)
    shape = local_scale.shape
    return (
        local_scale
        if global_scale is None
        else local_scale.view(
            tuple(
                global_scale.shape[i] if j == 0 else local_scale.shape[i] // global_scale.shape[i]
                for i in range(0, len(global_scale.shape), 2)
                for j in range(2)
            )
        ).mul(global_scale)
    ).view(shape)
