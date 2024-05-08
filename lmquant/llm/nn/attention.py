# -*- coding: utf-8 -*-
"""Llama model patcher."""

import functools
import logging

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import rotate_half

from lmquant.utils.patch import copy_func

__all__ = ["patch_attention", "RotaryEmbedding"]


def update_rotary_cos_sin(
    cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.LongTensor, unsqueeze_dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update the cos and sin tensors with new position_ids.

    Args:
        cos (torch.Tensor): Cosine tensor.
        sin (torch.Tensor): Sine tensor.
        position_ids (torch.LongTensor): Position ids.

    Returns:
        tuple[torch.Tensor]: Updated cos and sin tensors.
    """
    assert unsqueeze_dim in (1, 2), f"unsqueeze_dim must be 1 or 2, got {unsqueeze_dim}"
    if position_ids is None:
        if cos.ndim == 2:
            cos = cos.unsqueeze(0)
        if sin.ndim == 2:
            sin = sin.unsqueeze(0)
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
    else:
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim] if unsqueeze_dim == 1
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # [bs, seq_len, 1, dim] if unsqueeze_dim == 2
    assert cos.ndim == 4, f"cos must have 4 dimensions, got {cos.ndim}"
    assert sin.ndim == 4, f"sin must have 4 dimensions, got {sin.ndim}"
    return cos, sin


class RotaryEmbedding(nn.Module):
    """Rotary embedding for attention."""

    def __init__(self) -> None:
        """Initialize the class."""
        super().__init__()

    def forward(
        self, states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
    ) -> torch.Tensor:
        """Apply rotary embedding to states.

        Args:
            states (torch.Tensor): States.
            cos (torch.Tensor): Cosine tensor.
            sin (torch.Tensor): Sine tensor.

        Returns:
            torch.Tensor: States with rotary embedding.
        """
        states = (states * cos) + (rotate_half(states) * sin)
        if unsqueeze_dim == 1:
            batch_size, num_heads, seq_len, head_dim = states.shape
            states = states.transpose(1, 2)
        else:
            batch_size, seq_len, num_heads, head_dim = states.shape
        return states.view(batch_size, seq_len, num_heads * head_dim)


def apply_rotary_pos_emb(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.LongTensor = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    assert unsqueeze_dim == 1 or unsqueeze_dim == 2, f"unsqueeze_dim must be 1 or 2, got {unsqueeze_dim}"
    if unsqueeze_dim == 1:
        batch_size, _, seq_len, head_dim = q.shape
    else:
        batch_size, seq_len, _, head_dim = q.shape
    cos, sin = update_rotary_cos_sin(cos, sin, position_ids=position_ids, unsqueeze_dim=unsqueeze_dim)
    q = self.q_rotary_emb(q, cos=cos, sin=sin, unsqueeze_dim=unsqueeze_dim)
    k = self.k_rotary_emb(k, cos=cos, sin=sin, unsqueeze_dim=unsqueeze_dim)
    q = q.view(batch_size, seq_len, -1, head_dim)
    k = k.view(batch_size, seq_len, -1, head_dim)
    if unsqueeze_dim == 1:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
    return q, k


def patch_attention(model: nn.Module) -> nn.Module:
    """Patch attention."""
    logger = logging.getLogger(f"{__name__}.ModelPatcher")
    for module_name, module in model.named_modules():
        classname = type(module).__name__
        if classname.lower().endswith("attention"):
            forward_name = ""
            if isinstance(module.forward, functools.partial):
                if hasattr(module, "_lmquant_orig_forward"):
                    logger.info(f"- Attention in {module_name} has already been patched")
                else:
                    # this module has been wrapped in ``accelerate`` package
                    assert hasattr(module, "_old_forward")
                    assert module._old_forward is module.forward.__wrapped__
                    if "apply_rotary_pos_emb" in module._old_forward.__func__.__globals__:
                        forward_name = "_old_forward"
            else:
                if "apply_rotary_pos_emb" in module.forward.__func__.__globals__:
                    forward_name = "forward"
            if forward_name:
                logger.info(f"- Patching {classname}.{forward_name} in {module_name}")
                module.q_rotary_emb = RotaryEmbedding()
                module.k_rotary_emb = RotaryEmbedding()
                module.apply_rotary_pos_emb = functools.partial(apply_rotary_pos_emb, module)
                module._lmquant_orig_forward = getattr(module, forward_name)

                orig_forward = module._lmquant_orig_forward.__func__
                new_globals = dict(orig_forward.__globals__)
                new_globals["apply_rotary_pos_emb"] = module.apply_rotary_pos_emb
                new_forward = copy_func(orig_forward, new_globals)
                setattr(module, forward_name, new_forward.__get__(module))

    return model
