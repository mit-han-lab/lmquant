# -*- coding: utf-8 -*-
"""Sparse attention module."""

import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ScaleDotProductAttention"]


class ScaleDotProductAttention(nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: tp.Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: tp.Optional[float] = None,
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
        )
