# -*- coding: utf-8 -*-

import typing as tp

import diffusers
import packaging.version
import torch
import torch.nn as nn
from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor2_0,
    FluxAttnProcessor2_0,
    JointAttnProcessor2_0,
)

from deepcompressor.nn.patch.sdpa import ScaleDotProductAttention

__all__ = ["DiffusionAttentionProcessor"]


if packaging.version.Version(diffusers.__version__) >= packaging.version.Version("0.31"):
    from diffusers.models.embeddings import apply_rotary_emb

    def apply_flux_rope(query, key, image_rotary_emb):
        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)
        return query, key

else:
    from diffusers.models.attention_processor import apply_rope as apply_flux_rope


class DiffusionAttentionProcessor(nn.Module):
    def __init__(
        self,
        orig: AttnProcessor2_0 | FluxAttnProcessor2_0 | JointAttnProcessor2_0,
        sdpa: ScaleDotProductAttention | None = None,
    ) -> None:
        super().__init__()
        self.orig = orig
        if orig.__class__.__name__.startswith("Flux"):
            self.rope = apply_flux_rope
        elif isinstance(orig, (AttnProcessor2_0, JointAttnProcessor2_0)):
            self.rope = None
        else:
            raise NotImplementedError(f"Unsupported AttentionProcessor: {orig}")
        self.sdpa = sdpa or ScaleDotProductAttention()

    def __call__(  # noqa: C901
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: tp.Optional[torch.Tensor] = None,
        attention_mask: tp.Optional[torch.Tensor] = None,
        image_rotary_emb: tp.Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert len(args) == 0 and kwargs.get("scale", None) is None
        assert attn.spatial_norm is None
        assert attn.group_norm is None
        assert attn.norm_cross is None
        assert not attn.residual_connection
        assert attn.rescale_output_factor == 1.0
        heads = attn.heads
        head_dim = attn.inner_dim // heads
        kv_heads = attn.inner_kv_dim // head_dim
        assert attn.scale == head_dim**-0.5

        input_ndim, input_shape = hidden_states.dim(), hidden_states.size()
        if input_ndim > 3:
            hidden_states = hidden_states.view(input_shape[0], input_shape[1], -1).transpose(1, 2)
        batch_size, input_length, _ = hidden_states.shape
        context_ndim, context_shape, context_length = None, None, None
        if encoder_hidden_states is not None:
            context_ndim, context_shape = encoder_hidden_states.ndim, encoder_hidden_states.shape
            assert context_shape[0] == batch_size
            if context_ndim > 3:
                encoder_hidden_states = encoder_hidden_states.view(batch_size, context_shape[1], -1).transpose(1, 2)
            context_length = encoder_hidden_states.shape[1]

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, context_length or input_length, batch_size)
            attention_mask = attention_mask.view(batch_size, heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key, value, add_query, add_key, add_value = None, None, None, None, None
        if hasattr(attn, "add_k_proj"):
            if attn.to_k is not None:
                key = attn.to_k(hidden_states)
                value = attn.to_v(hidden_states)
            add_key = attn.add_k_proj(encoder_hidden_states)
            add_value = attn.add_v_proj(encoder_hidden_states)
            if hasattr(attn, "add_q_proj"):
                add_query = attn.add_q_proj(encoder_hidden_states)
        else:
            if attn.is_cross_attention:
                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)
            else:
                assert encoder_hidden_states is None
                key = attn.to_k(hidden_states)
                value = attn.to_v(hidden_states)
        hidden_states, encoder_hidden_states = None, None

        query = query.view(batch_size, -1, heads, head_dim).transpose(1, 2)
        if key is not None:
            key = key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        if add_query is not None:
            add_query = add_query.view(batch_size, -1, heads, head_dim).transpose(1, 2)
        if add_key is not None:
            add_key = add_key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
            add_value = add_value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)

        if kv_heads != heads:
            heads_per_kv_head = heads // kv_heads
            if key is not None:
                key = torch.repeat_interleave(key, heads_per_kv_head, dim=1)
                value = torch.repeat_interleave(value, heads_per_kv_head, dim=1)
            if add_key is not None:
                add_key = torch.repeat_interleave(add_key, heads_per_kv_head, dim=1)
                add_value = torch.repeat_interleave(add_value, heads_per_kv_head, dim=1)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
            key = attn.norm_k(key)
        if attn.norm_added_q is not None:
            add_query = attn.norm_added_q(add_query)
            add_key = attn.norm_added_k(add_key)

        if add_query is not None:
            query = torch.cat([add_query, query], dim=2)
        if add_key is not None:
            if key is None:
                key, value = add_key, add_value
            else:
                key = torch.cat([add_key, key], dim=2)
                value = torch.cat([add_value, value], dim=2)
        del add_query, add_key, add_value

        if image_rotary_emb is not None:
            query, key = self.rope(query, key, image_rotary_emb)

        hidden_states = self.sdpa(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.inner_dim)
        hidden_states = hidden_states.to(query.dtype)

        if hidden_states.shape[1] > input_length:
            encoder_hidden_states = hidden_states[:, :context_length]
            hidden_states = hidden_states[:, context_length:]

        if hasattr(attn, "to_out"):
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
        if hasattr(attn, "to_add_out"):
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim > 3:
            hidden_states = hidden_states.transpose(-1, -2).reshape(input_shape)
        if encoder_hidden_states is not None and context_ndim > 3:
            assert encoder_hidden_states.ndim == 3
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(context_shape)

        if encoder_hidden_states is None:
            return hidden_states
        return hidden_states, encoder_hidden_states
