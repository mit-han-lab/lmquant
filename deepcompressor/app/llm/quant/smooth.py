# -*- coding: utf-8 -*-
"""LLM smooth quantization module."""

import typing as tp

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from deepcompressor.calib.smooth import smooth_attention, smooth_linear_modules
from deepcompressor.data.cache import IOTensorsCache
from deepcompressor.quantizer.processor import Quantizer
from deepcompressor.utils import tools

from ..nn.struct import LlmModelStruct, LlmTransformerBlockStruct
from .config import LlmQuantConfig
from .utils import get_needs_inputs_fn, get_needs_outputs_fn

__all__ = ["smooth_llm"]


@torch.inference_mode()
def smooth_llm_layer(  # noqa: C901
    layer: LlmTransformerBlockStruct,
    config: LlmQuantConfig,
    smooth_cache: dict[str, torch.Tensor],
    layer_cache: dict[str, IOTensorsCache] | None = None,
    layer_kwargs: dict[str, tp.Any] | None = None,
) -> None:
    """Smooth a large language model layer.

    Args:
        layer (`LlmTransformerBlockStruct`):
            Large language model layer to smooth.
        config (`LlmQuantConfig`):
            Quantization configuration.
        smooth_cache (`dict[str, torch.Tensor]`):
            Smoothing scale caches.
        layer_caches (`dict[str, IOTensorsCache]` or `None`, *optional*, defaults to `None`):
            Activation caches of the layer.
        layer_kwargs (`dict[str, tp.Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments for the layer.
    """
    logger = tools.logging.getLogger(f"{__name__}.SmoothQuant")
    logger.debug("- Smoothing %s", layer.name)
    tools.logging.Formatter.indent_inc()
    layer_cache = layer_cache or {}
    layer_kwargs = layer_kwargs or {}
    attn, ffn = layer.attn_struct, layer.ffn_struct
    # region attention qk
    if config.smooth.enabled_attn:
        logger.debug("- %s.%s", attn.name, attn.k_rkey)
        cache_key = f"{attn.name}.{attn.k_rkey}"
        smooth_cache[cache_key] = smooth_attention(
            k_proj=attn.k_proj,
            q_proj=attn.q_proj,
            scale=smooth_cache.get(cache_key, None),
            config=config.smooth.attn,
            query_quantizer=Quantizer(config.opts, channels_dim=-1, key=attn.q_key),
            key_quantizer=Quantizer(config.opts, channels_dim=-1, key=attn.k_key),
            queries=layer_cache[attn.q_name].outputs if layer_cache else None,
            keys=layer_cache[attn.k_name].outputs if layer_cache else None,
            attn_q=attn.q,
            attn_k=attn.k,
            eval_inputs=layer_cache[attn.name].inputs if layer_cache else None,
            eval_module=attn,
            eval_kwargs=attn.filter_kwargs(layer_kwargs),
            num_heads=attn.config.num_query_heads,
            num_head_repeats=attn.config.num_head_repeats,
            with_rope=attn.config.with_rope,
            develop_dtype=config.develop_dtype,
        )
    # endregion
    # region qkv projection
    if (
        config.smooth.enabled_proj
        and attn.config.do_norm_before
        and config.smooth.proj.is_enabled_for(attn.qkv_proj_key)
    ):
        logger.debug("- %s.%s", attn.name, attn.qkv_proj_rkey)
        cache_key = attn.v_proj_name
        smooth_cache[cache_key] = smooth_linear_modules(
            attn.parent.attn_norms[attn.idx],
            attn.qkv_proj,
            scale=smooth_cache.get(cache_key, None),
            config=config.smooth.proj,
            weight_quantizer=Quantizer(config.wgts, key=attn.qkv_proj_key),
            input_quantizer=Quantizer(config.ipts, channels_dim=-1, key=attn.qkv_proj_key),
            inputs=layer_cache[attn.q_proj_name].inputs if layer_cache else None,
            eval_inputs=layer_cache[attn.name].inputs if layer_cache else None,
            eval_module=attn,
            eval_kwargs=attn.filter_kwargs(layer_kwargs),
            develop_dtype=config.develop_dtype,
        )
    # endregion
    # region output projection
    if config.smooth.enabled_proj and config.smooth.proj.is_enabled_for(attn.out_proj_key):
        logger.debug("- %s.%s", attn.name, attn.out_proj_rkey)
        cache_key = attn.o_proj_name
        smooth_cache[cache_key] = smooth_linear_modules(
            attn.v_proj,
            attn.o_proj,
            scale=smooth_cache.get(cache_key, None),
            config=config.smooth.proj,
            weight_quantizer=Quantizer(config.wgts, key=attn.out_proj_key),
            input_quantizer=Quantizer(config.ipts, channels_dim=-1, key=attn.out_proj_key),
            inputs=layer_cache[cache_key].inputs if layer_cache else None,
            eval_inputs=layer_cache[cache_key].inputs if layer_cache else None,
            eval_module=attn.o_proj,
            num_heads=attn.config.num_query_heads,
            num_head_repeats=attn.config.num_head_repeats,
            develop_dtype=config.develop_dtype,
        )
    # endregion
    num_experts = ffn.config.num_experts
    # region up projection
    if config.smooth.enabled_proj and ffn.config.do_norm_before and config.smooth.proj.is_enabled_for(ffn.up_proj_key):
        logger.debug("- %s.%s", ffn.name, ffn.up_proj_rkey)
        cache_key = ffn.name
        smooth_cache[cache_key] = smooth_linear_modules(
            ffn.parent.ffn_norm,
            ffn.up_projs,
            scale=smooth_cache.get(cache_key, None),
            config=config.smooth.proj,
            weight_quantizer=Quantizer(config.wgts, key=ffn.up_proj_key),
            input_quantizer=Quantizer(config.ipts, channels_dim=-1, key=ffn.up_proj_key),
            inputs=layer_cache[ffn.name].inputs if layer_cache else None,
            eval_inputs=layer_cache[ffn.name].inputs if layer_cache else None,
            eval_module=ffn,
            extra_modules=[ffn.moe_gate] if num_experts > 1 else None,
            develop_dtype=config.develop_dtype,
        )
    # endregion
    # region down projection
    if config.smooth.enabled_proj and config.smooth.proj.is_enabled_for(ffn.down_proj_key):
        for expert_idx in range(num_experts):
            logger.debug("- %s.%s", ffn.expert_names[expert_idx], ffn.down_proj_rkey)
            cache_key = ffn.down_proj_names[expert_idx]
            smooth_cache[cache_key] = smooth_linear_modules(
                ffn.up_projs[expert_idx],
                ffn.down_projs[expert_idx],
                scale=smooth_cache.get(cache_key, None),
                config=config.smooth.proj,
                weight_quantizer=Quantizer(config.wgts, key=ffn.down_proj_key),
                input_quantizer=Quantizer(config.ipts, channels_dim=-1, key=ffn.down_proj_key),
                inputs=layer_cache[ffn.down_proj_names[expert_idx]].inputs if layer_cache else None,
                eval_inputs=layer_cache[ffn.down_proj_names[expert_idx]].inputs if layer_cache else None,
                eval_module=ffn.down_projs[expert_idx],
                develop_dtype=config.develop_dtype,
            )
    # endregion
    tools.logging.Formatter.indent_dec()


@torch.inference_mode()
def smooth_llm(
    model: nn.Module | LlmModelStruct,
    /,
    config: LlmQuantConfig,
    tokenizer: PreTrainedTokenizer | None = None,
    smooth_cache: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Smooth the large language model.

    Args:
        model (`nn.Module` or `LlmStruct`):
            Model to be smoothed.
        config (`LlmQuantConfig`):
            Quantization configuration.
        tokenizer (`PreTrainedTokenizer`, *optional*, defaults to `None`):
            Tokenizer.
        smooth_cache (`dict[str, torch.Tensor]`, *optional*, defaults to `None`):
            Smoothing scale caches.

    Returns:
        `dict[str, torch.Tensor]`:
            Dictionary mapping module names to smoothing scales.
    """
    if not isinstance(model, LlmModelStruct):
        model = LlmModelStruct.construct(model)
    assert isinstance(model, LlmModelStruct)
    smooth_cache = smooth_cache or {}
    if not smooth_cache:
        with tools.logging.redirect_tqdm():
            for _, (layer, layer_cache, layer_kwargs) in tqdm(
                config.calib.build_loader(tokenizer).iter_layer_activations(
                    model,
                    needs_inputs_fn=get_needs_inputs_fn(model=model, config=config),
                    needs_outputs_fn=get_needs_outputs_fn(model=model, config=config),
                ),
                desc="smoothing",
                leave=False,
                total=len(model.backbone_struct.layer_structs),
                dynamic_ncols=True,
            ):
                smooth_llm_layer(
                    layer=layer,
                    config=config,
                    smooth_cache=smooth_cache,
                    layer_cache=layer_cache,
                    layer_kwargs=layer_kwargs,
                )
    else:
        for layer in model.backbone_struct.layer_structs:
            smooth_llm_layer(layer=layer, config=config, smooth_cache=smooth_cache)
    return smooth_cache
