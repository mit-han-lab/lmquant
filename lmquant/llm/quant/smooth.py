# -*- coding: utf-8 -*-
"""LLM smooth quantization module."""

import logging
import typing as tp

import torch
import torch.nn as nn
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from lmquant.dataset import IOActivationsCache
from lmquant.quant.calib.smooth import smooth_attention, smooth_linear_modules
from lmquant.quant.quantizer.base import Quantizer
from lmquant.utils import tools

from ..dataset import LlmCalibConfig, LlmCalibrationCache
from ..nn import LlmDecoderLayerStruct, LlmModelStruct
from ..utils import get_needs_inputs_fn, get_needs_outputs_fn
from .config import LlmModuleKey, LlmQuantConfig

__all__ = ["smooth_llm"]


@torch.inference_mode()
def smooth_llm_decoder_layer(  # noqa: C901
    layer: LlmDecoderLayerStruct,
    config: LlmQuantConfig,
    smooth_cache: dict[str, torch.Tensor],
    layer_cache: dict[str, IOActivationsCache] | None = None,
    layer_kwargs: dict[str, tp.Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Smooth a large foundation model decoder layer.

    Args:
        layer (LlmLayerStruct): LLM decoder layer to smooth.
        config (LlmModelQuantConfig): Quantization configuration.
        smooth_cache (dict[str, torch.Tensor]): Smooth quantization scale caches.
        layer_caches (dict[str, IOActivationsCache]): Activation caches of the decoder layer. Defaults to ``None``.
        layer_kwargs (dict[str, tp.Any]): Keyword arguments for the decoder layer. Defaults to ``None``.

    Returns:
        dict[str, torch.Tensor]: Dictionary mapping module names to scales.
    """
    logger = logging.getLogger(f"{__name__}.SmoothQuant")
    logger.debug("- Smooth Quantizing Decoder Layer %s", layer.full_name)
    tools.logging.Formatter.indent_inc()
    layer_cache = layer_cache or {}
    layer_kwargs = layer_kwargs or {}
    # attention qk
    attn_block_name = layer.attn_block_full_name
    key = LlmModuleKey.ATTN_QK
    if config.smooth.enabled_smooth_yx and config.smooth.yx.enabled_for(key):
        logger.debug("- %s.%s", attn_block_name, key)
        cache_key = f"{attn_block_name}.{key}"
        attn_q_name = config.keywords_o[LlmModuleKey.ATTN_Q][0]
        attn_k_name = config.keywords_o[LlmModuleKey.ATTN_K][0]
        attn_q = getattr(layer.attn_block, attn_q_name)
        attn_k = getattr(layer.attn_block, attn_k_name)
        attn_q_key = f"{attn_block_name}.{attn_q_name}"
        attn_k_key = f"{attn_block_name}.{attn_k_name}"
        quant_config = config.specialize_for(key, layer_idx=layer.idx)
        smooth_cache[cache_key] = smooth_attention(
            proj_k=layer.proj_k,
            proj_q=layer.proj_q,
            scale=smooth_cache.get(cache_key, None),
            smooth_config=config.smooth.yx,
            q_quantizer=Quantizer(quant_config.ipts, channels_dim=-1, key=LlmModuleKey.ATTN_Q),
            k_quantizer=Quantizer(quant_config.opts, channels_dim=-1, key=LlmModuleKey.ATTN_K),
            qs=layer_cache.get(attn_q_key, IOActivationsCache()).outputs,
            ks=layer_cache.get(attn_k_key, IOActivationsCache()).outputs,
            q_mod=attn_q,
            k_mod=attn_k,
            eval_ipt=layer_cache.get(attn_block_name, IOActivationsCache()).inputs,
            eval_mod=layer.attn_block,
            eval_kwargs=layer.filter_layer_kwargs_to_attn_kwargs(layer_kwargs),
            num_heads=layer.config.num_query_heads,
            num_head_repeats=layer.config.num_head_repeats,
            with_rope=layer.config.with_rope,
            develop_dtype=config.develop_dtype,
        )
    # qkv projection
    key = LlmModuleKey.PROJ_QKV
    if config.smooth.enabled_smooth_xw and layer.config.do_norm_before and config.smooth.xw.enabled_for(key):
        logger.debug("- %s.%s", attn_block_name, key)
        cache_key = layer.proj_v_full_name
        quant_config = config.specialize_for(key, layer_idx=layer.idx)
        smooth_cache[cache_key] = smooth_linear_modules(
            layer.attn_ln,
            layer.proj_qkv,
            scale=smooth_cache.get(cache_key, None),
            smooth_config=config.smooth.xw,
            wgts_quantizer=Quantizer(quant_config.wgts, quant_config.wgts.calib_kernel, key=key),
            ipts_quantizer=Quantizer(quant_config.ipts, channels_dim=-1, key=key),
            ipts=layer_cache.get(cache_key, IOActivationsCache()).inputs,
            eval_ipt=layer_cache.get(attn_block_name, IOActivationsCache()).inputs,
            eval_mod=layer.attn_block,
            eval_kwargs=layer.filter_layer_kwargs_to_attn_kwargs(layer_kwargs),
            develop_dtype=config.develop_dtype,
        )
    # output projection
    key = LlmModuleKey.PROJ_OUT
    if config.smooth.enabled_smooth_xw and config.smooth.xw.enabled_for(key):
        logger.debug("- %s.%s", attn_block_name, key)
        cache_key = layer.proj_o_full_name
        quant_config = config.specialize_for(key, layer_idx=layer.idx)
        smooth_cache[cache_key] = smooth_linear_modules(
            layer.proj_v,
            layer.proj_o,
            scale=smooth_cache.get(cache_key, None),
            smooth_config=config.smooth.xw,
            wgts_quantizer=Quantizer(quant_config.wgts, quant_config.wgts.calib_kernel, key=key),
            ipts_quantizer=Quantizer(quant_config.ipts, channels_dim=-1, key=key),
            ipts=layer_cache.get(cache_key, IOActivationsCache()).inputs,
            eval_ipt=layer_cache.get(cache_key, IOActivationsCache()).inputs,
            eval_mod=layer.proj_o,
            num_heads=layer.config.num_query_heads,
            num_head_repeats=layer.config.num_head_repeats,
            develop_dtype=config.develop_dtype,
        )
    num_experts = layer.num_experts
    # ffn 1st layer projection
    key = LlmModuleKey.PROJ_1ST
    if config.smooth.enabled_smooth_xw and layer.config.do_norm_before and config.smooth.xw.enabled_for(key):
        ffn_block_name = layer.ffn_block_full_name
        logger.debug("- %s.%s", ffn_block_name, key)
        cache_key = ffn_block_name
        quant_config = config.specialize_for(key, layer_idx=layer.idx)
        smooth_cache[cache_key] = smooth_linear_modules(
            layer.ffn_ln,
            layer.proj_1st,
            scale=smooth_cache.get(cache_key, None),
            smooth_config=config.smooth.xw,
            wgts_quantizer=Quantizer(quant_config.wgts, quant_config.wgts.calib_kernel, key=key),
            ipts_quantizer=Quantizer(quant_config.ipts, channels_dim=-1, key=key),
            ipts=layer_cache.get(cache_key, IOActivationsCache()).inputs,
            eval_ipt=layer_cache.get(ffn_block_name, IOActivationsCache()).inputs,
            eval_mod=layer.ffn_block,
            extra_second_modules=[layer.router] if num_experts > 1 else None,
            develop_dtype=config.develop_dtype,
        )
    # ffn 2nd layer projection
    key = LlmModuleKey.PROJ_2ND
    if config.smooth.enabled_smooth_xw and config.smooth.xw.enabled_for(key):
        quant_config = config.specialize_for(key, layer_idx=layer.idx)
        for expert_idx in range(num_experts):
            logger.debug("- %s.%s", layer.expert_full_names[expert_idx], key)
            cache_key = layer.proj_2nd_full_names[expert_idx]
            smooth_cache[cache_key] = smooth_linear_modules(
                layer.proj_1st[expert_idx],
                layer.proj_2nd[expert_idx],
                scale=smooth_cache.get(cache_key, None),
                smooth_config=config.smooth.xw,
                wgts_quantizer=Quantizer(quant_config.wgts, quant_config.wgts.calib_kernel, key=key),
                ipts_quantizer=Quantizer(quant_config.ipts, channels_dim=-1, key=key),
                ipts=layer_cache.get(cache_key, IOActivationsCache()).inputs,
                eval_ipt=layer_cache.get(cache_key, IOActivationsCache()).inputs,
                eval_mod=layer.proj_2nd[expert_idx],
                develop_dtype=config.develop_dtype,
            )
    tools.logging.Formatter.indent_dec()
    return smooth_cache


@torch.inference_mode()
def smooth_llm(
    model: nn.Module | LlmModelStruct,
    /,
    quant_config: LlmQuantConfig,
    tokenizer: nn.Module | None = None,
    calib_config: LlmCalibConfig | None = None,
    smooth_cache: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Smooth the large foundation model.

    Args:
        model (nn.Module | LlmStruct): Model to be smoothed.
        tokenizer (nn.Module): Tokenizer.
        quant_config (LlmModelQuantConfig): Module quantization config.
        calib_config (LlmCalibrationConfig, optional): Calibration config. Defaults to ``LMCalibrationConfig().``
        smooth_cache (dict[str, torch.Tensor], optional): Smooth quantization scale caches. Defaults to ``None``.

    Returns:
        dict[str, torch.Tensor]: Smooth quantization scale caches.
    """
    if not isinstance(model, LlmModelStruct):
        model = LlmModelStruct.build(model)
    assert isinstance(model, LlmModelStruct)
    smooth_cache = smooth_cache or {}
    if not smooth_cache:
        with logging_redirect_tqdm():
            for _, (layer, layer_cache, layer_kwargs) in tqdm(
                LlmCalibrationCache(calib_config).iter_layer_activations(
                    model,
                    tokenizer,
                    needs_inputs_fn=get_needs_inputs_fn(config=quant_config),
                    needs_outputs_fn=get_needs_outputs_fn(config=quant_config),
                    needs_samples_caching=False,
                ),
                desc="smooth quantization",
                leave=False,
                total=model.config.num_hidden_layers,
            ):
                smooth_llm_decoder_layer(
                    layer=layer,
                    config=quant_config,
                    smooth_cache=smooth_cache,
                    layer_cache=layer_cache,
                    layer_kwargs=layer_kwargs,
                )
    else:
        for layer in model.backbone_struct.layer_structs:
            smooth_llm_decoder_layer(layer=layer, config=quant_config, smooth_cache=smooth_cache)
    return smooth_cache
