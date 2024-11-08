# -*- coding: utf-8 -*-
"""Diffusion smooth quantization module."""

import typing as tp

import torch
import torch.nn as nn
from tqdm import tqdm

from deepcompressor.calib.smooth import ActivationSmoother, smooth_linear_modules
from deepcompressor.data.cache import IOTensorsCache
from deepcompressor.quantizer import Quantizer
from deepcompressor.utils import tools
from deepcompressor.utils.hooks import KeyedInputPackager

from ..nn.struct import (
    DiffusionAttentionStruct,
    DiffusionBlockStruct,
    DiffusionFeedForwardStruct,
    DiffusionModelStruct,
    DiffusionTransformerBlockStruct,
)
from .config import DiffusionQuantConfig
from .utils import get_needs_inputs_fn, wrap_joint_attn

__all__ = ["smooth_diffusion"]


@torch.inference_mode()
def smooth_diffusion_attention(
    attn: DiffusionAttentionStruct,
    config: DiffusionQuantConfig,
    smooth_cache: dict[str, torch.Tensor],
    block_cache: dict[str, IOTensorsCache] | None = None,
    block_kwargs: dict[str, tp.Any] | None = None,
) -> dict[str, torch.Tensor]:
    logger = tools.logging.getLogger(f"{__name__}.SmoothQuant")
    # attention qk
    if config.smooth.enabled_attn:
        logger.debug("- %s.k", attn.name)
        raise NotImplementedError("Not implemented yet")
    return smooth_cache


@torch.inference_mode()
def smooth_diffusion_qkv_proj(
    attn: DiffusionAttentionStruct,
    config: DiffusionQuantConfig,
    smooth_cache: dict[str, torch.Tensor],
    block_cache: dict[str, IOTensorsCache] | None = None,
    block_kwargs: dict[str, tp.Any] | None = None,
) -> dict[str, torch.Tensor]:
    logger = tools.logging.getLogger(f"{__name__}.SmoothQuant")
    # region qkv projection
    module_key = attn.qkv_proj_key
    needs_quant = config.enabled_wgts and config.wgts.is_enabled_for(module_key)
    needs_quant = needs_quant or (config.enabled_ipts and config.ipts.is_enabled_for(module_key))
    if needs_quant and config.smooth.enabled_proj and config.smooth.proj.is_enabled_for(module_key):
        logger.debug("- %s.qkv_proj", attn.name)
        prevs = None
        if attn.parent.norm_type.startswith("layer_norm"):
            if not hasattr(attn.parent.module, "pos_embed") or attn.parent.module.pos_embed is None:
                prevs = attn.parent.attn_norms[attn.idx]
                assert isinstance(prevs, nn.LayerNorm)
        cache_key = attn.q_proj_name
        smooth_cache[cache_key] = smooth_linear_modules(
            prevs,
            attn.qkv_proj,
            scale=smooth_cache.get(cache_key, None),
            config=config.smooth.proj,
            weight_quantizer=Quantizer(config.wgts, key=module_key, low_rank=config.wgts.low_rank),
            input_quantizer=Quantizer(config.ipts, channels_dim=-1, key=module_key),
            inputs=block_cache[attn.q_proj_name].inputs if block_cache else None,
            eval_inputs=block_cache[attn.name].inputs if block_cache else None,
            eval_module=attn,
            eval_kwargs=attn.filter_kwargs(block_kwargs),
            develop_dtype=config.develop_dtype,
        )
        if prevs is None:
            # we need to register forward pre hook to smooth inputs
            if attn.module.group_norm is None and attn.module.spatial_norm is None:
                ActivationSmoother(
                    smooth_cache[cache_key],
                    channels_dim=-1,
                    input_packager=KeyedInputPackager(attn.module, [0]),
                ).as_hook().register(attn.module)
            else:
                ActivationSmoother(smooth_cache[cache_key], channels_dim=-1).as_hook().register(attn.qkv_proj)
        for m in attn.qkv_proj:
            m.in_smooth_cache_key = cache_key
    # endregion
    if attn.is_self_attn():
        return smooth_cache
    # region additional qkv projection
    module_key = attn.add_qkv_proj_key
    needs_quant = config.enabled_wgts and config.wgts.is_enabled_for(module_key)
    needs_quant = needs_quant or (config.enabled_ipts and config.ipts.is_enabled_for(module_key))
    needs_quant = needs_quant and attn.add_k_proj is not None
    if needs_quant and config.smooth.enabled_proj and config.smooth.proj.is_enabled_for(module_key):
        logger.debug("- %s add_qkv_proj", attn.name)
        prevs = None
        add_attn_norm = attn.parent.add_attn_norms[attn.idx]
        if isinstance(add_attn_norm, nn.LayerNorm):
            prevs = add_attn_norm
        cache_key = attn.add_k_proj_name
        smooth_cache[cache_key] = smooth_linear_modules(
            prevs,
            attn.add_qkv_proj,
            scale=smooth_cache.get(cache_key, None),
            config=config.smooth.proj,
            weight_quantizer=Quantizer(config.wgts, key=module_key, low_rank=config.wgts.low_rank),
            input_quantizer=Quantizer(config.ipts, channels_dim=-1, key=module_key),
            inputs=block_cache[attn.add_k_proj_name].inputs if block_cache else None,
            eval_inputs=block_cache[attn.name].inputs if block_cache else None,
            eval_module=wrap_joint_attn(attn, indexes=1) if attn.is_joint_attn() else attn,
            eval_kwargs=attn.filter_kwargs(block_kwargs),
            develop_dtype=config.develop_dtype,
        )
        if prevs is None:
            # we need to register forward pre hook to smooth inputs
            ActivationSmoother(smooth_cache[cache_key], channels_dim=-1).as_hook().register(attn.add_qkv_proj)
        for m in attn.add_qkv_proj:
            m.in_smooth_cache_key = cache_key
    # endregion
    return smooth_cache


@torch.inference_mode()
def smooth_diffusion_out_proj(
    attn: DiffusionAttentionStruct,
    config: DiffusionQuantConfig,
    smooth_cache: dict[str, torch.Tensor],
    block_cache: dict[str, IOTensorsCache] | None = None,
    block_kwargs: dict[str, tp.Any] | None = None,
) -> dict[str, torch.Tensor]:
    logger = tools.logging.getLogger(f"{__name__}.SmoothQuant")
    module_keys = []
    for module_key in (attn.out_proj_key, attn.add_out_proj_key) if attn.is_joint_attn() else (attn.out_proj_key,):
        needs_quant = config.enabled_wgts and config.wgts.is_enabled_for(module_key)
        needs_quant = needs_quant or (config.enabled_ipts and config.ipts.is_enabled_for(module_key))
        if needs_quant and config.smooth.enabled_proj and config.smooth.proj.is_enabled_for(module_key):
            module_keys.append(module_key)
    if not module_keys:
        return smooth_cache
    exclusive = False
    if config.enabled_wgts and config.wgts.enabled_low_rank:
        exclusive = config.wgts.low_rank.exclusive
        config.wgts.low_rank.exclusive = True
    if len(module_keys) == 1 and module_keys[0] == attn.out_proj_key:
        logger.debug("- %s.out_proj", attn.name)
        module_key = attn.out_proj_key
        cache_key = attn.o_proj_name
        if attn.is_joint_attn():
            prevs = [attn.v_proj, attn.add_v_proj]
        elif attn.is_cross_attn():
            prevs = [attn.add_v_proj]
        else:
            prevs = [attn.v_proj]
        smooth_cache[cache_key] = smooth_linear_modules(
            prevs,
            attn.o_proj,
            scale=smooth_cache.get(cache_key, None),
            config=config.smooth.proj,
            weight_quantizer=Quantizer(config.wgts, key=module_key, low_rank=config.wgts.low_rank),
            input_quantizer=Quantizer(config.ipts, channels_dim=-1, key=module_key),
            inputs=block_cache[attn.o_proj_name].inputs if block_cache else None,
            eval_inputs=block_cache[attn.o_proj_name].inputs if block_cache else None,
            eval_module=attn.o_proj,
            extra_modules=[attn.add_o_proj] if attn.is_joint_attn() else None,
            develop_dtype=config.develop_dtype,
        )
    elif len(module_keys) == 1 and module_keys[0] == attn.add_out_proj_key:
        assert attn.is_joint_attn()
        logger.debug("- %s.add_out_proj", attn.name)
        module_key = attn.add_out_proj_key
        cache_key = attn.add_o_proj_name
        prevs = [attn.v_proj, attn.add_v_proj]
        smooth_cache[cache_key] = smooth_linear_modules(
            prevs,
            attn.add_o_proj,
            scale=smooth_cache.get(cache_key, None),
            config=config.smooth.proj,
            weight_quantizer=Quantizer(config.wgts, key=module_key, low_rank=config.wgts.low_rank),
            input_quantizer=Quantizer(config.ipts, channels_dim=-1, key=module_key),
            inputs=block_cache[attn.add_o_proj_name].inputs if block_cache else None,
            eval_inputs=block_cache[attn.add_o_proj_name].inputs if block_cache else None,
            eval_module=attn.add_o_proj,
            extra_modules=[attn.o_proj],
            develop_dtype=config.develop_dtype,
        )
    else:
        assert attn.is_joint_attn()
        logger.debug("- %s.out_proj + %s.add_out_proj", attn.name, attn.name)
        module_key = attn.out_proj_key
        cache_key = attn.o_proj_name
        prevs = [attn.v_proj, attn.add_v_proj]
        smooth_cache[cache_key] = smooth_linear_modules(
            prevs,
            [attn.o_proj, attn.add_o_proj],
            scale=smooth_cache.get(cache_key, None),
            config=config.smooth.proj,
            weight_quantizer=Quantizer(config.wgts, key=module_key, low_rank=config.wgts.low_rank),
            input_quantizer=Quantizer(config.ipts, channels_dim=-1, key=module_key),
            inputs=block_cache[attn.o_proj_name].inputs if block_cache else None,
            eval_inputs=block_cache[attn.name].inputs if block_cache else None,
            eval_module=wrap_joint_attn(attn, indexes=(0, 1)),
            eval_kwargs=attn.filter_kwargs(block_kwargs),
            develop_dtype=config.develop_dtype,
        )
    if config.enabled_wgts and config.wgts.enabled_low_rank:
        config.wgts.low_rank.exclusive = exclusive
    for prev in prevs:
        prev.out_smooth_cache_key = cache_key
    attn.o_proj.in_smooth_cache_key = cache_key
    if attn.add_o_proj is not None:
        attn.add_o_proj.in_smooth_cache_key = cache_key
    return smooth_cache


@torch.inference_mode()
def smooth_diffusion_up_proj(
    ffn_norm: nn.Module,
    ffn: DiffusionFeedForwardStruct,
    config: DiffusionQuantConfig,
    smooth_cache: dict[str, torch.Tensor],
    block_cache: dict[str, IOTensorsCache] | None = None,
) -> dict[str, torch.Tensor]:
    assert len(ffn.up_projs) == 1
    logger = tools.logging.getLogger(f"{__name__}.SmoothQuant")
    # ffn up projection
    module_key = ffn.up_proj_key
    needs_quant = config.enabled_wgts and config.wgts.is_enabled_for(module_key)
    needs_quant = needs_quant or (config.enabled_ipts and config.ipts.is_enabled_for(module_key))
    if needs_quant and config.smooth.enabled_proj and config.smooth.proj.is_enabled_for(module_key):
        logger.debug("- %s.up_proj", ffn.name)
        prevs = None
        if isinstance(ffn_norm, nn.LayerNorm) and ffn.parent.norm_type in ["ada_norm", "layer_norm"]:
            prevs = ffn_norm
        cache_key = ffn.up_proj_name
        smooth_cache[cache_key] = smooth_linear_modules(
            prevs,
            ffn.up_projs,
            scale=smooth_cache.get(cache_key, None),
            config=config.smooth.proj,
            weight_quantizer=Quantizer(config.wgts, key=module_key, low_rank=config.wgts.low_rank),
            input_quantizer=Quantizer(config.ipts, channels_dim=-1, key=module_key),
            inputs=block_cache[ffn.up_proj_name].inputs if block_cache else None,
            eval_inputs=block_cache[ffn.up_proj_name].inputs if block_cache else None,
            eval_module=ffn.up_proj,
            develop_dtype=config.develop_dtype,
        )
        if prevs is None:
            ActivationSmoother(smooth_cache[cache_key], channels_dim=-1).as_hook().register(ffn.up_proj)
        for proj in ffn.up_projs:
            proj.in_smooth_cache_key = cache_key
    return smooth_cache


@torch.inference_mode()
def smooth_diffusion_down_proj(
    ffn: DiffusionFeedForwardStruct,
    config: DiffusionQuantConfig,
    smooth_cache: dict[str, torch.Tensor],
    block_cache: dict[str, IOTensorsCache] | None = None,
) -> dict[str, torch.Tensor]:
    logger = tools.logging.getLogger(f"{__name__}.SmoothQuant")
    # ffn down projection
    module_key = ffn.down_proj_key.upper()
    needs_quant = config.enabled_wgts and config.wgts.is_enabled_for(module_key)
    needs_quant = needs_quant or (config.enabled_ipts and config.ipts.is_enabled_for(module_key))
    if needs_quant and config.smooth.enabled_proj and config.smooth.proj.is_enabled_for(module_key):
        logger.debug("- %s.down_proj", ffn.name)
        prev = None
        if ffn.config.intermediate_act_type.endswith("_glu"):
            prev = ffn.up_proj
        cache_key = ffn.down_proj_name
        unsigned_ipts = getattr(ffn.down_proj, "unsigned", False)
        config_ipts = config.unsigned_ipts if unsigned_ipts else config.ipts
        smooth_cache[cache_key] = smooth_linear_modules(
            prev,
            ffn.down_proj,
            scale=smooth_cache.get(cache_key, None),
            config=config.smooth.proj,
            weight_quantizer=Quantizer(config.wgts, key=module_key, low_rank=config.wgts.low_rank),
            input_quantizer=Quantizer(config_ipts, channels_dim=-1, key=module_key),
            inputs=block_cache[ffn.down_proj_name].inputs if block_cache else None,
            eval_inputs=block_cache[ffn.down_proj_name].inputs if block_cache else None,
            eval_module=ffn.down_proj,
            develop_dtype=config.develop_dtype,
        )
        ffn.down_proj.in_smooth_cache_key = cache_key
        if prev is None:
            ActivationSmoother(smooth_cache[cache_key], channels_dim=-1).as_hook().register(ffn.down_proj)
        else:
            prev.out_smooth_cache_key = cache_key
    return smooth_cache


@torch.inference_mode()
def smooth_diffusion_parallel_qkv_up_proj(
    block: DiffusionTransformerBlockStruct,
    config: DiffusionQuantConfig,
    smooth_cache: dict[str, torch.Tensor],
    block_cache: dict[str, IOTensorsCache] | None = None,
    block_kwargs: dict[str, tp.Any] | None = None,
) -> dict[str, torch.Tensor]:
    assert block.parallel
    assert len(block.ffn_struct.up_projs) == 1
    logger = tools.logging.getLogger(f"{__name__}.SmoothQuant")
    # region qkv proj + up proj
    attn, ffn = block.attn_structs[0], block.ffn_struct
    module_key = attn.qkv_proj_key
    needs_quant = config.enabled_wgts and config.wgts.is_enabled_for(module_key)
    needs_quant = needs_quant or (config.enabled_ipts and config.ipts.is_enabled_for(module_key))
    if needs_quant and config.smooth.enabled_proj and config.smooth.proj.is_enabled_for(module_key):
        logger.debug("- %s.qkv_proj + %s.up_proj", attn.name, ffn.name)
        cache_key = attn.q_proj_name
        modules = attn.qkv_proj + ffn.up_projs
        smooth_cache[cache_key] = smooth_linear_modules(
            None,
            modules,
            scale=smooth_cache.get(cache_key, None),
            config=config.smooth.proj,
            weight_quantizer=Quantizer(config.wgts, key=module_key, low_rank=config.wgts.low_rank),
            input_quantizer=Quantizer(config.ipts, channels_dim=-1, key=module_key),
            inputs=block_cache[attn.q_proj_name].inputs if block_cache else None,
            eval_inputs=block_cache[block.name].inputs if block_cache else None,
            eval_module=block,
            eval_kwargs=block_kwargs,
            splits=[len(attn.qkv_proj)],
            develop_dtype=config.develop_dtype,
        )
        ActivationSmoother(smooth_cache[cache_key], channels_dim=-1).as_hook().register(modules)
        for m in modules:
            m.in_smooth_cache_key = cache_key
    # endregion
    # region add qkv proj + add up proj
    if attn.is_self_attn():
        if block.add_ffn_struct is not None:
            smooth_cache = smooth_diffusion_up_proj(
                ffn_norm=block.add_ffn_norm,
                ffn=block.add_ffn_struct,
                config=config,
                smooth_cache=smooth_cache,
                block_cache=block_cache,
            )
        return smooth_cache
    module_key = attn.add_qkv_proj_key
    needs_quant = config.enabled_wgts and config.wgts.is_enabled_for(module_key)
    needs_quant = needs_quant or (config.enabled_ipts and config.ipts.is_enabled_for(module_key))
    if needs_quant and config.smooth.enabled_proj and config.smooth.proj.is_enabled_for(module_key):
        add_ffn = block.add_ffn_struct
        cache_key = attn.add_k_proj_name
        modules = attn.add_qkv_proj
        if add_ffn is None:
            logger.debug("- %s.add_qkv_proj", attn.name)
        else:
            logger.debug("- %s.add_qkv_proj + %s.up_proj", attn.name, add_ffn.name)
            modules = modules + add_ffn.up_projs
        smooth_cache[cache_key] = smooth_linear_modules(
            None,
            modules,
            scale=smooth_cache.get(cache_key, None),
            config=config.smooth.proj,
            weight_quantizer=Quantizer(config.wgts, key=module_key, low_rank=config.wgts.low_rank),
            input_quantizer=Quantizer(config.ipts, channels_dim=-1, key=module_key),
            inputs=block_cache[attn.add_k_proj_name].inputs if block_cache else None,
            eval_inputs=block_cache[block.name].inputs if block_cache else None,
            eval_module=block,
            eval_kwargs=block_kwargs,
            develop_dtype=config.develop_dtype,
        )
        ActivationSmoother(smooth_cache[cache_key], channels_dim=-1).as_hook().register(modules)
        for m in modules:
            m.in_smooth_cache_key = cache_key
    # endregion
    return smooth_cache


@torch.inference_mode()
def smooth_diffusion_sequential_transformer_block(
    block: DiffusionTransformerBlockStruct,
    config: DiffusionQuantConfig,
    smooth_cache: dict[str, torch.Tensor],
    block_cache: dict[str, IOTensorsCache] | None = None,
    block_kwargs: dict[str, tp.Any] | None = None,
) -> dict[str, torch.Tensor]:
    assert not block.parallel
    for attn in block.attn_structs:
        smooth_cache = smooth_diffusion_attention(
            attn=attn, config=config, smooth_cache=smooth_cache, block_cache=block_cache, block_kwargs=block_kwargs
        )
        smooth_cache = smooth_diffusion_qkv_proj(
            attn=attn, config=config, smooth_cache=smooth_cache, block_cache=block_cache, block_kwargs=block_kwargs
        )
        smooth_cache = smooth_diffusion_out_proj(
            attn=attn, config=config, smooth_cache=smooth_cache, block_cache=block_cache, block_kwargs=block_kwargs
        )
    if block.ffn_struct is not None:
        smooth_cache = smooth_diffusion_up_proj(
            ffn_norm=block.ffn_norm,
            ffn=block.ffn_struct,
            config=config,
            smooth_cache=smooth_cache,
            block_cache=block_cache,
        )
        smooth_cache = smooth_diffusion_down_proj(
            ffn=block.ffn_struct, config=config, smooth_cache=smooth_cache, block_cache=block_cache
        )
    if block.add_ffn_struct is not None:
        smooth_cache = smooth_diffusion_up_proj(
            ffn_norm=block.add_ffn_norm,
            ffn=block.add_ffn_struct,
            config=config,
            smooth_cache=smooth_cache,
            block_cache=block_cache,
        )
        smooth_cache = smooth_diffusion_down_proj(
            ffn=block.add_ffn_struct, config=config, smooth_cache=smooth_cache, block_cache=block_cache
        )
    return smooth_cache


@torch.inference_mode()
def smooth_diffusion_parallel_transformer_block(
    block: DiffusionTransformerBlockStruct,
    config: DiffusionQuantConfig,
    smooth_cache: dict[str, torch.Tensor],
    block_cache: dict[str, IOTensorsCache] | None = None,
    block_kwargs: dict[str, tp.Any] | None = None,
) -> dict[str, torch.Tensor]:
    assert block.parallel
    assert block.ffn_struct is not None
    for attn in block.attn_structs:
        smooth_cache = smooth_diffusion_attention(
            attn=attn, config=config, smooth_cache=smooth_cache, block_cache=block_cache, block_kwargs=block_kwargs
        )
        if attn.idx == 0:
            smooth_cache = smooth_diffusion_parallel_qkv_up_proj(
                block=block,
                config=config,
                smooth_cache=smooth_cache,
                block_cache=block_cache,
                block_kwargs=block_kwargs,
            )
        else:
            smooth_cache = smooth_diffusion_qkv_proj(
                attn=attn, config=config, smooth_cache=smooth_cache, block_cache=block_cache, block_kwargs=block_kwargs
            )
        smooth_cache = smooth_diffusion_out_proj(
            attn=attn, config=config, smooth_cache=smooth_cache, block_cache=block_cache, block_kwargs=block_kwargs
        )
    smooth_cache = smooth_diffusion_down_proj(
        ffn=block.ffn_struct, config=config, smooth_cache=smooth_cache, block_cache=block_cache
    )
    if block.add_ffn_struct is not None:
        smooth_cache = smooth_diffusion_down_proj(
            ffn=block.add_ffn_struct, config=config, smooth_cache=smooth_cache, block_cache=block_cache
        )
    return smooth_cache


@torch.inference_mode()
def smooth_diffusion_linear(
    module_key: str,
    module_name: str,
    module: nn.Linear | nn.Conv2d,
    config: DiffusionQuantConfig,
    smooth_cache: dict[str, torch.Tensor],
    layer_cache: dict[str, IOTensorsCache] | None = None,
) -> dict[str, torch.Tensor]:
    assert isinstance(module, (nn.Linear, nn.Conv2d))
    logger = tools.logging.getLogger(f"{__name__}.SmoothQuant")
    needs_quant = config.enabled_wgts and config.wgts.is_enabled_for(module_key)
    needs_quant = needs_quant or (config.enabled_ipts and config.ipts.is_enabled_for(module_key))
    if needs_quant and config.smooth.enabled_proj and config.smooth.proj.is_enabled_for(module_key):
        logger.debug("- %s", module_name)
        cache_key = module_name
        channels_dim = -1 if isinstance(module, nn.Linear) else 1
        smooth_cache[cache_key] = smooth_linear_modules(
            None,
            module,
            scale=smooth_cache.get(cache_key, None),
            config=config.smooth.proj,
            weight_quantizer=Quantizer(config.wgts, key=module_key),
            input_quantizer=Quantizer(config.ipts, channels_dim=channels_dim, key=module_key),
            inputs=layer_cache[module_name].inputs if layer_cache else None,
            eval_inputs=layer_cache[module_name].inputs if layer_cache else None,
            eval_module=module,
            develop_dtype=config.develop_dtype,
        )
        ActivationSmoother(smooth_cache[cache_key], channels_dim=channels_dim).as_hook().register(module)
        module.in_smooth_cache_key = cache_key
    return smooth_cache


@torch.inference_mode()
def smooth_diffusion_layer(
    layer: DiffusionBlockStruct,
    config: DiffusionQuantConfig,
    smooth_cache: dict[str, torch.Tensor],
    layer_cache: dict[str, IOTensorsCache] | None = None,
    layer_kwargs: dict[str, tp.Any] | None = None,
) -> None:
    """Smooth a single diffusion model block.

    Args:
        layer (`DiffusionBlockStruct`):
            The diffusion block.
        config (`DiffusionQuantConfig`):
            The quantization configuration.
        smooth_cache (`dict[str, torch.Tensor]`):
            The smoothing scales cache.
        layer_cache (`dict[str, IOTensorsCache]`, *optional*):
            The layer cache.
        layer_kwargs (`dict[str, tp.Any]`, *optional*):
            The layer keyword arguments.
    """
    logger = tools.logging.getLogger(f"{__name__}.SmoothQuant")
    logger.debug("- Smoothing Diffusion Block %s", layer.name)
    tools.logging.Formatter.indent_inc()
    layer_cache = layer_cache or {}
    layer_kwargs = layer_kwargs or {}
    # We skip resnets since we currently cannot scale the Swish function
    visited: set[str] = set()
    for module_key, module_name, module, parent, _ in layer.named_key_modules():
        if isinstance(parent, (DiffusionAttentionStruct, DiffusionFeedForwardStruct)):
            block = parent.parent
            assert isinstance(block, DiffusionTransformerBlockStruct)
            if block.name not in visited:
                logger.debug("- Smoothing Transformer Block %s", block.name)
                visited.add(block.name)
                tools.logging.Formatter.indent_inc()
                if block.parallel:
                    smooth_cache = smooth_diffusion_parallel_transformer_block(
                        block=block,
                        config=config,
                        smooth_cache=smooth_cache,
                        block_cache=layer_cache,
                        block_kwargs=layer_kwargs,
                    )
                else:
                    smooth_cache = smooth_diffusion_sequential_transformer_block(
                        block=block,
                        config=config,
                        smooth_cache=smooth_cache,
                        block_cache=layer_cache,
                        block_kwargs=layer_kwargs,
                    )
                tools.logging.Formatter.indent_dec()
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            logger.debug("- Smoothing Module %s", module_name)
            tools.logging.Formatter.indent_inc()
            smooth_cache = smooth_diffusion_linear(
                module_key=module_key,
                module_name=module_name,
                module=module,
                config=config,
                smooth_cache=smooth_cache,
                layer_cache=layer_cache,
            )
            tools.logging.Formatter.indent_dec()
        else:
            needs_quant = config.enabled_wgts and config.wgts.is_enabled_for(module_key)
            needs_quant = needs_quant or (config.enabled_ipts and config.ipts.is_enabled_for(module_key))
            if needs_quant and config.smooth.enabled_proj and config.smooth.proj.is_enabled_for(module_key):
                raise NotImplementedError(f"Module {module_name} is not supported for smoothing")
            logger.debug("- Skipping Module %s", module_name)
    tools.logging.Formatter.indent_dec()


@torch.inference_mode()
def smooth_diffusion(
    model: nn.Module | DiffusionModelStruct,
    config: DiffusionQuantConfig,
    smooth_cache: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Smooth the diffusion model.

    Args:
        model (`nn.Module` or `DiffusionModelStruct`):
            The diffusion model.
        config (`DiffusionQuantConfig`):
            The quantization configuration.
        smooth_cache (`dict[str, torch.Tensor]`, *optional*):
            The smoothing scales cache.

    Returns:
        `dict[str, torch.Tensor]`:
            The smoothing scales cache.
    """
    if not isinstance(model, DiffusionModelStruct):
        model = DiffusionModelStruct.construct(model)
    assert isinstance(model, DiffusionModelStruct)
    smooth_cache = smooth_cache or {}
    if not smooth_cache:
        with tools.logging.redirect_tqdm():
            for _, (layer, layer_cache, layer_kwargs) in tqdm(
                config.calib.build_loader().iter_layer_activations(
                    model,
                    needs_inputs_fn=get_needs_inputs_fn(model, config),
                    skip_pre_modules=True,
                    skip_post_modules=True,
                ),
                desc="smoothing",
                leave=False,
                total=model.num_blocks,
                dynamic_ncols=True,
            ):
                smooth_diffusion_layer(
                    layer=layer,
                    config=config,
                    smooth_cache=smooth_cache,
                    layer_cache=layer_cache,
                    layer_kwargs=layer_kwargs,
                )
    else:
        for layer in model.block_structs:
            smooth_diffusion_layer(layer=layer, config=config, smooth_cache=smooth_cache)
    return smooth_cache
