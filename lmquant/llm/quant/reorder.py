# -*- coding: utf-8 -*-
"""LLM quantization channel reordering module."""

import gc
import logging
import typing as tp

import torch
import torch.nn as nn
import torch.utils
import torch.utils.hooks
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from lmquant.dataset import IOActivationsCache
from lmquant.dataset.transform import TransformFn
from lmquant.quant.calib.calibrator.reorder import ChannelOrderCalibrator
from lmquant.quant.quantizer.base import Quantizer

from ..dataset import LlmCalibConfig, LlmCalibrationCache
from ..nn import LlmDecoderLayerStruct, LlmModelStruct
from ..utils import get_needs_inputs_fn
from .config import LlmModuleKey, LlmQuantConfig

__all__ = ["reorder_llm"]


def _extend_out_params_(
    params: list[tuple[nn.Parameter, int]],
    modules: list[nn.Linear | nn.Embedding, nn.LayerNorm],
    channels_dim: int,
) -> list[tuple[nn.Parameter, int]]:
    for module in modules:
        if module is None:
            continue
        params.append((module.weight, channels_dim))
        if hasattr(module, "bias") and module.bias is not None:
            params.append((module.bias, 0))
    return params


@torch.inference_mode()
def reorder_llm_decoder_layer(  # noqa: C901
    layer: LlmDecoderLayerStruct,
    config: LlmQuantConfig,
    reorder_cache: dict[str, torch.Tensor],
    residual_calibrator: ChannelOrderCalibrator | None = None,
    layer_cache: dict[str, IOActivationsCache] | None = None,
    layer_kwargs: dict[str, tp.Any] | None = None,
) -> dict[str, list[torch.utils.hooks.RemovableHandle]]:
    """Calibrate the weight quantization groupping of modules in a decoder layer.

    Args:
        layer (LlmLayerStruct): LLM decoder layer to be reordered.
        config (LlmModelQuantConfig): Module quantization config.
        reorder_cache (dict[str, torch.Tensor]): Reorder index caches.
        residual_calibrator (ChannelOrderCalibrator, optional): Residual calibrator. Defaults to ``None``.
        layer_cache (dict[str, IOActivationsCache], optional): Layer cache. Defaults to ``None``.
        layer_kwargs (dict[str, tp.Any], optional): Layer keyword arguments. Defaults to ``None``.
    """
    logger = logging.getLogger(f"{__name__}.Reorder")
    layer_cache = layer_cache or {}
    hooks: dict[str, list[torch.utils.hooks.RemovableHandle]] = {}
    num_heads, num_head_repeats = layer.config.num_query_heads, layer.config.num_head_repeats
    num_experts = layer.num_experts
    proj_qkv, proj_out, proj_1st, proj_2nd = layer.proj_qkv, layer.proj_out, layer.proj_1st, layer.proj_2nd
    key = LlmModuleKey.PROJ_QKV
    if config.reorder.dynamic and config.reorder.enabled_for(key):
        logger.debug("- Reordering module %s", layer.attn_block_full_name)
        cache_key = layer.attn_block_full_name
        if cache_key not in reorder_cache:
            quant_config = config.specialize_for(key, layer_idx=layer.idx)
            index = ChannelOrderCalibrator(
                calib_config=config.reorder,
                wgts_quantizer=Quantizer(quant_config.wgts, quant_config.wgts.calib_kernel, key=key),
                ipts_quantizer=Quantizer(quant_config.ipts, channels_dim=-1, key=key),
                develop_dtype=config.develop_dtype,
            ).calibrate(
                ipt_wgts=[m.weight for m in proj_qkv],
                ipts=layer_cache.get(layer.proj_v_full_name, IOActivationsCache()).inputs,
                eval_ipt=layer_cache.get(layer.attn_block_full_name, IOActivationsCache()).inputs,
                eval_mod=layer.attn_block,
                ipt_mods=proj_qkv,
                eval_kwargs=layer.filter_layer_kwargs_to_attn_kwargs(layer_kwargs),
                reorder_wgts=[(m.weight, 1) for m in proj_qkv],
                reorder_ipt_mods=[(layer.attn_block, -1, None, None)],
                reorder_opt_mods=[],
            )
            reorder_cache[cache_key] = index.to(device=torch.device("cpu"))
        index = reorder_cache[cache_key]
        for proj in proj_qkv:
            index = index.to(proj.weight.device)
            proj.weight.data = proj.weight.data.index_select(1, index)

        def reorder_attn_hook(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            assert isinstance(inputs, tuple)
            x = inputs[0]
            assert isinstance(x, torch.Tensor)
            x = x.index_select(-1, index.to(x.device))
            return (x, *inputs[1:])

        hooks[layer.attn_block_full_name] = [layer.attn_block.register_forward_pre_hook(reorder_attn_hook)]
        gc.collect()
        torch.cuda.empty_cache()
    key = LlmModuleKey.PROJ_OUT
    if config.reorder.enabled_for(key):
        logger.debug("- Reordering module %s", layer.proj_out_full_name)
        cache_key = layer.proj_out_full_name
        if cache_key not in reorder_cache:
            quant_config = config.specialize_for(key, layer_idx=layer.idx)
            index = ChannelOrderCalibrator(
                calib_config=config.reorder,
                wgts_quantizer=Quantizer(quant_config.wgts, quant_config.wgts.calib_kernel, key=key),
                ipts_quantizer=Quantizer(quant_config.ipts, channels_dim=-1, key=key),
                num_heads=num_heads,
                num_head_repeats=num_head_repeats,
                develop_dtype=config.develop_dtype,
            ).calibrate(
                ipt_wgts=[proj_out.weight],
                ipts=layer_cache.get(layer.proj_out_full_name, IOActivationsCache()).inputs,
                eval_ipt=layer_cache.get(layer.proj_out_full_name, IOActivationsCache()).inputs,
                eval_mod=proj_out,
                ipt_mods=[proj_out],
                reorder_wgts=[(proj_out.weight, 1)],
                reorder_ipt_mods=[(proj_out, -1, None, None)],
                reorder_opt_mods=[],
            )
            reorder_cache[cache_key] = index.to(device=torch.device("cpu"))
        index = reorder_cache[cache_key]
        index = index.to(proj_out.weight.device)
        proj_out.weight.data = proj_out.weight.data.index_select(1, index)
        proj_v = proj_qkv[2]
        if num_heads > 1 and num_head_repeats > 1:
            num_channels = index.numel()
            head_channels = num_channels // num_heads
            index = index.view(num_heads, head_channels)
            delta = torch.arange(0, num_channels, head_channels, device=index.device).view(num_heads, 1)
            index = index - delta
            num_v_channels = num_channels // num_head_repeats
            num_v_heads = num_heads // num_head_repeats
            index = index.view(num_v_heads, num_head_repeats, head_channels)[:, 0, :]
            delta = torch.arange(0, num_v_channels, head_channels, device=index.device).view(num_v_heads, 1)
            index = index + delta
            index = index.view(-1)
        proj_v.weight.data = proj_v.weight.data.index_select(0, index.to(proj_v.weight.device))
        if proj_v.bias is not None:
            proj_v.bias.data = proj_v.bias.data[index.to(proj_v.bias.device)].contiguous()
        gc.collect()
        torch.cuda.empty_cache()
    key = LlmModuleKey.PROJ_1ST
    if config.reorder.dynamic and config.reorder.enabled_for(key):
        logger.debug("- Reordering module %s", layer.ffn_block_full_name)
        cache_key = layer.ffn_block_full_name
        if cache_key not in reorder_cache:
            quant_config = config.specialize_for(key, layer_idx=layer.idx)
            index = ChannelOrderCalibrator(
                calib_config=config.reorder,
                wgts_quantizer=Quantizer(quant_config.wgts, quant_config.wgts.calib_kernel, key=key),
                ipts_quantizer=Quantizer(quant_config.ipts, channels_dim=-1, key=key),
                develop_dtype=config.develop_dtype,
            ).calibrate(
                ipt_wgts=[m.weight for m in proj_1st],
                ipts=layer_cache.get(layer.ffn_block_full_name, IOActivationsCache()).inputs,
                eval_ipt=layer_cache.get(layer.ffn_block_full_name, IOActivationsCache()).inputs,
                eval_mod=layer.ffn_block,
                ipt_mods=proj_1st,
                reorder_wgts=[(m.weight, 1) for m in proj_1st],
                reorder_ipt_mods=[(layer.ffn_block, -1, None, None)],
                reorder_opt_mods=[],
            )
            reorder_cache[cache_key] = index.to(device=torch.device("cpu"))
        index = reorder_cache[cache_key]
        index = index.to(device=proj_1st[0].weight.device)
        for fc in proj_1st:
            fc.weight.data = fc.weight.data.index_select(1, index.to(fc.weight.device))
        router = layer.router
        if router is not None:
            router.weight.data = router.weight.data.index_select(1, index.to(router.weight.device))

        def reorder_ffn_hook(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            assert isinstance(inputs, tuple)
            x = inputs[0]
            assert isinstance(x, torch.Tensor)
            x = x.index_select(-1, index.to(x.device))
            return (x, *inputs[1:])

        hooks[layer.ffn_block_full_name] = [layer.ffn_block.register_forward_pre_hook(reorder_ffn_hook)]
    key = LlmModuleKey.PROJ_2ND
    if config.reorder.enabled_for(key):
        for expert_idx, (proj_2nd_name, fc2) in enumerate(zip(layer.proj_2nd_full_names, proj_2nd)):
            logger.debug("- Reordering module %s", proj_2nd_name)
            cache_key = proj_2nd_name
            if cache_key not in reorder_cache:
                quant_config = config.specialize_for(key, layer_idx=layer.idx)
                index = ChannelOrderCalibrator(
                    calib_config=config.reorder,
                    wgts_quantizer=Quantizer(quant_config.wgts, quant_config.wgts.calib_kernel, key=key),
                    ipts_quantizer=Quantizer(quant_config.ipts, channels_dim=-1, key=key),
                    develop_dtype=config.develop_dtype,
                ).calibrate(
                    ipt_wgts=[fc2.weight],
                    ipts=layer_cache.get(proj_2nd_name, IOActivationsCache()).inputs,
                    eval_ipt=layer_cache.get(proj_2nd_name, IOActivationsCache()).inputs,
                    eval_mod=fc2,
                    ipt_mods=[fc2],
                    reorder_wgts=[(fc2.weight, 1)],
                    reorder_ipt_mods=[(fc2, -1, None, None)],
                    reorder_opt_mods=[],
                )
                reorder_cache[cache_key] = index.to(device=torch.device("cpu"))
            index = reorder_cache[cache_key]
            index = index.to(fc2.weight.device)
            fc2.weight.data = fc2.weight.data.index_select(1, index.to(fc2.weight.device))
            for fc1 in proj_1st[expert_idx::num_experts]:
                fc1.weight.data = fc1.weight.data.index_select(0, index.to(fc1.weight.device))
                if fc1.bias is not None:
                    fc1.bias.data = fc1.bias.data[index.to(fc1.bias.device)].contiguous()
        gc.collect()
        torch.cuda.empty_cache()
    if residual_calibrator is not None:
        residual_calibrator.update_channel_metrics(
            wgts=[m.weight for m in proj_qkv], ipts=layer_cache.get(layer.proj_v_full_name, IOActivationsCache()).inputs
        )
        for expert_idx in range(num_experts):
            residual_calibrator.update_channel_metrics(
                wgts=[m.weight for m in proj_1st[expert_idx::num_experts]],
                ipts=layer_cache.get(layer.proj_1st_full_names[expert_idx], IOActivationsCache()).inputs,
            )
    return hooks


@torch.inference_mode()
def reorder_llm(  # noqa: C901
    model: nn.Module | LlmModelStruct,
    quant_config: LlmQuantConfig,
    tokenizer: nn.Module | None = None,
    calib_config: LlmCalibConfig | None = None,
    reorder_cache: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, list[torch.utils.hooks.RemovableHandle]]]:
    """Quantize the large foundation model weights.

    Args:
        model (nn.Module | LlmStruct): Model to be smoothed.
        quant_config (LlmModelQuantConfig): Module quantization config.
        tokenizer (nn.Module, optional): Tokenizer. Defaults to ``None``.
        calib_config (LlmCalibrationConfig, optional): Calibration config. Defaults to ``None``.
        reorder_cache (dict[str, torch.Tensor], optional): Reorder index caches. Defaults to ``None``.

    Returns:
        tuple[dict[str, torch.Tensor], dict[str, list[torch.utils.hooks.RemovableHandle]]]: Reorder index caches
            and reorder hooks.
    """
    if not isinstance(model, LlmModelStruct):
        model = LlmModelStruct.build(model)
    assert isinstance(model, LlmModelStruct)
    logger = logging.getLogger(f"{__name__}.Reorder")
    reorder_cache = {} if reorder_cache is None else reorder_cache
    calib_cache = LlmCalibrationCache(calib_config)
    residual_cache_key = "residual"
    needs_residual = not quant_config.reorder.dynamic and quant_config.reorder.enabled_for("residual")
    residual_calibrator = None
    if residual_cache_key not in reorder_cache and needs_residual:
        residual_calibrator = ChannelOrderCalibrator(
            calib_config=quant_config.reorder,
            wgts_quantizer=Quantizer(quant_config.wgts, quant_config.wgts.calib_kernel),
            ipts_quantizer=Quantizer(quant_config.ipts, channels_dim=-1),
            develop_dtype=quant_config.develop_dtype,
        )
    hooks: dict[str, list[torch.utils.hooks.RemovableHandle]] = {}
    with logging_redirect_tqdm():
        for _, (layer, layer_cache, layer_kwargs) in tqdm(
            calib_cache.iter_layer_activations(
                model,
                tokenizer,
                needs_inputs_fn=get_needs_inputs_fn(config=quant_config),
                needs_samples_caching=residual_calibrator is not None,
            ),
            desc="reordering",
            leave=False,
            total=model.config.num_hidden_layers,
        ):
            block_hooks = reorder_llm_decoder_layer(
                layer=layer,
                config=quant_config,
                reorder_cache=reorder_cache,
                residual_calibrator=residual_calibrator,
                layer_cache=layer_cache,
                layer_kwargs=layer_kwargs,
            )
            hooks.update(block_hooks)
            gc.collect()
            torch.cuda.empty_cache()
    if not needs_residual:
        return reorder_cache, hooks
    # region add extra params to be reordered
    backbone = model.backbone_struct
    ipt_mods: list[nn.Linear] = []
    reorder_wgts: list[tuple[nn.Parameter, int]] = []
    for layer in backbone.layer_structs:
        ipt_mods.extend(layer.proj_qkv)
        ipt_mods.extend(layer.proj_1st)
        _extend_out_params_(reorder_wgts, [layer.attn_ln, layer.proj_out, layer.ffn_ln, *layer.proj_2nd], 0)
        if layer.router is not None:
            reorder_wgts.append((layer.router.weight, 1))
    need_reorder_final_fc = True
    _extend_out_params_(reorder_wgts, backbone.embeddings, channels_dim=1)
    _extend_out_params_(reorder_wgts, [backbone.first_ln, backbone.final_ln], channels_dim=0)
    if backbone.proj_in is not None:
        reorder_wgts.append((backbone.proj_in.weight, 1))
        _extend_out_params_(reorder_wgts, [backbone.proj_in], channels_dim=0)
    if backbone.proj_out is not None:
        reorder_wgts.append((backbone.proj_out.weight, 1))
        need_reorder_final_fc = False
    logger.debug("- Reordering residual modules")
    ipt_wgts = [m.weight for m in ipt_mods]
    reorder_wgts.extend([(m.weight, 1) for m in ipt_mods])
    if residual_cache_key not in reorder_cache:
        residual_calibrator.init_channel_indexes()
        index = residual_calibrator.calibrate(
            ipt_wgts=ipt_wgts,
            ipts=None,
            eval_ipt=([x for x in calib_cache.cached_samples], -1, TransformFn()),
            eval_mod=model.backbone,
            ipt_mods=ipt_mods,
            reorder_wgts=reorder_wgts,
            reorder_ipt_mods=[],
            reorder_opt_mods=[(model.backbone, -1, None, None)] if need_reorder_final_fc else [],
        )
        reorder_cache[residual_cache_key] = index.to(device=torch.device("cpu"))
        del ipt_mods, residual_calibrator, calib_cache
        gc.collect()
        torch.cuda.empty_cache()
    index = reorder_cache[residual_cache_key]
    for w, d in reorder_wgts:
        w.data = w.data.index_select(dim=d, index=index.to(w.data.device))
    if need_reorder_final_fc and not model.config.tie_word_embeddings:
        fc = model.fc
        fc.weight.data = fc.weight.data.index_select(dim=1, index=index.to(fc.weight.device))
    gc.collect()
    torch.cuda.empty_cache()
    return reorder_cache, hooks
