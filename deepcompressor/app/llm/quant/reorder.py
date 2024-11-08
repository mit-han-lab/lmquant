# -*- coding: utf-8 -*-
"""LLM quantization channel reordering module."""

import gc
import typing as tp

import torch
import torch.nn as nn
import torch.utils
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from deepcompressor.calib.reorder import ChannelOrderCalibrator, ChannelReorderer
from deepcompressor.data.cache import IOTensorsCache, TensorCache, TensorsCache
from deepcompressor.quantizer.processor import Quantizer
from deepcompressor.utils import tools

from ..nn import LlmModelStruct, LlmTransformerBlockStruct
from .config import LlmQuantConfig
from .utils import get_needs_inputs_fn

__all__ = ["reorder_llm"]


def _extend_params_(
    params: list[tuple[nn.Parameter, int]],
    modules: list[nn.Linear | nn.Embedding, nn.LayerNorm],
    out_channels_dim: int | None = None,
    in_channels_dim: int | None = None,
) -> list[tuple[nn.Parameter, int]]:
    """Extend the parameters to be reordered."""
    if out_channels_dim is not None:
        assert in_channels_dim is None
    else:
        assert in_channels_dim is not None
    for module in modules:
        if module is None:
            continue
        if out_channels_dim is not None:
            params.append((module.weight, out_channels_dim))
            if hasattr(module, "bias") and module.bias is not None:
                params.append((module.bias, 0))
        else:
            params.append((module.weight, in_channels_dim))
    return params


@torch.inference_mode()
def reorder_llm_layer(  # noqa: C901
    layer: LlmTransformerBlockStruct,
    config: LlmQuantConfig,
    reorder_cache: dict[str, torch.Tensor],
    residual_calibrator: ChannelOrderCalibrator | None = None,
    layer_cache: dict[str, IOTensorsCache] | None = None,
    layer_kwargs: dict[str, tp.Any] | None = None,
) -> ChannelOrderCalibrator | None:
    """Calibrate the channel order in a layer.

    Args:
        layer (`LlmTransformerBlockStruct`):
            Large language model layer to be reordered.
        config (`LlmQuantConfig`):
            Quantization config.
        reorder_cache (`dict[str, torch.Tensor]`):
            Reorder indexes cache.
        residual_calibrator (`ChannelOrderCalibrator` or `None`, *optional*, defaults to `None`):
            Channel order calibrator for residual modules.
        layer_cache (`dict[str, IOTensorsCache]`, *optional*, defaults to `None`):
            Layer activations cache.
        layer_kwargs (`dict[str, tp.Any]`, *optional*, defaults to `None`):
            Layer keyword arguments.

    Returns:
        `ChannelOrderCalibrator` or `None`:
            Channel order calibrator for residual modules.
    """
    logger = tools.logging.getLogger(f"{__name__}.Reorder")
    layer_cache = layer_cache or {}

    attn = layer.attn_struct
    qkv_proj, out_proj = attn.qkv_proj, attn.out_proj
    num_heads, num_head_repeats = attn.config.num_query_heads, attn.config.num_head_repeats
    # region reorder in attention module
    if config.reorder.dynamic and config.reorder.is_enabled_for(attn.qkv_proj_key):
        logger.debug("- Reordering %s", attn.qkv_proj_names)
        cache_key = attn.name
        if cache_key not in reorder_cache:
            index = ChannelOrderCalibrator(
                config=config.reorder,
                weight_quantizer=Quantizer(config.wgts, key=attn.qkv_proj_key),
                input_quantizer=Quantizer(config.ipts, channels_dim=-1, key=attn.qkv_proj_key),
                develop_dtype=config.develop_dtype,
            ).calibrate(
                x_wgts=[m.weight for m in qkv_proj],
                x_acts=layer_cache[attn.v_proj_name].inputs if layer_cache else None,
                x_mods=qkv_proj,
                eval_inputs=layer_cache[cache_key].inputs if layer_cache else None,
                eval_module=attn.module,
                eval_kwargs=attn.filter_kwargs(layer_kwargs),
                reorder_wgts=[(m.weight, 1) for m in qkv_proj],
                reorder_ipt_mods=[(attn.module, -1, None)],
                reorder_opt_mods=[],
            )
            reorder_cache[cache_key] = index.to(device=torch.device("cpu"))
        index = reorder_cache[cache_key]
        for proj in qkv_proj:
            index = index.to(proj.weight.device)
            proj.weight.data = proj.weight.data.index_select(1, index)
        ChannelReorderer(index, channels_dim=-1).as_hook().register(attn.module)
        gc.collect()
        torch.cuda.empty_cache()
    if config.reorder.is_enabled_for(attn.out_proj_key):
        logger.debug("- Reordering %s", attn.out_proj_name)
        cache_key = attn.out_proj_name
        if cache_key not in reorder_cache:
            index = ChannelOrderCalibrator(
                config=config.reorder,
                weight_quantizer=Quantizer(config.wgts, key=attn.out_proj_key),
                input_quantizer=Quantizer(config.ipts, channels_dim=-1, key=attn.out_proj_key),
                num_heads=num_heads,
                num_head_repeats=num_head_repeats,
                develop_dtype=config.develop_dtype,
            ).calibrate(
                x_wgts=[out_proj.weight],
                x_acts=layer_cache[cache_key].inputs if layer_cache else None,
                x_mods=[out_proj],
                eval_inputs=layer_cache[cache_key].inputs if layer_cache else None,
                eval_module=out_proj,
                reorder_wgts=[(out_proj.weight, 1)],
                reorder_ipt_mods=[(out_proj, -1, None)],
                reorder_opt_mods=[],
            )
            reorder_cache[cache_key] = index.to(device=torch.device("cpu"))
        index = reorder_cache[cache_key]
        index = index.to(out_proj.weight.device)
        out_proj.weight.data = out_proj.weight.data.index_select(1, index)
        v_proj = qkv_proj[2]
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
        v_proj.weight.data = v_proj.weight.data.index_select(0, index.to(v_proj.weight.device))
        if v_proj.bias is not None:
            v_proj.bias.data = v_proj.bias.data[index.to(v_proj.bias.device)].contiguous()
        gc.collect()
        torch.cuda.empty_cache()
    # endregion
    ffn = layer.ffn_struct
    num_experts = ffn.config.num_experts
    up_proj, down_proj = ffn.up_projs, ffn.down_projs
    # region reorder in feed-forward module
    if config.reorder.dynamic and config.reorder.is_enabled_for(ffn.up_proj_key):
        logger.debug("- Reordering %s", ffn.name)
        cache_key = ffn.name
        if cache_key not in reorder_cache:
            index = ChannelOrderCalibrator(
                config=config.reorder,
                weight_quantizer=Quantizer(config.wgts, key=ffn.up_proj_key),
                input_quantizer=Quantizer(config.ipts, channels_dim=-1, key=ffn.up_proj_key),
                develop_dtype=config.develop_dtype,
            ).calibrate(
                x_wgts=[m.weight for m in up_proj],
                x_acts=layer_cache[cache_key].inputs if layer_cache else None,
                x_mods=up_proj,
                eval_inputs=layer_cache[cache_key].inputs if layer_cache else None,
                eval_module=ffn.module,
                reorder_wgts=[(m.weight, 1) for m in up_proj],
                reorder_ipt_mods=[(ffn.module, -1, None)],
                reorder_opt_mods=[],
            )
            reorder_cache[cache_key] = index.to(device=torch.device("cpu"))
        index = reorder_cache[cache_key]
        index = index.to(device=up_proj[0].weight.device)
        for fc in up_proj:
            fc.weight.data = fc.weight.data.index_select(1, index.to(fc.weight.device))
        moe_gate = ffn.moe_gate
        if moe_gate is not None:
            moe_gate.weight.data = moe_gate.weight.data.index_select(1, index.to(moe_gate.weight.device))
        ChannelReorderer(index, channels_dim=-1).as_hook().register(ffn.module)
    if config.reorder.is_enabled_for(ffn.down_proj_key):
        for expert_idx, (fc2_name, fc2) in enumerate(zip(ffn.down_proj_names, down_proj, strict=True)):
            logger.debug("- Reordering module %s", fc2_name)
            cache_key = fc2_name
            if cache_key not in reorder_cache:
                index = ChannelOrderCalibrator(
                    config=config.reorder,
                    weight_quantizer=Quantizer(config.wgts, key=ffn.down_proj_key),
                    input_quantizer=Quantizer(config.ipts, channels_dim=-1, key=ffn.down_proj_key),
                    develop_dtype=config.develop_dtype,
                ).calibrate(
                    x_wgts=[fc2.weight],
                    x_acts=layer_cache[cache_key].inputs if layer_cache else None,
                    x_mods=[fc2],
                    eval_inputs=layer_cache[cache_key].inputs if layer_cache else None,
                    eval_module=fc2,
                    reorder_wgts=[(fc2.weight, 1)],
                    reorder_ipt_mods=[(fc2, -1, None)],
                    reorder_opt_mods=[],
                )
                reorder_cache[cache_key] = index.to(device=torch.device("cpu"))
            index = reorder_cache[cache_key]
            index = index.to(fc2.weight.device)
            fc2.weight.data = fc2.weight.data.index_select(1, index.to(fc2.weight.device))
            for fc1 in up_proj[expert_idx::num_experts]:
                fc1.weight.data = fc1.weight.data.index_select(0, index.to(fc1.weight.device))
                if fc1.bias is not None:
                    fc1.bias.data = fc1.bias.data[index.to(fc1.bias.device)].contiguous()
        gc.collect()
        torch.cuda.empty_cache()
    # endregion

    if residual_calibrator is not None and (
        config.reorder.dynamic
        or not config.reorder.is_enabled_for(attn.qkv_proj_key)
        or not config.reorder.is_enabled_for(ffn.up_proj_key)
    ):
        residual_calibrator = None
    if residual_calibrator is not None and "residual" not in reorder_cache:
        residual_calibrator.update_channel_metrics(
            weights=[m.weight for m in qkv_proj],
            inputs=layer_cache[attn.v_proj_name].inputs if layer_cache else None,
        )
        for expert_idx in range(num_experts):
            residual_calibrator.update_channel_metrics(
                weights=[m.weight for m in up_proj[expert_idx::num_experts]],
                inputs=layer_cache[ffn.up_proj_names[expert_idx]].inputs if layer_cache else None,
            )
    return residual_calibrator


@torch.inference_mode()
def reorder_llm(  # noqa: C901
    model: nn.Module | LlmModelStruct,
    config: LlmQuantConfig,
    tokenizer: PreTrainedTokenizer | None = None,
    reorder_cache: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Quantize the large foundation model weights.

    Args:
        model (`nn.Module` or `LlmStruct`):
            Model to be reordered.
        config (`LlmQuantConfig`):
            Quantization config.
        tokenizer (`PreTrainedTokenizer` or `None`, *optional*, defaults to `None`):
            Tokenizer.
        reorder_cache (`dict[str, torch.Tensor]`, *optional*, defaults to `None`):
            Reorder indexes cache.

    Returns:
        `dict[str, torch.Tensor]`:
            Reorder indexes cache.
    """
    if not isinstance(model, LlmModelStruct):
        model = LlmModelStruct.construct(model)
    assert isinstance(model, LlmModelStruct)
    logger = tools.logging.getLogger(f"{__name__}.Reorder")
    reorder_cache = {} if reorder_cache is None else reorder_cache
    residual_calibrator = None
    if "residual" not in reorder_cache and not config.reorder.dynamic and config.reorder.is_enabled_for("residual"):
        residual_calibrator = ChannelOrderCalibrator(
            config=config.reorder,
            weight_quantizer=Quantizer(config.wgts),
            input_quantizer=Quantizer(config.ipts, channels_dim=-1),
            develop_dtype=config.develop_dtype,
        )
    with tools.logging.redirect_tqdm():
        if not reorder_cache:
            calib_cache = config.calib.build_loader(tokenizer)
            for _, (layer, layer_cache, layer_kwargs) in tqdm(
                calib_cache.iter_layer_activations(
                    model,
                    needs_inputs_fn=get_needs_inputs_fn(model=model, config=config),
                ),
                desc="reordering",
                leave=False,
                total=len(model.backbone_struct.layer_structs),
                dynamic_ncols=True,
            ):
                residual_calibrator = reorder_llm_layer(
                    layer=layer,
                    config=config,
                    reorder_cache=reorder_cache,
                    residual_calibrator=residual_calibrator,
                    layer_cache=layer_cache,
                    layer_kwargs=layer_kwargs,
                )
                gc.collect()
                torch.cuda.empty_cache()
        else:
            calib_cache = None
            for layer in tqdm(
                model.backbone_struct.layer_structs,
                desc="reordering",
                leave=False,
                dynamic_ncols=True,
            ):
                residual_calibrator = reorder_llm_layer(
                    layer=layer,
                    config=config,
                    reorder_cache=reorder_cache,
                    residual_calibrator=residual_calibrator,
                )
    if residual_calibrator is None:
        return reorder_cache
    # region add extra params to be reordered
    backbone = model.backbone_struct
    x_mods: list[nn.Linear] = []
    reorder_wgts: list[tuple[nn.Parameter, int]] = []
    for layer in backbone.layer_structs:
        x_mods.extend(layer.attn_struct.qkv_proj)
        x_mods.extend(layer.ffn_struct.up_projs)
        _extend_params_(
            reorder_wgts,
            [layer.attn_norm, layer.attn_struct.out_proj, layer.ffn_norm, *layer.ffn_struct.down_projs],
            out_channels_dim=0,
        )
        _extend_params_(reorder_wgts, [layer.ffn_struct.moe_gate], in_channels_dim=1)
    need_reorder_head = model.head is not None
    if backbone.proj_in is not None:
        _extend_params_(reorder_wgts, [backbone.proj_in], out_channels_dim=0)
        _extend_params_(reorder_wgts, [backbone.embed_positions], out_channels_dim=1)
    else:
        _extend_params_(reorder_wgts, [backbone.embed_tokens, backbone.embed_positions], out_channels_dim=1)
    _extend_params_(reorder_wgts, [backbone.norm_in, backbone.norm_out], out_channels_dim=0)
    if backbone.proj_out is not None:
        _extend_params_(reorder_wgts, [backbone.proj_out], in_channels_dim=1)
        need_reorder_head = False
    logger.debug("- Reordering residual modules")
    _extend_params_(reorder_wgts, x_mods, in_channels_dim=1)
    if "residual" not in reorder_cache:
        calib_cache = calib_cache or config.calib.build_loader(tokenizer)
        residual_calibrator.init_channel_indexes()
        index = residual_calibrator.calibrate(
            x_wgts=[m.weight for m in x_mods],
            x_acts=None,
            eval_inputs=TensorsCache(TensorCache(calib_cache.dataset.data, channels_dim=-1, orig_device="cuda")),
            eval_module=model.backbone,
            x_mods=x_mods,
            reorder_wgts=reorder_wgts,
            reorder_ipt_mods=[],
            reorder_opt_mods=[(model.backbone, -1, None)] if need_reorder_head else [],
        )
        reorder_cache["residual"] = index.to(device=torch.device("cpu"))
        del x_mods, residual_calibrator, calib_cache
        gc.collect()
        torch.cuda.empty_cache()
    index = reorder_cache["residual"]
    for wgt, dim in reorder_wgts:
        wgt.data = wgt.data.index_select(dim=dim, index=index.to(wgt.data.device))
    if need_reorder_head and not model.config.tie_word_embeddings:
        model.head.weight.data = model.head.weight.data.index_select(dim=1, index=index.to(model.head.weight.device))
    gc.collect()
    torch.cuda.empty_cache()
    return reorder_cache
