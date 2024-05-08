# -*- coding: utf-8 -*-
"""LLM activation quantization calibration module."""

import gc
import logging
import typing as tp

import torch
import torch.nn as nn
import torch.utils.hooks
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from lmquant.dataset import IOActivationsCache
from lmquant.quant.calib.config import QuantTensorType
from lmquant.quant.quantizer.activation import ActivationQuantizer
from lmquant.utils import tools

from ..dataset import LlmCalibConfig, LlmCalibrationCache
from ..nn import LlmDecoderLayerStruct, LlmModelStruct
from ..utils import get_needs_inputs_fn
from .config import LlmModuleKey, LlmQuantConfig

__all__ = ["quantize_llm_activations"]


@torch.inference_mode()
def quantize_llm_decoder_layer_activations(  # noqa: C901
    layer: LlmDecoderLayerStruct,
    config: LlmQuantConfig,
    quant_cache: dict[str, dict[str, torch.Tensor | float | None]],
    layer_cache: dict[str, IOActivationsCache] | None = None,
    layer_kwargs: dict[str, tp.Any] | None = None,
    orig_state_dict: dict[str, torch.Tensor] | None = None,
    return_with_quantizers: bool = False,
) -> tuple[dict[str, ActivationQuantizer], dict[str, torch.utils.hooks.RemovableHandle]]:
    """Calibrate the activation quantization ranges of modules in a decoder layer.

    Args:
        layer (LlmDecoderLayerStruct): Decoder layer.
        config (LlmQuantConfig): Quantization configuration.
        quant_cache (dict[str, dict[str, torch.Tensor | float | None]], optional): Quantization cache.
        layer_cache (dict[str, IOActivationsCache], optional): Layer cache. Defaults to ``None``.
        layer_kwargs (dict[str, tp.Any], optional): Layer kwargs. Defaults to ``None``.
        orig_state_dict (dict[str, torch.Tensor], optional): Original state dict. Defaults to ``None``.
        return_with_quantizers (bool, optional): Whether to return the quantizers. Defaults to ``False``.

    Returns:
        tuple[dict[str, ActivationQuantizer], list[torch.utils.hooks.RemovableHandle]]: Quantizers, and hooks.
    """
    logger = logging.getLogger(f"{__name__}.ActivationQuant")
    logger.debug("- Quantizing decoder layer %s", layer.full_name)
    layer_cache = layer_cache or {}
    layer_kwargs = layer_kwargs or {}
    orig_state_dict = orig_state_dict or {}
    args_caches: list[
        tuple[
            LlmModuleKey,  # key
            QuantTensorType,
            list[nn.Module],  # modules
            str,  # module name
            nn.Module,  # eval module
            str,  # eval name
            dict[str, tp.Any],  # eval kwargs
            list[tuple[nn.Parameter, torch.Tensor]],  # original wgts
        ]
    ] = []
    In, Out = QuantTensorType.Inputs, QuantTensorType.Outputs
    # region attn
    attn_kwargs = layer.filter_layer_kwargs_to_attn_kwargs(layer_kwargs)
    attn_block_name = layer.attn_block_full_name
    if orig_state_dict:
        orig_wgts = [
            (module.weight, orig_state_dict[f"{module_name}.weight"])
            for module_name, module in zip(layer.proj_qkv_full_names, layer.proj_qkv)
        ] + [(layer.proj_out.weight, orig_state_dict[f"{layer.proj_out_full_name}.weight"])]
    else:
        orig_wgts = None
    # region proj_qkv (Inputs)
    module_name = layer.proj_v_full_name
    key, cache_key, modules = LlmModuleKey.PROJ_QKV, f"{module_name}.input", layer.proj_qkv
    args_caches.append((key, In, modules, module_name, layer.attn_block, attn_block_name, attn_kwargs, orig_wgts))
    # endregion
    # region attn_qkv (Outputs)
    orig_proj_wgts = (orig_wgts + orig_wgts) if orig_wgts else None
    for idx, (key, name) in enumerate(
        [
            (LlmModuleKey.ATTN_Q, config.keywords_o[LlmModuleKey.ATTN_Q][0]),
            (LlmModuleKey.ATTN_K, config.keywords_o[LlmModuleKey.ATTN_K][0]),
            (LlmModuleKey.ATTN_V, layer.proj_v_name),
        ]
    ):
        module_name, module = f"{attn_block_name}.{name}", getattr(layer.attn_block, name)
        cache_key = f"{module_name}.output"
        orig_wgts = orig_proj_wgts[idx : idx + 4] if orig_proj_wgts else None
        args_caches.append((key, Out, [module], module_name, layer.attn_block, attn_block_name, attn_kwargs, orig_wgts))
    # endregion
    # region proj_out (Inputs)
    module_name, module = layer.proj_out_full_name, layer.proj_out
    key, cache_key = LlmModuleKey.PROJ_OUT, f"{module_name}.input"
    orig_wgts = [(module.weight, orig_state_dict[f"{module_name}.weight"])] if orig_state_dict else None
    args_caches.append((key, In, [module], module_name, module, module_name, None, orig_wgts))
    # endregion
    del orig_wgts
    # endregion
    # region ffn block
    # region router (Inputs)
    if layer.router is not None:
        module_name = layer.ffn_block_full_name
        key, cache_key = LlmModuleKey.ROUTER, f"{module_name}.input"
        if orig_state_dict:
            orig_wgts = [(layer.router.weight, orig_state_dict[f"{layer.router_full_name}.weight"])]
        else:
            orig_wgts = None
        args_caches.append((key, In, [layer.router], module_name, layer.ffn_block, module_name, None, orig_wgts))
    # endregion
    # region ffn block projections
    for expert_idx in range(layer.config.num_experts):
        expert = layer.experts[expert_idx]
        expert_name = layer.expert_full_names[expert_idx]
        # region proj 1st in expert (Inputs)
        module_name, modules = layer.proj_1st_full_names[expert_idx], layer.proj_1st[:: layer.config.num_experts]
        key, cache_key = LlmModuleKey.PROJ_1ST, f"{module_name}.input"
        if orig_state_dict:
            orig_wgts = [
                (module.weight, orig_state_dict[f"{expert_name}.{layer.proj_1st_names[module_idx]}.weight"])
                for module_idx, module in enumerate(modules)
            ]
        else:
            orig_wgts = None
        args_caches.append((key, In, modules, module_name, expert, module_name, None, orig_wgts))
        # endregion
        # region proj 2nd in expert (Inputs)
        module_name, module = layer.proj_2nd_full_names[expert_idx], layer.proj_2nd[expert_idx]
        key, cache_key = LlmModuleKey.PROJ_2ND, f"{module_name}.input"
        if orig_state_dict:
            orig_wgts = [(module.weight, orig_state_dict[f"{module_name}.weight"])]
        else:
            orig_wgts = None
        args_caches.append((key, In, [module], module_name, module, module_name, None, orig_wgts))
        # endregion
    # endregion
    # endregion
    quantizers: dict[str, ActivationQuantizer] = {}
    tools.logging.Formatter.indent_inc()
    for key, tensor_type, modules, module_name, eval_module, eval_name, eval_kwargs, orig_wgts in args_caches:
        if tensor_type == QuantTensorType.Inputs:
            cache_key = f"{module_name}.input"
            quantizer_config = config.specialize_for(key, layer_idx=layer.idx).ipts
            acts = layer_cache.get(module_name, IOActivationsCache()).inputs
        else:
            cache_key = f"{module_name}.output"
            quantizer_config = config.specialize_for(key, layer_idx=layer.idx).opts
            acts = layer_cache.get(module_name, IOActivationsCache()).outputs
        quantizer = ActivationQuantizer(
            quantizer_config, channels_dim=-1, develop_dtype=config.develop_dtype, key=key, tensor_type=tensor_type
        )
        if quantizer.enabled:
            quantizers[cache_key] = quantizer
            if cache_key not in quant_cache:
                logger.debug("- Calibrating %s", cache_key)
                quantizer.calibrate_dynamic_range(
                    modules=modules,
                    activations=acts,
                    eval_module=eval_module,
                    eval_inputs=layer_cache.get(eval_name, IOActivationsCache()).inputs,
                    eval_kwargs=eval_kwargs,
                    orig_weights=orig_wgts,
                )
                quant_cache[cache_key] = quantizer.state_dict()
                gc.collect()
                torch.cuda.empty_cache()
            else:
                quantizer.load_state_dict(quant_cache[cache_key], device=acts[0].orig_device)
            if tensor_type == QuantTensorType.Inputs:
                if layer.proj_v_name in cache_key:
                    for proj_name in [layer.proj_q_name, layer.proj_k_name]:
                        quantizers[cache_key.replace(layer.proj_v_name, proj_name)] = quantizer
                if layer.proj_1st_names[0] in cache_key:
                    for fc_name in layer.proj_1st_names[1:]:
                        quantizers[cache_key.replace(layer.proj_1st_names[0], fc_name)] = quantizer
        del quantizer
    hooks: dict[str, list[torch.utils.hooks.RemovableHandle]] = {}
    for name, module in layer.module.named_modules():
        module_name = f"{layer.full_name}.{name}"
        ipts_quantizer = quantizers.get(f"{module_name}.input", None)
        opts_quantizer = quantizers.get(f"{module_name}.output", None)
        needs_quant_ipts = ipts_quantizer is not None and ipts_quantizer.enabled
        needs_quant_opts = opts_quantizer is not None and opts_quantizer.enabled
        if needs_quant_ipts or needs_quant_opts:
            logger.debug(
                "- Quantizing %s (%s)",
                module_name,
                ("inputs" if needs_quant_ipts else "")
                + (" and " if needs_quant_ipts and needs_quant_opts else "")
                + ("outputs" if needs_quant_opts else ""),
            )
            if needs_quant_ipts:
                hooks[f"{module_name}.input"] = [ipts_quantizer.quantize_module_inputs(module)]
            if needs_quant_opts:
                hooks[f"{module_name}.output"] = [opts_quantizer.quantize_module_outputs(module)]
    tools.logging.Formatter.indent_dec()
    if return_with_quantizers:
        return quantizers, hooks
    else:
        return {}, hooks


@torch.inference_mode()
def quantize_llm_activations(
    model: nn.Module | LlmModelStruct,
    quant_config: LlmQuantConfig,
    tokenizer: nn.Module | None = None,
    calib_config: LlmCalibConfig | None = None,
    quant_cache: dict[str, dict[str, torch.Tensor | float | None]] | None = None,
    orig_state_dict: dict[str, torch.Tensor] | None = None,
    return_with_quantizers: bool = False,
) -> tuple[
    dict[str, dict[str, torch.Tensor | float | None]],
    dict[str, ActivationQuantizer],
    dict[str, list[torch.utils.hooks.RemovableHandle]],
]:
    """Quantize the large foundation model activations.

    Args:
        model (nn.Module): Model to be smoothed.
        quant_config (LlmModelQuantConfig): Module quantization config.
        tokenizer (nn.Module, optional): Tokenizer. Defaults to ``None``.
        calib_config (LMCalibrationConfig, optional): Calibration config. Defaults to ``None``.
        quant_cache (dict[str, dict[str, torch.Tensor | float | None]], optional): Quantization cache.
        orig_state_dict (dict[str, torch.Tensor], optional): Original state dict. Defaults to ``None``.
        return_with_quantizers (bool, optional): Whether to return the quantizers. Defaults to ``False``.

    Returns:
        tuple[
            dict[str, dict[str, torch.Tensor | float | None]],
            dict[str, ActivationQuantizer],
            dict[str, list[torch.utils.hooks.RemovableHandle]]
        ]: Quantization cache, quantizers, and hooks.
    """
    if not isinstance(model, LlmModelStruct):
        model = LlmModelStruct.build(model)
    assert isinstance(model, LlmModelStruct)
    with logging_redirect_tqdm():
        quant_cache = quant_cache or {}
        quantizers: dict[str, ActivationQuantizer] = {}
        hooks: dict[str, list[torch.utils.hooks.RemovableHandle]] = {}
        for _, (layer, layer_cache, layer_kwargs) in tqdm(
            LlmCalibrationCache(calib_config).iter_layer_activations(
                model,
                tokenizer,
                needs_inputs_fn=get_needs_inputs_fn(config=quant_config),
                needs_outputs_fn=quant_config.needs_quant_outputs,
                needs_samples_caching=False,
            ),
            desc="quantizing activations",
            leave=False,
            total=model.config.num_hidden_layers,
        ):
            block_quantizers, block_hooks = quantize_llm_decoder_layer_activations(
                layer=layer,
                config=quant_config,
                quant_cache=quant_cache,
                layer_cache=layer_cache,
                layer_kwargs=layer_kwargs,
                orig_state_dict=orig_state_dict,
                return_with_quantizers=return_with_quantizers,
            )
            quantizers.update(block_quantizers)
            hooks.update(block_hooks)
    return quant_cache, quantizers, hooks
