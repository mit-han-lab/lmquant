# -*- coding: utf-8 -*-
"""LLM activation quantization calibration module."""

import gc
import typing as tp

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from deepcompressor.data.cache import IOTensorsCache
from deepcompressor.data.common import TensorType
from deepcompressor.utils import tools

from ..nn import LlmModelStruct, LlmTransformerBlockStruct
from .config import LlmQuantConfig
from .quantizer import LlmActivationQuantizer
from .utils import get_needs_inputs_fn, get_needs_outputs_fn

__all__ = ["quantize_llm_activations"]


@torch.inference_mode()
def quantize_llm_layer_activations(  # noqa: C901
    layer: LlmTransformerBlockStruct,
    config: LlmQuantConfig,
    quantizer_state_dict: dict[str, tp.Any],
    layer_cache: dict[str, IOTensorsCache] | None = None,
    layer_kwargs: dict[str, tp.Any] | None = None,
    orig_state_dict: dict[str, torch.Tensor] | None = None,
) -> None:
    """Calibrate the activation quantization ranges of modules in a layer.

    Args:
        layer (`LlmTransformerBlockStruct`):
            Layer.
        config (`LlmQuantConfig`):
            Quantization configuration.
        quantizer_state_dict (`dict[str, Any]`):
            Activation quantizer state dict.
        layer_cache (`dict[str, IOTensorsCache]` or `None`, *optional*, defaults to `None`):
            Layer activations cache.
        layer_kwargs (`dict[str, tp.Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments for the layer.
        orig_state_dict (`dict[str, torch.Tensor]` or `None`, *optional*, defaults to `None`):
            Original weight state dict.
    """
    logger = tools.logging.getLogger(f"{__name__}.ActivationQuant")
    logger.debug("- Quantizing layer %s", layer.name)
    layer_cache = layer_cache or {}
    layer_kwargs = layer_kwargs or {}
    orig_state_dict = orig_state_dict or {}
    args_caches: list[
        tuple[
            str,  # key
            TensorType,
            list[nn.Linear],  # modules
            str,  # module name
            nn.Module,  # eval module
            str,  # eval name
            dict[str, tp.Any],  # eval kwargs
            list[tuple[nn.Parameter, torch.Tensor]],  # original wgts
        ]
    ] = []
    In, Out = TensorType.Inputs, TensorType.Outputs

    attn, ffn = layer.attn_struct, layer.ffn_struct
    # region attn
    attn_kwargs = attn.filter_kwargs(layer_kwargs)
    if orig_state_dict:
        orig_wgts = [
            (module.weight, orig_state_dict[f"{module_name}.weight"])
            for module_name, module in zip(attn.qkv_proj_names, attn.qkv_proj, strict=True)
        ] + [(attn.out_proj.weight, orig_state_dict[f"{attn.out_proj_name}.weight"])]
    else:
        orig_wgts = None
    # region qkv_proj (Inputs)
    module_name = attn.v_proj_name
    module_key, cache_key, modules = attn.qkv_proj_key, f"{module_name}.input", attn.qkv_proj
    args_caches.append((module_key, In, modules, module_name, attn, attn.name, attn_kwargs, orig_wgts))
    # endregion
    # region qkv_attn (Outputs)
    orig_proj_wgts = (orig_wgts + orig_wgts) if orig_wgts else None
    for idx, module_key in enumerate((attn.q_key, attn.k_key, attn.v_key)):
        module = getattr(attn, "qkv"[idx])
        module_name = getattr(attn, f"{'qkv'[idx]}_name")
        cache_key = f"{module_name}.output"
        orig_wgts = orig_proj_wgts[idx : idx + 4] if orig_proj_wgts else None
        args_caches.append((module_key, Out, [module], module_name, attn, attn.name, attn_kwargs, orig_wgts))
    # endregion
    # region out_proj (Inputs)
    module_name, module = attn.out_proj_name, attn.out_proj
    module_key, cache_key = attn.out_proj_key, f"{module_name}.input"
    orig_wgts = [(module.weight, orig_state_dict[f"{module_name}.weight"])] if orig_state_dict else None
    args_caches.append((module_key, In, [module], module_name, module, module_name, None, orig_wgts))
    # endregion
    del orig_wgts
    # endregion
    # region ffn
    # region ffn block projections
    for expert_idx in range(ffn.config.num_experts):
        expert = ffn.experts[expert_idx]
        expert_name = ffn.expert_names[expert_idx]
        # region proj 1st in expert (Inputs)
        module_name = ffn.up_proj_names[expert_idx]
        modules = ffn.up_projs[expert_idx :: ffn.config.num_experts]
        module_key, cache_key = ffn.up_proj_key, f"{module_name}.input"
        if orig_state_dict:
            orig_wgts = [
                (module.weight, orig_state_dict[f"{expert_name}.{ffn.up_proj_rnames[module_idx]}.weight"])
                for module_idx, module in enumerate(modules)
            ]
        else:
            orig_wgts = None
        args_caches.append((module_key, In, modules, module_name, expert, module_name, None, orig_wgts))
        # endregion
        # region proj 2nd in expert (Inputs)
        module_name, module = ffn.down_proj_names[expert_idx], ffn.down_projs[expert_idx]
        module_key, cache_key = ffn.down_proj_key, f"{module_name}.input"
        if orig_state_dict:
            orig_wgts = [(module.weight, orig_state_dict[f"{module_name}.weight"])]
        else:
            orig_wgts = None
        args_caches.append((module_key, In, [module], module_name, module, module_name, None, orig_wgts))
        # endregion
    # endregion
    # endregion
    quantizers: dict[str, LlmActivationQuantizer] = {}
    tools.logging.Formatter.indent_inc()
    for module_key, tensor_type, modules, module_name, eval_module, eval_name, eval_kwargs, orig_wgts in args_caches:
        if tensor_type == TensorType.Inputs:
            cache_key = f"{module_name}.input"
            quantizer_config = config.ipts
            activations = layer_cache.get(module_name, IOTensorsCache()).inputs
            device = modules[0].weight.device
        else:
            cache_key = f"{module_name}.output"
            quantizer_config = config.opts
            activations = layer_cache.get(module_name, IOTensorsCache()).outputs
            device = attn.out_proj.weight.device
        quantizer = LlmActivationQuantizer(
            quantizer_config,
            channels_dim=-1,
            develop_dtype=config.develop_dtype,
            key=module_key,
            tensor_type=tensor_type,
        )
        if quantizer.is_enabled():
            quantizers[cache_key] = quantizer
            if cache_key not in quantizer_state_dict:
                logger.debug("- Calibrating %s", cache_key)
                quantizer.calibrate_dynamic_range(
                    modules=modules,
                    activations=activations,
                    eval_module=eval_module,
                    eval_inputs=layer_cache[eval_name].inputs if layer_cache else None,
                    eval_kwargs=eval_kwargs,
                    orig_weights=orig_wgts,
                )
                quantizer_state_dict[cache_key] = quantizer.state_dict()
                gc.collect()
                torch.cuda.empty_cache()
            else:
                quantizer.load_state_dict(quantizer_state_dict[cache_key], device=device)
            if tensor_type == TensorType.Inputs:
                if attn.v_proj_rname in cache_key:
                    for proj_name in [attn.q_proj_rname, attn.k_proj_rname]:
                        quantizers[cache_key.replace(attn.v_proj_rname, proj_name)] = quantizer
                if ffn.up_proj_rnames[0] in cache_key:
                    for proj_name in ffn.up_proj_rnames[1:]:
                        quantizers[cache_key.replace(ffn.up_proj_rnames[0], proj_name)] = quantizer
        del quantizer
    for name, module in layer.module.named_modules():
        module_name = f"{layer.name}.{name}"
        ipts_quantizer = quantizers.get(f"{module_name}.input", None)
        opts_quantizer = quantizers.get(f"{module_name}.output", None)
        needs_quant_ipts = ipts_quantizer is not None and ipts_quantizer.is_enabled()
        needs_quant_opts = opts_quantizer is not None and opts_quantizer.is_enabled()
        if needs_quant_ipts or needs_quant_opts:
            logger.debug(
                "- Quantizing %s (%s)",
                module_name,
                ("inputs" if needs_quant_ipts else "")
                + (" and " if needs_quant_ipts and needs_quant_opts else "")
                + ("outputs" if needs_quant_opts else ""),
            )
            if needs_quant_ipts:
                ipts_quantizer.as_hook(is_output=False).register(module)
            if needs_quant_opts:
                opts_quantizer.as_hook(is_output=True).register(module)
    tools.logging.Formatter.indent_dec()


@torch.inference_mode()
def quantize_llm_activations(
    model: nn.Module | LlmModelStruct,
    config: LlmQuantConfig,
    tokenizer: PreTrainedTokenizer | None = None,
    quantizer_state_dict: dict[str, tp.Any] | None = None,
    orig_state_dict: dict[str, torch.Tensor] | None = None,
) -> dict[str, tp.Any]:
    """Quantize the large foundation model activations.

    Args:
        model (`nn.Module` or `LlmStruct`):
            Model to be quantized.
        config (`LlmQuantConfig`):
            Quantization configuration.
        tokenizer (`PreTrainedTokenizer`, *optional*, defaults to `None`):
            Tokenizer.
        quantizer_state_dict (`dict[str, Any]`, *optional*, defaults to `None`):
            Activation quantizer state dict cache.
        orig_state_dict (`dict[str, torch.Tensor]`, *optional*, defaults to `None`):
            Original weight state dict

    Returns:
        `dict[str, Any]`:
            Activation quantizer state dict cache.
    """
    if not isinstance(model, LlmModelStruct):
        model = LlmModelStruct.construct(model)
    assert isinstance(model, LlmModelStruct)
    quantizer_state_dict = quantizer_state_dict or {}
    with tools.logging.redirect_tqdm():
        if not quantizer_state_dict and config.needs_acts_quantizer_cache:
            for _, (layer, layer_cache, layer_kwargs) in tqdm(
                config.calib.build_loader(tokenizer).iter_layer_activations(
                    model,
                    needs_inputs_fn=get_needs_inputs_fn(model=model, config=config),
                    needs_outputs_fn=get_needs_outputs_fn(model=model, config=config),
                ),
                desc="quantizing activations",
                leave=False,
                total=len(model.backbone_struct.layer_structs),
                dynamic_ncols=True,
            ):
                quantize_llm_layer_activations(
                    layer=layer,
                    config=config,
                    quantizer_state_dict=quantizer_state_dict,
                    layer_cache=layer_cache,
                    layer_kwargs=layer_kwargs,
                    orig_state_dict=orig_state_dict,
                )
        else:
            for layer in tqdm(
                model.backbone_struct.layer_structs,
                desc="quantizing activations",
                leave=False,
                dynamic_ncols=True,
            ):
                quantize_llm_layer_activations(
                    layer=layer,
                    config=config,
                    quantizer_state_dict=quantizer_state_dict,
                    orig_state_dict=orig_state_dict,
                )
    return quantizer_state_dict
