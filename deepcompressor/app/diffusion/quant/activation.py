# -*- coding: utf-8 -*-
"""Diffusion model activation quantization calibration module."""

import gc
import typing as tp

import torch
import torch.nn as nn
from tqdm import tqdm

from deepcompressor.data.cache import IOTensorsCache
from deepcompressor.data.common import TensorType
from deepcompressor.data.dtype import QuantDataType
from deepcompressor.utils import tools

from ..nn.struct import (
    DiffusionAttentionStruct,
    DiffusionBlockStruct,
    DiffusionModelStruct,
    DiffusionModuleStruct,
    DiffusionTransformerBlockStruct,
)
from .config import DiffusionQuantConfig
from .quantizer import DiffusionActivationQuantizer
from .utils import get_needs_inputs_fn, get_needs_outputs_fn

__all__ = ["quantize_diffusion_activations"]


@torch.inference_mode()
def quantize_diffusion_block_activations(  # noqa: C901
    layer: DiffusionBlockStruct | DiffusionModuleStruct,
    config: DiffusionQuantConfig,
    quantizer_state_dict: dict[str, dict[str, torch.Tensor | float | None]],
    layer_cache: dict[str, IOTensorsCache] | None = None,
    layer_kwargs: dict[str, tp.Any] | None = None,
    orig_state_dict: dict[str, torch.Tensor] | None = None,
) -> dict[str, DiffusionActivationQuantizer]:
    """Quantize the activations of a diffusion model block.

    Args:
        layer (`DiffusionBlockStruct` or `DiffusionModuleStruct`):
            The diffusion model block.
        config (`DiffusionQuantConfig`):
            The quantization configuration.
        quantizer_state_dict (`dict[str, dict[str, torch.Tensor | float | None]]`):
            The activation quantizers state dict cache.
        layer_cache (`dict[str, IOTensorsCache]`, *optional*):
            The layer cache.
        layer_kwargs (`dict[str, Any]`, *optional*):
            The layer keyword arguments.
        orig_state_dict (`dict[str, torch.Tensor]`, *optional*):
            The original state dictionary.

    Returns:
        `dict[str, DiffusionActivationQuantizer]`:
            The activation quantizers.
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
            list[str],  # module names
            nn.Module,  # eval module
            str,  # eval name
            dict[str, tp.Any],  # eval kwargs
            list[tuple[nn.Parameter, torch.Tensor]],  # original wgts
        ]
    ] = []
    In, Out = TensorType.Inputs, TensorType.Outputs  # noqa: F841

    used_modules: set[nn.Module] = set()
    for module_key, module_name, module, parent, field_name in layer.named_key_modules():
        modules, orig_struct_wgts = None, {}
        if field_name in ("k_proj", "v_proj", "add_q_proj", "add_v_proj"):
            continue
        if field_name in ("q_proj", "add_k_proj", "up_proj"):
            grandparent = parent.parent
            assert isinstance(grandparent, DiffusionTransformerBlockStruct)
            if grandparent.parallel and parent.idx == 0:
                if orig_state_dict:
                    orig_struct_wgts = {
                        proj_module: (proj_module.weight, orig_state_dict[f"{proj_name}.weight"])
                        for _, proj_name, proj_module, _, _ in grandparent.named_key_modules()
                    }
                if field_name == "q_proj":
                    assert isinstance(parent, DiffusionAttentionStruct)
                    assert module_name == parent.q_proj_name
                    modules, module_names = parent.qkv_proj, parent.qkv_proj_names
                    if grandparent.ffn_struct is not None:
                        modules.append(grandparent.ffn_struct.up_proj)
                        module_names.append(grandparent.ffn_struct.up_proj_name)
                elif field_name == "add_k_proj":
                    assert isinstance(parent, DiffusionAttentionStruct)
                    assert module_name == parent.add_k_proj_name
                    modules, module_names = parent.add_qkv_proj, parent.add_qkv_proj_names
                    if grandparent.add_ffn_struct is not None:
                        modules.append(grandparent.add_ffn_struct.up_proj)
                        module_names.append(grandparent.add_ffn_struct.up_proj_name)
                else:
                    assert field_name == "up_proj"
                    if module in used_modules:
                        continue
                    assert module_name == grandparent.add_ffn_struct.up_proj_name
                    assert grandparent.attn_structs[0].is_self_attn()
                eval_module, eval_name, eval_kwargs = grandparent.module, grandparent.name, layer_kwargs
            elif isinstance(parent, DiffusionAttentionStruct):
                eval_module, eval_name = parent.module, parent.name
                eval_kwargs = parent.filter_kwargs(layer_kwargs) if layer_kwargs else {}
                if orig_state_dict:
                    orig_struct_wgts = {
                        proj_module: (proj_module.weight, orig_state_dict[f"{proj_name}.weight"])
                        for _, proj_name, proj_module, _, _ in parent.named_key_modules()
                    }
                if field_name == "q_proj":
                    assert module_name == parent.q_proj_name
                    modules, module_names = parent.qkv_proj, parent.qkv_proj_names
                else:
                    assert field_name == "add_k_proj"
                    assert module_name == parent.add_k_proj_name
                    modules, module_names = parent.add_qkv_proj, parent.add_qkv_proj_names
        if modules is None:
            assert module not in used_modules
            used_modules.add(module)
            orig_wgts = [(module.weight, orig_state_dict[f"{module_name}.weight"])] if orig_state_dict else None
            args_caches.append((module_key, In, [module], [module_name], module, module_name, None, orig_wgts))
        else:
            orig_wgts = []
            for proj_module in modules:
                assert proj_module not in used_modules
                used_modules.add(proj_module)
                if orig_state_dict:
                    orig_wgts.append(orig_struct_wgts.pop(proj_module))
            orig_wgts.extend(orig_struct_wgts.values())
            orig_wgts = None if not orig_wgts else orig_wgts
            args_caches.append((module_key, In, modules, module_names, eval_module, eval_name, eval_kwargs, orig_wgts))
    # endregion
    quantizers: dict[str, DiffusionActivationQuantizer] = {}
    tools.logging.Formatter.indent_inc()
    for module_key, tensor_type, modules, module_names, eval_module, eval_name, eval_kwargs, orig_wgts in args_caches:
        if isinstance(modules[0], nn.Linear):
            channels_dim = -1
            assert all(isinstance(m, nn.Linear) for m in modules)
        elif isinstance(modules[0], nn.Conv2d):
            channels_dim = 1
            assert all(isinstance(m, nn.Conv2d) for m in modules)
        else:
            raise ValueError(f"Unknown module type: {type(modules[0])}")
        if tensor_type == TensorType.Inputs:
            cache_keys = [f"{name}.input" for name in module_names]
            quantizer_config = config.ipts
            if getattr(modules[0], "unsigned", False):
                if isinstance(config.ipts.dtype, QuantDataType) and config.ipts.dtype.is_integer:
                    quantizer_config = config.unsigned_ipts
            activations = layer_cache.get(module_names[0], IOTensorsCache()).inputs
        else:
            cache_keys = [f"{name}.output" for name in module_names]
            quantizer_config = config.opts
            activations = layer_cache.get(module_names[0], IOTensorsCache()).outputs
        quantizer = DiffusionActivationQuantizer(
            quantizer_config,
            channels_dim=channels_dim,
            develop_dtype=config.develop_dtype,
            key=module_key,
            tensor_type=tensor_type,
        )
        if quantizer.is_enabled():
            if cache_keys[0] not in quantizer_state_dict:
                logger.debug("- Calibrating %s", ", ".join(cache_keys))
                quantizer.calibrate_dynamic_range(
                    modules=modules,
                    activations=activations,
                    eval_module=eval_module,
                    eval_inputs=layer_cache[eval_name].inputs if layer_cache else None,
                    eval_kwargs=eval_kwargs,
                    orig_weights=orig_wgts,
                )
                quantizer_state_dict[cache_keys[0]] = quantizer.state_dict()
                gc.collect()
                torch.cuda.empty_cache()
            else:
                quantizer.load_state_dict(quantizer_state_dict[cache_keys[0]], device=modules[0].weight.device)
            for cache_key in cache_keys:
                quantizers[cache_key] = quantizer
        del quantizer
    tools.logging.Formatter.indent_dec()
    return quantizers


@torch.inference_mode()
def quantize_diffusion_activations(
    model: nn.Module | DiffusionModelStruct,
    config: DiffusionQuantConfig,
    quantizer_state_dict: dict[str, dict[str, torch.Tensor | float | None]] | None = None,
    orig_state_dict: dict[str, torch.Tensor] | None = None,
) -> dict[str, dict[str, torch.Tensor | float | None]]:
    """Quantize the activations of a diffusion model.

    Args:
        model (`nn.Module` or `DiffusionModelStruct`):
            The diffusion model.
        config (`DiffusionQuantConfig`):
            The quantization configuration.
        quantizer_state_dict (`dict[str, dict[str, torch.Tensor | float | None]]`, *optional*, defaults to `None`):
            The activation quantizers state dict cache.
        orig_state_dict (`dict[str, torch.Tensor]`, *optional*, defaults to `None`):
            The original state dictionary.

    Returns:
        `dict[str, dict[str, torch.Tensor | float | None]]`:
            The activation quantizers state dict cache.
    """
    logger = tools.logging.getLogger(f"{__name__}.ActivationQuant")
    if not isinstance(model, DiffusionModelStruct):
        model = DiffusionModelStruct.construct(model)
    assert isinstance(model, DiffusionModelStruct)
    quantizer_state_dict = quantizer_state_dict or {}
    quantizers: dict[str, DiffusionActivationQuantizer] = {}
    skip_pre_modules = all(key in config.ipts.skips for key in model.get_prev_module_keys())
    skip_post_modules = all(key in config.ipts.skips for key in model.get_post_module_keys())
    if not quantizer_state_dict and config.needs_acts_quantizer_cache:
        with tools.logging.redirect_tqdm():
            for _, (layer, layer_cache, layer_kwargs) in tqdm(
                config.calib.build_loader().iter_layer_activations(
                    model,
                    needs_inputs_fn=get_needs_inputs_fn(model, config=config),
                    needs_outputs_fn=get_needs_outputs_fn(model, config=config),
                    skip_pre_modules=skip_pre_modules,
                    skip_post_modules=skip_post_modules,
                ),
                desc="quantizing activations",
                leave=False,
                total=model.num_blocks + int(not skip_post_modules) + int(not skip_pre_modules) * 3,
                dynamic_ncols=True,
            ):
                block_quantizers = quantize_diffusion_block_activations(
                    layer=layer,
                    config=config,
                    quantizer_state_dict=quantizer_state_dict,
                    layer_cache=layer_cache,
                    layer_kwargs=layer_kwargs,
                    orig_state_dict=orig_state_dict,
                )
                quantizers.update(block_quantizers)
    else:
        for _, layer in model.get_named_layers(
            skip_pre_modules=skip_pre_modules, skip_post_modules=skip_post_modules
        ).items():
            block_quantizers = quantize_diffusion_block_activations(
                layer=layer,
                config=config,
                quantizer_state_dict=quantizer_state_dict,
                orig_state_dict=orig_state_dict,
            )
            quantizers.update(block_quantizers)
    for _, module_name, module, _, _ in model.named_key_modules():
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
    return quantizer_state_dict
