# -*- coding: utf-8 -*-
"""LLM weight quantization calibration module."""
import gc
import logging
import typing as tp

import torch
import torch.nn as nn
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from lmquant.dataset import IOActivationsCache
from lmquant.quant.quantizer.weight import WeightQuantizer
from lmquant.utils import tools

from ..dataset import LlmCalibConfig, LlmCalibrationCache
from ..nn import LlmDecoderLayerStruct, LlmModelStruct
from ..utils import get_needs_inputs_fn
from .config import LlmQuantConfig

__all__ = ["quantize_llm_weights"]


@torch.inference_mode()
def quantize_llm_decoder_layer_weights(  # noqa: C901
    layer: LlmDecoderLayerStruct,
    config: LlmQuantConfig,
    quant_cache: dict[str, dict[str, torch.Tensor | float | None]],
    layer_cache: dict[str, IOActivationsCache] | None = None,
    layer_kwargs: dict[str, tp.Any] | None = None,
    return_with_quantizers: bool = False,
    return_with_scale_state_dict: bool = False,
) -> tuple[dict[str, WeightQuantizer], dict[str, torch.Tensor | float | None]]:
    """Calibrate the weight quantization ranges of modules in a decoder layer.

    Args:
        layer (LlmDecoderLayerStruct): Decoder layer.
        config (LlmModelQuantConfig): Model quantization config.
        quant_cache (dict[str, dict[str, torch.Tensor | float | None]], optional): Quantization cache. Defaults to ``None``.
        layer_cache (dict[str, IOActivationsCache], optional): Layer cache. Defaults to ``None``.
        layer_kwargs (dict[str, tp.Any], optional): Layer keyword arguments. Defaults to ``None``.
        return_with_quantizers (bool, optional): Whether to return with quantizers. Defaults to ``False``.
        return_with_scale_state_dict (bool, optional): Whether to return with scale state dict. Defaults to ``False``.

    Returns:
        tuple[dict[str, WeightQuantizer], dict[str, torch.Tensor | float | None]: Quantizers, and scale state dict.
    """
    logger = logging.getLogger(f"{__name__}.WeightQuant")
    logger.debug("- Quantizing decoder layer %s", layer.full_name)
    tools.logging.Formatter.indent_inc()
    layer_cache = layer_cache or {}
    layer_kwargs = layer_kwargs or {}

    args_caches: list[tuple[str, nn.Module, str, nn.Module, str, dict]] = []
    # region proj_q and proj_k
    attn_kwargs = layer.filter_layer_kwargs_to_attn_kwargs(layer_kwargs)
    key = "proj_qkv"
    for module_name, module in [(layer.proj_q_full_name, layer.proj_q), (layer.proj_k_full_name, layer.proj_k)]:
        args_caches.append((key, module, module_name, layer.attn_block, layer.attn_block_full_name, attn_kwargs))
    # endregion
    # region proj_v and proj_o
    for key, module_name, module in [
        ("proj_qkv", layer.proj_v_full_name, layer.proj_v),
        ("proj_out", layer.proj_o_full_name, layer.proj_o),
    ]:
        args_caches.append((key, module, module_name, module, module_name, None))
    # endregion
    # region ffn block
    if layer.router is not None:
        key, module_name, module = "router", layer.router_full_name, layer.router
        args_caches.append((key, module, module_name, module, module_name, None))
    num_experts = layer.config.num_experts
    for expert_idx in range(num_experts):
        key = "proj_1st"
        for module_name, module in zip(
            layer.proj_1st_full_names[expert_idx::num_experts], layer.proj_1st[expert_idx::num_experts]
        ):
            args_caches.append((key, module, module_name, module, module_name, None))
        key, module_name, module = "proj_2nd", layer.proj_2nd_full_names[expert_idx], layer.proj_2nd[expert_idx]
        args_caches.append((key, module, module_name, module, module_name, None))
    # endregion
    for key, module, module_name, eval_module, eval_name, eval_kwargs in args_caches:
        quantizer_config = config.specialize_for(key, layer_idx=layer.idx).wgts
        quantizer = WeightQuantizer(quantizer_config, develop_dtype=config.develop_dtype, key=key)
        if quantizer.enabled:
            if module_name not in quant_cache:
                logger.debug("- Calibrating %s.weight", module_name)
                quantizer.calibrate_dynamic_range(
                    module=module,
                    inputs=layer_cache.get(module_name, IOActivationsCache()).inputs,
                    eval_inputs=layer_cache.get(eval_name, IOActivationsCache()).inputs,
                    eval_module=eval_module,
                    eval_kwargs=eval_kwargs,
                )
                quant_cache[module_name] = quantizer.state_dict()
                gc.collect()
                torch.cuda.empty_cache()
    quantizers: dict[str, WeightQuantizer] = {}
    scale_state_dict: dict[str, torch.Tensor | float | None] = {}
    for key, module, module_name, eval_module, eval_name, eval_kwargs in args_caches:
        quantizer_config = config.specialize_for(key, layer_idx=layer.idx).wgts
        quantizer = WeightQuantizer(quantizer_config, develop_dtype=config.develop_dtype, key=key)
        param_name = f"{module_name}.weight"
        if quantizer.enabled:
            logger.debug("- Quantizing %s", param_name)
            quantizer.load_state_dict(quant_cache[module_name], device=module.weight.device)
            result = quantizer.quantize(
                module.weight.data,
                inputs=layer_cache.get(module_name, IOActivationsCache()).inputs,
                return_with_dequant=True,
                return_with_quant=return_with_scale_state_dict,
            )
            module.weight.data = result.data
            if return_with_quantizers:
                quantizers[param_name] = quantizer
            if return_with_scale_state_dict:
                scale_naem = f"{param_name}.scale"
                zero_name = f"{param_name}.zero"
                for level in range(result.scale.num_levels):
                    scale_state_dict[f"{scale_naem}.{level}"] = result.scale.get_level_scale(level).to("cpu")
                if isinstance(result.zero, torch.Tensor):
                    scale_state_dict[zero_name] = result.zero.to("cpu")
                else:
                    scale_state_dict[zero_name] = result.zero
            del result
            gc.collect()
            torch.cuda.empty_cache()
    del args_caches
    tools.logging.Formatter.indent_dec()
    return quantizers, scale_state_dict


@torch.inference_mode()
def quantize_llm_weights(
    model: nn.Module | LlmModelStruct,
    quant_config: LlmQuantConfig,
    tokenizer: nn.Module | None = None,
    calib_config: LlmCalibConfig | None = None,
    quant_cache: dict[str, dict[str, torch.Tensor | float | None]] | None = None,
    return_with_quantizers: bool = False,
    return_with_scale_state_dict: bool = False,
) -> tuple[
    dict[str, dict[str, torch.Tensor | float | None]],
    dict[str, WeightQuantizer],
    dict[str, torch.Tensor | float | None],
]:
    """Quantize the large foundation model weights.

    Args:
        model (nn.Module | LlmModelStruct): Model to be smoothed.
        quant_config (LlmModelQuantConfig): Module quantization config.
        tokenizer (nn.Module, optional): Tokenizer. Defaults to ``None``.
        calib_config (LlmCalibrationConfig, optional): Calibration config. Defaults to ``None``.
        quant_cache (dict[str, dict[str, torch.Tensor | float | None]], optional): Quantization cache.
            Defaults to ``None``.
        return_with_quantizers (bool, optional): Whether to return with quantizers. Defaults to ``False``.
        return_with_scale_state_dict (bool, optional): Whether to return with scale state dict. Defaults to ``False``.

    Returns:
        tuple[
            dict[str, dict[str, torch.Tensor | float | None]],
            dict[str, WeightQuantizer],
            dict[str, torch.Tensor | float | None
        ]: Quantization cache, quantizers, and scale state dict.
    """
    if not isinstance(model, LlmModelStruct):
        model = LlmModelStruct.build(model)
    assert isinstance(model, LlmModelStruct)
    quant_cache = quant_cache or {}
    quantizers: dict[str, WeightQuantizer] = {}
    scale_state_dict: dict[str, torch.Tensor | float | None] = {}
    if not quant_cache or quant_config.wgts.enabled_calib_kernel:
        with logging_redirect_tqdm():
            for _, (layer, layer_cache, layer_kwargs) in tqdm(
                LlmCalibrationCache(calib_config).iter_layer_activations(
                    model,
                    tokenizer,
                    needs_inputs_fn=get_needs_inputs_fn(config=quant_config),
                    needs_samples_caching=False,
                ),
                desc="quantizing weights",
                leave=False,
                total=model.config.num_hidden_layers,
            ):
                block_quantizers, block_state_dict = quantize_llm_decoder_layer_weights(
                    layer=layer,
                    config=quant_config,
                    quant_cache=quant_cache,
                    layer_cache=layer_cache,
                    layer_kwargs=layer_kwargs,
                    return_with_quantizers=return_with_quantizers,
                    return_with_scale_state_dict=return_with_scale_state_dict,
                )
                quantizers.update(block_quantizers)
                scale_state_dict.update(block_state_dict)
    else:
        for layer in model.backbone_struct.layer_structs:
            block_quantizers, block_state_dict = quantize_llm_decoder_layer_weights(
                layer=layer,
                config=quant_config,
                quant_cache=quant_cache,
                return_with_quantizers=return_with_quantizers,
                return_with_scale_state_dict=return_with_scale_state_dict,
            )
            quantizers.update(block_quantizers)
            scale_state_dict.update(block_state_dict)
    return quant_cache, quantizers, scale_state_dict
