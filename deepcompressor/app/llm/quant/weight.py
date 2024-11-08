# -*- coding: utf-8 -*-
"""LLM weight quantization calibration module."""

import gc
import typing as tp

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from deepcompressor.data.cache import IOTensorsCache
from deepcompressor.data.zero import ZeroPointDomain
from deepcompressor.utils import tools

from ..nn.struct import LlmModelStruct, LlmSelfAttentionStruct, LlmTransformerBlockStruct
from .config import LlmQuantConfig
from .quantizer import LlmWeightQuantizer
from .utils import get_needs_inputs_fn

__all__ = ["quantize_llm_weights"]


@torch.inference_mode()
def quantize_llm_layer_weights(  # noqa: C901
    layer: LlmTransformerBlockStruct,
    config: LlmQuantConfig,
    quantizer_state_dict: dict[str, tp.Any],
    layer_cache: dict[str, IOTensorsCache] | None = None,
    layer_kwargs: dict[str, tp.Any] | None = None,
    return_with_scale_state_dict: bool = False,
) -> dict[str, torch.Tensor | float | None]:
    """Calibrate the weight quantization ranges of modules in a layer.

    Args:
        layer (`LlmTransformerBlockStruct`):
            Layer.
        config (`LlmQuantConfig`):
            Quantization config.
        quantizer_state_dict (`dict[str, Any]`):
            Weight quantizer.
        layer_cache (`dict[str, IOTensorsCache]` or `None`, *optional*, defaults to `None`):
            Layer activations cache.
        layer_kwargs (`dict[str, tp.Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments for the layer.
        return_with_scale_state_dict (bool, *optional*, defaults to `False`):
            Whether to return with scale state dict.

    Returns:
        `dict[str, torch.Tensor | float | None]`:
            Scale state dict.
    """
    logger = tools.logging.getLogger(f"{__name__}.WeightQuant")
    logger.debug("- Quantizing layer %s", layer.name)
    tools.logging.Formatter.indent_inc()
    layer_cache = layer_cache or {}
    layer_kwargs = layer_kwargs or {}
    for module_key, module_name, module, parent, field_name in layer.named_key_modules():
        assert isinstance(module, nn.Linear)
        if field_name in ("q_proj", "k_proj"):
            assert isinstance(parent, LlmSelfAttentionStruct)
            eval_name, eval_module, eval_kwargs = parent.name, parent.module, parent.filter_kwargs(layer_kwargs)
        else:
            eval_name, eval_module, eval_kwargs = module_name, module, None
        quantizer = LlmWeightQuantizer(config.wgts, develop_dtype=config.develop_dtype, key=module_key)
        if quantizer.is_enabled():
            if module_name not in quantizer_state_dict:
                logger.debug("- Calibrating %s.weight", module_name)
                quantizer.calibrate_dynamic_range(
                    module=module,
                    inputs=layer_cache[module_name].inputs if layer_cache else None,
                    eval_inputs=layer_cache[eval_name].inputs if layer_cache else None,
                    eval_module=eval_module,
                    eval_kwargs=eval_kwargs,
                )
                quantizer_state_dict[module_name] = quantizer.state_dict()
                gc.collect()
                torch.cuda.empty_cache()
    scale_state_dict: dict[str, torch.Tensor | float | None] = {}
    for module_key, module_name, module, _, _ in layer.named_key_modules():
        assert isinstance(module, nn.Linear)
        quantizer = LlmWeightQuantizer(config.wgts, develop_dtype=config.develop_dtype, key=module_key)
        param_name = f"{module_name}.weight"
        if quantizer.is_enabled():
            logger.debug("- Quantizing %s", param_name)
            quantizer.load_state_dict(quantizer_state_dict[module_name], device=module.weight.device)
            result = quantizer.quantize(
                module.weight.data,
                inputs=layer_cache[module_name].inputs.front() if layer_cache else None,
                return_with_dequant=True,
                return_with_quant=return_with_scale_state_dict,
            )
            module.weight.data = result.data
            if return_with_scale_state_dict:
                scale_state_dict.update(result.scale.state_dict(f"{param_name}.scale"))
                zero_name = "scaled_zero" if config.wgts.zero_domain is ZeroPointDomain.PostScale else "zero"
                if isinstance(result.zero, torch.Tensor):
                    scale_state_dict[f"{param_name}.{zero_name}"] = result.zero.to("cpu")
                else:
                    scale_state_dict[f"{param_name}.{zero_name}"] = result.zero
            del result
            gc.collect()
            torch.cuda.empty_cache()
    tools.logging.Formatter.indent_dec()
    return scale_state_dict


@torch.inference_mode()
def quantize_llm_weights(
    model: nn.Module | LlmModelStruct,
    config: LlmQuantConfig,
    tokenizer: PreTrainedTokenizer | None = None,
    quantizer_state_dict: dict[str, tp.Any] | None = None,
    return_with_scale_state_dict: bool = False,
) -> tuple[dict[str, tp.Any], dict[str, torch.Tensor | float | None]]:
    """Quantize the large language model weights.

    Args:
        model (`nn.Module` or `LlmStruct`):
            Model to be quantized.
        config (`LlmQuantConfig`):
            Quantization configuration.
        tokenizer (`PreTrainedTokenizer`, *optional*, defaults to `None`):
            Tokenizer.
        quantizer_state_dict (`dict[str, Any]`, *optional*, defaults to `None`):
            Weight quantizer state dict.
        return_with_scale_state_dict (bool, *optional*, defaults to `False`):
            Whether to return with scale state dict.

    Returns:
        `tuple[dict[str, Any], dict[str, torch.Tensor | float | None]`:
            Weight quantizer cache and scale state dict.
    """
    if not isinstance(model, LlmModelStruct):
        model = LlmModelStruct.construct(model)
    assert isinstance(model, LlmModelStruct)
    quantizer_state_dict = quantizer_state_dict or {}
    scale_state_dict: dict[str, torch.Tensor | float | None] = {}
    with tools.logging.redirect_tqdm():
        if not quantizer_state_dict and config.wgts.needs_calib_data:
            for _, (layer, layer_cache, layer_kwargs) in tqdm(
                config.calib.build_loader(tokenizer).iter_layer_activations(
                    model,
                    needs_inputs_fn=get_needs_inputs_fn(model=model, config=config),
                ),
                desc="quantizing weights",
                leave=False,
                total=len(model.backbone_struct.layer_structs),
                dynamic_ncols=True,
            ):
                scale_state_dict.update(
                    quantize_llm_layer_weights(
                        layer=layer,
                        config=config,
                        quantizer_state_dict=quantizer_state_dict,
                        layer_cache=layer_cache,
                        layer_kwargs=layer_kwargs,
                        return_with_scale_state_dict=return_with_scale_state_dict,
                    )
                )
        else:
            for layer in tqdm(
                model.backbone_struct.layer_structs, desc="quantizing weights", leave=False, dynamic_ncols=True
            ):
                scale_state_dict.update(
                    quantize_llm_layer_weights(
                        layer=layer,
                        config=config,
                        quantizer_state_dict=quantizer_state_dict,
                        return_with_scale_state_dict=return_with_scale_state_dict,
                    )
                )
    return quantizer_state_dict, scale_state_dict
