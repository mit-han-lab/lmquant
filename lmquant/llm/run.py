# -*- coding: utf-8 -*-
"""Evaluate a large language model."""

import gc
import json
import logging
import os
import pprint

import torch
import torch.utils.hooks
from transformers import AutoModelForCausalLM, AutoTokenizer

from lmquant.quant.quantizer import ActivationQuantizer, WeightQuantizer
from lmquant.utils import tools

from .config import LlmRunConfig
from .nn import LlmModelStruct
from .quant import quantize_llm_activations, quantize_llm_weights, reorder_llm, rotate_llm, smooth_llm

__all__ = ["run"]


def run(  # noqa: C901
    config: LlmRunConfig,
    return_with_quantizers: bool = False,
    return_with_scale_state_dict: bool = False,
    logging_level: int = logging.DEBUG,
) -> tuple[
    AutoModelForCausalLM,
    AutoTokenizer,
    dict[str, dict[str, list[torch.utils.hooks.RemovableHandle]]],
    dict[str, WeightQuantizer],
    dict[str, ActivationQuantizer],
    dict[str, torch.Tensor],
]:
    """Evaluate a large language model with the given arguments.

    Args:
        config (LlmRunConfig): The configuration.
        return_with_quantizers (bool, optional): Whether to return with quantizers. Defaults to ``False``.
        return_with_scale_state_dict (bool, optional): Whether to return with scale state dict. Defaults to ``False``.
        logging_level (int, optional): The logging level. Defaults to ``logging.DEBUG``.

    Returns:
        tuple[
            AutoModelForCausalLM,
            AutoTokenizer,
            dict[str, dict[str, list[torch.utils.hooks.RemovableHandle]]],
            dict[str, WeightQuantizer],
            dict[str, ActivationQuantizer],
            dict[str, torch.Tensor],
        ]: The model, tokenizer, hooks, weight quantizers, activation quantizers, and scale state dict.
    """
    output_dirpath = config.output_dirpath + ".RUNNING"
    os.makedirs(output_dirpath, exist_ok=True)
    config.dump(path=os.path.join(output_dirpath, "config.yaml"))
    tools.logging.setup(path=os.path.join(output_dirpath, "run.log"), level=logging_level)
    logger = logging.getLogger(__name__)
    # region log configurations
    logger.info("=== Configurations ===")
    tools.logging.info(config.formatted_str(), logger=logger)
    logger.info("=== Dumped Configurations ===")
    tools.logging.info(pprint.pformat(config.dump(), indent=2, width=120), logger=logger)
    logger.info("=== Output Directory ===")
    logger.info(config.output_dirpath)
    # endregion
    logger.info("=== Start Evaluating ===")
    quant_wgts = config.quant.enabled_wgts
    quant_ipts = config.quant.enabled_ipts
    quant_opts = config.quant.enabled_opts
    quant_acts = quant_ipts or quant_opts
    quant = quant_wgts or quant_acts
    needs_rotation = quant and config.quant.enabled_rotation
    # region rotate model
    if needs_rotation:
        logger.info(f"* Building model {config.model.name} from {config.model.path}")
        model, tokenizer = config.model.build(dtype=torch.float32, cpu=config.model.size > 30)
        model = LlmModelStruct.build(model)
        config.quant.num_hidden_layers = model.config.num_hidden_layers
        if config.quant.develop_dtype is None:
            config.quant.develop_dtype = torch.float64
        logger.info(f"* Development dtype is {config.quant.develop_dtype}")
        logger.info("* Rotating model")
        tools.logging.Formatter.indent_inc()
        if config.cache_path.rotation and os.path.exists(config.cache_path.rotation):
            logger.info(f"- Loading rotation from {config.cache_path.rotation}")
            rotation = torch.load(config.cache_path.rotation)
            rotate_llm(model, config.quant.rotation, rotation=rotation)
        else:
            logger.info("- Generating rotation")
            rotation = rotate_llm(model, config.quant.rotation)
            if config.cache_path.rotation:
                logger.info(f"- Saving rotation to {config.cache_path.rotation}")
                os.makedirs(config.cache_dirpath.rotation, exist_ok=True)
                torch.save(rotation, config.cache_path.rotation)
        tools.logging.Formatter.indent_dec()
        gc.collect()
        torch.cuda.empty_cache()
        rotated_state_dict = {name: param.data.cpu() for name, param in model.module.named_parameters()}
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        model, tokenizer = config.model.build()
        model = LlmModelStruct.build(model)
        logger.info("* Rotating model in float16 again")
        tools.logging.Formatter.indent_inc()
        rotate_llm(model, config.quant.rotation, rotation=rotation)
        for name, param in model.module.named_parameters():
            if name in rotated_state_dict:
                param.data = rotated_state_dict[name].data.to(dtype=param.dtype, device=param.device)
        del rotated_state_dict, rotation
        tools.logging.Formatter.indent_dec()
        gc.collect()
        torch.cuda.empty_cache()
    else:
        logger.info(f"* Building model {config.model.name} from {config.model.path}")
        model, tokenizer = config.model.build(dtype=torch.float16)
        model = LlmModelStruct.build(model)
        config.quant.num_hidden_layers = model.config.num_hidden_layers
        if config.quant.develop_dtype is None:
            config.quant.develop_dtype = torch.float32
        logger.info(f"* Development dtype is {config.quant.develop_dtype}")
    # endregion
    hooks: dict[str, dict[str, list[torch.utils.hooks.RemovableHandle]]] = dict(reorder={}, activation={})
    # region reorder channels
    if quant and config.quant.enabled_reorder:
        logger.info("* Reordering channels")
        tools.logging.Formatter.indent_inc()
        if config.cache_path.reorder and os.path.exists(config.cache_path.reorder):
            logger.info(f"- Loading channel indices from {config.cache_path.reorder}")
            reorder_cache = torch.load(config.cache_path.reorder)
            _, reorder_hooks = reorder_llm(
                model, config.quant, tokenizer, calib_config=config.calib, reorder_cache=reorder_cache
            )
        else:
            logger.info("- Generating channel indices")
            reorder_cache, reorder_hooks = reorder_llm(model, config.quant, tokenizer, calib_config=config.calib)
            if config.cache_path.reorder:
                logger.info(f"- Saving channel indices to {config.cache_path.reorder}")
                os.makedirs(config.cache_dirpath.reorder, exist_ok=True)
                torch.save(reorder_cache, config.cache_path.reorder)
        hooks["reorder"] = reorder_hooks
        del reorder_cache, reorder_hooks
        tools.logging.Formatter.indent_dec()
        gc.collect()
        torch.cuda.empty_cache()
    # endregion
    # region smooth quantization
    if quant and config.quant.enabled_smooth:
        logger.info("* Smooth quantizing model")
        tools.logging.Formatter.indent_inc()
        if config.cache_path.smooth and os.path.exists(config.cache_path.smooth):
            logger.info(f"- Loading smooth scales from {config.cache_path.smooth}")
            smooth_llm(model, config.quant, smooth_cache=torch.load(config.cache_path.smooth))
        else:
            logger.info("- Generating smooth scales")
            smooth_cache = smooth_llm(model, config.quant, tokenizer=tokenizer, calib_config=config.calib)
            if config.cache_path.smooth:
                logger.info(f"- Saving smooth scales to {config.cache_path.smooth}")
                os.makedirs(config.cache_dirpath.smooth, exist_ok=True)
                torch.save(smooth_cache, config.cache_path.smooth)
            del smooth_cache
        tools.logging.Formatter.indent_dec()
        gc.collect()
        torch.cuda.empty_cache()
    # endregion
    # region collect original state dict
    if config.quant.needs_orig_wgts:
        orig_state_dict: dict[str, torch.Tensor] = {
            name: param.detach().clone()
            for name, param in model.module.named_parameters()
            if param.ndim > 1 and config.quant.needs_quant_weights(name)
        }
    else:
        orig_state_dict = None
    # endregion
    if quant_wgts:
        logger.info("* Quantizing weights")
        tools.logging.Formatter.indent_inc()
        if config.cache_path.wgts and os.path.exists(config.cache_path.wgts):
            logger.info(f"- Loading weight settings from {config.cache_path.wgts}")
            _, weight_quantizers, scale_state_dict = quantize_llm_weights(
                model,
                config.quant,
                tokenizer=tokenizer,
                calib_config=config.calib,
                quant_cache=torch.load(config.cache_path.wgts),
                return_with_quantizers=return_with_quantizers,
                return_with_scale_state_dict=return_with_scale_state_dict or config.save_model,
            )
        else:
            logger.info("- Generating weight settings")
            quant_cache, weight_quantizers, scale_state_dict = quantize_llm_weights(
                model,
                config.quant,
                tokenizer=tokenizer,
                calib_config=config.calib,
                return_with_quantizers=return_with_quantizers,
                return_with_scale_state_dict=return_with_scale_state_dict or config.save_model,
            )
            if config.cache_dirpath.wgts:
                logger.info(f"- Saving weight settings to {config.cache_path.wgts}")
                os.makedirs(config.cache_dirpath.wgts, exist_ok=True)
                torch.save(quant_cache, config.cache_path.wgts)
            del quant_cache
        if config.save_model:
            logger.info(f"- Saving model to {config.output_dirpath}")
            torch.save(scale_state_dict, os.path.join(output_dirpath, "scale.pt"))
            torch.save(model.module.state_dict(), os.path.join(output_dirpath, "model.pt"))
            if not return_with_scale_state_dict:
                scale_state_dict = {}
        tools.logging.Formatter.indent_dec()
        gc.collect()
        torch.cuda.empty_cache()
    else:
        weight_quantizers: dict[str, WeightQuantizer] = {}
        scale_state_dict: dict[str, torch.Tensor] = {}
    if quant_acts:
        logger.info("  * Quantizing activations")
        tools.logging.Formatter.indent_inc()
        if config.cache_path.acts and os.path.exists(config.cache_path.acts):
            logger.info(f"- Loading activation settings from {config.cache_path.acts}")
            _, activation_quantizers, activation_hooks = quantize_llm_activations(
                model,
                config.quant,
                tokenizer=tokenizer,
                calib_config=config.calib,
                quant_cache=torch.load(config.cache_path.acts),
                orig_state_dict=orig_state_dict,
                return_with_quantizers=return_with_quantizers,
            )
        else:
            logger.info("- Generating activation settings")
            quant_cache, activation_quantizers, activation_hooks = quantize_llm_activations(
                model,
                config.quant,
                tokenizer=tokenizer,
                calib_config=config.calib,
                orig_state_dict=orig_state_dict,
                return_with_quantizers=return_with_quantizers,
            )
            if config.cache_dirpath.acts and quant_cache is not None:
                logger.info(f"- Saving activation settings to {config.cache_path.acts}")
                os.makedirs(config.cache_dirpath.acts, exist_ok=True)
                torch.save(quant_cache, config.cache_path.acts)
            del quant_cache
        hooks["activation"] = activation_hooks
        tools.logging.Formatter.indent_dec()
        del orig_state_dict, activation_hooks
        gc.collect()
        torch.cuda.empty_cache()
    else:
        activation_quantizers: dict[str, ActivationQuantizer] = {}
    # region evaluate model
    logger.info("* Evaluating model")
    tools.logging.Formatter.indent_inc()
    results = config.eval.evaluate(model.module, tokenizer, config.model.name)
    tools.logging.Formatter.indent_dec()
    logger.info(f"* Saving results to {config.output_dirpath}")
    # dump results
    with open(os.path.join(output_dirpath, "results.json"), "w") as f:
        for rst in results.values():
            rst["config"]["model"] = config.model.name
        json.dump(results, f, indent=2)
    # endregion
    assert output_dirpath, "Output directory is not set"
    assert os.path.exists(output_dirpath), f"Output directory {output_dirpath} does not exist"
    os.rename(output_dirpath, config.output_dirpath)
    return model, tokenizer, hooks, weight_quantizers, activation_quantizers, scale_state_dict


if __name__ == "__main__":
    config, _, unknown_args = LlmRunConfig.parse_args()
    assert len(unknown_args) == 0, f"Unknown arguments: {unknown_args}"
    run(config)
