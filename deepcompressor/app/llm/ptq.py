# -*- coding: utf-8 -*-
"""Evaluate a large language model."""

import gc
import json
import os
import pprint
import traceback

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from deepcompressor.utils import tools

from .config import LlmCacheConfig, LlmPtqRunConfig, LlmQuantCacheConfig, LlmQuantConfig
from .nn import LlmModelStruct
from .quant import quantize_llm_activations, quantize_llm_weights, reorder_llm, rotate_llm, smooth_llm

__all__ = ["ptq"]


def ptq(  # noqa: C901
    model: PreTrainedModel | LlmModelStruct,
    /,
    tokenizer: PreTrainedTokenizer,
    config: LlmQuantConfig,
    cache: LlmCacheConfig | None = None,
    load_dirpath: str = "",
    save_dirpath: str = "",
    copy_on_save: bool = False,
    save_model: bool = False,
) -> PreTrainedModel:
    """Post-training quantization of a large language model.

    Args:
        model (`PreTrainedModel` or `LlmStruct`):
            The large language model.
        tokenizer (`PreTrainedTokenizer`):
            The large language model tokenizer.
        config (`LlmQuantConfig`):
            The large language model post-training quantization configuration.
        cache (`LlmCacheConfig`, *optional*, defaults to `None`):
            The large language model quantization cache path configuration.
        load_dirpath (`str`, *optional*, defaults to `""`):
            The directory path to load the quantization checkpoint.
        save_dirpath (`str`, *optional*, defaults to `""`):
            The directory path to save the quantization checkpoint.
        copy_on_save (`bool`, *optional*, defaults to `False`):
            Whether to copy the cache to the save directory.
        save_model (`bool`, *optional*, defaults to `False`):
            Whether to save the quantized model checkpoint.

    Returns:
        `PreTrainedModel`:
            The quantized model.
    """
    logger = tools.logging.getLogger(__name__)
    if not isinstance(model, LlmModelStruct):
        model = LlmModelStruct.construct(model)
    assert isinstance(model, LlmModelStruct)

    quant_wgts = config.enabled_wgts
    quant_ipts = config.enabled_ipts
    quant_opts = config.enabled_opts
    quant_acts = quant_ipts or quant_opts
    quant = quant_wgts or quant_acts
    needs_rotation = quant and config.enabled_rotation
    needs_reorder = quant and config.enabled_reorder
    needs_smooth = quant and config.enabled_smooth

    load_model_path, load_path, save_path = "", None, None
    if load_dirpath:
        load_path = LlmQuantCacheConfig(
            rotation=os.path.join(load_dirpath, "rotation.pt"),
            reorder=os.path.join(load_dirpath, "reorder.pt"),
            smooth=os.path.join(load_dirpath, "smooth.pt"),
            wgts=os.path.join(load_dirpath, "wgts.pt"),
            acts=os.path.join(load_dirpath, "acts.pt"),
        )
        load_model_path = os.path.join(load_dirpath, "model.pt")
        if os.path.exists(load_model_path):
            logger.info(f"* Found the model from {load_model_path}")
            load_model = True
            save_dirpath = ""  # do not save the model if loading
            if needs_reorder and not config.reorder.dynamic:
                needs_reorder = False
                logger.info("* Safe to skip reordering the model")
            if needs_smooth:
                needs_smooth = False
                logger.info("* Safe to skip smoothing the model")
        else:
            logger.warning(f"Model checkpoint {load_model_path} does not exist")
            load_model, load_model_path = False, ""
    else:
        load_model = False
    if save_dirpath:
        os.makedirs(save_dirpath, exist_ok=True)
        save_path = LlmQuantCacheConfig(
            rotation=os.path.join(save_dirpath, "rotation.pt"),
            reorder=os.path.join(save_dirpath, "reorder.pt"),
            smooth=os.path.join(save_dirpath, "smooth.pt"),
            wgts=os.path.join(save_dirpath, "wgts.pt"),
            acts=os.path.join(save_dirpath, "acts.pt"),
        )
    else:
        save_model = False

    # region rotate model
    if needs_rotation:
        logger.info("* Rotating model")
        tools.logging.Formatter.indent_inc()
        load_from = ""
        if load_path and os.path.exists(load_path.rotation):
            load_from = load_path.rotation
        elif cache and cache.path.rotation and os.path.exists(cache.path.rotation):
            load_from = cache.path.rotation
        if load_from:
            logger.info(f"- Loading rotation from {load_from}")
            rotation = torch.load(load_from)
            rotate_llm(model, config.rotation, rotation=rotation)
        else:
            logger.info("- Generating rotation")
            rotation = rotate_llm(model, config.rotation)
            if cache and cache.path.rotation:
                logger.info(f"- Saving rotation to {cache.path.rotation}")
                os.makedirs(cache.dirpath.rotation, exist_ok=True)
                torch.save(rotation, cache.path.rotation)
                load_from = cache.path.rotation
        if save_path:
            if not copy_on_save and load_from:
                logger.info(f"- Linking rotation to {save_path.rotation}")
                os.symlink(os.path.relpath(load_from, save_dirpath), save_path.rotation)
            else:
                logger.info(f"- Saving rotation to {save_path.rotation}")
                torch.save(rotation, save_path.rotation)
        del rotation
        tools.logging.Formatter.indent_dec()
        gc.collect()
        torch.cuda.empty_cache()
    logger.info(f"* Development dtype is {config.develop_dtype}")
    # endregion
    # region reorder channels
    if needs_reorder:
        logger.info("* Reordering channels")
        tools.logging.Formatter.indent_inc()
        load_from = ""
        if load_path and os.path.exists(load_path.reorder):
            load_from = load_path.reorder
        elif cache and cache.path.reorder and os.path.exists(cache.path.reorder):
            load_from = cache.path.reorder
        if load_from:
            logger.info(f"- Loading reorder indices from {load_from}")
            reorder_cache = torch.load(load_from)
            reorder_llm(model, config, tokenizer, reorder_cache=reorder_cache)
        else:
            logger.info("- Generating reorder indices")
            reorder_cache = reorder_llm(model, config, tokenizer)
            if cache and cache.path.reorder:
                logger.info(f"- Saving reorder indices to {cache.path.reorder}")
                os.makedirs(cache.dirpath.reorder, exist_ok=True)
                torch.save(reorder_cache, cache.path.reorder)
                load_from = cache.path.reorder
        if save_path:
            if not copy_on_save and load_from:
                logger.info(f"- Linking reorder indices to {save_path.reorder}")
                os.symlink(os.path.relpath(load_from, save_dirpath), save_path.reorder)
            else:
                logger.info(f"- Saving reorder indices to {save_path.reorder}")
                torch.save(reorder_cache, save_path.reorder)
        del reorder_cache
        tools.logging.Formatter.indent_dec()
        gc.collect()
        torch.cuda.empty_cache()
    # endregion
    # region smooth quantization
    if needs_smooth:
        logger.info("* Smoothing model for quantization")
        tools.logging.Formatter.indent_inc()
        load_from = ""
        if load_path and os.path.exists(load_path.smooth):
            load_from = load_path.smooth
        elif cache and cache.path.smooth and os.path.exists(cache.path.smooth):
            load_from = cache.path.smooth
        if load_from:
            logger.info(f"- Loading smooth scales from {load_from}")
            smooth_cache = torch.load(load_from)
            smooth_llm(model, config, smooth_cache=smooth_cache)
        else:
            logger.info("- Generating smooth scales")
            smooth_cache = smooth_llm(model, config, tokenizer=tokenizer)
            if cache and cache.path.smooth:
                logger.info(f"- Saving smooth scales to {cache.path.smooth}")
                os.makedirs(cache.dirpath.smooth, exist_ok=True)
                torch.save(smooth_cache, cache.path.smooth)
                load_from = cache.path.smooth
        if save_path:
            if not copy_on_save and load_from:
                logger.info(f"- Linking smooth scales to {save_path.smooth}")
                os.symlink(os.path.relpath(load_from, save_dirpath), save_path.smooth)
            else:
                logger.info(f"- Saving smooth scales to {save_path.smooth}")
                torch.save(smooth_cache, save_path.smooth)
        del smooth_cache
        tools.logging.Formatter.indent_dec()
        gc.collect()
        torch.cuda.empty_cache()
    # endregion
    # region collect original state dict
    if config.needs_acts_quantizer_cache:
        if load_path and os.path.exists(load_path.acts):
            orig_state_dict = None
        elif cache and cache.path.acts and os.path.exists(cache.path.acts):
            orig_state_dict = None
        else:
            orig_state_dict: dict[str, torch.Tensor] = {
                name: param.detach().clone() for name, param in model.module.named_parameters() if param.ndim > 1
            }
    else:
        orig_state_dict = None
    # endregion
    if load_model:
        logger.info(f"* Loading model checkpoint from {load_model_path}")
        model.module.load_state_dict(torch.load(load_model_path))
        gc.collect()
        torch.cuda.empty_cache()
    elif quant_wgts:
        logger.info("* Quantizing weights")
        tools.logging.Formatter.indent_inc()
        load_from = ""
        if load_path and os.path.exists(load_path.wgts):
            load_from = load_path.wgts
        elif cache and cache.path.wgts and os.path.exists(cache.path.wgts):
            load_from = cache.path.wgts
        if load_from:
            logger.info(f"- Loading weight quantizer settings from {load_from}")
            quantizer_state_dict = torch.load(load_from)
            _, scale_state_dict = quantize_llm_weights(
                model,
                config,
                tokenizer=tokenizer,
                quantizer_state_dict=quantizer_state_dict,
                return_with_scale_state_dict=save_model,
            )
        else:
            logger.info("- Generating weight quantizer settings")
            quantizer_state_dict, scale_state_dict = quantize_llm_weights(
                model, config, tokenizer=tokenizer, return_with_scale_state_dict=save_model
            )
            if cache and cache.dirpath.wgts:
                logger.info(f"- Saving weight quantizer settings to {cache.path.wgts}")
                os.makedirs(cache.dirpath.wgts, exist_ok=True)
                torch.save(quantizer_state_dict, cache.path.wgts)
                load_from = cache.path.wgts
        if save_path:
            if not copy_on_save and load_from:
                logger.info(f"- Linking weight quantizer settings to {save_path.wgts}")
                os.symlink(os.path.relpath(load_from, save_dirpath), save_path.wgts)
            else:
                logger.info(f"- Saving weight quantizer settings to {save_path.wgts}")
                torch.save(quantizer_state_dict, save_path.wgts)
        if save_model:
            logger.info(f"- Saving model checkpoint to {save_dirpath}")
            torch.save(scale_state_dict, os.path.join(save_dirpath, "scale.pt"))
            torch.save(model.module.state_dict(), os.path.join(save_dirpath, "model.pt"))
        del quantizer_state_dict, scale_state_dict
        tools.logging.Formatter.indent_dec()
        gc.collect()
        torch.cuda.empty_cache()
    if quant_acts:
        logger.info("  * Quantizing activations")
        tools.logging.Formatter.indent_inc()
        if config.needs_acts_quantizer_cache:
            load_from = ""
            if load_path and os.path.exists(load_path.acts):
                load_from = load_path.acts
            elif cache and cache.path.acts and os.path.exists(cache.path.acts):
                load_from = cache.path.acts
            if load_from:
                logger.info(f"- Loading activation quantizer settings from {load_from}")
                quantizer_state_dict = torch.load(load_from)
                quantize_llm_activations(
                    model,
                    config,
                    tokenizer=tokenizer,
                    quantizer_state_dict=quantizer_state_dict,
                    orig_state_dict=orig_state_dict,
                )
            else:
                logger.info("- Generating activation quantizer settings")
                quantizer_state_dict = quantize_llm_activations(
                    model, config, tokenizer=tokenizer, orig_state_dict=orig_state_dict
                )
                if cache and cache.dirpath.acts:
                    logger.info(f"- Saving activation quantizer settings to {cache.path.acts}")
                    os.makedirs(cache.dirpath.acts, exist_ok=True)
                    torch.save(quantizer_state_dict, cache.path.acts)
                    load_from = cache.path.acts
            if save_dirpath:
                if not copy_on_save and load_from:
                    logger.info(f"- Linking activation quantizer settings to {save_path.acts}")
                    os.symlink(os.path.relpath(load_from, save_dirpath), save_path.acts)
                else:
                    logger.info(f"- Saving activation quantizer settings to {save_path.acts}")
                    torch.save(quantizer_state_dict, save_path.acts)
            del quantizer_state_dict
        else:
            logger.info("- No need to generate/load activation quantizer settings")
            quantize_llm_activations(model, config, tokenizer=tokenizer, orig_state_dict=orig_state_dict)
        tools.logging.Formatter.indent_dec()
        del orig_state_dict
        gc.collect()
        torch.cuda.empty_cache()
    return model.module


def main(config: LlmPtqRunConfig, logging_level: int = tools.logging.DEBUG) -> None:  # noqa: C901
    """Post-training quantization and evaluation of a large language model.

    Args:
        config (`LlmPtqConfig`):
            The large language model post-training quantization configuration.
        logging_level (`int`, *optional*, defaults to `logging.DEBUG`):
            The logging level.
    """
    config.output.lock()
    config.dump(path=config.output.get_running_job_path("config.yaml"))
    tools.logging.setup(path=config.output.get_running_job_path("run.log"), level=logging_level)
    logger = tools.logging.getLogger(__name__)
    # region log configurations
    logger.info("=== Configurations ===")
    tools.logging.info(config.formatted_str(), logger=logger)
    logger.info("=== Dumped Configurations ===")
    tools.logging.info(pprint.pformat(config.dump(), indent=2, width=120), logger=logger)
    logger.info("=== Output Directory ===")
    logger.info(config.output.job_dirpath)
    # endregion
    logger.info("=== Start Evaluating ===")
    logger.info(f"* Building model {config.model.name} from {config.model.path}")
    tools.logging.Formatter.indent_inc()
    model, tokenizer = config.model.build()
    tools.logging.Formatter.indent_dec()
    save_dirpath = os.path.join(config.output.running_job_dirpath, "cache")
    if config.save_model:
        if config.save_model.lower() in ("false", "none", "null", "nil"):
            save_model = False
        elif config.save_model.lower() in ("true", "default"):
            save_dirpath, save_model = os.path.join(config.output.running_job_dirpath, "model"), True
        else:
            save_dirpath, save_model = config.save_model, True
    else:
        save_model = False
    model = ptq(
        model,
        tokenizer=tokenizer,
        config=config.quant,
        cache=config.cache,
        load_dirpath=config.load_from,
        save_dirpath=save_dirpath,
        copy_on_save=config.copy_on_save,
        save_model=save_model,
    )
    # region evaluate model
    if not config.skip_eval:
        logger.info("* Evaluating model")
        tools.logging.Formatter.indent_inc()
        results = config.eval.evaluate(model, tokenizer, model_name=config.model.name)
        tools.logging.Formatter.indent_dec()
        logger.info(f"* Saving results to {config.output.job_dirpath}")
        # dump results
        with open(os.path.join(config.output.get_running_job_path("results.json")), "w") as f:
            json.dump(results, f, indent=2)
        # endregion
    config.output.unlock()


if __name__ == "__main__":
    config, _, unused_cfgs, unused_args, unknown_args = LlmPtqRunConfig.get_parser().parse_known_args()
    if len(unused_cfgs) > 0:
        tools.logging.warning(f"Unused configurations: {unused_cfgs}")
    if unused_args is not None:
        tools.logging.warning(f"Unused arguments: {unused_args}")
    assert len(unknown_args) == 0, f"Unknown arguments: {unknown_args}"
    try:
        main(config, logging_level=tools.logging.DEBUG)
    except Exception as e:
        tools.logging.Formatter.indent_reset()
        tools.logging.error("=== Error ===")
        tools.logging.error(traceback.format_exc())
        tools.logging.shutdown()
        traceback.print_exc()
        config.output.unlock(error=True)
        raise e
