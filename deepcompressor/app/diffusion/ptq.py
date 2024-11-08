import gc
import json
import os
import pprint
import traceback

import torch
from diffusers import DiffusionPipeline

from deepcompressor.app.llm.ptq import ptq as llm_ptq
from deepcompressor.utils import tools

from .config import DiffusionPtqCacheConfig, DiffusionPtqRunConfig, DiffusionQuantCacheConfig, DiffusionQuantConfig
from .nn.struct import DiffusionModelStruct
from .quant import (
    load_diffusion_weights_state_dict,
    quantize_diffusion_activations,
    quantize_diffusion_weights,
    smooth_diffusion,
)

__all__ = ["ptq"]


def ptq(  # noqa: C901
    model: DiffusionModelStruct,
    config: DiffusionQuantConfig,
    cache: DiffusionPtqCacheConfig | None = None,
    load_dirpath: str = "",
    save_dirpath: str = "",
    copy_on_save: bool = False,
    save_model: bool = False,
) -> DiffusionModelStruct:
    """Post-training quantization of a diffusion model.

    Args:
        model (`DiffusionModelStruct`):
            The diffusion model.
        config (`DiffusionQuantConfig`):
            The diffusion model post-training quantization configuration.
        cache (`DiffusionPtqCacheConfig`, *optional*, defaults to `None`):
            The diffusion model quantization cache path configuration.
        load_dirpath (`str`, *optional*, defaults to `""`):
            The directory path to load the quantization checkpoint.
        save_dirpath (`str`, *optional*, defaults to `""`):
            The directory path to save the quantization checkpoint.
        copy_on_save (`bool`, *optional*, defaults to `False`):
            Whether to copy the cache to the save directory.
        save_model (`bool`, *optional*, defaults to `False`):
            Whether to save the quantized model checkpoint.

    Returns:
        `DiffusionModelStruct`:
            The quantized diffusion model.
    """
    logger = tools.logging.getLogger(__name__)
    if not isinstance(model, DiffusionModelStruct):
        model = DiffusionModelStruct.construct(model)
    assert isinstance(model, DiffusionModelStruct)

    quant_wgts = config.enabled_wgts
    quant_ipts = config.enabled_ipts
    quant_opts = config.enabled_opts
    quant_acts = quant_ipts or quant_opts
    quant = quant_wgts or quant_acts

    load_model_path, load_path, save_path = "", None, None
    if load_dirpath:
        load_path = DiffusionQuantCacheConfig(
            smooth=os.path.join(load_dirpath, "smooth.pt"),
            branch=os.path.join(load_dirpath, "branch.pt"),
            wgts=os.path.join(load_dirpath, "wgts.pt"),
            acts=os.path.join(load_dirpath, "acts.pt"),
        )
        load_model_path = os.path.join(load_dirpath, "model.pt")
        if os.path.exists(load_model_path):
            if config.enabled_wgts and config.wgts.enabled_low_rank:
                if os.path.exists(load_path.branch):
                    load_model = True
                else:
                    logger.warning(f"Model low-rank branch checkpoint {load_path.branch} does not exist")
                    load_model = False
            else:
                load_model = True
            if load_model:
                logger.info(f"* Loading model from {load_model_path}")
                save_dirpath = ""  # do not save the model if loading
        else:
            logger.warning(f"Model checkpoint {load_model_path} does not exist")
            load_model = False
    else:
        load_model = False
    if save_dirpath:
        os.makedirs(save_dirpath, exist_ok=True)
        save_path = DiffusionQuantCacheConfig(
            smooth=os.path.join(save_dirpath, "smooth.pt"),
            branch=os.path.join(save_dirpath, "branch.pt"),
            wgts=os.path.join(save_dirpath, "wgts.pt"),
            acts=os.path.join(save_dirpath, "acts.pt"),
        )
    else:
        save_model = False

    # region smooth quantization
    if quant and config.enabled_smooth:
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
            smooth_diffusion(model, config, smooth_cache=smooth_cache)
        else:
            logger.info("- Generating smooth scales")
            smooth_cache = smooth_diffusion(model, config)
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
        load_diffusion_weights_state_dict(
            model,
            config,
            state_dict=torch.load(load_model_path),
            branch_state_dict=torch.load(load_path.branch) if os.path.exists(load_path.branch) else None,
        )
        gc.collect()
        torch.cuda.empty_cache()
    elif quant_wgts:
        logger.info("* Quantizing weights")
        tools.logging.Formatter.indent_inc()
        quantizer_state_dict, quantizer_load_from = None, ""
        if load_path and os.path.exists(load_path.wgts):
            quantizer_load_from = load_path.wgts
        elif cache and cache.path.wgts and os.path.exists(cache.path.wgts):
            quantizer_load_from = cache.path.wgts
        if quantizer_load_from:
            logger.info(f"- Loading weight settings from {quantizer_load_from}")
            quantizer_state_dict = torch.load(quantizer_load_from)
        branch_state_dict, branch_load_from = None, ""
        if load_path and os.path.exists(load_path.branch):
            branch_load_from = load_path.branch
        elif cache and cache.path.branch and os.path.exists(cache.path.branch):
            branch_load_from = cache.path.branch
        if branch_load_from:
            logger.info(f"- Loading branch settings from {branch_load_from}")
            branch_state_dict = torch.load(branch_load_from)
        if not quantizer_load_from:
            logger.info("- Generating weight settings")
        if not branch_load_from:
            logger.info("- Generating branch settings")
        quantizer_state_dict, branch_state_dict, scale_state_dict = quantize_diffusion_weights(
            model,
            config,
            quantizer_state_dict=quantizer_state_dict,
            branch_state_dict=branch_state_dict,
            return_with_scale_state_dict=bool(save_dirpath),
        )
        if not quantizer_load_from and cache and cache.dirpath.wgts:
            logger.info(f"- Saving weight settings to {cache.path.wgts}")
            os.makedirs(cache.dirpath.wgts, exist_ok=True)
            torch.save(quantizer_state_dict, cache.path.wgts)
            quantizer_load_from = cache.path.wgts
        if not branch_load_from and cache and cache.dirpath.branch:
            logger.info(f"- Saving branch settings to {cache.path.branch}")
            os.makedirs(cache.dirpath.branch, exist_ok=True)
            torch.save(branch_state_dict, cache.path.branch)
            branch_load_from = cache.path.branch
        if save_path:
            if not copy_on_save and quantizer_load_from:
                logger.info(f"- Linking weight settings to {save_path.wgts}")
                os.symlink(os.path.relpath(quantizer_load_from, save_dirpath), save_path.wgts)
            else:
                logger.info(f"- Saving weight settings to {save_path.wgts}")
                torch.save(quantizer_state_dict, save_path.wgts)
            if not copy_on_save and branch_load_from:
                logger.info(f"- Linking branch settings to {save_path.branch}")
                os.symlink(os.path.relpath(branch_load_from, save_dirpath), save_path.branch)
            else:
                logger.info(f"- Saving branch settings to {save_path.branch}")
                torch.save(branch_state_dict, save_path.branch)
        if save_model:
            logger.info(f"- Saving model to {save_dirpath}")
            torch.save(scale_state_dict, os.path.join(save_dirpath, "scale.pt"))
            torch.save(model.module.state_dict(), os.path.join(save_dirpath, "model.pt"))
        del quantizer_state_dict, branch_state_dict, scale_state_dict
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
                logger.info(f"- Loading activation settings from {load_from}")
                quantizer_state_dict = torch.load(load_from)
                quantize_diffusion_activations(
                    model, config, quantizer_state_dict=quantizer_state_dict, orig_state_dict=orig_state_dict
                )
            else:
                logger.info("- Generating activation settings")
                quantizer_state_dict = quantize_diffusion_activations(model, config, orig_state_dict=orig_state_dict)
                if cache and cache.dirpath.acts and quantizer_state_dict is not None:
                    logger.info(f"- Saving activation settings to {cache.path.acts}")
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
            quantize_diffusion_activations(model, config, orig_state_dict=orig_state_dict)
        tools.logging.Formatter.indent_dec()
        del orig_state_dict
        gc.collect()
        torch.cuda.empty_cache()
    return model


def main(config: DiffusionPtqRunConfig, logging_level: int = tools.logging.DEBUG) -> DiffusionPipeline:
    """Post-training quantization of a diffusion model.

    Args:
        config (`DiffusionPtqRunConfig`):
            The diffusion model post-training quantization configuration.
        logging_level (`int`, *optional*, defaults to `logging.DEBUG`):
            The logging level.

    Returns:
        `DiffusionPipeline`:
            The diffusion pipeline with quantized model.
    """
    config.output.lock()
    config.dump(path=config.output.get_running_job_path("config.yaml"))
    tools.logging.setup(path=config.output.get_running_job_path("run.log"), level=logging_level)
    logger = tools.logging.getLogger(__name__)

    logger.info("=== Configurations ===")
    tools.logging.info(config.formatted_str(), logger=logger)
    logger.info("=== Dumped Configurations ===")
    tools.logging.info(pprint.pformat(config.dump(), indent=2, width=120), logger=logger)
    logger.info("=== Output Directory ===")
    logger.info(config.output.job_dirpath)

    logger.info("=== Start Evaluating ===")
    logger.info("* Building diffusion model pipeline")
    tools.logging.Formatter.indent_inc()
    pipeline = config.pipeline.build()
    model = DiffusionModelStruct.construct(pipeline)
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
        config.quant,
        cache=config.cache,
        load_dirpath=config.load_from,
        save_dirpath=save_dirpath,
        copy_on_save=config.copy_on_save,
        save_model=save_model,
    )
    if config.pipeline.lora is not None:
        load_from = ""
        if config.quant.enabled_smooth:
            if config.load_from and os.path.exists(os.path.join(config.load_from, "smooth.pt")):
                load_from = os.path.join(config.load_from, "smooth.pt")
            elif config.cache.path and os.path.exists(config.cache.path.smooth):
                load_from = config.cache.path.smooth
            elif os.path.exists(os.path.join(save_dirpath, "smooth.pt")):
                load_from = os.path.join(save_dirpath, "smooth.pt")
            logger.info(f"* Loading smooth scales from {load_from}")
        config.pipeline.load_lora(pipeline, smooth_cache=torch.load(load_from) if load_from else None)
    if config.text is not None and config.text.is_enabled():
        for encoder_name, encoder, tokenizer in config.pipeline.extract_text_encoders(pipeline):
            logger.info(f"* Post-training quantizing the text encoder {encoder_name}")
            save_dirpath = os.path.join(save_dirpath, "encoder")
            setattr(
                pipeline,
                encoder_name,
                llm_ptq(
                    encoder,
                    tokenizer,
                    config.text,
                    cache=config.text_cache,
                    load_dirpath=os.path.join(config.load_from, "encoder") if config.load_from else "",
                    save_dirpath=save_dirpath,
                    copy_on_save=config.copy_on_save,
                    save_model=save_model,
                ),
            )
    config.eval.gen_root = config.eval.gen_root.format(
        output=config.output.running_dirpath, job=config.output.running_job_dirname
    )
    if config.skip_eval:
        if not config.skip_gen:
            logger.info("* Generating image")
            tools.logging.Formatter.indent_inc()
            config.eval.generate(pipeline)
            tools.logging.Formatter.indent_dec()
    else:
        logger.info(f"* Evaluating model {'(skipping generation)' if config.skip_gen else ''}")
        tools.logging.Formatter.indent_inc()
        results = config.eval.evaluate(pipeline, skip_gen=config.skip_gen)
        tools.logging.Formatter.indent_dec()
        if results is not None:
            logger.info(f"* Saving results to {config.output.job_dirpath}")
            with open(config.output.get_running_job_path("results.json"), "w") as f:
                json.dump(results, f, indent=2, sort_keys=True)
    config.output.unlock()


if __name__ == "__main__":
    config, _, unused_cfgs, unused_args, unknown_args = DiffusionPtqRunConfig.get_parser().parse_known_args()
    assert isinstance(config, DiffusionPtqRunConfig)
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
