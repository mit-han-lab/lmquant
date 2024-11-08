# -*- coding: utf-8 -*-
"""Diffusion model weight quantization calibration module."""

import gc
import typing as tp

import torch
import torch.nn as nn
from tqdm import tqdm

from deepcompressor.data.cache import IOTensorsCache
from deepcompressor.data.zero import ZeroPointDomain
from deepcompressor.nn.patch.lowrank import LowRankBranch
from deepcompressor.utils import tools

from ..nn.struct import DiffusionAttentionStruct, DiffusionBlockStruct, DiffusionModelStruct, DiffusionModuleStruct
from .config import DiffusionQuantConfig
from .quantizer import DiffusionActivationQuantizer, DiffusionWeightQuantizer
from .utils import get_needs_inputs_fn, wrap_joint_attn

__all__ = ["quantize_diffusion_weights", "load_diffusion_weights_state_dict"]


@torch.inference_mode()
def calibrate_diffusion_block_low_rank_branch(  # noqa: C901
    layer: DiffusionModuleStruct | DiffusionBlockStruct,
    config: DiffusionQuantConfig,
    branch_state_dict: dict[str, dict[str, torch.Tensor]],
    layer_cache: dict[str, IOTensorsCache] = None,
    layer_kwargs: dict[str, tp.Any] = None,
) -> None:
    """Calibrate low-rank branches for a block of a diffusion model.

    Args:
        layer (`DiffusionModuleStruct` or `DiffusionBlockStruct`):
            The block to calibrate.
        config (`DiffusionQuantConfig`):
            The quantization configuration.
        branch_state_dict (`dict[str, dict[str, torch.Tensor]]`):
            The state dict of the low-rank branches.
        layer_cache (`dict[str, IOTensorsCache]`, *optional*, defaults to `None`):
            The cache of the layer.
        layer_kwargs (`dict[str, tp.Any]`, *optional*, defaults to `None`):
            The keyword arguments for the layer.
    """
    assert config.wgts.low_rank is not None
    logger = tools.logging.getLogger(f"{__name__}.WeightQuantSVD")
    logger.debug("- Calibrating low-rank branches of block %s", layer.name)
    layer_cache = layer_cache or {}
    layer_kwargs = layer_kwargs or {}
    for module_key, module_name, module, parent, field_name in layer.named_key_modules():
        modules, module_names = [module], [module_name]
        if not config.wgts.low_rank.exclusive:
            if field_name.endswith(("q_proj", "k_proj", "v_proj")):
                assert isinstance(parent, DiffusionAttentionStruct)
                if parent.is_self_attn():
                    if field_name == "q_proj":
                        modules, module_names = parent.qkv_proj, parent.qkv_proj_names
                    else:
                        continue
                elif parent.is_cross_attn():
                    if field_name == "add_k_proj":
                        modules.append(parent.add_v_proj)
                        module_names.append(parent.add_v_proj_name)
                    elif field_name != "q_proj":
                        continue
                else:
                    assert parent.is_joint_attn()
                    if field_name == "q_proj":
                        modules, module_names = parent.qkv_proj, parent.qkv_proj_names
                    elif field_name == "add_k_proj":
                        modules, module_names = parent.add_qkv_proj, parent.add_qkv_proj_names
                    else:
                        continue
        if field_name.endswith(("q_proj", "k_proj")):
            assert isinstance(parent, DiffusionAttentionStruct)
            if parent.parent.parallel and parent.idx == 0:
                eval_module = parent.parent.module
                eval_name = parent.parent.name
                eval_kwargs = layer_kwargs
            else:
                eval_module = parent.module
                eval_name = parent.name
                eval_kwargs = parent.filter_kwargs(layer_kwargs)
            if parent.is_joint_attn() and "add_" in field_name:
                eval_module = wrap_joint_attn(eval_module, indexes=1)
        else:
            eval_module, eval_name, eval_kwargs = module, module_name, None
        quantizer = DiffusionWeightQuantizer(config.wgts, develop_dtype=config.develop_dtype, key=module_key)
        if quantizer.is_enabled() and quantizer.is_enabled_low_rank():
            assert isinstance(module, nn.Linear), "Only Linear modules are supported for low-rank branch calibration"
            if module_name not in branch_state_dict:
                logger.debug("- Calibrating low-rank branch for %s", ", ".join(module_names))
                tools.logging.Formatter.indent_inc()
                branch_state_dict[module_name] = quantizer.calibrate_low_rank(
                    input_quantizer=DiffusionActivationQuantizer(config.ipts, key=module_key, channels_dim=-1),
                    modules=modules,
                    inputs=layer_cache[module_name].inputs if layer_cache else None,
                    eval_inputs=layer_cache[eval_name].inputs if layer_cache else None,
                    eval_module=eval_module,
                    eval_kwargs=eval_kwargs,
                ).state_dict()
                tools.logging.Formatter.indent_dec()
                gc.collect()
                torch.cuda.empty_cache()
            shared_branch = LowRankBranch(
                in_features=module.weight.shape[1],
                out_features=sum(m.weight.shape[0] for m in modules),
                rank=config.wgts.low_rank.rank,
            )
            shared_branch.to(device=module.weight.device, dtype=module.weight.dtype)
            shared_branch.load_state_dict(branch_state_dict[module_name])
            logger.debug("  + Adding low-rank branches to %s", ", ".join(module_names))
            if len(modules) > 1:
                oc_idx = 0
                for module in modules:
                    branch = LowRankBranch(
                        in_features=module.weight.shape[1],
                        out_features=module.weight.shape[0],
                        rank=config.wgts.low_rank.rank,
                    )
                    branch.a = shared_branch.a
                    branch.b.to(dtype=module.weight.dtype, device=module.weight.device)
                    branch.b.weight.copy_(shared_branch.b.weight[oc_idx : oc_idx + module.weight.shape[0]])
                    oc_idx += module.weight.shape[0]
                    module.weight.data.sub_(branch.get_effective_weight())
                    branch.as_hook().register(module)
            else:
                module.weight.data.sub_(shared_branch.get_effective_weight())
                shared_branch.as_hook().register(module)
            del shared_branch
            gc.collect()
            torch.cuda.empty_cache()


@torch.inference_mode()
def update_diffusion_block_weight_quantizer_state_dict(
    layer: DiffusionModuleStruct | DiffusionBlockStruct,
    config: DiffusionQuantConfig,
    quantizer_state_dict: dict[str, dict[str, torch.Tensor | float | None]],
    layer_cache: dict[str, IOTensorsCache],
    layer_kwargs: dict[str, tp.Any],
):
    """Update the state dict of the weight quantizers for a block of a diffusion model.

    Args:
        layer (`DiffusionModuleStruct` or `DiffusionBlockStruct`):
            The block to update.
        config (`DiffusionQuantConfig`):
            The quantization configuration.
        quantizer_state_dict (`dict[str, dict[str, torch.Tensor | float | None]]`):
            The state dict of the weight quantizers.
        layer_cache (`dict[str, IOTensorsCache]`):
            The cache of the layer.
        layer_kwargs (`dict[str, tp.Any]`):
            The keyword arguments for the layer.
    """
    logger = tools.logging.getLogger(f"{__name__}.WeightQuant")
    logger.debug("- Calibrating weights: block %s", layer.name)
    tools.logging.Formatter.indent_inc()
    for module_key, module_name, module, parent, field_name in layer.named_key_modules():
        if field_name.endswith(("q_proj", "k_proj")):
            assert isinstance(parent, DiffusionAttentionStruct)
            if parent.parent.parallel and parent.idx == 0:
                eval_module = parent.parent.module
                eval_name = parent.parent.name
                eval_kwargs = layer_kwargs
            else:
                eval_module = parent.module
                eval_name = parent.name
                eval_kwargs = parent.filter_kwargs(layer_kwargs)
            if parent.is_joint_attn() and "add_" in field_name:
                eval_module = wrap_joint_attn(eval_module, indexes=1)
        else:
            eval_module, eval_name, eval_kwargs = module, module_name, None
        quantizer = DiffusionWeightQuantizer(config.wgts, develop_dtype=config.develop_dtype, key=module_key)
        if quantizer.is_enabled():
            if module_name not in quantizer_state_dict:
                logger.debug("- Calibrating %s.weight quantizer", module_name)
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
            else:
                logger.debug("- Loading %s.weight quantizer", module_name)
        else:
            logger.debug("- Skipping %s.weight", module_name)
            if module_name in quantizer_state_dict:
                quantizer_state_dict.pop(module_name)
    tools.logging.Formatter.indent_dec()


@torch.inference_mode()
def quantize_diffusion_block_weights(
    layer: DiffusionModuleStruct | DiffusionBlockStruct,
    config: DiffusionQuantConfig,
    quantizer_state_dict: dict[str, dict[str, torch.Tensor | float | None]],
    layer_cache: dict[str, IOTensorsCache] = None,
    return_with_scale_state_dict: bool = False,
) -> dict[str, torch.Tensor | float | None]:
    """Quantize the weights of a block of a diffusion model.

    Args:
        layer (`DiffusionModuleStruct` or `DiffusionBlockStruct`):
            The block to quantize.
        config (`DiffusionQuantConfig`):
            The quantization configuration.
        quantizer_state_dict (`dict[str, dict[str, torch.Tensor | float | None]]`):
            The state dict of the weight quantizers.
        layer_cache (`dict[str, IOTensorsCache]`, *optional*, defaults to `None`):
            The cache of the layer.
        return_with_scale_state_dict (`bool`, *optional*, defaults to `False`):
            Whether to return the scale state dict.

    Returns:
        `dict[str, torch.Tensor | float | None]`:
            The scale state dict.
    """
    logger = tools.logging.getLogger(f"{__name__}.WeightQuant")
    logger.debug("- Quantizing weights: block %s", layer.name)
    layer_cache = layer_cache or {}

    scale_state_dict: dict[str, torch.Tensor | float | None] = {}

    tools.logging.Formatter.indent_inc()
    for module_key, module_name, module, _, _ in layer.named_key_modules():
        if module_name in quantizer_state_dict:
            param_name = f"{module_name}.weight"
            logger.debug("- Quantizing %s", param_name)
            quantizer = DiffusionWeightQuantizer(config.wgts, develop_dtype=config.develop_dtype, key=module_key)
            quantizer.load_state_dict(quantizer_state_dict[module_name], device=module.weight.device)
            result = quantizer.quantize(
                module.weight.data,
                inputs=layer_cache[module_name].inputs.front() if layer_cache else None,
                return_with_dequant=True,
                return_with_quant=return_with_scale_state_dict,
            )
            if (
                config.wgts.enabled_low_rank
                and config.wgts.low_rank.is_enabled_for(module_key)
                and config.wgts.low_rank.compensate
                and config.wgts.low_rank.num_iters <= 1
            ):
                logger.debug("- Adding compensate low-rank branch to %s (side)", module_name)
                LowRankBranch(
                    in_features=module.weight.shape[1],
                    out_features=module.weight.shape[0],
                    rank=config.wgts.low_rank.rank,
                    weight=module.weight.data - result.data,
                ).as_hook().register(module)
            module.weight.data = result.data
            if return_with_scale_state_dict:
                scale_state_dict.update(result.scale.state_dict(f"{param_name}.scale"))
                zero_name = "scaled_zero" if config.wgts.zero_point is ZeroPointDomain.PostScale else "zero"
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
def quantize_diffusion_weights(
    model: nn.Module | DiffusionModelStruct,
    config: DiffusionQuantConfig,
    quantizer_state_dict: dict[str, dict[str, torch.Tensor | float | None]] | None = None,
    branch_state_dict: dict[str, dict[str, torch.Tensor]] | None = None,
    return_with_scale_state_dict: bool = False,
) -> tuple[
    dict[str, dict[str, torch.Tensor | float | None]],
    dict[str, dict[str, torch.Tensor]],
    dict[str, torch.Tensor | float | None],
]:
    """Quantize the weights of a diffusion model.

    Args:
        model (`nn.Module` or `DiffusionModelStruct`):
            The diffusion model to quantize.
        config (`DiffusionQuantConfig`):
            The quantization configuration.
        quantizer_state_dict (`dict[str, dict[str, torch.Tensor | float | None]]`, *optional*, defaults to `None`):
            The state dict of the weight quantizers.
        branch_state_dict (`dict[str, dict[str, torch.Tensor]]`, *optional*, defaults to `None`):
            The state dict of the low-rank branches.
        return_with_scale_state_dict (`bool`, *optional*, defaults to `False`):
            Whether to return the scale state dict.

    Returns:
        `tuple[
            dict[str, dict[str, torch.Tensor | float | None]],
            dict[str, dict[str, torch.Tensor]],
            dict[str, torch.Tensor | float | None]
        ]`:
            The state dict of the weight quantizers, the state dict of the low-rank branches, and the scale state dict.
    """
    logger = tools.logging.getLogger(f"{__name__}.WeightQuant")
    if not isinstance(model, DiffusionModelStruct):
        model = DiffusionModelStruct.construct(model)
    assert isinstance(model, DiffusionModelStruct)
    quantizer_state_dict = quantizer_state_dict or {}
    branch_state_dict = branch_state_dict or {}

    if config.wgts.enabled_low_rank and (not config.wgts.low_rank.compensate or config.wgts.low_rank.num_iters > 1):
        logger.info("* Adding low-rank branches to weights")
        tools.logging.Formatter.indent_inc()
        with tools.logging.redirect_tqdm():
            if branch_state_dict:
                for _, layer in tqdm(
                    model.get_named_layers(skip_pre_modules=True, skip_post_modules=True).items(),
                    desc="adding low-rank branches",
                    leave=False,
                    dynamic_ncols=True,
                ):
                    calibrate_diffusion_block_low_rank_branch(
                        layer=layer, config=config, branch_state_dict=branch_state_dict
                    )
            else:
                for _, (layer, layer_cache, layer_kwargs) in tqdm(
                    config.calib.build_loader().iter_layer_activations(
                        model,
                        needs_inputs_fn=get_needs_inputs_fn(model, config),
                        skip_pre_modules=True,
                        skip_post_modules=True,
                    ),
                    desc="calibrating low-rank branches",
                    leave=False,
                    total=model.num_blocks,
                    dynamic_ncols=True,
                ):
                    calibrate_diffusion_block_low_rank_branch(
                        layer=layer,
                        config=config,
                        branch_state_dict=branch_state_dict,
                        layer_cache=layer_cache,
                        layer_kwargs=layer_kwargs,
                    )
        tools.logging.Formatter.indent_dec()

    skip_pre_modules = all(key in config.wgts.skips for key in model.get_prev_module_keys())
    skip_post_modules = all(key in config.wgts.skips for key in model.get_post_module_keys())
    if not quantizer_state_dict:
        with tools.logging.redirect_tqdm():
            if config.wgts.needs_calib_data:
                for _, (layer, layer_cache, layer_kwargs) in tqdm(
                    config.calib.build_loader().iter_layer_activations(
                        model,
                        needs_inputs_fn=get_needs_inputs_fn(model, config),
                        skip_pre_modules=skip_pre_modules,
                        skip_post_modules=skip_post_modules,
                    ),
                    desc="calibrating weight quantizers",
                    leave=False,
                    total=model.num_blocks + int(not skip_post_modules) + int(not skip_pre_modules) * 3,
                    dynamic_ncols=True,
                ):
                    update_diffusion_block_weight_quantizer_state_dict(
                        layer=layer,
                        config=config,
                        quantizer_state_dict=quantizer_state_dict,
                        layer_cache=layer_cache,
                        layer_kwargs=layer_kwargs,
                    )
            else:
                for _, layer in tqdm(
                    model.get_named_layers(
                        skip_pre_modules=skip_pre_modules, skip_post_modules=skip_post_modules
                    ).items(),
                    desc="calibrating weight quantizers",
                    leave=False,
                    total=model.num_blocks + int(not skip_post_modules) + int(not skip_pre_modules) * 3,
                    dynamic_ncols=True,
                ):
                    update_diffusion_block_weight_quantizer_state_dict(
                        layer=layer,
                        config=config,
                        quantizer_state_dict=quantizer_state_dict,
                        layer_cache={},
                        layer_kwargs={},
                    )
    scale_state_dict: dict[str, torch.Tensor | float | None] = {}
    for _, layer in tqdm(
        model.get_named_layers(skip_pre_modules=skip_pre_modules, skip_post_modules=skip_post_modules).items(),
        desc="quantizing weights",
        leave=False,
        dynamic_ncols=True,
    ):
        layer_scale_state_dict = quantize_diffusion_block_weights(
            layer=layer,
            config=config,
            quantizer_state_dict=quantizer_state_dict,
            return_with_scale_state_dict=return_with_scale_state_dict,
        )
        scale_state_dict.update(layer_scale_state_dict)
    return quantizer_state_dict, branch_state_dict, scale_state_dict


@torch.inference_mode()
def load_diffusion_weights_state_dict(
    model: nn.Module | DiffusionModelStruct,
    config: DiffusionQuantConfig,
    state_dict: dict[str, torch.Tensor],
    branch_state_dict: dict[str, dict[str, torch.Tensor]] | None = None,
) -> None:
    """Load the state dict of the weights of a diffusion model.

    Args:
        model (`nn.Module` or `DiffusionModelStruct`):
            The diffusion model to load the weights.
        config (`DiffusionQuantConfig`):
            The quantization configuration.
        state_dict (`dict[str, torch.Tensor]`):
            The state dict of the weights.
        branch_state_dict (`dict[str, dict[str, torch.Tensor]]`):
            The state dict of the low-rank branches.
    """
    if not isinstance(model, DiffusionModelStruct):
        model = DiffusionModelStruct.construct(model)
    assert isinstance(model, DiffusionModelStruct)
    if config.enabled_wgts and config.wgts.enabled_low_rank:
        assert branch_state_dict is not None
        for _, layer in tqdm(
            model.get_named_layers(skip_pre_modules=True, skip_post_modules=True).items(),
            desc="adding low-rank branches",
            leave=False,
            dynamic_ncols=True,
        ):
            calibrate_diffusion_block_low_rank_branch(layer=layer, config=config, branch_state_dict=branch_state_dict)
    model.module.load_state_dict(state_dict)
    gc.collect()
    torch.cuda.empty_cache()
