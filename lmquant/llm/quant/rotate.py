# -*- coding: utf-8 -*-
"""Large Language Model Rotation module."""

import gc
import logging

import torch
import torch.nn as nn
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from lmquant.quant.calib.config import QuantRotationConfig
from lmquant.quant.calib.rotate import (
    get_rotation_matrix,
    hadamard_in_channels,
    rotate_in_channels,
    rotate_out_channels,
    transform_norm_and_linear,
)
from lmquant.utils import tools

from ..nn import LlmModelStruct

__all__ = ["rotate_llm"]


@torch.inference_mode()
def rotate_llm(  # noqa: C901
    model: nn.Module | LlmModelStruct,
    /,
    config: QuantRotationConfig,
    rotation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rotate the weights the large foundation model.

    Args:
        model (nn.Module | LlmStruct): Model to be rotated.
        config (QuantRotationConfig): Rotation configuration.
        rotation (torch.Tensor, optional): Rotation matrix. Defaults to ``None``.
    """
    if not isinstance(model, LlmModelStruct):
        model = LlmModelStruct.build(model)
    assert isinstance(model, LlmModelStruct)
    assert model.config.do_norm_before, "Rotation is only supported for models with norm before matmul."
    logger = logging.getLogger(f"{__name__}.Rotate")
    backbone = model.backbone_struct
    layers = backbone.layer_structs
    # region transform norm and linear
    if backbone.first_ln is None:
        if backbone.proj_in is None:
            prev_modules = backbone.embeddings
            prev_out_channels_dims = 1
        elif backbone.embed_positions is None:
            prev_modules = [backbone.proj_in]
            prev_out_channels_dims = 0
        else:
            prev_modules = [backbone.proj_in, backbone.embed_positions]
            prev_out_channels_dims = [0, 1]
    else:
        prev_modules = [backbone.first_ln]
        prev_out_channels_dims = 0
    with logging_redirect_tqdm():
        for layer in tqdm(layers, desc="Transforming norm and linear", total=model.config.num_hidden_layers):
            logger.debug(f"- Transforming norm and linear in {layer.full_name}")
            transform_norm_and_linear(
                parent=layer.module,
                norm_name=layer.attn_ln_name,
                next_modules=layer.proj_qkv,
                prev_modules=prev_modules,
                prev_out_channels_dims=prev_out_channels_dims,
            )
            prev_modules = [layer.proj_o]
            prev_out_channels_dims = 0
            transform_norm_and_linear(
                parent=layer.module,
                norm_name=layer.ffn_ln_name,
                next_modules=layer.proj_1st + ([layer.router] if layer.router is not None else []),
                prev_modules=prev_modules,
                prev_out_channels_dims=prev_out_channels_dims,
            )
            prev_modules = layer.proj_2nd
            prev_out_channels_dims = 0
            gc.collect()
            torch.cuda.empty_cache()
    logger.debug(f"- Transforming {backbone.final_ln_full_name}")
    transform_norm_and_linear(
        parent=backbone.module,
        norm_name=backbone.final_ln_name,
        next_modules=[model.fc if backbone.proj_out is None else backbone.proj_out],
        prev_modules=prev_modules,
        prev_out_channels_dims=prev_out_channels_dims,
    )
    # endregion
    if rotation is None:
        rotation = get_rotation_matrix(backbone.config.num_hidden_channels, random=config.random)
    # region rotate embeddings
    if backbone.proj_in is None:
        logger.debug(f"- Rotating {backbone.embed_tokens_full_name}")
        weight = backbone.embed_tokens.weight
        rotation = rotation.to(weight.device)
        rotate_in_channels(weight, rotation=rotation)
    else:
        logger.debug(f"- Rotating {backbone.proj_in_full_name} (out)")
        weight = backbone.proj_in.weight
        rotation = rotation.to(weight.device)
        rotate_out_channels(weight, rotation=rotation, bias=backbone.proj_in.bias)
    if backbone.embed_positions is not None:
        logger.debug(f"- Rotating {backbone.embed_positions_full_name}")
        weight = backbone.embed_positions.weight
        rotation = rotation.to(weight.device)
        rotate_in_channels(weight, rotation=rotation)
    # endregion
    proj_out, proj_2nd = [], []
    # region rotate decoder layers
    with logging_redirect_tqdm():
        for layer in tqdm(layers, desc="Rotating decoder layers", total=model.config.num_hidden_layers):
            logger.debug(f"- Rotating {layer.full_name}")
            tools.logging.Formatter.indent_inc()
            for proj_name, proj in zip(layer.proj_qkv_full_names, layer.proj_qkv):
                logger.debug(f"- Rotating {proj_name} (in)")
                rotation = rotation.to(proj.weight.device)
                rotate_in_channels(proj.weight, rotation=rotation)
            logger.debug(f"- Rotating {layer.proj_o_full_name} (out)")
            rotation = rotation.to(layer.proj_o.weight.device)
            rotate_out_channels(layer.proj_o.weight, rotation=rotation, bias=layer.proj_o.bias)
            proj_out.append(layer.proj_o)
            logger.debug(f"- Rotating {layer.full_name}.proj_1st (in)")
            for fc in layer.proj_1st:
                rotation = rotation.to(fc.weight.device)
                rotate_in_channels(fc.weight, rotation=rotation)
            if layer.router is not None:
                logger.debug(f"- Rotating {layer.full_name}.router (in)")
                rotation = rotation.to(layer.router.weight.device)
                rotate_in_channels(layer.router.weight, rotation=rotation)
            logger.debug(f"- Rotating {layer.full_name}.proj_2nd (out)")
            for fc in layer.proj_2nd:
                rotation = rotation.to(fc.weight.device)
                rotate_out_channels(fc.weight, rotation=rotation, bias=fc.bias)
            proj_2nd.extend(layer.proj_2nd)
            tools.logging.Formatter.indent_dec()
            gc.collect()
            torch.cuda.empty_cache()
    if backbone.proj_out is not None:
        logger.debug(f"- Rotating {backbone.proj_out_full_name} (in)")
        weight = backbone.proj_out.weight
        rotation = rotation.to(weight.device)
        rotate_in_channels(weight, rotation=rotation)
        logger.debug(f"- Rotating {backbone.proj_out_full_name} (out)")
        rotation = rotation.to(weight.device)
        rotate_out_channels(weight, rotation=rotation, bias=backbone.proj_out.bias)
    # endregion
    if "proj_out" in config.transforms:
        logger.debug(f"- Applying Hadamard transform on {backbone.layers_full_name}.proj_out (in)")
        hadamard_in_channels(proj_out)
    if "proj_2nd" in config.transforms:
        logger.debug(f"- Applying Hadamard transform on {backbone.layers_full_name}.proj_2nd (in)")
        hadamard_in_channels(proj_2nd)
    logger.debug(f"- Rotating {model.fc_full_name} (in)")
    weight = model.fc.weight
    rotation = rotation.to(weight.device)
    rotate_in_channels(weight, rotation=rotation)
    return rotation.cpu()
