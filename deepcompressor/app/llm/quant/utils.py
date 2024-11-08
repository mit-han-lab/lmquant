# -*- coding: utf-8 -*-
"""LLM quantization utils module."""

import typing as tp

import torch.nn as nn

from ..nn.struct import LlmModelStruct
from .quantizer.config import LlmModuleQuantizerConfig

__all__ = ["get_needs_inputs_fn", "get_needs_outputs_fn"]


def get_needs_inputs_fn(model: LlmModelStruct, config: LlmModuleQuantizerConfig) -> tp.Callable[[str, nn.Module], bool]:
    """Get function that checks if the module needs to cache the inputs.

    Args:
        model (`LlmStruct`):
            Model struct.
        config (`LlmModuleQuantizerConfig`):
            Module quantization config.

    Returns:
        `Callable[[str, nn.Module], bool]`:
            Function to check if the module needs to cache the inputs.
    """

    needs_inputs_names = set()

    example_layer = model.backbone_struct.layer_structs[0]
    attn, ffn = example_layer.attn_struct, example_layer.ffn_struct
    if (config.enabled_wgts and config.wgts.is_enabled_for(attn.qkv_proj_key)) or (
        config.enabled_ipts and config.ipts.is_enabled_for(attn.qkv_proj_key)
    ):
        needs_inputs_names.add(attn.rname)
        needs_inputs_names.add(attn.v_proj_rname)
    if (config.enabled_wgts and config.wgts.is_enabled_for(attn.out_proj_key)) or (
        config.enabled_ipts and config.ipts.is_enabled_for(attn.out_proj_key)
    ):
        needs_inputs_names.add(attn.o_proj_rname)
    if (config.enabled_wgts and config.wgts.is_enabled_for(ffn.up_proj_key)) or (
        config.enabled_ipts and config.ipts.is_enabled_for(ffn.up_proj_key)
    ):
        needs_inputs_names.add(ffn.rname)
        needs_inputs_names.add(ffn.up_proj_rnames[0])
    if (config.enabled_wgts and config.wgts.is_enabled_for(ffn.down_proj_key)) or (
        config.enabled_ipts and config.ipts.is_enabled_for(ffn.down_proj_key)
    ):
        needs_inputs_names.add(ffn.down_proj_rnames[0])
    if config.enabled_opts:
        needs_inputs_names.add(attn.rname)

    needs_inputs_names = tuple(needs_inputs_names)

    def needs_inputs(name: str, module: nn.Module) -> bool:
        return name.endswith(needs_inputs_names)

    return needs_inputs


def get_needs_outputs_fn(
    model: LlmModelStruct, config: LlmModuleQuantizerConfig
) -> tp.Callable[[str, nn.Module], bool]:
    """Get function that checks if the module needs to cache the outputs.

    Args:
        model (`LlmStruct`):
            Model struct.
        config (`LlmModuleQuantizerConfig`):
            Module quantization config.

    Returns:
        `Callable[[str, nn.Module], bool]`:
            Function to check if the module needs to cache the outputs.
    """

    attn = model.backbone_struct.layer_structs[0].attn_struct
    needs_outputs_names = set()
    if config.enabled_opts:
        needs_outputs_names.add(attn.q_rname)
        needs_outputs_names.add(attn.k_rname)
        needs_outputs_names.add(attn.v_rname)
    needs_outputs_names = tuple(needs_outputs_names)

    def needs_outputs(name: str, module: nn.Module) -> bool:
        return name.endswith(needs_outputs_names)

    return needs_outputs
