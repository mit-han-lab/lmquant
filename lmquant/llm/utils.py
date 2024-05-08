# -*- coding: utf-8 -*-
"""LLM quantization utils module."""

import typing as tp

import torch.nn as nn

from .quant.config import LlmQuantConfig

__all__ = ["get_needs_inputs_fn", "get_needs_outputs_fn"]


def get_needs_inputs_fn(config: LlmQuantConfig) -> tp.Callable[[str, nn.Module], bool]:
    """Get needs inputs function.

    Args:
        config (LlmModelQuantConfig): Module quantization config.

    Returns:
        tp.Callable[[str, nn.Module], bool]: Needs inputs function.
    """

    def needs_inputs(name: str, module: nn.Module) -> bool:
        if name.endswith("self_attn"):  # self attention block
            return True
        elif name.endswith("block_sparse_moe") or name.endswith("mlp"):  # feed forward block
            return True
        elif name.endswith("q_proj") or name.endswith("k_proj") or name.endswith("v_proj"):
            if config.needs_quant_weights(name, module) or config.needs_quant_inputs(name, module):
                if name.endswith("v_proj"):
                    return True
            return False
        elif name.endswith("gate_proj") or name.endswith("up_proj") or name.endswith("w1") or name.endswith("w3"):
            if config.needs_quant_weights(name, module) or config.needs_quant_inputs(name, module):
                if name.endswith("up_proj") or name.endswith("w3"):
                    return True
            return False
        else:
            return config.needs_quant_inputs(name, module) or config.needs_quant_weights(name, module)

    return needs_inputs


def get_needs_outputs_fn(config: LlmQuantConfig) -> tp.Callable[[str, nn.Module], bool]:
    """Get needs outputs function.

    Args:
        config (LlmModelQuantConfig): Module quantization config.

    Returns:
        tp.Callable[[str, nn.Module], bool]: Needs outputs function.
    """

    def needs_outputs(name: str, module: nn.Module) -> bool:
        for key in config.keywords_o["attn_q"]:
            if name.endswith(key):
                return True
        for key in config.keywords_o["attn_k"]:
            if name.endswith(key):
                return True
        for key in config.keywords_o["attn_v"]:
            if name.endswith(key):
                return True

    return needs_outputs
