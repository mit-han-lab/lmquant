# -*- coding: utf-8 -*-
"""Bias correction for LLM quantization."""

import torch
import torch.nn as nn

from lmquant.dataset.cache import IOActivationsCache

from ..nn.struct import LlmModelStruct
from .config import LlmQuantConfig

__all__ = ["correct_llm_bias", "quantize_llm_bias"]


def correct_llm_bias(
    model: nn.Module | LlmModelStruct,
    config: LlmQuantConfig,
    activations: dict[str, IOActivationsCache],
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Bias correction for LLM quantization."""
    raise NotImplementedError("Not implemented yet")
