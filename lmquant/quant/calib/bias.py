# -*- coding: utf-8 -*-
"""Bias correction for quantization."""
import gc

import torch
import torch.nn as nn

from ...dataset.cache import ActivationsCache

__all__ = ["correct_bias"]


def correct_bias(
    module: nn.Module, inputs: ActivationsCache, orig_weights: torch.Tensor, weight: nn.Parameter | None = None
) -> None:
    """Bias correction for quantization.

    Args:
        module (nn.Module): Module.
        inputs (ActivationsCache): Input activations cache.
        orig_weights (torch.Tensor): Original weights.
        weight (nn.Parameter | None, optional): Weight. Defaults to ``None``.
    """
    assert inputs.num_sources == 1, f"Only one input source is supported, got {inputs.num_sources}"
    if weight is None:
        weight = module.weight
    w = weight.data
    dw = orig_weights.to(w.device, dtype=torch.float32) - w.float()
    x, fn = inputs[0].cached, inputs[0].transform
    x = x[0].to(w.device, dtype=torch.float32)
    x = x if fn is None else fn(x, True)
    dw = dw.view(w.shape[0], -1)
    x = x.mean(dim=0).view(dw.shape[1], 1)
    db = (dw @ x).to(w.dtype)
    if not hasattr(module, "bias") or module.bias is None:
        module.bias = nn.Parameter(db.view(-1))
    else:
        module.bias.data += db.view(-1)
    gc.collect()
    torch.cuda.empty_cache()
