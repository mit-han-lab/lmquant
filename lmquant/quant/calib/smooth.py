# -*- coding: utf-8 -*-
"""Smooth quantization module."""

import gc
import typing as tp

import torch
import torch.nn as nn

from ...dataset import ActivationsCache
from ..quantizer.base import Quantizer
from .calibrator.smooth import SmoothAttentionCalibrator, SmoothLinearCalibrator
from .config import QuantSmoothCalibConfig, QuantTensorType

__all__ = ["smooth_linear_modules", "smooth_attention"]


def _smooth_modules(
    prev_modules: list[nn.Module],
    modules: list[nn.Module],
    scale: torch.Tensor,
    tensor_type: QuantTensorType,
    num_heads: int = 1,
    num_head_repeats: int = 1,
) -> None:
    view_shape = (1, -1) if tensor_type == QuantTensorType.Weights else (-1, 1)
    curr_scale = scale
    scale_dtype = scale.dtype
    for module in modules:
        param: nn.Parameter = module.weight
        curr_scale = curr_scale.to(device=param.device).view(*view_shape, *([1] * (param.ndim - 2)))
        dtype = param.dtype
        param.data = param.data.to(dtype=scale_dtype).mul_(curr_scale).to(dtype=dtype)
        assert not param.data.isnan().any(), f"NaN found in {module}"
        assert not param.data.isinf().any(), f"Inf found in {module}"
    if num_heads > 1 and num_head_repeats > 1:
        head_channels = scale.numel() // num_heads
        num_unique_heads = num_heads // num_head_repeats
        prev_scale = scale.view(num_unique_heads, num_head_repeats, head_channels)[:, 0, :].reshape(-1)
    else:
        prev_scale = scale
    for module in prev_modules:
        param: nn.Parameter = module.weight
        prev_scale = prev_scale.to(device=param.device).view(-1, *([1] * (param.ndim - 1)))
        dtype = param.dtype
        param.data = param.data.to(dtype=scale_dtype).div_(prev_scale).to(dtype=dtype)
        assert not param.data.isnan().any(), f"NaN found in {module}"
        assert not param.data.isinf().any(), f"Inf found in {module}"
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data = module.bias.data.to(dtype=scale_dtype).div_(prev_scale.view(-1)).to(dtype=dtype)
            assert not module.bias.data.isnan().any(), f"NaN found in {module} bias"
            assert not module.bias.data.isinf().any(), f"Inf found in {module} bias"


@torch.inference_mode()
def smooth_linear_modules(
    prev_modules: list[nn.Module] | nn.Module,
    modules: list[nn.Module] | nn.Module,
    *,
    scale: torch.Tensor | None,
    smooth_config: QuantSmoothCalibConfig | None = None,
    wgts_quantizer: Quantizer | None = None,
    ipts_quantizer: Quantizer | None = None,
    wgts: list[nn.Parameter] | None = None,
    ipts: ActivationsCache | None = None,
    eval_ipt: ActivationsCache | None = None,
    eval_mod: nn.Module = None,
    num_heads: int = 1,
    num_head_repeats: int = 1,
    eval_kwargs: dict[str, tp.Any] = None,
    extra_second_modules: list[nn.Module] | nn.Module = None,
    develop_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Smooth two consecutive modules.

    Args:
        prev_modules (list[nn.Module] | nn.Module): First modules.
        modules (list[nn.Module] | nn.Module): Second modules.
        scale (torch.Tensor, optional): The smooth quantization scale.
        smooth_config (QuantSmoothConfig): The smooth quantization configuration.
        wgts_quantizer (KernelQuantizer, optional): The quantizer for weights. Defaults to ``None``.
        ipts_quantizer (KernelQuantizer, optional): The quantizer for inputs. Defaults to ``None``.
        ipts (ActivationsCache, optional): The cache of the input activations. Defaults to ``None``.
        eval_ipt (ActivationsCache, optional): The cache of the inputs corresponding to the evaluation module.
        eval_mod (nn.Module, optional): The module to evaluate the quantization error. Defaults to ``None``.
        num_heads (int, optional): The number of heads. Defaults to ``1``.
        num_head_repeats (int, optional): The number of head repeats. Defaults to ``1``.
        extra_second_modules (list[nn.Module] | nn.Module, optional): Extra second modules. Defaults to ``None``.
        develop_dtype (torch.dtype, optional): The develop dtype. Defaults to ``torch.float32``.

    Returns:
        torch.Tensor: The smooth quantization scale in CPU.
    """
    if not isinstance(prev_modules, (list, tuple)):
        prev_modules = [prev_modules]
    if not isinstance(modules, (list, tuple)):
        modules = [modules]
    if extra_second_modules is None:
        extra_second_modules = []
    else:
        if not isinstance(extra_second_modules, (list, tuple)):
            extra_second_modules = [extra_second_modules]
    if scale is None:
        assert ipts is not None or eval_ipt is not None, "ftrs or ipts must be provided"
        scale = SmoothLinearCalibrator(
            calib_config=smooth_config,
            wgts_quantizer=wgts_quantizer,
            ipts_quantizer=ipts_quantizer,
            num_heads=num_heads,
            num_head_repeats=num_head_repeats,
            develop_dtype=develop_dtype,
        ).calibrate(
            ipt_wgts=[module.weight for module in modules] if wgts is None else wgts,
            ipts=ipts,
            eval_ipt=eval_ipt,
            eval_mod=eval_mod,
            ipt_mods=modules,
            eval_kwargs=eval_kwargs,
        )
        gc.collect()
        torch.cuda.empty_cache()
    _smooth_modules(
        prev_modules=prev_modules,
        modules=modules + extra_second_modules,
        scale=scale,
        tensor_type=QuantTensorType.Weights,
        num_heads=num_heads,
        num_head_repeats=num_head_repeats,
    )
    return scale.to(device="cpu")


@torch.inference_mode()
def smooth_attention(
    proj_k: nn.Linear,
    proj_q: nn.Linear,
    *,
    scale: torch.Tensor | None,
    smooth_config: QuantSmoothCalibConfig | None = None,
    q_quantizer: Quantizer | None = None,
    k_quantizer: Quantizer | None = None,
    qs: ActivationsCache | None = None,
    ks: ActivationsCache | None = None,
    q_mod: nn.Module | None = None,
    k_mod: nn.Module | None = None,
    eval_ipt: ActivationsCache | None = None,
    eval_mod: nn.Module = None,
    eval_kwargs: dict[str, tp.Any] = None,
    num_heads: int = 1,
    num_head_repeats: int = 1,
    with_rope: bool = True,
    develop_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Smooth attention modules.

    Args:
        proj_k (nn.Linear): The projection module of keys.
        proj_q (nn.Linear): The projection module of queries.
        scale (torch.Tensor, optional): The smooth quantization scale.
        smooth_config (QuantSmoothConfig): The smooth quantization configuration.
        q_quantizer (KernelQuantizer, optional): The quantizer for queries. Defaults to ``None``.
        k_quantizer (KernelQuantizer, optional): The quantizer for keys. Defaults to ``None``.
        qs (ActivationsCache, optional): The cache of the queries. Defaults to ``None``.
        ks (ActivationsCache, optional): The cache of the keys. Defaults to ``None``.
        q_mod (nn.Module, optional): The module for queries. Defaults to ``None``.
        k_mod (nn.Module, optional): The module for keys. Defaults to ``None``.
        eval_ipt (ActivationsCache, optional): The cache of the inputs corresponding to the evaluation module.
        eval_mod (nn.Module, optional): The module to evaluate the quantization error. Defaults to ``None``.
        eval_kwargs (dict[str, tp.Any], optional): The keyword arguments for evaluation. Defaults to ``None``.
        num_heads (int, optional): The number of heads. Defaults to ``1``.
        num_head_repeats (int, optional): The number of head repeats. Defaults to ``1``.
        post_rope (bool, optional): Whether to apply the post-ROPE. Defaults to ``True``.
        develop_dtype (torch.dtype, optional): The develop dtype. Defaults to ``torch.float32``.

    Returns:
        torch.Tensor: The smooth quantization scale in CPU.
    """
    if scale is None:
        assert qs is not None and ks is not None and eval_ipt is not None, "ftrs or ipts must be provided"
        assert q_mod is not None and k_mod is not None, "modules must be provided"
        scale = SmoothAttentionCalibrator(
            calib_config=smooth_config,
            q_quantizer=q_quantizer,
            k_quantizer=k_quantizer,
            num_heads=num_heads,
            num_head_repeats=num_head_repeats,
            with_rope=with_rope,
            develop_dtype=develop_dtype,
        ).calibrate(
            q_wgt=proj_q.weight,
            k_wgt=proj_k.weight,
            qs=qs,
            ks=ks,
            q_mod=q_mod,
            k_mod=k_mod,
            eval_ipt=eval_ipt,
            eval_mod=eval_mod,
            eval_kwargs=eval_kwargs,
        )
        gc.collect()
        torch.cuda.empty_cache()
    _smooth_modules(
        prev_modules=[proj_k],
        modules=[proj_q],
        scale=scale,
        tensor_type=QuantTensorType.Outputs,
        num_heads=num_heads,
        num_head_repeats=num_head_repeats,
    )
    return scale.to(device="cpu")
