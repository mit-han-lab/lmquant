# -*- coding: utf-8 -*-
"""GPTQ Quantization kernel."""

import gc
import logging
import math

import torch

from ...dataset import ActivationsCache
from ..data.dtype import QuantDataType
from ..data.range import QuantRange, RangeBound
from .config import QuantGPTQConfig
from .simple import simple_quantize

__all__ = ["gptq_quantize"]


@torch.no_grad()
def gptq_quantize(  # noqa: C901
    tensor: torch.Tensor,
    *,
    view_shape: torch.Size,
    quant_dtype: QuantDataType,
    gptq_config: QuantGPTQConfig,
    scale: torch.Tensor,
    zero: torch.Tensor,
    inputs: ActivationsCache,
    quant_range: QuantRange = None,
    range_bound: RangeBound = None,
    round_delta: torch.Tensor = None,
) -> torch.Tensor:
    """Quantize the tensor using the GPTQ quantization kernel.

    Args:
        tensor (torch.Tensor): The tensor to be quantized.
        view_shape (torch.Size): The view shape.
        quant_dtype (QuantDataType): The quantization data type.
        gptq_config (QuantGPTQConfig): The GPTQ configuration.
        scale (torch.Tensor): The scale tensor.
        zero (torch.Tensor): The zero point tensor.
        inputs (ActivationsCache): The input activations.
        quant_range (QuantRange, optional): The quantization range. Defaults to ``None``.
        range_bound (RangeBound, optional): The range bound. Defaults to ``None``.
        round_delta (torch.Tensor, optional): The rounding delta. Defaults to ``None``.

    Returns:
        torch.Tensor: The quantized tensor in the shape of ``view_shape``.
    """
    view_tensor = tensor.view(view_shape)
    # region step 1: reshape the tensor
    len_view_shape = len(view_shape)
    # view_tensor: (#g0, gs0, #g1, gs1, #g2, gs2, ...) -> (#g0, gs0, #g1, #g2, ..., gs1, gs2, ...)
    reshaped_tensor = view_tensor.permute(0, 1, *range(2, len_view_shape, 2), *range(3, len_view_shape, 2))
    # reshaped_tensor: (#g0 * gs0, #g1 * #g2 * ... * gs1 * gs2 * ...)
    reshaped_tensor = reshaped_tensor.reshape(view_shape[0] * view_shape[1], -1)
    num_row_groups, num_column_groups = view_shape[0], view_shape[2::2].numel()
    row_group_size, column_group_size = view_shape[1], view_shape[3::2].numel()
    num_rows, num_columns = reshaped_tensor.shape
    reshaped_scale = scale.view(num_row_groups, 1, num_column_groups)
    zero_is_number = isinstance(zero, (int, float)) or zero.numel() == 1
    reshaped_zero = zero if zero_is_number else zero.view(num_row_groups, 1, num_column_groups)
    # endregion
    # region step 2: get Hessian matrix
    assert inputs.num_sources == 1, f"GPTQ requires only one input source, got {inputs.num_sources}."
    num_samples = inputs.num_samples
    xs, dim, fn = inputs[0].cached, inputs[0].channels_dim, inputs[0].transform
    hessian = torch.zeros((num_columns, num_columns), device=view_tensor.device, dtype=view_tensor.dtype)
    for x in xs:
        x: torch.Tensor = fn(x.view(-1, *x.shape[dim:]))
        if gptq_config.hessian_block_size > 0 and x.shape[0] > gptq_config.hessian_block_size:
            for b in range(0, x.shape[0], gptq_config.hessian_block_size):
                _x = x[b : min(b + gptq_config.hessian_block_size, x.shape[0])]
                _x = math.sqrt(2 / num_samples) * _x.to(device=view_tensor.device, dtype=view_tensor.dtype)
                hessian += torch.matmul(_x.t(), _x)
        else:
            x = math.sqrt(2 / num_samples) * x.to(device=view_tensor.device, dtype=view_tensor.dtype)
            hessian += torch.matmul(x.t(), x)
    dead = hessian.diagonal() == 0
    hessian[dead, dead] = 1
    reshaped_tensor[:, dead] = 0
    del xs, dim, fn, x, inputs, num_samples, dead
    gc.collect()
    torch.cuda.empty_cache()
    # endregion
    # region step 3: permute the Hessian matrix
    importance = torch.diag(hessian)  # (#g1 * #g2 * ... * gs1 * gs2 * ..., )
    permute = torch.argsort(importance, descending=True)
    hessian = hessian[permute][:, permute]
    reshaped_tensor = reshaped_tensor[:, permute]
    inverse_permute = torch.argsort(permute)
    del importance
    # endregion
    # region step 4: apply dampening to avoid numerical instability
    hessian_diag = hessian.diagonal()
    hessian_diag_mean = hessian_diag.mean()
    hessian_diag += gptq_config.damp_percentage * hessian_diag_mean
    # endregion
    # region step 5: get the inverse of the Hessian matrix
    stable_inv, num_inv_tries = False, 0
    hessian_inv = None
    while (not stable_inv) and num_inv_tries < gptq_config.num_inv_tries:
        num_inv_tries += 1
        try:
            hessian_inv = torch.linalg.cholesky(hessian)
            hessian_inv = torch.cholesky_inverse(hessian_inv)
            hessian_inv = torch.linalg.cholesky(hessian_inv, upper=True)
        except RuntimeError:
            hessian_diag += gptq_config.damp_percentage * hessian_diag_mean
            continue
        stable_inv = True
    if num_inv_tries > 1:
        logger = logging.getLogger(f"{__name__}.GPTQ")
        logger.debug(
            "        - GPTQ Hessian is not stable %s %d tries.", "until" if stable_inv else "after", num_inv_tries
        )
    assert stable_inv and hessian_inv is not None, "GPTQ Hessian is not stable! Consider increase damp_percentage."
    assert not hessian_inv.isinf().any(), "Inverse of Hessian matrix contains Inf."
    assert not hessian_inv.isnan().any(), "Inverse of Hessian matrix contains NaN."
    del hessian, hessian_diag, hessian_diag_mean, num_inv_tries
    # endregion
    # region step 6: quantize the tensor
    tensor_hat = torch.zeros_like(reshaped_tensor)
    for c_start in range(0, num_columns, gptq_config.block_size):
        c_end = min(c_start + gptq_config.block_size, num_columns)
        block_tensor = reshaped_tensor[:, c_start:c_end].clone()
        block_tensor_hat = tensor_hat[:, c_start:c_end]
        block_hessian_inv = hessian_inv[c_start:c_end, c_start:c_end]
        block_error = torch.zeros_like(block_tensor)
        for _c in range(c_end - c_start):
            c = c_start + _c
            column = block_tensor[:, _c]  # (#g0 * gs0, )
            pos_diag = block_hessian_inv[_c, _c]
            column_group_index = permute[c] // column_group_size
            column_scale = reshaped_scale[:, :, column_group_index]  # (#g0, 1)
            column_zero = reshaped_zero if zero_is_number else reshaped_zero[:, :, column_group_index]
            column_hat = column.view(num_row_groups, row_group_size).clone()  # (#g0, gs0)
            if range_bound is not None and range_bound.is_set():
                column_hat = column_hat.clamp_(min=range_bound.min, max=range_bound.max)
            column_hat = column_hat.div_(column_scale).add_(column_zero)
            column_hat = simple_quantize(
                column_hat, quant_dtype=quant_dtype, quant_range=quant_range, round_delta=round_delta
            )
            column_hat = column_hat.sub_(column_zero).mul_(column_scale)
            column_hat = column_hat.view(column.shape)
            block_tensor_hat[:, _c] = column_hat.view(-1)
            column_error = column.sub_(column_hat).div_(pos_diag)
            block_error[:, _c] = column_error.view(-1)
            block_tensor[:, _c:] -= column_error.view(-1, 1).matmul(block_hessian_inv[_c, _c:].view(1, -1))
        reshaped_tensor[:, c_end:] -= block_error.matmul(hessian_inv[c_start:c_end, c_end:])
    tensor_hat = tensor_hat[:, inverse_permute]
    # endregion
    # region step 7: reshape the tensor
    _view_shape = view_shape[:2] + view_shape[2::2] + view_shape[3::2]
    # tensor_hat: (#g0 * gs0, #g1 * #g2 * ... * gs1 * gs2, ...) -> (#g0, gs0, #g1, #g2, ..., gs1, gs2, ...)
    tensor_hat = tensor_hat.reshape(_view_shape)
    # tensor_hat: (#g0, gs0, #g1, #g2, ..., gs1, gs2, ...) -> (#g0, gs0, #g1, gs1, #g2, gs2, ...)
    permute_dims = [0, 1]
    for i in range(1, len_view_shape // 2):
        permute_dims.append(1 + i)
        permute_dims.append(len_view_shape // 2 + i)
    view_tensor_hat = tensor_hat.permute(*permute_dims).reshape(view_shape)
    del tensor_hat
    # endregion
    assert not view_tensor_hat.isnan().any(), "GPTQ Quantized tensor contains NaN."
    assert not view_tensor_hat.isinf().any(), "GPTQ Quantized tensor contains Inf."
    return view_tensor_hat
