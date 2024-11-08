# -*- coding: utf-8 -*-
"""GPTQ Quantization kernel."""

import gc
import math
from dataclasses import dataclass

import torch
from omniconfig import configclass

from ...data.cache import TensorCache
from ...data.dtype import QuantDataType
from ...data.range import QuantRange, RangeBound
from ...data.zero import ZeroPointDomain
from ...utils import tools
from ...utils.common import num2str
from ..config.kernel import BaseQuantKernel, BaseQuantKernelConfig
from ..impl.simple import simple_quantize

__all__ = ["gptq_quantize"]


@configclass
@dataclass
class QuantGptqConfig(BaseQuantKernelConfig):
    """Configuration for GPTQ quantization.

    Args:
        damp_percentage (`float`, *optional*, defaults to `0.01`):
            The percentage of damping.
        block_size (`int`, *optional*, defaults to `128`):
            The block size of the GPTQ quantization.
        num_inv_tries (`int`, *optional*, defaults to `200`):
            The number of tries for the inverse.
        hessian_block_size (`int`, *optional*, defaults to `-1`):
            The block size when calculing the Hessian.
    """

    damp_percentage: float = 0.01
    block_size: int = 128
    num_inv_tries: int = 200
    hessian_block_size: int = -1

    @property
    def name(self) -> str:
        return "GPTQ"

    def build(self) -> "QuantGptqKernel":
        return QuantGptqKernel(self)

    def generate_dirnames(self, *, prefix: str = "", **kwargs) -> list[str]:
        """Generate the directory names of the configuration.

        Args:
            prefix (`str`, *optional*, defaults to `""`):
                The prefix for the directory names.

        Returns:
            `list[str]`:
                The directory names.
        """
        name = f"gptq.d{num2str(self.damp_percentage)}.b{num2str(self.block_size)}"
        return [f"{prefix}.{name}" if prefix else name]


class QuantGptqKernel(BaseQuantKernel):
    def __init__(self, config: "QuantGptqConfig"):
        self.config = config

    def quantize(
        self,
        tensor: torch.Tensor,
        *,
        view_shape: torch.Size,
        quant_dtype: QuantDataType,
        zero_domain: ZeroPointDomain | None,
        scale: torch.Tensor,
        zero: torch.Tensor,
        inputs: TensorCache,
        quant_range: QuantRange | None = None,
        range_bound: RangeBound | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Quantize the tensor.

        Args:
            tensor (`torch.Tensor`):
                The tensor to quantize.
            view_shape (`torch.Size`):
                The view shape when quantizing the tensor.
            quant_dtype (`QuantDataType`):
                The quantization data type.
            zero_domain (`ZeroPointDomain` or `None`):
                The zero point domain.
            scale (`torch.Tensor`):
                The scale tensor.
            zero (`torch.Tensor`):
                The zero point tensor.
            inputs (`TensorCache`):
                The input activations.
            quant_range (`QuantRange` or `None`, *optional*, defaults to `None`):
                The quantization range.
            range_bound (`RangeBound` or `None`, *optional*, defaults to `None`):
                The range bound.
            **kwargs: Other keyword arguments.

        Returns:
            `torch.Tensor`:
                The quantized tensor in the shape of ``view_shape``.
        """
        assert not tensor.requires_grad, "tensor must not require gradient."
        assert not scale.data.requires_grad, "scale must not require gradient."
        assert not zero.data.requires_grad, "zero must not require gradient."
        return gptq_quantize(
            tensor,
            view_shape=view_shape,
            quant_dtype=quant_dtype,
            zero_domain=zero_domain,
            scale=scale,
            zero=zero,
            gptq_config=self.config,
            inputs=inputs,
            quant_range=quant_range,
            range_bound=range_bound,
        )


@torch.no_grad()
def gptq_quantize(  # noqa: C901
    tensor: torch.Tensor,
    *,
    view_shape: torch.Size,
    quant_dtype: QuantDataType,
    zero_domain: ZeroPointDomain | None,
    scale: torch.Tensor,
    zero: torch.Tensor,
    gptq_config: QuantGptqConfig,
    inputs: TensorCache,
    quant_range: QuantRange | None = None,
    range_bound: RangeBound | None = None,
) -> torch.Tensor:
    """Quantize the tensor using the GPTQ quantization kernel.

    Args:
        tensor (`torch.Tensor`):
            The tensor to quantize.
        view_shape (`torch.Size`):
            The view shape when quantizing the tensor.
        quant_dtype (`QuantDataType`):
            The quantization data type.
        zero_domain (`ZeroPointDomain` or `None`):
            The zero point domain.
        scale (`torch.Tensor`):
            The scale tensor.
        zero (`torch.Tensor`):
            The zero point tensor.
        inputs (`TensorCache`):
            The input activations.
        quant_range (`QuantRange` or `None`, *optional*, defaults to `None`):
            The quantization range.
        range_bound (`RangeBound` or `None`, *optional*, defaults to `None`):
            The range bound.

    Returns:
        `torch.Tensor`:
            The quantized tensor in the shape of ``view_shape``.
    """
    view_tensor = tensor.view(view_shape)
    view_shape = view_tensor.shape  # remove any -1 in the view_shape
    # region step 1: reshape the tensor to (#g0 * gs0, #g1 * #g2 * ... * gs1 * gs2, ...)
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
    hessian = torch.zeros((num_columns, num_columns), device=view_tensor.device, dtype=view_tensor.dtype)
    for x in inputs.data:
        x: torch.Tensor = inputs.reshape(x.view(-1, *x.shape[inputs.channels_dim :]))
        if gptq_config.hessian_block_size > 0 and x.shape[0] > gptq_config.hessian_block_size:
            for b in range(0, x.shape[0], gptq_config.hessian_block_size):
                _x = x[b : min(b + gptq_config.hessian_block_size, x.shape[0])]
                _x = math.sqrt(2 / inputs.num_samples) * _x.to(device=view_tensor.device, dtype=view_tensor.dtype)
                hessian += torch.matmul(_x.t(), _x)
        else:
            x = math.sqrt(2 / inputs.num_samples) * x.to(device=view_tensor.device, dtype=view_tensor.dtype)
            hessian += torch.matmul(x.t(), x)
    dead = hessian.diagonal() == 0
    hessian[dead, dead] = 1
    reshaped_tensor[:, dead] = 0
    del x, inputs, dead
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
    while (not stable_inv) and num_inv_tries < gptq_config.num_inv_tries:
        num_inv_tries += 1
        try:
            hessian_inv = torch.linalg.cholesky(hessian)
            hessian_inv = torch.cholesky_inverse(hessian_inv)
            hessian_inv = torch.linalg.cholesky(hessian_inv, upper=True)
        except RuntimeError:
            hessian_diag += (gptq_config.damp_percentage * 0.1) * hessian_diag_mean
            continue
        stable_inv = True
    if num_inv_tries > 1:
        logger = tools.logging.getLogger(f"{__name__}.GPTQ")
        logger.debug("        - Hessian is not stable %s %d tries.", "until" if stable_inv else "after", num_inv_tries)
    assert not hessian_inv.isinf().any(), "Inverse of Hessian matrix contains Inf."
    assert not hessian_inv.isnan().any(), "Inverse of Hessian matrix contains NaN."
    del hessian, hessian_diag, hessian_diag_mean, num_inv_tries
    # endregion
    # region step 6: quantize the tensor
    qtensor = torch.zeros_like(reshaped_tensor)
    for c_start in range(0, num_columns, gptq_config.block_size):
        c_end = min(c_start + gptq_config.block_size, num_columns)
        block_tensor = reshaped_tensor[:, c_start:c_end].clone()
        block_qtensor = qtensor[:, c_start:c_end]
        block_hessian_inv = hessian_inv[c_start:c_end, c_start:c_end]
        block_error = torch.zeros_like(block_tensor)
        for _c in range(c_end - c_start):
            c = c_start + _c
            column = block_tensor[:, _c]  # (#g0 * gs0, )
            pos_diag = block_hessian_inv[_c, _c]
            column_group_index = permute[c] // column_group_size
            column_scale = reshaped_scale[:, :, column_group_index]  # (#g0, 1)
            column_zero = reshaped_zero if zero_is_number else reshaped_zero[:, :, column_group_index]
            qcolumn = column.view(num_row_groups, row_group_size).clone()  # (#g0, gs0)
            if range_bound is not None and range_bound.is_set():
                qcolumn = qcolumn.clamp_(min=range_bound.min, max=range_bound.max)
            if zero_domain == ZeroPointDomain.PostScale:
                qcolumn = qcolumn.add_(column_zero)
            qcolumn = qcolumn.div_(column_scale)
            if zero_domain == ZeroPointDomain.PreScale:
                qcolumn = qcolumn.add_(column_zero)
            qcolumn = simple_quantize(
                qcolumn, quant_dtype=quant_dtype, has_zero_point=zero_domain is not None, quant_range=quant_range
            )
            block_qtensor[:, _c] = qcolumn.view(-1)  # ! copy the quantized column
            if zero_domain == ZeroPointDomain.PreScale:
                qcolumn = qcolumn.sub_(column_zero)
            qcolumn = qcolumn.mul_(column_scale)
            if zero_domain == ZeroPointDomain.PostScale:
                qcolumn = qcolumn.sub_(column_zero)
            column_error = column.sub_(qcolumn.view(column.shape)).div_(pos_diag)
            block_error[:, _c] = column_error.view(-1)
            block_tensor[:, _c:] -= column_error.view(-1, 1).matmul(block_hessian_inv[_c, _c:].view(1, -1))
        reshaped_tensor[:, c_end:] -= block_error.matmul(hessian_inv[c_start:c_end, c_end:])
    qtensor = qtensor[:, inverse_permute]
    # endregion
    # region step 7: reshape the tensor back to (#g0, gs0, #g1, gs1, #g2, gs2, ...)
    _view_shape = view_shape[:2] + view_shape[2::2] + view_shape[3::2]
    # qtensor: (#g0 * gs0, #g1 * #g2 * ... * gs1 * gs2, ...) -> (#g0, gs0, #g1, #g2, ..., gs1, gs2, ...)
    qtensor = qtensor.reshape(_view_shape)
    # qtensor: (#g0, gs0, #g1, #g2, ..., gs1, gs2, ...) -> (#g0, gs0, #g1, gs1, #g2, gs2, ...)
    permute_dims = [0, 1]
    for i in range(1, len_view_shape // 2):
        permute_dims.append(1 + i)
        permute_dims.append(len_view_shape // 2 + i)
    qtensor = qtensor.permute(*permute_dims).reshape(view_shape)
    # endregion
    assert not qtensor.isnan().any(), "GPTQ Quantized tensor contains NaN."
    assert not qtensor.isinf().any(), "GPTQ Quantized tensor contains Inf."
    return qtensor
