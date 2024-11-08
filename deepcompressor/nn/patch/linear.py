# -*- coding: utf-8 -*-
"""Concat Linear Module."""

import torch
import torch.nn as nn

__all__ = ["ConcatLinear", "ShiftedLinear"]


class ConcatLinear(nn.Module):
    def __init__(
        self,
        in_features_list: list[int],
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        assert len(in_features_list) > 1, "ConcatLinear requires at least 2 input features"
        self.in_features_list = in_features_list
        self.in_features = sum(in_features_list)
        self.out_features = out_features
        num_linears = len(in_features_list)
        self.linears = nn.ModuleList(
            [
                nn.Linear(
                    in_features,
                    out_features,
                    bias if idx == num_linears - 1 else False,
                    device,
                    dtype,
                )
                for idx, in_features in enumerate(in_features_list)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # slice x based on in_features_list
        x_splits: list[torch.Tensor] = x.split(self.in_features_list, dim=-1)
        # apply each linear to each slice (we have to make contiguous input for quantization)
        out_splits = [linear(x_split.contiguous()) for linear, x_split in zip(self.linears, x_splits, strict=True)]
        # sum the results
        return sum(out_splits)

    @staticmethod
    def from_linear(linear: nn.Linear, splits: list[int]) -> "ConcatLinear":
        splits.append(linear.in_features - sum(splits))
        splits = [s for s in splits if s > 0]
        assert len(splits) > 1, "ConcatLinear requires at least 2 input features"
        concat_linear = ConcatLinear(
            in_features_list=splits,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        used_in_features = 0
        for sub_linear in concat_linear.linears:
            assert isinstance(sub_linear, nn.Linear)
            in_features = sub_linear.in_features
            sub_linear.weight.data.copy_(linear.weight[:, used_in_features : used_in_features + in_features])
            used_in_features += in_features
        if linear.bias is not None:
            assert sub_linear.bias is not None
            sub_linear.bias.data.copy_(linear.bias)
        return concat_linear


class ShiftedLinear(nn.Module):
    shift: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        shift: float | torch.Tensor,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias, device, dtype)
        self.linear.shifted = True
        device, dtype = self.linear.weight.device, self.linear.weight.dtype
        if not isinstance(shift, torch.Tensor):
            shift = torch.tensor(shift, device=device, dtype=dtype)
        shift = shift.flatten().to(device=device, dtype=dtype)
        shift_features = shift.numel()
        if shift_features > 1:
            assert in_features >= shift_features and in_features % shift_features == 0
            shift = shift.view(-1, 1).expand(-1, in_features // shift_features).flatten()
        self.register_buffer("shift", shift)

    @property
    def in_features(self) -> int:
        return self.linear.in_features

    @property
    def out_features(self) -> int:
        return self.linear.out_features

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.linear(input + self.shift.view([1] * (input.dim() - 1) + [-1]))

    @staticmethod
    def from_linear(linear: nn.Linear, shift: float | torch.Tensor) -> "ShiftedLinear":
        device, dtype = linear.weight.device, linear.weight.dtype
        shifted = ShiftedLinear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            shift=shift,
            bias=True,
            device=device,
            dtype=dtype,
        )
        shifted.linear.weight.data.copy_(linear.weight)
        shift = shifted.shift
        if shift.numel() == 1:
            shifted_bias = linear.weight.double().sum(dim=1) * shift.double()
        else:
            shifted_bias = torch.matmul(linear.weight.double(), shift.view(1, -1).double())
        shifted_bias = shifted_bias.view(shifted.linear.bias.size())
        if linear.bias is not None:
            shifted.linear.bias.data.copy_((linear.bias.data.double() - shifted_bias).to(dtype))
        else:
            shifted.linear.bias.data.copy_(shifted_bias.to(dtype).neg_())
        return shifted
