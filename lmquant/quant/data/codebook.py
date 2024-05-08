# -*- coding: utf-8 -*-
"""Codebook for quantization."""

import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import repeat

import bitsandbytes.functional as bnb
import torch

__all__ = ["Codebook"]


@dataclass
class Codebook:
    """A codebook for quantization.

    Attributes:
        size (int): Number of values in the codebook.
        normalized (bool): Whether the values are normalized.
        norm_value (float | None): Normalization value.
        value_book (torch.FloatTensor): A value book in ascending order.
        binary_book (torch.ByteTensor): A binary book containing the binary representation of the value.
    """

    size: int
    normalized: bool
    norm_value: float | None
    value_book: torch.FloatTensor
    binary_book: torch.ByteTensor | torch.ShortTensor | torch.IntTensor | torch.LongTensor

    def __post_init__(self):
        assert self.size <= self.value_book.numel(), "Codebook size is larger than the value book size"
        assert self.value_book.shape == self.binary_book.shape, "Value book and binary book must have the same shape"
        if self.normalized:
            assert self.norm_value is not None, "Normalization value must be given"
            assert self.norm_value > 0, "Normalization value must be positive"

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize a tensor with a codebook.

        Args:
            tensor (torch.Tensor): A tensor to quantize.

        Returns:
            torch.Tensor: A quantized tensor.
        """
        dtype, shape, numel = tensor.dtype, tensor.shape, tensor.numel()
        tensor = tensor.contiguous().to(torch.float32)
        if self.normalized:
            tensor = tensor.div(self.norm_value)
        block_size = 128 * 512 * 4096
        if numel > block_size:
            tensor = tensor.view(-1)
            out = torch.empty_like(tensor)
            for i in range(0, numel, block_size):
                start, end = i, min(i + block_size, numel)
                bnb.dequantize_no_absmax(
                    bnb.quantize_no_absmax(tensor[start:end], code=self.value_book),
                    code=self.value_book,
                    out=out[start:end],
                )
            out = out.view(shape)
        else:
            out = bnb.dequantize_no_absmax(bnb.quantize_no_absmax(tensor, code=self.value_book), code=self.value_book)
        if self.normalized:
            out = out.mul_(self.norm_value)
        return out.to(dtype=dtype)

    @staticmethod
    def build(
        values: list[tuple[float, int]],
        code_bits: int,
        *,
        normalize: bool | float | None = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> "Codebook":
        """Create a map of values to a code of `code_bits` bits.

        Args:
            values (list[tuple[float, int]]): A list of tuples of (value, binary of the value).
            code_bits (int): Number of bits for the code.
            normalize (bool | float | None, optional): Normalization value. Defaults to ``None``.
            device (torch.device, optional): Device to put the codebook and binarybook on.
                                            Defaults to ``torch.device("cpu")``.
            dtype (torch.dtype, optional): Dtype of the codebook. Defaults to ``torch.float32``.

        Returns:
            Codebook: A codebook.

        Raises:
            AssertionError: If the number of values is greater than 2**code_bits.
                            Or if normalize value is smaller than codebook max absolute value.
        """
        assert len(values) <= 2**code_bits, "Too many values for the code bits"
        codebook_size = len(values)
        num_gaps = 2**code_bits - codebook_size
        values.sort(key=lambda x: abs(x[0]))
        values.extend(repeat(values[0], num_gaps))
        values.sort(key=lambda x: x[0])
        value_book = torch.tensor([v[0] for v in values], device=device, dtype=dtype)
        binary_book = torch.tensor(
            [v[1] for v in values],
            dtype=torch.uint8 if code_bits <= 8 else (torch.int16 if code_bits < 16 else torch.int32),
            device=device,
        )
        if normalize:
            if isinstance(normalize, bool):
                normalize = value_book.abs().max().item()
            assert isinstance(normalize, (float, int)), "Normalization value must be a float or an int"
            assert value_book.abs().max() <= normalize, "The maximum value is larger than the given normalization value"
            value_book.div_(normalize)
        else:
            normalize = None
        return Codebook(
            size=codebook_size,
            normalized=normalize is not None,
            norm_value=normalize,
            value_book=value_book,
            binary_book=binary_book,
        )

    @staticmethod
    def build_with_splits(
        values: list[tuple[float, int]],
        code_bits: int,
        *,
        normalize: bool,
        split_mask: int | None = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> list["Codebook"]:
        """Create a map of values to a code of `code_bits` bits.

        Args:
            values (list[tuple[float, int]]): A list of tuples of (value, binary of the value).
            code_bits (int): Number of bits for the code.
            normalize (bool): Whether to normalize the values based on the maximum value.
            split_mask (int, optional): A mask to split the values into multiple codebooks. Defaults to ``None``.
            device (torch.device, optional): Device to put the codebook on. Defaults to ``torch.device("cpu")``.
            dtype (torch.dtype, optional): Dtype of the codebook. Defaults to ``torch.float32``.

        Returns:
            list[Codebook]: A list of codebooks.
        """
        if split_mask is None:
            value_splits = [values]
            max_value = max(abs(v) for v, _ in values)
        else:
            _value_splits, max_value = defaultdict(list), -float("inf")
            for value, binary in values:
                split = binary & split_mask
                _value_splits[split].append((value, binary))
                max_value = max(max_value, abs(value))
            value_splits = [_value_splits[split] for split in sorted(_value_splits)]
        return [
            Codebook.build(
                values=value_split,
                code_bits=code_bits,
                normalize=max_value if normalize else None,
                device=device,
                dtype=dtype,
            )
            for value_split in value_splits
        ]

    @staticmethod
    def build_fp_with_splits(
        *,
        total_bits: int,
        exponent_bits: int,
        signed: bool = True,
        has_subnormal: bool = True,
        has_inf: bool = False,
        has_nan: bool = False,
        code_bits: int = 8,
        normalize: bool = False,
        split_mask: int | None = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> list["Codebook"]:
        """Create a map of floating point values to a code of `code_bits` bits.

        Args:
            total_bits (int, optional): Number of bits for the floating point value.
            exponent_bits (int, optional): Number of bits for the exponent.
            signed (bool, optional): Whether to use signed code. Defaults to ``True``.
            has_inf (bool, optional): Whether to include infinity. Defaults to ``False``.
            has_nan (bool, optional): Whether to include NaN. Defaults to ``False``.
            code_bits (int, optional): Number of bits for the code. Defaults to ``8``.
            normalize (bool, optional): Whether to normalize the values based on the maximum value. Defaults to ``False``.
            split_mask (int, optional): A mask to split the values into multiple codebooks. Defaults to ``None``.
            device (torch.device, optional): Device to put the codebook on. Defaults to ``torch.device("cpu")``.
            dtype (torch.dtype, optional): Dtype of the codebook. Defaults to ``torch.float32``.

        Returns:
            list[Codebook]: A list of codebooks.
        """
        mantissa_bits = total_bits - exponent_bits - int(signed)
        assert exponent_bits > 0, "Exponent bits must be positive"
        assert mantissa_bits >= 0, "Mantissa bits must be non-negative"
        assert (
            total_bits <= code_bits
        ), f"Too many bits ({exponent_bits} + {mantissa_bits} + {int(signed)} = {total_bits}) for {code_bits}-bit code"
        has_nan = has_inf or has_nan

        sign_mask = 1 << (total_bits - 1)
        if mantissa_bits > 0:
            end_evalue = 2**exponent_bits - int(has_inf)
        else:
            end_evalue = 2**exponent_bits - int(has_nan)
        end_mvalue = 2**mantissa_bits
        bias = 2 ** (exponent_bits - 1) - 1
        values, binary = [], 0
        for evalue in range(end_evalue):
            for mvalue in range(end_mvalue):
                if evalue == 0 and has_subnormal:
                    value = (mvalue / end_mvalue) * (2 ** (1 - bias))
                else:
                    value = (1 + mvalue / end_mvalue) * (2 ** (evalue - bias))
                values.append((value, binary))
                if signed:
                    values.append((-value, binary | sign_mask))
                binary += 1
        if mantissa_bits > 0 and not has_inf and has_nan:
            values = values[: -(1 + int(signed))]
        return Codebook.build_with_splits(
            values, code_bits, normalize=normalize, split_mask=split_mask, device=device, dtype=dtype
        )

    @staticmethod
    def build_int_with_splits(
        *,
        total_bits: int,
        signed: bool = True,
        magnitude: bool = False,
        symmetric: bool = False,
        code_bits: int = 8,
        normalize: bool = False,
        split_mask: int | None = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> list["Codebook"]:
        """Create a map of integer values to a code of `code_bits` bits.

        Args:
            total_bits (int): Number of bits for the integer value.
            signed (bool, optional): Whether to use signed code. Defaults to ``True``.
            magnitude (bool, optional): Whether to use magnitude code. Defaults to ``False``.
            symmetric (bool, optional): Whether to use symmetric code. Defaults to ``False``.
            code_bits (int, optional): Number of bits for the code. Defaults to ``8``.
            normalize (bool, optional): Whether to normalize the values based on the maximum value. Defaults to ``False``.
            split_mask (int, optional): A mask to split the values into multiple codebooks. Defaults to ``None``.
            device (torch.device, optional): Device to put the codebook on. Defaults to ``torch.device("cpu")``.
            dtype (torch.dtype, optional): Dtype of the codebook. Defaults to ``torch.float32``.

        Returns:
            list[Codebook]: A list of codebooks.
        """
        if signed:
            end_value = 2 ** (total_bits - 1)
            min_value = -end_value + (1 if magnitude or symmetric else 0)
        else:
            end_value = 2**total_bits
            min_value = 0
        values = []
        for value in range(min_value, end_value):
            if value >= 0:
                binary = value
            elif magnitude:
                binary = end_value - value
            else:
                binary = end_value + end_value + value
            values.append((value, binary))
        return Codebook.build_with_splits(
            values,
            code_bits,
            normalize=normalize,
            split_mask=split_mask,
            device=device,
            dtype=dtype,
        )

    def split(
        self,
        split_mask: int | None,
        normalize: bool | None = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> list["Codebook"]:
        """Split a codebook into multiple codebooks.

        Args:
            split_mask (int, optional): A mask to split the values into multiple codebooks.
            normalize (bool, optional): Whether to normalize the values based on the maximum value. Defaults to ``None``.
            device (torch.device, optional): Device to put the codebook on. Defaults to ``torch.device("cpu")``.
            dtype (torch.dtype, optional): Dtype of the codebook. Defaults to ``torch.float32``.

        Returns:
            list[Codebook]: A list of codebooks.
        """
        value_book = self.value_book.view(-1)
        binary_book = self.binary_book.view(-1)
        code_bits = int(math.log2(binary_book.numel()))
        if self.normalized:
            value_book = value_book.mul(self.norm_value)
        values = [(value.item(), binary.item()) for value, binary in zip(value_book, binary_book)]
        return Codebook.build_with_splits(
            values,
            code_bits,
            normalize=self.normalized if normalize is None else normalize,
            split_mask=split_mask,
            device=device,
            dtype=dtype,
        )
