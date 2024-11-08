# -*- coding: utf-8 -*-
"""Codebook for quantization."""

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
        size (`int`):
            Number of values in the codebook.
        norm_value (`float` or `None`):
            Normalization value.
        value_bits (`int`):
            Number of bits for the value.
        code_bits (`int`):
            Number of bits for the binary code.
        values (`torch.FloatTensor`):
            A value book in ascending order.
        codes (`torch.ByteTensor`):
            A binary book containing the binary representation of the value.
    """

    size: int
    norm_value: float | None
    value_bits: int
    code_bits: int
    values: torch.Tensor
    codes: torch.Tensor

    def __post_init__(self):
        assert self.size <= self.values.numel(), "Codebook size is larger than the values size"
        assert self.values.shape == self.codes.shape, "Values and Codes must have the same shape"
        assert self.codes.numel() == 2**self.code_bits, "Codebook size must be 2**code_bits"
        if self.norm_value is not None:
            assert self.norm_value > 0, "Normalization value must be positive"

    @property
    def normalized(self) -> bool:
        """Check if the codebook is normalized.

        Returns:
            bool:
                `True` if the codebook is normalized, `False` otherwise.
        """
        return self.norm_value is not None

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize a tensor with a codebook.

        Args:
            tensor (`torch.Tensor`):
                A tensor to quantize.

        Returns:
            `torch.Tensor`:
                A quantized tensor.
        """
        dtype, shape, numel = tensor.dtype, tensor.shape, tensor.numel()
        tensor = tensor.contiguous().to(torch.float32)
        if self.norm_value is not None:
            tensor = tensor.div(self.norm_value)
        block_size = 128 * 512 * 4096
        if numel > block_size:
            tensor = tensor.view(-1)
            out = torch.empty_like(tensor)
            for i in range(0, numel, block_size):
                start, end = i, min(i + block_size, numel)
                bnb.dequantize_no_absmax(
                    bnb.quantize_no_absmax(tensor[start:end], code=self.values),
                    code=self.values,
                    out=out[start:end],
                )
            out = out.view(shape)
        else:
            out = bnb.dequantize_no_absmax(bnb.quantize_no_absmax(tensor, code=self.values), code=self.values)
        if self.norm_value is not None:
            out = out.mul_(self.norm_value)
        return out.to(dtype=dtype)

    @staticmethod
    def construct(
        maps: list[tuple[float, int]],
        *,
        value_bits: int,
        code_bits: int,
        normalize: bool | float | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "Codebook":
        """Create a map of values to a code of `code_bits` bits.

        Args:
            maps (`list[tuple[float, int]]`):
                A list of tuples of (value, binary code).
            value_bits (`int`):
                Number of bits for the value.
            code_bits (`int`):
                Number of bits for the binary code.
            normalize (`bool` or `float` or `None`, *optional*, defaults to `None`):
                Normalization value. If `True`, normalize the values based on the maximum value.
            device (`torch.device` or str, *optional*, defaults to `"cpu"`):
                Device to put the codebook and binarybook on.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Dtype of the codebook.

        Returns:
            `Codebook`:
                A codebook.

        Raises:
            `AssertionError`:
                If the number of values is greater than 2**code_bits,
                or if normalize value is smaller than codebook max absolute value.
        """
        if code_bits > 32:
            raise NotImplementedError("Codebook with more than 32 bits is not supported")
        assert len(maps) <= 2**code_bits, "Too many (value, code) maps for the code bits"
        size = len(maps)
        maps.sort(key=lambda x: abs(x[0]))
        maps.extend(repeat(maps[0], 2**code_bits - size))  # fill the gap with the value of the smallest magnitude
        maps.sort(key=lambda x: x[0])
        values = torch.tensor([v[0] for v in maps], device=device, dtype=dtype)
        codes = torch.tensor(
            [v[1] for v in maps],
            dtype=torch.uint8 if code_bits <= 8 else (torch.int16 if code_bits < 16 else torch.int32),
            device=device,
        )
        if normalize:
            if isinstance(normalize, bool):
                normalize = values.abs().max().item()
            assert isinstance(normalize, (float, int)), "Normalization value must be a float or an int"
            assert values.abs().max() <= normalize, "The maximum value is larger than the given normalization value"
            assert normalize > 0, "Normalization value must be positive"
            values.div_(normalize)
        else:
            normalize = None
        return Codebook(
            size=size,
            norm_value=normalize,
            value_bits=value_bits,
            code_bits=code_bits,
            values=values,
            codes=codes,
        )

    @staticmethod
    def build_with_splits(
        maps: list[tuple[float, int]],
        *,
        value_bits: int,
        code_bits: int,
        normalize: bool,
        split_mask: int | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> list["Codebook"]:
        """Create a map of values to a code of `code_bits` bits.

        Args:
            maps (`list[tuple[float, int]]`): A list of tuples of (value, binary code).
            value_bits (`int`): Number of bits for the value.
            code_bits (`int`): Number of bits for the binary code.
            normalize (`bool`): Whether to normalize the values based on the maximum value.
            split_mask (`int` or `None`, *optional*, defaults to `None`):
                A mask to split the values into multiple codebooks.
            device (`torch.device` or str, *optional*, defaults to `"cpu"`):
                Device to put the codebook and binarybook on.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Dtype of the codebook.

        Returns:
            `list[Codebook]`: A list of codebooks.
        """
        if split_mask is None:
            split_maps = [maps]
            max_value = max(abs(v) for v, _ in maps)
        else:
            _split_maps: dict[int, list[tuple[float, int]]] = defaultdict(list)
            max_value = -float("inf")
            for value, code in maps:
                split = code & split_mask
                _split_maps[split].append((value, code))
                max_value = max(max_value, abs(value))
            split_maps = [_split_maps[split] for split in sorted(_split_maps)]
        return [
            Codebook.construct(
                maps=split,
                value_bits=value_bits,
                code_bits=code_bits,
                normalize=max_value if normalize else None,
                device=device,
                dtype=dtype,
            )
            for split in split_maps
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
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> list["Codebook"]:
        """Create a map of floating point values to a code of `code_bits` bits.

        Args:
            total_bits (`int`):
                Number of bits for the floating point value.
            exponent_bits (`int`):
                Number of bits for the exponent.
            signed (`bool`, *optional*, defaults to `True`):
                Whether to use signed code.
            has_inf (`bool`, *optional*, defaults to `False`):
                Whether to include infinity.
            has_nan (`bool`, *optional*, defaults to `False`):
                Whether to include NaN.
            code_bits (`int`, *optional*, defaults to `8`):
                Number of bits for the code.
            normalize (`bool`, *optional*, defaults to `False`):
                Whether to normalize the values based on the maximum value.
            split_mask (`int`, *optional*, defaults to `None`):
                A mask to split the values into multiple codebooks.
            device (`torch.device` or str, *optional*, defaults to `"cpu"`):
                Device to put the codebook on.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Dtype of the codebook.

        Returns:
            `list[Codebook]`:
                A list of codebooks.
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
        maps, code = [], 0
        for evalue in range(end_evalue):
            for mvalue in range(end_mvalue):
                if evalue == 0 and has_subnormal:
                    value = (mvalue / end_mvalue) * (2 ** (1 - bias))
                else:
                    value = (1 + mvalue / end_mvalue) * (2 ** (evalue - bias))
                maps.append((value, code))
                if signed:
                    maps.append((-value, code | sign_mask))
                code += 1
        if mantissa_bits > 0 and not has_inf and has_nan:
            maps = maps[: -(1 + int(signed))]
        return Codebook.build_with_splits(
            maps,
            value_bits=total_bits,
            code_bits=code_bits,
            normalize=normalize,
            split_mask=split_mask,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def build_int_with_splits(
        *,
        total_bits: int,
        signed: bool = True,
        magnitude: bool = False,
        code_bits: int = 8,
        normalize: bool = False,
        split_mask: int | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> list["Codebook"]:
        """Create a map of integer values to a code of `code_bits` bits.

        Args:
            total_bits (`int`):
                Number of bits for the integer value.
            signed (`bool`, *optional*, defaults to `True`):
                Whether to use signed code.
            magnitude (`bool`, *optional*, defaults to `False`):
                Whether to use magnitude-based integer.
            code_bits (`int`, *optional*, defaults to `8`):
                Number of bits for the code.
            normalize (`bool`, *optional*, defaults to `False`):
                Whether to normalize the values based on the maximum value.
            split_mask (`int`, *optional*, defaults to `None`):
                A mask to split the values into multiple codebooks.
            device (`torch.device` or `str`, *optional*, defaults to `"cpu"`):
                Device to put the codebook on.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Dtype of the codebook.

        Returns:
            `list[Codebook]`:
                A list of codebooks.
        """
        if signed:
            end_value = 2 ** (total_bits - 1)
            min_value = -end_value + int(magnitude)
        else:
            end_value = 2**total_bits
            min_value = 0
        maps = []
        for value in range(min_value, end_value):
            if value >= 0:
                code = value
            elif magnitude:
                code = end_value - value
            else:
                code = end_value + end_value + value
            maps.append((value, code))
        return Codebook.build_with_splits(
            maps,
            value_bits=total_bits,
            code_bits=code_bits,
            normalize=normalize,
            split_mask=split_mask,
            device=device,
            dtype=dtype,
        )

    def split(
        self,
        split_mask: int | None,
        normalize: bool | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> list["Codebook"]:
        """Split a codebook into multiple codebooks.

        Args:
            split_mask (`int` or `None`):
                A mask to split the values into multiple codebooks.
            normalize (`bool`, *optional*, defaults to `None`):
                Whether to normalize the values based on the maximum value.
            device (`torch.device` or str, *optional*, defaults to `"cpu"`):
                Device to put the codebook on.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Dtype of the codebook.

        Returns:
            `list[Codebook]`:
                A list of codebooks.
        """
        values = self.values.view(-1)
        codes = self.codes.view(-1)
        if self.norm_value is not None:
            values = values.mul(self.norm_value)
        return Codebook.build_with_splits(
            [(float(value.item()), int(code.item())) for value, code in zip(values, codes, strict=True)],
            value_bits=self.value_bits,
            code_bits=self.code_bits,
            normalize=self.normalized if normalize is None else normalize,
            split_mask=split_mask,
            device=device,
            dtype=dtype,
        )
