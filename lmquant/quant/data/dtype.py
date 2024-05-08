# -*- coding: utf-8 -*-
"""Quantization data type."""

import typing as tp

import torch

from .codebook import Codebook

__all__ = ["QuantDataType", "QDType"]


class QuantDataType:
    """Quantization data type."""

    _instances: dict[str, "QuantDataType"] = {}

    def __new__(
        cls,
        *args: tp.Any,
        **kwargs: tp.Any,
    ) -> "QuantDataType":
        """Create a new instance of the QuantDataType.

        Returns:
            QuantDataType: The new instance.
        """
        obj = super().__new__(cls)
        # initialize the instance
        obj.__init__(*args, **kwargs)
        s = str(obj)
        return cls._instances.setdefault(s, obj)

    def __init__(
        self,
        total_bits: int,
        *,
        signed: bool = True,
        exponent_bits: int = 0,
        has_subnormal: bool = True,
        has_nan: bool = False,
        has_inf: bool = False,
        has_zero_point: bool = False,
        magnitude: bool = False,
        codebook: Codebook | None = None,
        codebook_name: str = "",
    ):
        """Initialize the quantization data type.

        Args:
            total_bits (int): Total number of bits. Must be greater than 0.
            signed (bool, optional): Whether the data type is signed. Defaults to ``True``.
            exponent_bits (int, optional): Number of bits for the exponent. Defaults to ``0``.
            has_nan (bool, optional): Whether the data type has NaN if it is float. Defaults to ``False``.
            has_inf (bool, optional): Whether the data type has Inf if it is float. Defaults to ``False``.
            has_zero (bool, optional): Whether the data type has non-zero zero-point. Defaults to ``False``.
            magnitude (bool, optional): Whether the data type is magnitude-based if it is int. Defaults to ``False``.
        """
        self.__signed = signed
        # region set bit widths
        self.__total_bits = total_bits
        self.__exponent_bits = exponent_bits
        assert self.__total_bits > 0, "Total bits must be greater than 0."
        assert self.__exponent_bits >= 0, "Exponent bits must be non-negative."
        self.__mantissa_bits = self.__total_bits - self.__exponent_bits - int(self.__signed)
        # endregion
        # region set data type properties
        if self.__exponent_bits > 0:
            # for floating-point data type
            self.__has_subnormal = has_subnormal
            self.__has_inf = has_inf
            self.__has_nan = has_inf or has_nan
            self.__magnitude = True
            if self.__mantissa_bits == 0:
                assert not self.__has_inf, "Inf is not supported for exponent-only floating-point data type."
                if self.__exponent_bits == 1:
                    assert not self.__has_nan, "NaN is not supported for 1-bit exponent-only floating-point data type."
        else:
            # for integer data type
            self.__has_subnormal = False
            self.__has_inf = False
            self.__has_nan = False
            self.__magnitude = magnitude
        # endregion
        self.__has_zero_point = has_zero_point
        # region set codebook
        if codebook is not None:
            assert self.is_float, "Codebook is only supported for floating-point data type."
            self.__codebook = codebook
            assert codebook_name, "Codebook name must be specified."
            self.__codebook_name = codebook_name
        else:
            self.__codebook = None
            self.__codebook_name = ""
        # endregion
        # region set split codebooks
        self.__split_codebooks: dict[tuple[int, bool, int, torch.device, torch.dtype], list[Codebook]] = {}
        # endregion

    # region properties
    @property
    def signed(self) -> bool:
        """Whether the data type is signed."""
        return self.__signed

    @property
    def total_bits(self) -> int:
        """Total number of bits."""
        return self.__total_bits

    @property
    def exponent_bits(self) -> int:
        """Number of bits for the exponent."""
        return self.__exponent_bits

    @property
    def mantissa_bits(self) -> int:
        """Number of bits for the mantissa."""
        return self.__mantissa_bits

    @property
    def has_subnormal(self) -> bool:
        """Whether the data type has subnormal."""
        return self.__has_subnormal

    @property
    def has_inf(self) -> bool:
        """Whether the data type has Inf."""
        return self.__has_inf

    @property
    def has_nan(self) -> bool:
        """Whether the data type has NaN."""
        return self.__has_nan

    @property
    def magnitude(self) -> bool:
        """Whether the data type is magnitude-based."""
        return self.__magnitude

    @property
    def has_zero_point(self) -> bool:
        """Whether the data type has non-zero zero-point."""
        return self.__has_zero_point

    @property
    def is_float(self) -> bool:
        """Whether the data type is floating-point."""
        return self.exponent_bits > 0

    @property
    def is_int(self) -> bool:
        """Whether the data type is integer."""
        return self.exponent_bits == 0

    @property
    def is_exp(self) -> bool:
        """Whether the data type is exponent-only floating-point."""
        return self.exponent_bits > 0 and self.mantissa_bits == 0 and not self.has_subnormal

    @property
    def symmetric(self) -> bool:
        """Whether the data type is symmetric around zero."""
        return self.signed and (self.magnitude or not self.has_zero_point)

    @property
    def exponent_mask(self) -> int:
        """Bit mask for the exponent."""
        return ((1 << self.exponent_bits) - 1) << self.mantissa_bits

    @property
    def mantissa_mask(self) -> int:
        """Bit mask for the mantissa."""
        return (1 << self.mantissa_bits) - 1

    @property
    def _end_mantissa_value(self) -> int:
        return 2**self.mantissa_bits

    @property
    def _end_exponent_value(self) -> int:
        if self.mantissa_bits > 0:
            return 2**self.exponent_bits - int(self.has_inf)
        else:
            return 2**self.exponent_bits - int(self.has_nan)

    @property
    def exponent_bias(self) -> int:
        """Exponent bias."""
        if self.is_float:
            return 2 ** (self.exponent_bits - 1) - 1
        else:
            return 0

    @property
    def max_exponent_value(self) -> int:
        """Maximum exponent value."""
        if self.is_float:
            return self._end_exponent_value - 1 - self.exponent_bias
        else:
            return self.total_bits - 1 - int(self.signed)

    @property
    def min_exponent_value(self) -> int:
        if self.is_float:
            return int(self.has_subnormal) - self.exponent_bias
        else:
            return 0

    @property
    def max_power_of_two(self) -> int:
        """Maximum positive exponent value."""
        return 2**self.max_exponent_value

    @property
    def min_power_of_two(self) -> int:
        if self.signed:
            return -self.max_power_of_two
        else:
            return 0

    @property
    def max_positive_normal(self) -> float:
        """Maximum positive normal value."""
        if self.is_float:
            if self.mantissa_bits > 0 and not self.has_inf and self.has_nan:
                t = 2 - 2 / self._end_mantissa_value
            else:
                t = 2 - 1 / self._end_mantissa_value
            return t * 2**self.max_exponent_value
            # e = self._end_exponent_value - 1 - self.exponent_bias
            # return t * 2**e
        else:
            return self._end_mantissa_value - 1

    @property
    def min_positive_normal(self) -> float:
        """Minimum positive normal value."""
        return 2**self.min_exponent_value
        # if self.is_float:
        #     return 2 ** (int(self.has_subnormal) - self.exponent_bias)
        # else:
        #     return 1

    @property
    def max_positive_subnormal(self) -> float:
        """Maximum positive subnormal value."""
        if self.is_float and self.has_subnormal and self.mantissa_bits > 0:
            t = 1 - 1 / self._end_mantissa_value
            e = 1 - self.exponent_bias
            return t * 2**e
        else:
            return 0

    @property
    def min_positive_subnormal(self) -> float:
        """Minimum non-negative subnormal value."""
        if self.is_float and self.has_subnormal and self.mantissa_bits > 0:
            t = 1 / self._end_mantissa_value
            e = 1 - self.exponent_bias
            return t * 2**e
        else:
            return 0

    @property
    def max_value(self) -> float:
        """Maximum value."""
        return self.max_positive_normal if self.__codebook is None else self.__codebook.value_book[-1].item()

    @property
    def min_value(self) -> float:
        """Minimum value."""
        if self.__codebook is not None:
            return self.__codebook.value_book[0].item()
        if self.signed:
            if self.magnitude or not self.__has_zero_point:
                return -self.max_value
            else:
                return -self.max_value - 1
        else:
            return 0

    # endregion

    def __build_split_codebooks(
        self,
        *,
        code_bits: int = 8,
        normalize: bool = False,
        split_mask: int | None = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> list[Codebook]:
        if self.is_float:
            return Codebook.build_fp_with_splits(
                total_bits=self.total_bits,
                exponent_bits=self.exponent_bits,
                signed=self.signed,
                has_subnormal=self.has_subnormal,
                has_inf=self.has_inf,
                has_nan=self.has_nan,
                code_bits=code_bits,
                normalize=normalize,
                split_mask=split_mask,
                device=device,
                dtype=dtype,
            )
        else:
            return Codebook.build_int_with_splits(
                total_bits=self.total_bits,
                signed=self.signed,
                magnitude=self.magnitude,
                symmetric=self.symmetric,
                code_bits=code_bits,
                normalize=normalize,
                split_mask=split_mask,
                device=device,
                dtype=dtype,
            )

    def get_split_codebooks(
        self,
        *,
        code_bits: int = 8,
        normalize: bool = False,
        split_mask: int | None = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> list[Codebook]:
        """Get a get_codebook of `code_bits` bits for the quantization.

        Args:
            code_bits (int, optional): Number of bits for the codebook. Defaults: ``8``.
            normalize (bool, optional): Whether to normalize the codebook values
                                        based on the maximum value. Defaults: ``False``.
            split_mask (int, optional): Bit mask to split the codebook into parts. Defaults: ``None``.
            device (torch.device, optional): Device to create the codebook on. Defaults: ``torch.device("cpu")``.
            dtype (torch.dtype, optional): Data type to create the codebook with. Defaults: ``torch.float32``.

        Returns:
            list[Codebook]: A list of codebooks.
        """
        device = torch.device("cpu") if device is None else torch.device(device)
        key = (code_bits, normalize, split_mask, device, dtype)
        if key not in self.__split_codebooks:
            if self.__codebook is not None:
                self.__split_codebooks[key] = self.__codebook.split(
                    split_mask=split_mask, normalize=normalize, device=device, dtype=dtype
                )
            else:
                self.__split_codebooks[key] = self.__build_split_codebooks(
                    code_bits=code_bits,
                    normalize=normalize,
                    split_mask=split_mask,
                    device=device,
                    dtype=dtype,
                )
        return self.__split_codebooks[key]

    def get_codebook(
        self,
        *,
        code_bits: int = 8,
        normalize: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> Codebook:
        """Get a get_codebook of `code_bits` bits for the quantization.

        Args:
            code_bits (int, optional): Number of bits for the codebook. Defaults: ``8``.
            normalize (bool, optional): Whether to normalize the codebook values
                                        based on the maximum value. Defaults: ``False``.
            device (torch.device, optional): Device to create the codebook on. Defaults: ``torch.device("cpu")``.
            dtype (torch.dtype, optional): Data type to create the codebook with. Defaults: ``torch.float32``.

        Returns:
            Codebook: The codebook.
        """
        return self.get_split_codebooks(code_bits=code_bits, normalize=normalize, device=device, dtype=dtype)[0]

    def __str__(self) -> str:
        if self.signed:
            s = "z" if self.has_zero_point else "s"
        else:
            s = "n" if self.has_zero_point else "u"
        if self.__codebook_name:
            return f"{s}{self.__codebook_name}{self.total_bits}"
        if self.is_float:
            if self.has_subnormal or self.mantissa_bits > 0:
                s += "fp" if self.has_subnormal else "fn"
                s += f"{self.total_bits}_e{self.exponent_bits}m{self.mantissa_bits}"
                s += "_inf" if self.has_inf else ("_nan" if self.has_nan else "_all")
            else:
                assert not self.has_inf, "Inf is not supported for exponent-only floating-point data type."
                s += f"exp{self.exponent_bits}"
                s += "_nan" if self.has_nan else "_all"
        else:
            s += "mag" if self.magnitude else "int"
            s += f"{self.total_bits}"
        return s

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def _from_str(s: str, /) -> "QuantDataType":
        s = s.strip().lower()
        if s[0] == "s":
            signed, has_zero_point = True, False
        elif s[0] == "z":
            signed, has_zero_point = True, True
        elif s[0] == "u":
            signed, has_zero_point = False, False
        elif s[0] == "n":
            signed, has_zero_point = False, True
        else:
            raise ValueError(f"Unknown QuantDataType {s}")
        s = s[1:]
        if s.startswith("int"):
            return QuantDataType(int(s[3:]), signed=signed, has_zero_point=has_zero_point)
        elif s.startswith("mag"):
            return QuantDataType(int(s[3:]), signed=signed, has_zero_point=has_zero_point, magnitude=True)
        elif s.startswith("exp"):
            ss = s.split("_")
            total_bits = int(ss[0][3:])
            if len(ss) >= 2:
                has_nan = ss[1] == "nan"
            else:
                has_nan = False
            return QuantDataType(
                total_bits=total_bits,
                signed=signed,
                exponent_bits=total_bits - int(signed),
                has_subnormal=False,
                has_nan=has_nan,
                has_zero_point=has_zero_point,
            )
        elif s.startswith("f"):
            ss = s.split("_")
            has_subnormal = s[1] == "p"
            total_bits = int(ss[0][2:])
            exponent_bits = int(ss[1][1 : ss[1].find("m")])
            if len(ss) >= 3:
                has_inf = ss[2] == "inf"
                has_nan = has_inf or (ss[2] == "nan")
            else:
                has_inf, has_nan = False, False
            return QuantDataType(
                total_bits=total_bits,
                signed=signed,
                exponent_bits=exponent_bits,
                has_subnormal=has_subnormal,
                has_inf=has_inf,
                has_nan=has_nan,
                has_zero_point=has_zero_point,
            )
        else:
            raise ValueError(f"Unknown QuantDataType {s}")

    @staticmethod
    def from_str(s: str, /) -> tp.Union["QuantDataType", torch.dtype, None]:
        """Get a QuantDataType from a string.

        Args:
            s (str): The string to parse.

        Returns:
            tp.Union[QuantDataType, torch.dtype, None]: The QuantDataType or torch.dtype.
        """
        s = s.lower()
        if not s or s == "none" or s == "null":
            return None
        elif s.startswith("torch"):
            return eval(s)
        _s, _d = s[:-2], s[-2:]
        if _d in ("16", "32", "64"):
            if _s == "fp" or _s == "float":
                return eval(f"torch.float{_d}")
            elif _s == "int":
                return eval(f"torch.int{_d}")
            elif _s == "uint":
                return eval(f"torch.uint{_d}")
        return QuantDataType._from_str(s)


class _QDTypeMeta(type):
    def __getattr__(cls, __name: str) -> tp.Any:
        if __name.startswith("_"):
            return super().__getattr__(__name)
        else:
            return QuantDataType.from_str(__name)


class QDType(metaclass=_QDTypeMeta):
    """QuantDataType class for easy access to QuantDataType by name."""

    pass
