# -*- coding: utf-8 -*-
"""TinyChat Extension."""

import os

from torch.utils.cpp_extension import load

__all__ = ["_C"]

dirpath = os.path.dirname(__file__)

_C = load(
    name="deepcompressor_tinychat_C",
    sources=[
        f"{dirpath}/pybind.cpp",
        f"{dirpath}/quantization/gemv/gemv_cuda.cu",
        f"{dirpath}/quantization/gemm/gemm_cuda.cu",
    ],
    extra_cflags=["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++20"],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++20",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_HALF2_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--ptxas-options=--allow-expensive-optimizations=true",
        "--threads=8",
    ],
)
