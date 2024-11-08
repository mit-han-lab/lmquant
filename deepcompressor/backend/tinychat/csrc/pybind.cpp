#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "quantization/gemm/gemm_cuda.h"
#include "quantization/gemv/gemv_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("awq_gemm_forward_cuda", &awq_gemm_forward_cuda, "AWQ quantized GEMM kernel.");
    m.def("awq_gemv_forward_cuda", &awq_gemv_forward_cuda, "AWQ quantized GEMV kernel.");
}
