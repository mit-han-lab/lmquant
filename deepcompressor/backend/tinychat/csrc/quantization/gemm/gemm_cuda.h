#include <torch/extension.h>

torch::Tensor awq_gemm_forward_cuda(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scales,
    torch::Tensor _zeros);
