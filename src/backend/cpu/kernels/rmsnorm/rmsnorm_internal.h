#ifndef AETHERMIND_BACKEND_CPU_KERNELS_RMSNORM_CPU_RMSNORM_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_RMSNORM_CPU_RMSNORM_INTERNAL_H

#include "aethermind/base/status.h"
#include "aethermind/base/tensor_view.h"

namespace aethermind::cpu::detail {

/// Backend-internal params for the CPU RMSNorm kernel.
///
/// Placement-constructed into a stack buffer by BuildRmsNormKernelParams
/// and consumed via KernelContext::kernel_params. Lifetime: the TensorView
/// storage referenced by these views must outlive the subsequent kernel
/// entry call. Operators never name this type directly.
struct RmsNormKernelParams {
    TensorView input_tensor{};
    TensorView weight_tensor{};
    MutableTensorView output_tensor{};
};

/// Pre-validated FP32 arguments for RMSNorm micro-kernels.
///
/// Produced by ValidateAndBuildRmsNormFp32Args from TensorView + epsilon attrs
/// and consumed by scalar/AVX2 implementations. Separates validation from
/// compute. `row_count` is the product of every input dimension except the
/// last, so rank-1 input is represented by one row.
struct RmsNormFp32KernelArgs {
    const float* input{};
    const float* weight{};
    float* output{};
    int64_t row_count{};
    int64_t hidden_size{};
    int64_t input_row_stride{};
    int64_t input_col_stride{1};
    int64_t weight_stride{1};
    int64_t output_row_stride{};
    int64_t output_col_stride{1};
    float eps{1.0e-5f};
};

Status RunRmsNormFp32Scalar(const RmsNormFp32KernelArgs& args) noexcept;

#if defined(AETHERMIND_HAS_RMSNORM_AVX2_FMA_KERNEL)
Status RunRmsNormFp32Avx2Fma(const RmsNormFp32KernelArgs& args) noexcept;
#endif

}// namespace aethermind::cpu::detail

#endif// AETHERMIND_BACKEND_CPU_KERNELS_RMSNORM_CPU_RMSNORM_INTERNAL_H
