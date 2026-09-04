#ifndef AETHERMIND_BACKEND_CPU_KERNELS_RMSNORM_CPU_RMSNORM_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_RMSNORM_CPU_RMSNORM_INTERNAL_H

/// @file rmsnorm_internal.h
/// @brief Backend-internal RMSNorm compute-ready params and FP32 micro-kernels.
///
/// Defines the pre-validated `RmsNormF32KernelArgs` consumed by the reference
/// and AVX2 RMSNorm micro-kernels. Args are prepared once per PreparedExecutionBindings by
/// the registered `KernelParamsBuilder` and passed through
/// `KernelContext::kernel_params` on every execution.

#include "aethermind/base/status.h"
#include "aethermind/base/tensor_view.h"

namespace aethermind::cpu::detail {

/// @brief Pre-validated FP32 arguments for RMSNorm micro-kernels.
///
/// Produced by `ValidateAndBuildCommonRmsNormF32Args` from the binding-time
/// `KernelParamsBuildContext` and epsilon attrs and consumed by reference/AVX2
/// implementations. Separates validation from compute. `row_count` is the
/// product of every input dimension except the last, so rank-1 input is
/// represented by one row.
struct RmsNormF32KernelArgs {
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

/// @brief Runs the reference FP32 RMSNorm micro-kernel.
///
/// @param args Pre-validated kernel arguments. All pointers must be non-null
///        and strides must be consistent with `row_count` and `hidden_size`.
/// @return Ok on success, or an error when arguments are invalid.
Status RunRmsNormF32Reference(const RmsNormF32KernelArgs& args) noexcept;

#if defined(RMSNORM_HAS_AVX2_FMA_KERNEL)
/// @brief Runs the AVX2/FMA FP32 RMSNorm micro-kernel.
///
/// Requires `FMA` and `AVX2` at runtime; the caller must have validated
/// the effective feature set before dispatch.
///
/// @param args Pre-validated kernel arguments. Same requirements as the
///        reference entry point. The `hidden_size` tail is handled by scalar
///        fallback.
/// @return Ok on success, or an error when arguments are invalid.
Status RunRmsNormF32Avx2Fma(const RmsNormF32KernelArgs& args) noexcept;
#endif

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_RMSNORM_CPU_RMSNORM_INTERNAL_H
