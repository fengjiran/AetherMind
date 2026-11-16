#ifndef AETHERMIND_BACKEND_CPU_KERNELS_RMSNORM_CPU_RMSNORM_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_RMSNORM_CPU_RMSNORM_INTERNAL_H

/// @file rmsnorm_internal.h
/// Internal declarations for the CPU RMSNorm kernel.
///
/// Defines the pre-validated FP32 compute-ready args (`RmsNormA32W32KernelArgs`)
/// and the FP32 reference/AVX2 micro-kernels. The shared templated
/// `ValidateAndBuildRmsNormArgs` validation core is TU-local to
/// rmsnorm_entry.cpp. Args are prepared once per PreparedExecutionBindings by
/// the registered `KernelParamsBuilder` and passed through
/// `KernelContext::kernel_params` on every execution.

#include "aethermind/base/status.h"
#include "aethermind/dtypes/data_type.h"

#include <cstddef>

namespace aethermind::cpu::detail {

/// @brief Pre-validated FP32 arguments for RMSNorm micro-kernels.
///
/// Produced by `ValidateAndBuildF32Args` (the F32 shell over the shared
/// `ValidateAndBuildRmsNormArgs` core) from the binding-time
/// `KernelParamsBuildContext` and epsilon attrs and consumed by reference/AVX2
/// implementations. Separates validation from compute. `row_count` is the
/// product of every input dimension except the last, so rank-1 input is
/// represented by one row.
struct RmsNormA32W32KernelArgs {
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
Status RunRmsNormF32Reference(const RmsNormA32W32KernelArgs& args) noexcept;

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
Status RunRmsNormF32Avx2Fma(const RmsNormA32W32KernelArgs& args) noexcept;
#endif

/// @brief Pre-validated A16 activation / FP32 weight RMSNorm arguments.
///
/// Activations are byte-addressed 16-bit elements with element strides; the
/// weight stays FP32 because retained-W16 conversion covers Norm roles with a
/// single FP32 copy (P2 Norm-only policy). The reference widens each activation
/// element, reduces the variance in F32, and narrows the normalized product
/// exactly once (frozen P4 rounding placement).
struct RmsNormA16W32KernelArgs {
    const std::byte* input{};
    const float* weight{};
    std::byte* output{};
    DataType act_dtype{};
    int64_t row_count{};
    int64_t hidden_size{};
    int64_t input_row_stride{};
    int64_t input_col_stride{1};
    int64_t weight_stride{1};
    int64_t output_row_stride{};
    int64_t output_col_stride{1};
    float eps{1.0e-5f};
};

/// @brief Runs the scalar A16 RMSNorm reference micro-kernel.
Status RunRmsNormA16Reference(const RmsNormA16W32KernelArgs& args) noexcept;

#if defined(A16_HAS_AVX2_FMA_KERNEL)
/// @brief Runs the A16 RMSNorm AVX2/FMA widening kernel.
///
/// Requires unit column strides for the vector path; other layouts delegate to
/// the scalar reference. The F32 variance reduction and the normalized product
/// use eight-lane FMA groups, so the summation order differs from the scalar
/// reference contract.
Status RunRmsNormA16Avx2Fma(const RmsNormA16W32KernelArgs& args) noexcept;
#endif

#if defined(A16_HAS_AVX512BF16_KERNEL)
/// @brief Runs the BF16 RMSNorm AVX512-BF16 kernel.
///
/// Widens BF16 with an exact left shift, reduces the variance in 16 F32 lanes,
/// and narrows with the native BF16 conversion. Requires unit column strides;
/// other layouts delegate to the scalar reference.
Status RunRmsNormA16Bf16Avx512Bf16(const RmsNormA16W32KernelArgs& args) noexcept;
#endif

#if defined(A16_HAS_AVX512FP16_KERNEL)
/// @brief Runs the F16 RMSNorm AVX512-FP16 kernel.
///
/// Widens and narrows FP16 with the native conversions and reduces the variance
/// in 16 F32 lanes. Requires unit column strides; other layouts delegate to the
/// scalar reference.
Status RunRmsNormA16F16Avx512Fp16(const RmsNormA16W32KernelArgs& args) noexcept;
#endif

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_RMSNORM_CPU_RMSNORM_INTERNAL_H
