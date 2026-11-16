#ifndef AETHERMIND_BACKEND_CPU_KERNELS_LINEAR_LINEAR_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_LINEAR_LINEAR_INTERNAL_H

#include "aethermind/base/status.h"
#include "aethermind/dtypes/data_type.h"
#include "backend/cpu/kernels/gemm/gemm_internal.h"

#include <cstddef>
#include <cstdint>

namespace aethermind::cpu::detail {

/// @brief Pre-validated FP32 arguments for one Linear kernel invocation.
///
/// Produced once by the registered KernelParamsBuilder and consumed on every
/// execution. `row_count` flattens input leading dimensions; rank-1 input is
/// represented by one row. The logical mapping is
/// `output[row, out] = sum(input[row, in] * weight[out, in])`.
struct LinearA32W32KernelArgs {
    const float* input{};
    const float* weight{};
    float* output{};
    int64_t row_count{};
    int64_t in_features{};
    int64_t out_features{};
    int64_t input_row_stride{};
    int64_t input_col_stride{1};
    int64_t weight_row_stride{};
    int64_t weight_col_stride{1};
    int64_t output_row_stride{};
    int64_t output_col_stride{1};
};

/// @brief Runs the scalar FP32 Linear reference kernel.
///
/// `args` must have been produced by the binding-time params builder. A zero
/// `row_count` or `out_features` is a no-op. With zero `in_features`, the
/// kernel writes zero to every output element without reading input or weight.
Status RunLinearF32Reference(const LinearA32W32KernelArgs& args) noexcept;

/// @brief Pre-validated FP32-activation, reduced-precision-weight arguments.
///
/// Weight strides are element strides; `weight` is byte-addressed because raw
/// model storage need not contain live C++ Half/BFloat16 objects or meet their
/// alignment. The reference widens each element and accumulates in FP32.
/// FP32 input/output addresses that are accessed must satisfy alignof(float).
struct LinearA32W16KernelArgs {
    const float* input{};
    const std::byte* weight{};
    float* output{};
    DataType weight_dtype{};
    int64_t row_count{};
    int64_t in_features{};
    int64_t out_features{};
    int64_t input_row_stride{};
    int64_t input_col_stride{1};
    int64_t weight_row_stride{};
    int64_t weight_col_stride{1};
    int64_t output_row_stride{};
    int64_t output_col_stride{1};
};

/// @brief Runs the scalar FP32-accumulating F16/BF16 Linear reference kernel.
Status RunLinearF32W16Reference(const LinearA32W16KernelArgs& args) noexcept;

/// @brief Pre-validated 16-bit activation and weight Linear arguments.
///
/// Both activation and weight are byte-addressed 16-bit elements; strides are
/// element strides. The reference widens every operand to FP32, accumulates in
/// ascending K order, and writes the output as either 16-bit (standard
/// projections) or FP32 (an A16 LMHead that must not narrow logits). Input and
/// output data may be arbitrarily aligned because every access uses memcpy.
struct LinearA16W16KernelArgs {
    const std::byte* input{};
    const std::byte* weight{};
    std::byte* output{};
    DataType act_dtype{};
    DataType output_dtype{};
    int64_t row_count{};
    int64_t in_features{};
    int64_t out_features{};
    int64_t input_row_stride{};
    int64_t input_col_stride{1};
    int64_t weight_row_stride{};
    int64_t weight_col_stride{1};
    int64_t output_row_stride{};
    int64_t output_col_stride{1};
};

/// @brief Runs the scalar A16 x W16 Linear reference kernel.
Status RunLinearA16Reference(const LinearA16W16KernelArgs& args) noexcept;

/// @brief Checks whether a contiguous BF16 dot-product segment is safe from
/// VDPBF16PS's unconditional DAZ/FTZ behavior.
/// @note This conservative integer-only check is baseline code. Rejected
///       segments must execute through the gradual-underflow reference.
bool IsBf16NativeDotProductSafe(const std::byte* input,
                                const std::byte* weight,
                                int64_t element_count) noexcept;

#if defined(A16_HAS_AVX2_FMA_KERNEL)
/// @brief Runs the A16 x W16 Linear AVX2/FMA widening kernel.
///
/// F16 uses F16C widening and BF16 uses an exact left-shift widening; both
/// accumulate in F32 with fused multiply-add. Non-unit column strides delegate
/// to the scalar reference so the prepared-args contract never fails on layout.
Status RunLinearA16Avx2Fma(const LinearA16W16KernelArgs& args) noexcept;
#endif

#if defined(A16_HAS_AVX512BF16_KERNEL)
/// @brief Runs the BF16 x BF16 Linear AVX512-BF16 kernel.
///
/// Uses VDPBF16PS: each instruction accumulates the pairwise products of 32
/// BF16 elements into 16 F32 lanes, and the lanes are reduced in ascending
/// order. The grouped summation order differs from the scalar ascending-K
/// reference and is verified independently against an FP64 oracle.
/// Segments outside IsBf16NativeDotProductSafe's domain use the reference,
/// including normal operands whose products can cancel into an F32 subnormal.
Status RunLinearA16Bf16Avx512Bf16(const LinearA16W16KernelArgs& args) noexcept;
#endif

#if defined(A16_HAS_AVX512FP16_KERNEL)
/// @brief Runs the F16 x F16 Linear AVX512-FP16 kernel.
///
/// Widens 16 FP16 elements to F32 with the native conversion and accumulates
/// with fused multiply-add over 16 lanes, then reduces lanes in ascending
/// order. The fused accumulation order differs from the scalar reference.
Status RunLinearA16F16Avx512Fp16(const LinearA16W16KernelArgs& args) noexcept;
#endif

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_LINEAR_LINEAR_INTERNAL_H
