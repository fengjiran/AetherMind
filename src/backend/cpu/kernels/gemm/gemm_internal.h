#ifndef AETHERMIND_BACKEND_CPU_KERNELS_GEMM_GEMM_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_GEMM_GEMM_INTERNAL_H

/// @file gemm_internal.h
/// @brief Backend-internal FP32 GEMM primitives.

#include "aethermind/base/status.h"

#include <cstdint>

namespace aethermind::cpu::detail {

/// @brief Pre-validated arguments for one strided FP32 matrix multiplication.
///
/// The logical mapping is `output[m, n] = sum(lhs[m, k] * rhs[k, n])`.
/// Callers own shape, stride, alias, and pointer validation; this primitive
/// only performs the compute-ready two-dimensional operation.
struct GemmF32Args {
    const float* lhs{};
    const float* rhs{};
    float* output{};
    int64_t m{};
    int64_t n{};
    int64_t k{};
    int64_t lhs_m_stride{};
    int64_t lhs_k_stride{};
    int64_t rhs_k_stride{};
    int64_t rhs_n_stride{};
    int64_t output_m_stride{};
    int64_t output_n_stride{};
};

/// @brief Runs the scalar FP32 reference GEMM.
///
/// A zero `m` or `n` is a no-op. A zero `k` writes `+0.0F` to every output
/// element without reading `lhs` or `rhs`.
Status RunGemmF32Reference(const GemmF32Args& args) noexcept;

/// @brief Runs the scalar-optimized FP32 GEMM candidate.
///
/// This candidate intentionally remains separate from the double-accumulation
/// reference oracle. It specializes pre-validated M=1, small-M, and generic-M
/// cases with unit-stride lhs K, unit-stride output N, and either contiguous
/// RHS K or N. Every other reference-legal layout falls back to
/// RunGemmF32Reference. It uses no intrinsics or heap allocation.
Status RunGemmF32Scalar(const GemmF32Args& args) noexcept;

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
/// @brief Runs the AVX2/FMA FP32 GEMM candidate with compatible fallback.
///
/// The direct SIMD path requires runtime-validated AVX2/FMA and only handles
/// `M=1`, nonzero K, unit-stride lhs K, K-contiguous RHS, and unit-stride
/// output N. Every other reference-legal input delegates to the scalar
/// candidate, which may in turn delegate to the reference oracle.
Status RunGemmF32Avx2Fma(const GemmF32Args& args) noexcept;
#endif

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_GEMM_GEMM_INTERNAL_H
