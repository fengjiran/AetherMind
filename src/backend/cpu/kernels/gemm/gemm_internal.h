#ifndef AETHERMIND_BACKEND_CPU_KERNELS_GEMM_GEMM_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_GEMM_GEMM_INTERNAL_H

/// @file gemm_internal.h
/// @brief Backend-internal GEMM arguments and precision-specific primitives.

#include "aethermind/base/status.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace aethermind::cpu::detail {

/// @brief Borrowed, pre-validated arguments for one strided matrix multiplication.
///
/// The logical mapping is `output[m, n] = sum(lhs[m, k] * rhs[k, n])`.
/// All strides count elements of the corresponding operand, not bytes. The
/// precision-specific entry defines operand formats, accumulation and rounding;
/// these pointers do not imply a dtype or relax alignment/object-access rules.
/// Callers validate dtype, shape, stride, alias and accessed address ranges,
/// and keep the borrowed storage alive until the invocation completes.
struct GemmArgs {
    const std::byte* lhs{};
    const std::byte* rhs{};
    std::byte* output{};
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

static_assert(std::is_trivially_copyable_v<GemmArgs>);
static_assert(alignof(GemmArgs) <= alignof(std::max_align_t));

/// @brief Pre-validated FP32 GEMM whose B operand is an opaque KC=512, NR=16 B-panel.
///
/// `gemm.n` is the output slice width, `weight_n_offset` selects its first
/// logical row in the packed [N,K] weight, and `logical_n`/`n_blocks` describe
/// that complete weight. `gemm.rhs` and rhs strides are unused. `packed_b`
/// borrows the exact F32 B-panel bytes, not a plain strided RHS.
struct PackedGemmF32Args {
    GemmArgs gemm{};
    const std::byte* packed_b{};
    size_t packed_nbytes{};
    int64_t logical_n{};
    int64_t logical_k{};
    int64_t weight_n_offset{};
    int64_t n_blocks{};
};

/// @brief Runs the scalar FP32 reference GEMM.
///
/// All RunGemmF32* entries interpret accessed operands as naturally aligned
/// float storage; byte pointers do not admit F16/BF16 or unaligned scalar
/// accesses. A zero `m` or `n` is a no-op. A zero `k` writes `+0.0F` to every
/// output element without reading `lhs` or `rhs`.
Status RunGemmF32Reference(const GemmArgs& args) noexcept;

/// @brief Runs the double-accumulation oracle over the KC=512, NR=16 packed-B layout.
Status RunGemmF32PackedBReference(const PackedGemmF32Args& args) noexcept;

/// @brief Runs the scalar-optimized FP32 GEMM candidate.
///
/// This candidate intentionally remains separate from the double-accumulation
/// reference oracle. It specializes pre-validated M=1, small-M, and generic-M
/// cases with unit-stride lhs K, unit-stride output N, and either contiguous
/// RHS K or N. Every other reference-legal layout falls back to
/// RunGemmF32Reference. The N-contiguous driver interleaves N tiles within K
/// panels, carrying ascending-K FP32 sums through bounded stack storage and
/// partial output; the first panel overwrites output without reading it.
/// It uses no intrinsics or heap allocation.
Status RunGemmF32Scalar(const GemmArgs& args) noexcept;

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
/// @brief Attempts F32 x BF16 computation without allocating or narrowing A.
/// @return False requests full reference recomputation, including any output
///         already written; callers must guarantee disjoint input/weight/output.
bool TryRunGemmA32Bf16Avx2Fma(const GemmArgs& args) noexcept;

#if defined(GEMM_HAS_A32_F16_AVX2_FMA_KERNEL)
/// @brief Same attempt contract for F32 x F16; requires AVX2/FMA/F16C.
bool TryRunGemmA32F16Avx2Fma(const GemmArgs& args) noexcept;
#endif

/// @brief Runs the AVX2 scan or blocked consumer for packed-B, with fallback.
Status RunGemmF32PackedB(const PackedGemmF32Args& args) noexcept;

/// @brief AVX2 packed-B scan driver, including partial N and arbitrary M.
Status RunGemmF32PackedBScan(const PackedGemmF32Args& args) noexcept;

/// @brief AVX2 packed-B blocked prefill driver with scan fallback.
Status RunGemmF32PackedBBlocked(const PackedGemmF32Args& args) noexcept;

/// @brief Runs the AVX2/FMA FP32 GEMM candidate with compatible fallback.
///
/// The caller must ensure the host supports AVX2 and FMA before invoking this
/// entry point (this primitive performs no capability detection itself). The
/// SIMD paths require nonzero K, unit-stride lhs K, K-contiguous RHS, and
/// unit-stride output N; `M=1..8` uses the row-pair fast path with a
/// multi-target core for a leftover odd row, while `M>=9` uses the blocked
/// 4x16 micro-kernel with A/B panel packing and row-pair edge handling. Every
/// other reference-legal input delegates to the scalar candidate, which may
/// in turn delegate to the reference oracle.
Status RunGemmF32Avx2Fma(const GemmArgs& args) noexcept;
#endif

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_GEMM_GEMM_INTERNAL_H
