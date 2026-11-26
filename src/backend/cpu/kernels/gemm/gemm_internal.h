#ifndef AETHERMIND_BACKEND_CPU_KERNELS_GEMM_GEMM_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_GEMM_GEMM_INTERNAL_H

/// @file gemm_internal.h
/// @brief Backend-internal GEMM arguments, entry declarations and ISA-neutral
///        numeric gates.
///
/// This header is included by translation units compiled without per-ISA flags
/// (scalar and reference GEMMs included), so it must stay ISA-neutral; keep
/// intrinsics and other ISA-specific code in per-ISA internal headers.

#include "aethermind/base/status.h"
#include "aethermind/dtypes/data_type.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
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

/// @brief Pre-validated GEMM parameters for same-type FP16 or BF16 operands.
///
/// `operand_dtype` describes both inputs. The output is either that dtype or
/// FP32. Drivers accumulate in FP32 and convert the complete sum at writeback.
/// The operands are byte-addressed and may be unaligned. As with GemmArgs,
/// callers validate shape, strides, ranges, and disjoint output storage.
struct GemmA16W16Args {
    GemmArgs gemm{};
    DataType operand_dtype{};
    DataType output_dtype{};
};

/// @brief Largest |input| over `k` elements.
///
/// Non-finite inputs report +inf so the reduction gate below rejects the
/// associated tile.
inline float GemmInputMagnitude(const float* input, int64_t k) noexcept {
    float magnitude = 0.0f;
    for (int64_t i = 0; i < k; ++i) {
        if (!std::isfinite(input[i])) {
            return std::numeric_limits<float>::infinity();
        }
        magnitude = std::max(magnitude, std::fabs(input[i]));
    }
    return magnitude;
}

/// @brief Whether every partial sum of a K-long FP32 reduction stays finite.
///
/// Bounding the sum of absolute products also bounds every scalar/SIMD partial
/// sum. Extreme inputs retain the reference's overflow semantics.
inline bool GemmReductionMagnitudeIsSafe(float input, float weight, int64_t k) noexcept {
    return std::isfinite(input) && std::isfinite(weight) &&
           static_cast<double>(input) * static_cast<double>(weight) * static_cast<double>(k) <=
                   static_cast<double>(std::numeric_limits<float>::max()) * 0.5;
}

/// @brief Tests a contiguous BF16 segment for safe VDPBF16PS execution.
/// Rejected segments require the reference path to preserve gradual underflow
/// and the FP32 product and accumulation semantics.
bool IsBf16NativeDotProductSafe(const std::byte* input,
                                const std::byte* weight,
                                int64_t element_count) noexcept;

/// @brief Runs the scalar FP32 reference GEMM.
///
/// The direct RunGemmA32W32* entries interpret accessed operands as naturally aligned
/// float storage; byte pointers do not admit F16/BF16 or unaligned scalar
/// accesses. A zero `m` or `n` is a no-op. A zero `k` writes `+0.0F` to every
/// output element without reading `lhs` or `rhs`.
Status RunGemmA32W32Reference(const GemmArgs& args) noexcept;

/// @brief Runs the scalar-optimized FP32 GEMM candidate.
///
/// This candidate intentionally remains separate from the double-accumulation
/// reference oracle. It specializes pre-validated M=1, small-M, and generic-M
/// cases with unit-stride lhs K, unit-stride output N, and either contiguous
/// RHS K or N. Every other reference-legal layout falls back to
/// RunGemmA32W32Reference. The N-contiguous driver interleaves N tiles within K
/// panels, carrying ascending-K FP32 sums through bounded stack storage and
/// partial output; the first panel overwrites output without reading it.
/// It uses no intrinsics or heap allocation.
Status RunGemmA32W32Scalar(const GemmArgs& args) noexcept;

/// @brief Runs the double-accumulation oracle over the KC=512, NR=16 packed-B layout.
Status RunGemmA32W32PackedBReference(const PackedGemmF32Args& args) noexcept;

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
/// @brief Attempts F32 x BF16 computation without allocating or narrowing A.
/// @return False requests full reference recomputation, including any output
///         already written; callers must guarantee disjoint input/weight/output.
bool TryRunGemmA32WBf16Avx2Fma(const GemmArgs& args) noexcept;

#if defined(GEMM_HAS_A32_F16_AVX2_FMA_KERNEL)
/// @brief Same attempt contract for F32 x F16; requires AVX2/FMA/F16C.
bool TryRunGemmA32WF16Avx2Fma(const GemmArgs& args) noexcept;
#endif

/// @brief Runs the AVX2 scan or blocked consumer for packed-B, with fallback.
Status RunGemmA32W32PackedB(const PackedGemmF32Args& args) noexcept;

/// @brief AVX2 packed-B scan driver, including partial N and arbitrary M.
Status RunGemmA32W32PackedBScan(const PackedGemmF32Args& args) noexcept;

/// @brief AVX2 packed-B blocked prefill driver with scan fallback.
Status RunGemmA32W32PackedBBlocked(const PackedGemmF32Args& args) noexcept;

/// @brief Returns whether the N-contiguous AVX2 driver amortizes its tile setup.
constexpr bool CanUseF32NContiguousAvx2(int64_t m, int64_t k, int64_t n) noexcept {
    return m >= 4 && k >= 128 && n >= 64;
}

/// @brief Runs the AVX2/FMA FP32 GEMM candidate with compatible fallback.
///
/// The caller must ensure the host supports AVX2 and FMA before invoking this
/// entry point (this primitive performs no capability detection itself). The
/// SIMD paths require nonzero K and unit-stride lhs K and output N.
/// For K-contiguous RHS, small M uses row pairs and a multi-target dot
/// product for an odd row. Larger shapes use transient B packing and a
/// blocked 6x16 microkernel. Eligible N-contiguous RHS shapes use direct
/// vector loads along N. Other layouts delegate to the scalar driver.
Status RunGemmA32W32Avx2Fma(const GemmArgs& args) noexcept;
#endif

/// @brief Ascending-K reference with separate FP32 multiplication and addition.
/// Empty output is a no-op. Zero K writes positive zero without reading A or B.
Status RunGemmA16W16Reference(const GemmA16W16Args& args) noexcept;

#if defined(A16_HAS_AVX2_FMA_KERNEL)
/// @brief AVX2 widening driver for both operand types.
/// Blocked shapes retain FP32 partial sums across K panels. Incompatible input
/// strides and exceptional blocked inputs fall back to the reference.
Status RunGemmA16W16Avx2Fma(const GemmA16W16Args& args) noexcept;
#endif

#if defined(A16_HAS_AVX512BF16_KERNEL)
/// @brief Native BF16 dot product with FP32 accumulation and safety fallback.
Status RunGemmABf16WBf16Avx512Bf16(const GemmA16W16Args& args) noexcept;
#endif

#if defined(A16_HAS_AVX512FP16_KERNEL)
/// @brief FP16 widening and FP32 FMA on the AVX-512 FP16 path.
Status RunGemmAF16WF16Avx512Fp16(const GemmA16W16Args& args) noexcept;
#endif

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_GEMM_GEMM_INTERNAL_H
