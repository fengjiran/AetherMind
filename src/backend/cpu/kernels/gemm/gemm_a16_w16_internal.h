#ifndef AETHERMIND_BACKEND_CPU_KERNELS_GEMM_GEMM_A16_W16_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_GEMM_GEMM_A16_W16_INTERNAL_H

#include "aethermind/dtypes/data_type.h"
#include "gemm_internal.h"

namespace aethermind::cpu::detail {

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

/// @brief Ascending-K reference with separate FP32 multiplication and addition.
/// Empty output is a no-op. Zero K writes positive zero without reading A or B.
Status RunGemmA16W16Reference(const GemmA16W16Args& args) noexcept;

/// @brief Tests a contiguous BF16 segment for safe VDPBF16PS execution.
/// Rejected segments require the reference path to preserve gradual underflow
/// and the FP32 product and accumulation semantics.
bool IsBf16NativeDotProductSafe(const std::byte* input,
                                const std::byte* weight,
                                int64_t element_count) noexcept;

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

#endif // AETHERMIND_BACKEND_CPU_KERNELS_GEMM_GEMM_A16_W16_INTERNAL_H
