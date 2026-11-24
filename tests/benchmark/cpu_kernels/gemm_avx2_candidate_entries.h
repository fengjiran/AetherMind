#ifndef AETHERMIND_TESTS_BENCHMARK_CPU_KERNELS_GEMM_AVX2_CANDIDATE_ENTRIES_H
#define AETHERMIND_TESTS_BENCHMARK_CPU_KERNELS_GEMM_AVX2_CANDIDATE_ENTRIES_H

#include "backend/cpu/kernels/gemm/gemm_internal.h"

namespace aethermind::cpu::detail {

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
/// @brief Explicit 6x16 blocked entry mirroring the production blocked branch
/// of RunGemmA32W32Avx2Fma. Measurement entry point only; not registered as a
/// kernel descriptor.
Status RunGemmA32W32Avx2FmaTile6(const GemmArgs& args) noexcept;

/// @brief Blocked 6x16 candidate over a broadcast-packed A panel (each scalar
/// replicated 8x) so the inner loop issues no shuffle-port operations.
/// Measurement entry point only; not registered for production.
Status RunGemmA32W32Avx2FmaTile6DupA(const GemmArgs& args) noexcept;
#endif

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_TESTS_BENCHMARK_CPU_KERNELS_GEMM_AVX2_CANDIDATE_ENTRIES_H