#ifndef AETHERMIND_TESTS_BENCHMARK_CPU_KERNELS_GEMM_AVX2_CANDIDATE_ENTRIES_H
#define AETHERMIND_TESTS_BENCHMARK_CPU_KERNELS_GEMM_AVX2_CANDIDATE_ENTRIES_H

#include "backend/cpu/kernels/gemm/gemm_internal.h"

// The DupA implementation is compiled into the unit-test and benchmark
// executables from gemm_avx2_candidates.cpp.

namespace aethermind::cpu::detail {

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
/// @brief Blocked 6x16 candidate over a broadcast-packed A panel (each scalar
/// replicated 8x) so the inner loop issues no shuffle-port operations.
/// Measurement entry point only; not registered for production.
Status RunGemmA32W32Avx2FmaTile6DupA(const GemmArgs& args) noexcept;
#endif

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_TESTS_BENCHMARK_CPU_KERNELS_GEMM_AVX2_CANDIDATE_ENTRIES_H
