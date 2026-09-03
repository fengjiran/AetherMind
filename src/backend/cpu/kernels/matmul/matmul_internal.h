#ifndef AETHERMIND_BACKEND_CPU_KERNELS_MATMUL_MATMUL_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_MATMUL_MATMUL_INTERNAL_H

/// @file matmul_internal.h
/// @brief Backend-internal compute-ready FP32 MatMul arguments.

#include "aethermind/base/shape_and_stride.h"
#include "aethermind/base/status.h"

#include <array>
#include <cstdint>

namespace aethermind::cpu::detail {

inline constexpr int32_t kMaxMatMulBatchRank =
        static_cast<int32_t>(ShapeAndStride::kMaxRank) - 2;

/// @brief Binding-specialized FP32 MatMul arguments.
///
/// Batch dimensions and input strides are right-aligned and broadcast-
/// normalized. A normalized zero input stride denotes a broadcast dimension.
/// Matrix strides describe the effective `[M, K] x [K, N]` operation, so
/// `transpose_rhs` is intentionally absent from this hot-path representation.
struct MatMulFp32KernelArgs {
    const float* lhs{};
    const float* rhs{};
    float* output{};
    int64_t batch_count{};
    int64_t m{};
    int64_t n{};
    int64_t k{};
    int64_t lhs_m_stride{};
    int64_t lhs_k_stride{};
    int64_t rhs_k_stride{};
    int64_t rhs_n_stride{};
    int64_t output_m_stride{};
    int64_t output_n_stride{};
    int32_t batch_rank{};
    std::array<int64_t, kMaxMatMulBatchRank> batch_dims{};
    std::array<int64_t, kMaxMatMulBatchRank> lhs_batch_strides{};
    std::array<int64_t, kMaxMatMulBatchRank> rhs_batch_strides{};
    std::array<int64_t, kMaxMatMulBatchRank> output_batch_strides{};
};

/// @brief Executes a binding-specialized batched FP32 MatMul reference kernel.
AM_NODISCARD Status RunMatMulFp32Reference(const MatMulFp32KernelArgs& args) noexcept;

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_MATMUL_MATMUL_INTERNAL_H
