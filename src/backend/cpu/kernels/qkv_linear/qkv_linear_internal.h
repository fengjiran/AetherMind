#ifndef AETHERMIND_BACKEND_CPU_KERNELS_QKV_LINEAR_QKV_LINEAR_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_QKV_LINEAR_QKV_LINEAR_INTERNAL_H

#include "aethermind/base/status.h"

#include <cstdint>

namespace aethermind::cpu::detail {

/// @brief Immutable split sizes derived from QkvLinearParams.
struct QkvLinearF32KernelMetadata {
    int64_t q_out_features{};
    int64_t k_out_features{};
    int64_t v_out_features{};
};

/// @brief Pre-validated arguments for the scalar packed FP32 QKV projection.
///
/// `q_weight`, `k_weight`, and `v_weight` reference the Q, K, and V row
/// ranges of the identity-packed `[Q + K + V, in_features]` artifact.
struct QkvLinearF32KernelArgs {
    const float* input{};
    const float* q_weight{};
    const float* k_weight{};
    const float* v_weight{};
    float* query{};
    float* key{};
    float* value{};
    int64_t row_count{};
    int64_t in_features{};
    int64_t q_out_features{};
    int64_t k_out_features{};
    int64_t v_out_features{};
    int64_t input_row_stride{};
    int64_t input_col_stride{1};
    int64_t query_row_stride{};
    int64_t query_col_stride{1};
    int64_t key_row_stride{};
    int64_t key_col_stride{1};
    int64_t value_row_stride{};
    int64_t value_col_stride{1};
};

/// @brief Executes the compute-ready scalar FP32 QKV projection.
Status RunQkvLinearF32Reference(const QkvLinearF32KernelArgs& args) noexcept;

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_QKV_LINEAR_QKV_LINEAR_INTERNAL_H
