#ifndef AETHERMIND_BACKEND_CPU_KERNELS_QKV_LINEAR_QKV_LINEAR_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_QKV_LINEAR_QKV_LINEAR_INTERNAL_H

#include "aethermind/base/status.h"
#include "aethermind/dtypes/data_type.h"
#include "backend/cpu/kernels/gemm/gemm_internal.h"

#include <cstddef>
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
struct QkvLinearA32W32KernelArgs {
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
Status RunQkvLinearF32Reference(const QkvLinearA32W32KernelArgs& args) noexcept;

/// @brief Pre-validated arguments for the scalar packed W16 QKV projection.
///
/// Weight pointers reference the Q, K, and V byte ranges of the
/// identity-packed `[Q + K + V, in_features]` artifact; strides are element
/// strides and the payload is byte-addressed like the plain W16 Linear args.
struct QkvLinearA32W16KernelArgs {
    const float* input{};
    const std::byte* q_weight{};
    const std::byte* k_weight{};
    const std::byte* v_weight{};
    DataType weight_dtype{};
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

/// @brief Executes the compute-ready scalar W16 QKV projection with FP32
///        accumulation, delegating each slice to the W16 Linear reference.
Status RunQkvLinearF32W16Reference(const QkvLinearA32W16KernelArgs& args) noexcept;

/// @brief Prepared bpanel args for the three output slices of packed QKV.
struct QkvLinearA32W32PackedBKernelArgs {
    PackedGemmF32Args query{};
    PackedGemmF32Args key{};
    PackedGemmF32Args value{};
};

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
Status RunQkvLinearF32PackedB(
        const QkvLinearA32W32PackedBKernelArgs& args) noexcept;
#endif

/// @brief Uses the shared Linear optimized driver and its compatible fallback.
Status RunQkvLinearF32Scalar(const QkvLinearA32W32KernelArgs& args) noexcept;
#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
/// @brief Uses the shared Linear optimized driver and its compatible fallback.
Status RunQkvLinearF32Avx2Fma(const QkvLinearA32W32KernelArgs& args) noexcept;
#endif
#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
/// @brief Uses the shared Linear optimized driver and its compatible fallback.
Status RunQkvLinearF32Bf16Avx2Fma(const QkvLinearA32W16KernelArgs& args) noexcept;
#endif
#if defined(GEMM_HAS_A32_F16_AVX2_FMA_KERNEL)
/// @brief Uses the shared Linear optimized driver and its compatible fallback.
Status RunQkvLinearF32F16Avx2Fma(const QkvLinearA32W16KernelArgs& args) noexcept;
#endif

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_QKV_LINEAR_QKV_LINEAR_INTERNAL_H
