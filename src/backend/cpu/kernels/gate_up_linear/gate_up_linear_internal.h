#ifndef AETHERMIND_BACKEND_CPU_KERNELS_GATE_UP_LINEAR_GATE_UP_LINEAR_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_GATE_UP_LINEAR_GATE_UP_LINEAR_INTERNAL_H

#include "aethermind/base/status.h"
#include "aethermind/dtypes/data_type.h"
#include "backend/cpu/kernels/gemm/gemm_internal.h"

#include <cstddef>
#include <cstdint>

namespace aethermind::cpu::detail {

/// @brief Immutable split sizes derived from GateUpLinearParams.
struct GateUpLinearF32KernelMetadata {
    int64_t gate_out_features{};
    int64_t up_out_features{};
};

/// @brief Pre-validated arguments for the scalar packed FP32 Gate-Up projection.
///
/// `gate_weight` and `up_weight` reference the Gate and Up row ranges of the
/// identity-packed `[gate_out_features + up_out_features, in_features]`
/// artifact. Activation and multiplication remain separate semantic nodes.
struct GateUpLinearF32KernelArgs {
    const float* input{};
    const float* gate_weight{};
    const float* up_weight{};
    float* gate{};
    float* up{};
    int64_t row_count{};
    int64_t in_features{};
    int64_t gate_out_features{};
    int64_t up_out_features{};
    int64_t input_row_stride{};
    int64_t input_col_stride{1};
    int64_t gate_row_stride{};
    int64_t gate_col_stride{1};
    int64_t up_row_stride{};
    int64_t up_col_stride{1};
};

/// @brief Executes the compute-ready scalar FP32 Gate-Up projection.
Status RunGateUpLinearF32Reference(const GateUpLinearF32KernelArgs& args) noexcept;

/// @brief Pre-validated arguments for the scalar packed W16 Gate-Up projection.
///
/// Weight pointers reference the Gate and Up byte ranges of the
/// identity-packed `[gate_out_features + up_out_features, in_features]`
/// artifact; strides are element strides and the payload is byte-addressed.
struct GateUpLinearF32W16KernelArgs {
    const float* input{};
    const std::byte* gate_weight{};
    const std::byte* up_weight{};
    DataType weight_dtype{};
    float* gate{};
    float* up{};
    int64_t row_count{};
    int64_t in_features{};
    int64_t gate_out_features{};
    int64_t up_out_features{};
    int64_t input_row_stride{};
    int64_t input_col_stride{1};
    int64_t gate_row_stride{};
    int64_t gate_col_stride{1};
    int64_t up_row_stride{};
    int64_t up_col_stride{1};
};

/// @brief Executes the compute-ready scalar W16 Gate-Up projection with FP32
///        accumulation, delegating each slice to the W16 Linear reference.
Status RunGateUpLinearF32W16Reference(const GateUpLinearF32W16KernelArgs& args) noexcept;

/// @brief Prepared bpanel args for Gate and Up output slices.
struct GateUpLinearF32PackedBKernelArgs {
    PackedGemmF32Args gate{};
    PackedGemmF32Args up{};
};

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
Status RunGateUpLinearF32PackedB(
        const GateUpLinearF32PackedBKernelArgs& args) noexcept;
#endif

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_GATE_UP_LINEAR_GATE_UP_LINEAR_INTERNAL_H
