#ifndef AETHERMIND_BACKEND_CPU_KERNELS_LINEAR_LINEAR_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_LINEAR_LINEAR_INTERNAL_H

#include "aethermind/base/status.h"

#include <cstdint>

namespace aethermind::cpu::detail {

/// @brief Pre-validated FP32 arguments for the scalar Linear reference kernel.
///
/// Produced once by the registered KernelParamsBuilder and consumed on every
/// execution. `row_count` flattens input leading dimensions; rank-1 input is
/// represented by one row. The logical mapping is
/// `output[row, out] = sum(input[row, in] * weight[out, in])`.
struct LinearF32KernelArgs {
    const float* input{};
    const float* weight{};
    float* output{};
    int64_t row_count{};
    int64_t in_features{};
    int64_t out_features{};
    int64_t input_row_stride{};
    int64_t input_col_stride{1};
    int64_t weight_row_stride{};
    int64_t weight_col_stride{1};
    int64_t output_row_stride{};
    int64_t output_col_stride{1};
};

/// @brief Runs the scalar FP32 Linear reference kernel.
///
/// `args` must have been produced by the binding-time params builder. A zero
/// `row_count` or `out_features` is a no-op. With zero `in_features`, the
/// kernel writes zero to every output element without reading input or weight.
AM_NODISCARD Status RunLinearF32Reference(const LinearF32KernelArgs& args) noexcept;

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_LINEAR_LINEAR_INTERNAL_H
