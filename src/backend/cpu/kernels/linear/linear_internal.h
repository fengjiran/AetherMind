#ifndef AETHERMIND_BACKEND_CPU_KERNELS_LINEAR_LINEAR_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_LINEAR_LINEAR_INTERNAL_H

#include "aethermind/base/status.h"
#include "aethermind/dtypes/data_type.h"
#include "backend/cpu/kernels/gemm/gemm_internal.h"

#include <cstddef>
#include <cstdint>

namespace aethermind::cpu::detail {

/// @brief Pre-validated FP32 arguments for one Linear kernel invocation.
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
Status RunLinearF32Reference(const LinearF32KernelArgs& args) noexcept;

/// @brief Pre-validated FP32-activation, reduced-precision-weight arguments.
///
/// Weight strides are element strides; `weight` is byte-addressed because raw
/// model storage need not contain live C++ Half/BFloat16 objects or meet their
/// alignment. The reference widens each element and accumulates in FP32.
struct LinearF32W16KernelArgs {
    const float* input{};
    const std::byte* weight{};
    float* output{};
    DataType weight_dtype{};
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

/// @brief Runs the scalar FP32-accumulating F16/BF16 Linear reference kernel.
Status RunLinearF32W16Reference(const LinearF32W16KernelArgs& args) noexcept;

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_LINEAR_LINEAR_INTERNAL_H
