#ifndef AETHERMIND_BACKEND_CPU_KERNELS_ADD_RMSNORM_ADD_RMSNORM_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_ADD_RMSNORM_ADD_RMSNORM_INTERNAL_H

/// @file add_rmsnorm_internal.h
/// @brief Compute-ready arguments and reference entry for CPU AddRmsNorm.

#include "aethermind/base/status.h"

#include <cstdint>

namespace aethermind::cpu::detail {

/// @brief Pre-validated FP32 arguments for fused AddRmsNorm.
///
/// The params builders specialize TensorViews once per PreparedExecutionBindings.
/// `new_residual` stores the FP32-rounded addition result before the second
/// pass uses it to compute the normalized output.
struct AddRmsNormF32KernelArgs {
    const float* input{};
    const float* residual{};
    const float* weight{};
    float* output{};
    float* new_residual{};
    int64_t row_count{};
    int64_t hidden_size{};
    int64_t input_row_stride{};
    int64_t input_col_stride{1};
    int64_t residual_row_stride{};
    int64_t residual_col_stride{1};
    int64_t weight_stride{1};
    int64_t output_row_stride{};
    int64_t output_col_stride{1};
    int64_t new_residual_row_stride{};
    int64_t new_residual_col_stride{1};
    float eps{1.0e-5F};
};

/// @brief Executes the scalar FP32 AddRmsNorm reference micro-kernel.
Status RunAddRmsNormF32Reference(const AddRmsNormF32KernelArgs& args) noexcept;

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_ADD_RMSNORM_ADD_RMSNORM_INTERNAL_H
