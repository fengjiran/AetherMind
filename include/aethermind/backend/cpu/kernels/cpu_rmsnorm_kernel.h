#ifndef AETHERMIND_BACKEND_CPU_KERNELS_CPU_RMSNORM_KERNEL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_CPU_RMSNORM_KERNEL_H

#include "aethermind/backend/kernel_types.h"
#include "aethermind/base/tensor_view.h"

#include <cstdint>

namespace aethermind {

struct CpuRmsNormAttrs {
    float Epsilon = 1e-5F;
};
static_assert(sizeof(CpuRmsNormAttrs) == sizeof(float));
static_assert(alignof(CpuRmsNormAttrs) <= alignof(float));

struct CpuRmsNormParams {
    TensorView Input{};
    TensorView Weight{};
    MutableTensorView Output{};
};

struct CpuRmsNormKernelArgs {
    const float* input_{};
    const float* weight_{};
    float* output_{};
    int64_t seq_len_{};
    int64_t hidden_size_{};
    int64_t input_row_stride_{};
    int64_t input_col_stride_{1};
    int64_t weight_stride_{1};
    int64_t output_row_stride_{};
    int64_t output_col_stride_{1};
    float epsilon_{1.0e-5F};
};

AM_NODISCARD Status CpuRmsNormKernel(const CpuRmsNormKernelArgs& args) noexcept;

AM_NODISCARD Status CpuRmsNormKernelEntry(const KernelContext& ctx) noexcept;

}// namespace aethermind

#endif
