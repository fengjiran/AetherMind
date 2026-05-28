#ifndef AETHERMIND_BACKEND_CPU_KERNELS_CPU_RMSNORM_KERNEL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_CPU_RMSNORM_KERNEL_H

#include "aethermind/backend/kernel_types.h"
#include "aethermind/base/tensor_view.h"

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

AM_NODISCARD Status CpuRmsNormKernel(const KernelInvocation& invocation,
                                     const KernelContext& op_ctx,
                                     const WorkspaceBinding& workspace) noexcept;

}// namespace aethermind

#endif
