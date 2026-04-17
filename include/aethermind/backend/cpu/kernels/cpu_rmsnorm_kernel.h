#ifndef AETHERMIND_BACKEND_CPU_KERNELS_CPU_RMSNORM_KERNEL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_CPU_RMSNORM_KERNEL_H

#include "aethermind/backend/kernel_types.h"

#include <cstddef>

namespace aethermind {

struct CpuRmsNormAttrs {
    float Epsilon = 1e-5F;
};

struct CpuRmsNormParams {
    const float* Input = nullptr;
    const float* Weight = nullptr;
    float* Output = nullptr;
    size_t HiddenSize = 0;
};

AM_NODISCARD Status CpuRmsNormKernel(const KernelInvocation& invocation,
                                     const OpKernelContext& op_ctx,
                                     const WorkspaceBinding& workspace) noexcept;

}// namespace aethermind

#endif
