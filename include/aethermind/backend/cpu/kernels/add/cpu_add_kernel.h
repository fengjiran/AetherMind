#ifndef AETHERMIND_BACKEND_CPU_KERNELS_ADD_CPU_ADD_KERNEL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_ADD_CPU_ADD_KERNEL_H

#include "aethermind/backend/kernel_types.h"
#include "aethermind/base/tensor_view.h"

namespace aethermind {

struct CpuAddParams {
    TensorView lhs_tensor{};
    TensorView rhs_tensor{};
    MutableTensorView output_tensor{};
};

AM_NODISCARD Status CpuAddKernel(const KernelContext& ctx) noexcept;

}// namespace aethermind

#endif