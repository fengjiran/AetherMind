/// Internal declarations for the CPU ElementwiseMul kernel.
///
/// Declares the backend-internal kernel entry point (ElementwiseMulKernel)
/// and its params struct (ElementwiseMulParams). Operator code never
/// includes this header; the KernelParamsBuilder indirection keeps
/// operators free of backend internals.

#ifndef AETHERMIND_BACKEND_CPU_KERNELS_ELEMENTWISE_MUL_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_ELEMENTWISE_MUL_INTERNAL_H

#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/base/tensor_view.h"

namespace aethermind::cpu::detail {

/// Backend-internal params struct for the CPU ElementwiseMul kernel.
///
/// Placement-constructed into a stack-allocated buffer by the
/// KernelParamsBuilder registered with this kernel (BuildElementwiseMulParams
/// in elementwise_mul_entry.cpp) and consumed by the subsequent
/// ElementwiseMulKernel call via KernelContext::kernel_params.
struct ElementwiseMulParams {
    TensorView lhs_tensor{};
    TensorView rhs_tensor{};
    MutableTensorView output_tensor{};
};

/// Kernel entry point registered via KernelDescriptor::kernel_func.
AM_NODISCARD Status ElementwiseMulKernel(const KernelContext& ctx) noexcept;

}// namespace aethermind::cpu::detail

#endif// AETHERMIND_BACKEND_CPU_KERNELS_ELEMENTWISE_MUL_INTERNAL_H
