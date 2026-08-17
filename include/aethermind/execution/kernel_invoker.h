#ifndef AETHERMIND_EXECUTION_KERNEL_INVOKER_H
#define AETHERMIND_EXECUTION_KERNEL_INVOKER_H

#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/resolved_kernel.h"
#include "aethermind/base/status.h"

#include <span>

namespace aethermind {

/// Invokes a frozen kernel with bindings for one ExecutionStep.
///
/// Any kernel params are constructed in stack storage whose lifetime spans
/// exactly this call. Kernels must not retain KernelContext::kernel_params.
AM_NODISCARD Status InvokeKernel(const ResolvedKernel& kernel,
                                 KernelContext& context,
                                 std::span<const TensorView> inputs,
                                 std::span<const MutableTensorView> outputs) noexcept;

}// namespace aethermind

#endif
