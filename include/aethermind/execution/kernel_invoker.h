#ifndef AETHERMIND_EXECUTION_KERNEL_INVOKER_H
#define AETHERMIND_EXECUTION_KERNEL_INVOKER_H

/// @file kernel_invoker.h
/// @brief Low-level frozen-kernel invocation for one execution step.

#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/resolved_kernel.h"
#include "aethermind/base/status.h"

#include <span>

namespace aethermind {

/// @brief Invokes a frozen kernel with bindings for one ExecutionStep.
///
/// @param kernel Frozen kernel prepared by a backend.
/// @param context Kernel execution context; workspace and packed weights are
///                populated by the caller.
/// @param inputs Input tensor views for the step.
/// @param outputs Output tensor views for the step.
/// @return Status::Ok() on success, or the kernel's error status.
/// @note Any kernel params are constructed in stack storage whose lifetime
///       spans exactly this call. Kernels must not retain
///       KernelContext::kernel_params.
Status InvokeKernel(const ResolvedKernel& kernel,
                    KernelContext& context,
                    std::span<const TensorView> inputs,
                    std::span<const MutableTensorView> outputs) noexcept;

} // namespace aethermind

#endif
