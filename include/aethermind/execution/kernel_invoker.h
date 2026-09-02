#ifndef AETHERMIND_EXECUTION_KERNEL_INVOKER_H
#define AETHERMIND_EXECUTION_KERNEL_INVOKER_H

/// @file kernel_invoker.h
/// @brief Low-level frozen-kernel invocation for one execution step.

#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/resolved_kernel.h"
#include "aethermind/base/status.h"

#include <span>

namespace aethermind {

/// @brief Invokes a frozen kernel with params prepared at BindingTable build
/// time.
///
/// @param kernel Frozen kernel prepared by a backend.
/// @param context Kernel execution context; workspace and packed weights are
///                populated by the caller.
/// @param prepared_params Params produced by `kernel.params_builder` while the
///                        owning `BindingTable` was built. Must be nullptr when
///                        the kernel registered no builder.
/// @return Status::Ok() on success, or the kernel's error status.
/// @note The prepared params are borrowed for the duration of this call only
///       and remain owned by the `BindingTable`.
Status InvokePreparedKernel(const ResolvedKernel& kernel,
                            KernelContext& context,
                            const void* prepared_params) noexcept;

/// @brief Builds params into stack storage and invokes one frozen kernel.
///
/// Testing and low-level diagnostics helper only: it stack-allocates a params
/// buffer and runs the builder for every call, so production execution paths
/// must never use it. Production code consumes params prepared at
/// BindingTable build time via `InvokePreparedKernel`.
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
Status BuildAndInvokeKernelForTesting(
        const ResolvedKernel& kernel,
        KernelContext& context,
        std::span<const TensorView> inputs,
        std::span<const MutableTensorView> outputs) noexcept;

} // namespace aethermind

#endif
