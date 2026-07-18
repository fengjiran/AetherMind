// Public SDK API for the CPU Add kernel.
//
// Provides a type-erased LaunchAdd entry point that accepts AddArgs (void*
// pointers for data, simple scalars for metadata) and dispatches to the
// internal AddKernel_Scalar. This is the stable external interface for
// SDK users who do not depend on TensorView or other engine internals.

#include "add_internal.h"
#include "aethermind/backend/cpu/kernels/add/cpu_add_kernel.h"
#include "aethermind/operators/add_op.h"

namespace aethermind {

Status LaunchAdd(const AddArgs& args) noexcept {
    if (!args.lhs_data || !args.rhs_data || !args.output_data) {
        return Status::InvalidArgument("LaunchAdd requires non-null data pointers");
    }

    if (!IsAddSupportedDType(args.dtype)) {
        return Status::InvalidArgument(
                MakeAddUnsupportedDTypeMessage("LaunchAdd"));
    }

    if (args.numel <= 0) {
        return Status::InvalidArgument("LaunchAdd requires positive numel");
    }

    cpu::detail::AddKernelArgs kernel_args{};
    kernel_args.lhs_data = args.lhs_data;
    kernel_args.rhs_data = args.rhs_data;
    kernel_args.output_data = args.output_data;
    kernel_args.dtype = args.dtype;
    kernel_args.numel = args.numel;
    // LaunchAdd assumes flat contiguous buffers (SDK-facing API).
    kernel_args.is_flat = true;

    return cpu::detail::AddKernel_Scalar(kernel_args);
}

}// namespace aethermind
