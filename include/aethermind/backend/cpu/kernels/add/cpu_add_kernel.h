// CPU Add kernel: backend-internal kernel entry and SDK launch surface.
//
// Exposes two layers:
//   - CpuAddKernel / CpuAddParams: backend-internal KernelFunc + params
//     registered through KernelDescriptor and consumed by the operator layer
//     via Operator::InvokeResolvedKernel (operators never include this header).
//   - LaunchAdd / AddArgs: type-erased SDK entry point for callers that do
//     not depend on TensorView or other engine internals.

#ifndef AETHERMIND_BACKEND_CPU_KERNELS_ADD_CPU_ADD_KERNEL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_ADD_CPU_ADD_KERNEL_H

#include "aethermind/backend/kernel_types.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/dtypes/data_type.h"

namespace aethermind {

/// Backend-specific params struct for the CPU Add kernel.
///
/// Passed to CpuAddKernel via KernelContext::kernel_params. Placement-
/// constructed into a stack-allocated buffer by the KernelParamsBuilder
/// registered with this kernel (BuildCpuAddParams in add_entry.cpp) and
/// consumed by the subsequent KernelFunc call. Operator code never names
/// this type directly; the builder indirection is what keeps operators
/// free of backend headers.
///
/// Lifetime invariant: the TensorView storage referenced by these views
/// must outlive the subsequent CpuAddKernel call that reads them.
struct CpuAddParams {
    TensorView lhs_tensor{};
    TensorView rhs_tensor{};
    MutableTensorView output_tensor{};
};

/// Type-erased arguments for the Add operation, suitable for SDK users
/// who do not depend on TensorView or other engine internals.
///
/// All data pointers are void*; the caller must ensure the backing memory
/// is correctly typed according to @p dtype and is at least @p numel elements
/// long. LaunchAdd assumes flat contiguous buffers (it hardcodes is_flat=true
/// internally); strided or broadcast data must go through the operator layer
/// instead. LaunchAdd validates non-null pointers, supported dtype, and
/// positive numel before dispatching to the internal AddKernel_Scalar.
struct AddArgs {
    const void* lhs_data = nullptr;
    const void* rhs_data = nullptr;
    void* output_data = nullptr;
    DataType dtype;
    int64_t numel = 0;
};

/// Kernel entry point registered via KernelDescriptor::kernel_func for the
/// CPU Add operator.
///
/// Reads a CpuAddParams from ctx.kernel_params (validated non-null), then
/// validates dtypes, shapes, broadcast compatibility, numel, and pointer/
/// offset bounds without dynamic allocation before dispatching to
/// cpu::detail::AddKernel_Scalar. Supported dtypes are governed by
/// kAddSupportedDTypes; mismatches return Status::InvalidArgument.
/// noexcept: errors are reported only through the return value.
AM_NODISCARD Status CpuAddKernel(const KernelContext& ctx) noexcept;

/// Public SDK entry point for the CPU Add operation.
///
/// Accepts type-erased AddArgs, validates the basic preconditions (non-null
/// pointers, supported dtype, positive numel), and forwards to the scalar
/// kernel implementation. Assumes flat contiguous buffers; strided or
/// broadcast data must go through the operator layer instead.
/// noexcept: errors are reported only through the return value.
AM_NODISCARD Status LaunchAdd(const AddArgs& args) noexcept;

}// namespace aethermind

#endif