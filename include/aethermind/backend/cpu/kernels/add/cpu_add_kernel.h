// CPU Add kernel: SDK launch surface.
//
// Exposes LaunchAdd / AddArgs: a type-erased SDK entry point for callers
// that do not depend on TensorView or other engine internals. The backend-
// internal kernel entry (AddKernel) and params struct (AddParams) live in
// src/backend/cpu/kernels/add/add_internal.h and are never included by
// operator code.

#ifndef AETHERMIND_BACKEND_CPU_KERNELS_ADD_CPU_ADD_KERNEL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_ADD_CPU_ADD_KERNEL_H

#include "aethermind/backend/kernel_types.h"
#include "aethermind/dtypes/data_type.h"

namespace aethermind {

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