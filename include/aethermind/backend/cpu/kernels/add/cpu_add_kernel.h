#ifndef AETHERMIND_BACKEND_CPU_KERNELS_ADD_CPU_ADD_KERNEL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_ADD_CPU_ADD_KERNEL_H

#include "aethermind/backend/kernel_types.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/dtypes/data_type.h"

namespace aethermind {

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
/// long. LaunchAdd performs basic validation (non-null pointers, supported
/// dtype, positive numel) and then dispatches to the internal AddKernel_Scalar.
struct AddArgs {
    const void* lhs_data = nullptr;
    const void* rhs_data = nullptr;
    void* output_data = nullptr;
    DataType dtype;
    int64_t numel = 0;
};

AM_NODISCARD Status CpuAddKernel(const KernelContext& ctx) noexcept;

/// Public SDK entry point for the CPU Add operation.
///
/// Accepts type-erased AddArgs, validates the basic preconditions, and
/// forwards to the scalar kernel implementation. This is the stable
/// external interface intended for SDK consumers.
AM_NODISCARD Status LaunchAdd(const AddArgs& args) noexcept;

}// namespace aethermind

#endif