/// Internal declarations for the CPU Add kernel.
///
/// Declares the backend-internal kernel entry point (AddKernel), its params
/// struct (AddParams), and the scalar implementation layer (AddKernelArgs +
/// AddKernel_Scalar). Operator code never includes this header; the
/// KernelParamsBuilder indirection keeps operators free of backend internals.

#ifndef AETHERMIND_BACKEND_CPU_KERNELS_ADD_ADD_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_ADD_ADD_INTERNAL_H

#include "aethermind/backend/kernel_types.h"
#include "aethermind/base/shape_and_stride.h"
#include "aethermind/base/status.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/dtypes/data_type.h"

#include <array>

namespace aethermind::cpu::detail {

constexpr uint32_t kMaxRank = ShapeAndStride::kMaxRank;

/// Backend-internal params struct for the CPU Add kernel.
///
/// Placement-constructed into a stack-allocated buffer by the
/// KernelParamsBuilder registered with this kernel (BuildAddParams in
/// add_entry.cpp) and consumed by the subsequent AddKernel call via
/// KernelContext::kernel_params. Operator code never names this type
/// directly; the builder indirection is what keeps operators free of
/// backend headers.
///
/// Lifetime invariant: the TensorView storage referenced by these views
/// must outlive the subsequent AddKernel call that reads them.
struct AddParams {
    TensorView lhs_tensor{};
    TensorView rhs_tensor{};
    MutableTensorView output_tensor{};
};

/// Pre-validated, type-erased kernel arguments for the Add implementation.
///
/// Produced by the entry layer (ValidateAndBuildArgs) and consumed by the
/// implementation layer (AddKernel_Scalar). Separates the "what to validate"
/// concern from the "how to compute" concern.
struct AddKernelArgs {
    const void* lhs_data = nullptr;
    const void* rhs_data = nullptr;
    void* output_data = nullptr;
    DataType dtype;
    int64_t numel = 0;
    bool is_flat = false;

    // Broadcast / strided path fields (only used when !is_flat).
    int32_t lhs_rank = 0;
    int32_t rhs_rank = 0;
    int32_t output_rank = 0;
    std::array<int64_t, kMaxRank> lhs_shape{};
    std::array<int64_t, kMaxRank> lhs_strides{};
    std::array<int64_t, kMaxRank> rhs_shape{};
    std::array<int64_t, kMaxRank> rhs_strides{};
    std::array<int64_t, kMaxRank> output_shape{};
    std::array<int64_t, kMaxRank> output_strides{};
};

/// Kernel entry point registered via KernelDescriptor::kernel_func.
///
/// Reads an AddParams from ctx.kernel_params, validates dtypes/shapes/
/// broadcast/numel/pointers without dynamic allocation, then dispatches to
/// AddKernel_Scalar. noexcept: errors are reported only through the return
/// value.
Status AddKernel(const KernelContext& ctx) noexcept;

/// Executes element-wise add via scalar loops for all supported dtypes.
///
/// Selects between a flat loop (when is_flat is true) and a stride-aware
/// scalar loop (for general broadcasts including non-contiguous output).
/// Templates are TU-local in add_scalar_impl.h, instantiated in add_scalar.cpp.
Status AddKernel_Scalar(const AddKernelArgs& args) noexcept;

}// namespace aethermind::cpu::detail

#endif// AETHERMIND_BACKEND_CPU_KERNELS_ADD_ADD_INTERNAL_H
