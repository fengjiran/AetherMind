/// Internal declarations for the CPU Add kernel (directory-structured).
///
/// Declares `CpuAddParams` (the kernel parameter struct) and the scalar
/// execution entry point. Validation and kernel registration live in
/// `add_entry.cpp`; the float32 scalar implementation lives in
/// `add_fp32_scalar.cpp`.

#ifndef AETHERMIND_BACKEND_CPU_KERNELS_ADD_ADD_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_ADD_ADD_INTERNAL_H

#include "aethermind/base/status.h"
#include "aethermind/base/tensor_view.h"

namespace aethermind::cpu {

/// Kernel parameters for the CPU Add operator.
struct CpuAddParams {
    TensorView lhs{};
    TensorView rhs{};
    MutableTensorView output{};
};

/// Executes float32 element-wise add via scalar loops.
///
/// Selects between a flat loop (when lhs/rhs/output share the same shape and
/// are all contiguous) and a stride-aware scalar loop (for general
/// broadcasts). The output must be contiguous; this is enforced by the entry
/// function before dispatch.
///
/// @param lhs    Validated Float32 lhs view.
/// @param rhs    Validated Float32 rhs view.
/// @param output Validated Float32 output view (must be contiguous).
/// @param numel  Total element count of the output (must be > 0).
Status AddKernel_CPU_FP32_Scalar(const TensorView& lhs,
                                 const TensorView& rhs,
                                 const MutableTensorView& output,
                                 int64_t numel) noexcept;

}// namespace aethermind::cpu

#endif// AETHERMIND_BACKEND_CPU_KERNELS_ADD_ADD_INTERNAL_H
