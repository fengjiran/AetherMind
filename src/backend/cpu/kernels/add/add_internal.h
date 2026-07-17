/// Internal declarations for the CPU Add kernel (directory-structured).
///
/// Declares the private scalar execution entry point. The public `CpuAddParams`
/// type and `CpuAddKernel` declaration are in the public header. Validation and
/// kernel registration live in `add_entry.cpp`; the scalar implementation lives
/// in `add_scalar.cpp`.

#ifndef AETHERMIND_BACKEND_CPU_KERNELS_ADD_ADD_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_ADD_ADD_INTERNAL_H

#include "aethermind/base/status.h"
#include "aethermind/base/tensor_view.h"

#include <cstdint>

namespace aethermind::cpu::detail {

/// Executes element-wise add via scalar loops for all supported dtypes.
///
/// Selects between a flat loop (when lhs/rhs/output share the same shape and
/// are all contiguous) and a stride-aware scalar loop (for general broadcasts
/// including non-contiguous output). Templates are TU-local in add_scalar.cpp.
///
/// @param lhs    Validated input view.
/// @param rhs    Validated input view.
/// @param output Validated output view (may be non-contiguous).
/// @param numel  Total element count of the output (must be > 0).
Status AddKernel_CPU_Scalar(const TensorView& lhs,
                            const TensorView& rhs,
                            const MutableTensorView& output,
                            int64_t numel) noexcept;

}// namespace aethermind::cpu::detail

#endif// AETHERMIND_BACKEND_CPU_KERNELS_ADD_ADD_INTERNAL_H