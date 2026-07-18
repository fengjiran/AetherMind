#ifndef AETHERMIND_BACKEND_KERNEL_TYPES_H
#define AETHERMIND_BACKEND_KERNEL_TYPES_H

#include "aethermind/base/status.h"
#include "aethermind/base/tensor_view.h"

#include <cstddef>
#include <span>

namespace aethermind {

struct KernelContext;

/// Type-erased kernel entry point.
///
/// Backends register one `KernelFunc` per kernel via `KernelDescriptor::kernel_func`.
/// The callee reads inputs from `KernelContext::kernel_params` (a `const void*`
/// pointing at a backend-specific params struct) and `KernelContext::attrs`.
/// Operators never assume a concrete params type; they populate `kernel_params`
/// indirectly through `KernelParamsBuilder`.
using KernelFunc = Status (*)(const KernelContext&) noexcept;

/// Backend-registered function that constructs a kernel-specific params struct
/// from operator-supplied tensor bindings.
///
/// `inputs` and `outputs` come from the operator's `RuntimeBindingContext` for
/// the current step. On success the builder placement-constructs its params
/// struct into `params_buffer`, which is caller-owned, stack-allocated, aligned
/// to `std::max_align_t`, and has capacity `kMaxKernelParamsSize` bytes.
///
/// Lifetime invariant: the constructed params must remain valid for the
/// duration of the subsequent `KernelFunc` call that consumes
/// `KernelContext::kernel_params`.
///
/// Registered via `KernelDescriptor::{params_builder, params_size}` and invoked
/// by `Operator::InvokeResolvedKernel`. This indirection is what lets operator
/// code call kernels without depending on backend-specific param structs.
///
/// Returns `Status::InvalidArgument` on input/output arity mismatch. `noexcept`:
/// errors are reported only through the return value.
using KernelParamsBuilder = Status (*)(std::span<const TensorView> inputs,
                                       std::span<const MutableTensorView> outputs,
                                       void* params_buffer) noexcept;

/// Upper bound on the byte size of any params struct passed to `KernelParamsBuilder`.
///
/// `Operator::InvokeResolvedKernel` stack-allocates a buffer of this size before
/// calling the builder. `KernelDescriptor` validation rejects kernels whose
/// `params_size` exceeds this constant. Raising the value increases per-call
/// stack usage for every operator `Run`.
inline constexpr size_t kMaxKernelParamsSize = 512;

}// namespace aethermind

#endif
