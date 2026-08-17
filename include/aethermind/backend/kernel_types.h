#ifndef AETHERMIND_BACKEND_KERNEL_TYPES_H
#define AETHERMIND_BACKEND_KERNEL_TYPES_H

#include "aethermind/base/status.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/operators/op_params.h"

#include <cstddef>
#include <span>
#include <vector>

namespace aethermind {

struct KernelContext;

/// Type-erased kernel entry point.
///
/// Backends register one `KernelFunc` per kernel via `KernelDescriptor::kernel_func`.
/// The callee reads inputs from `KernelContext::kernel_params` (a `const void*`
/// pointing at a backend-specific params struct) and `KernelContext::attrs`.
/// Kernel entries never assume a concrete parameter type beyond their own
/// registered `KernelParamsBuilder`.
using KernelFunc = Status (*)(const KernelContext&) noexcept;

/// Backend-registered function that constructs a kernel-specific params struct
/// from the current step's tensor bindings.
///
/// `inputs` and `outputs` come from the current step's runtime bindings. On
/// success the builder placement-constructs its params
/// struct into `params_buffer`, which is caller-owned, stack-allocated, aligned
/// to `std::max_align_t`, and has capacity `kMaxKernelParamsSize` bytes.
///
/// Lifetime invariant: the constructed params must remain valid for the
/// duration of the subsequent `KernelFunc` call that consumes
/// `KernelContext::kernel_params`. The params type must be trivially
/// destructible because the generic invoker does not retain type-erased
/// destruction metadata for its stack storage.
///
/// Registered via `KernelDescriptor::{params_builder, params_size}` and invoked
/// by execution's generic kernel invoker. This indirection keeps execution
/// independent of backend-specific parameter structs.
///
/// Returns `Status::InvalidArgument` on input/output arity mismatch. `noexcept`:
/// errors are reported only through the return value.
using KernelParamsBuilder = Status (*)(std::span<const TensorView> inputs,
                                       std::span<const MutableTensorView> outputs,
                                       void* params_buffer) noexcept;

/// Upper bound on the byte size of any params struct passed to `KernelParamsBuilder`.
///
/// The generic kernel invoker stack-allocates a buffer of this size before
/// calling the builder. `KernelDescriptor` validation rejects kernels whose
/// `params_size` exceeds this constant. Raising the value increases per-call
/// stack usage for every kernel step.
inline constexpr size_t kMaxKernelParamsSize = 512;

/// Builds immutable backend-specific metadata from typed semantic parameters.
///
/// This runs once while an ExecutionPlan is built, after a concrete kernel has
/// been selected. The output is copied into ResolvedKernel::attrs, so kernels
/// never need to inspect the typed semantic parameter variant during execution.
using KernelMetadataBuilder = Status (*)(const OpParams& params,
                                         std::vector<std::byte>& attrs);

}// namespace aethermind

#endif
