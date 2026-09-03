#ifndef AETHERMIND_BACKEND_KERNEL_TYPES_H
#define AETHERMIND_BACKEND_KERNEL_TYPES_H

/// @file kernel_types.h
/// @brief Type-erased backend kernel ABI contracts.
///
/// Defines the function-pointer aliases and the constant that form the boundary
/// between execution's generic kernel invoker and concrete backend kernels: the
/// kernel entry point (`KernelFunc`), the per-step params builder
/// (`KernelParamsBuilder`), the params size limit (`kMaxKernelParamsSize`), and
/// the metadata builder (`KernelMetadataBuilder`).

#include "aethermind/base/status.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/operators/op_params.h"

#include <cstddef>
#include <span>
#include <vector>

namespace aethermind {

struct KernelContext;

/// @brief Type-erased kernel entry point.
///
/// Backends register one `KernelFunc` per kernel via `KernelDescriptor::kernel_func`;
/// The callee reads inputs from `KernelContext::kernel_params` (a `const void*`
/// pointing at a backend-specific params struct) and `KernelContext::attrs`.
/// Kernel entries never assume a concrete parameter type beyond their own
/// registered `KernelParamsBuilder`.
using KernelFunc = Status (*)(const KernelContext&) noexcept;

/// @brief Binding-time context handed to a `KernelParamsBuilder`.
///
/// `inputs` and `outputs` are the per-step TensorViews cached in the caller's
/// `PreparedExecutionBindings`; their data pointers, shape, stride, and dtype are immutable
/// for the bindings' lifetime. `attrs` is the frozen per-kernel metadata owned by
/// the `ResolvedKernel`.
struct KernelParamsBuildContext {
    std::span<const TensorView> inputs{};
    std::span<const MutableTensorView> outputs{};
    std::span<const std::byte> attrs{};
};

/// @brief Cold-path binding specializer constructing a kernel-specific params
/// struct for one step.
///
/// Invoked exactly once per step while `PrepareExecutionBindings` prepares a
/// `PreparedExecutionBindings`, after generic shape and constraint validation. On success
/// the builder placement-constructs its params struct into `params_buffer`,
/// which is owned by the `PreparedExecutionBindings` params arena, aligned to
/// `std::max_align_t`, and sized `KernelDescriptor::params_size` (at most
/// `kMaxKernelParamsSize`).
///
/// Contract:
/// - The builder must not retain the `context` object itself. It may copy
///   context data (including TensorView data pointers) into the prepared
///   params because those views are stable for the whole `PreparedExecutionBindings`
///   lifetime.
/// - The prepared params live exactly as long as the owning `PreparedExecutionBindings`
///   and are consumed via `KernelContext::kernel_params` on every subsequent
///   execution of the step. They must not depend on any per-execution state
///   (sequence position, KV state, workspace); that data flows through
///   `KernelContext`.
/// - The params type must be trivially destructible: the arena is released as
///   a whole with the `PreparedExecutionBindings` and keeps no per-step destruction
///   metadata.
/// - On failure the builder must not have constructed anything into
///   `params_buffer`.
///
/// Registered via `KernelDescriptor::{params_builder, params_size}`. This
/// indirection keeps execution independent of backend-specific parameter
/// structs. `noexcept`: errors are reported only through the return value.
using KernelParamsBuilder = Status (*)(const KernelParamsBuildContext& context,
                                       void* params_buffer) noexcept;

/// @brief Upper bound on the byte size of any params struct passed to
/// `KernelParamsBuilder`.
///
/// Each step's params slot is carved from the `PreparedExecutionBindings` params arena
/// with at most this many bytes. `KernelDescriptor` validation rejects
/// kernels whose `params_size` exceeds this constant. Raising the value
/// increases per-step binding memory.
inline constexpr size_t kMaxKernelParamsSize = 512;

/// @brief Builds immutable backend-specific metadata from typed semantic
/// parameters.
///
/// This runs once while an ExecutionPlan is built, after a concrete kernel has
/// been selected. The output is copied into ResolvedKernel::attrs, so kernels
/// never need to inspect the typed semantic parameter variant during execution.
using KernelMetadataBuilder = Status (*)(const OpParams& params,
                                         std::vector<std::byte>& attrs);

} // namespace aethermind

#endif
