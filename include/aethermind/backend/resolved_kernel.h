#ifndef AETHERMIND_BACKEND_RESOLVED_KERNEL_H
#define AETHERMIND_BACKEND_RESOLVED_KERNEL_H

/// @file resolved_kernel.h
/// @brief Resolved kernel handle produced by planning-time backend selection.
///
/// A `ResolvedKernel` is the frozen result of `Backend::PrepareKernel`. It
/// owns immutable metadata and borrows a stable debug name, and is later
/// owned by an `ExecutionStep` for repeated invocation without further
/// backend lookup.

#include "aethermind/backend/kernel_types.h"
#include "aethermind/backend/packed_weights.h"
#include "aethermind/base/workspace_types.h"
#include "aethermind/operators/op_type.h"

#include <cstddef>
#include <vector>

namespace aethermind {

/// @brief Resolved kernel handle produced by `Backend::PrepareKernel`.
///
/// Holds the type-erased entry point, immutable metadata, scratch-space
/// requirement, and packing recipe for one execution step. The handle is
/// value-typed and is moved into an `ExecutionStep`; `KernelContext`
/// instances borrow `attrs` and `debug_name` for each call.
struct ResolvedKernel {
    /// Operator type resolved by this kernel.
    OpType op_type = OpType::kUnknown;

    /// Type-erased kernel entry point. Set by the backend during resolution.
    KernelFunc fn = nullptr;

    /// Immutable kernel metadata owned by the resolved kernel. Runtime
    /// `KernelContext` instances borrow a span from this storage for each
    /// call.
    std::vector<std::byte> attrs{};

    /// Debug name that borrows backend-owned stable storage. Must stay valid
    /// for the lifetime of the frozen execution plan.
    const char* name = nullptr;

    /// Optional builder for type-erased kernel params; null when param-less.
    KernelParamsBuilder params_builder = nullptr;

    /// Byte size of the params struct; 0 when `params_builder` is null.
    size_t params_size = 0;

    /// Immutable scratch-space requirement of this prepared kernel. Backends
    /// compute it after selecting the concrete implementation; execution
    /// planning assigns its offset before the kernel is frozen into a step.
    WorkspaceRequirement workspace_requirement{};

    /// Packing layout this kernel consumes from packed-weight artifacts. Empty
    /// (default) for non-packed selectors. Execution resolves artifacts by
    /// the exact `{binding, selector, recipe}` key built from this field.
    PackingRecipe expected_packing_recipe{};
};

} // namespace aethermind
#endif
