#ifndef AETHERMIND_BASE_WORKSPACE_H
#define AETHERMIND_BASE_WORKSPACE_H

/// @file workspace.h
/// @brief Base-layer workspace binding types shared by backend and execution.
///
/// WorkspaceRequirement lives in `base/workspace_types.h`; planning (offset
/// assignment) lives in `runtime/workspace.h`. WorkspaceBinding sits in the
/// base layer so that backend and execution contract headers can carry it
/// without depending on each other or on runtime headers.

#include "aethermind/base/macros.h"
#include "aethermind/base/workspace_types.h"

#include <cstddef>

namespace aethermind {

/// Actual workspace slice bound to a kernel at execution time.
///
/// Produced by WorkspaceArena::Bind() using a WorkspaceRequirement's offset.
/// Passed to kernel functions as the third parameter (WorkspaceBinding).
///
/// Lifetime:
/// - Borrowed from the underlying WorkspaceArena
/// - Valid only during that step's execution
/// - Do not store beyond a single kernel invocation
struct WorkspaceBinding {
    /// Pointer to the step's workspace slice.
    /// Already aligned according to WorkspaceRequirement.alignment.
    void* data = nullptr;

    /// Size of the slice in bytes. Matches WorkspaceRequirement.bytes.
    size_t size = 0;
};

}// namespace aethermind

#endif// AETHERMIND_BASE_WORKSPACE_H
