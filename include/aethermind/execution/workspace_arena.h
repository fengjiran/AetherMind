#ifndef AETHERMIND_BACKEND_WORKSPACE_ARENA_H
#define AETHERMIND_BACKEND_WORKSPACE_ARENA_H

/// @file workspace_arena.h
/// @brief Abstract workspace allocation arena for execution-time buffers.

#include "aethermind/runtime/workspace.h"

namespace aethermind {

/// @brief Arena that satisfies execution-time workspace requirements.
///
/// Bind() returns a borrowed binding that remains valid until Reset().
/// Implementations decide the allocation strategy (static pre-allocation,
/// ring buffer, etc.).
class WorkspaceArena {
public:
    virtual ~WorkspaceArena() = default;
    /// @brief Binds workspace for a requirement.
    ///
    /// @param requirement Required bytes and alignment.
    /// @return A binding of at least the required size.
    virtual WorkspaceBinding Bind(const WorkspaceRequirement& requirement) noexcept = 0;
    /// @brief Releases all bindings and makes them invalid.
    virtual void Reset() noexcept = 0;
};

}// namespace aethermind
#endif
