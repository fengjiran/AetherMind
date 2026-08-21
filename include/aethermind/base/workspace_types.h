#ifndef AETHERMIND_BASE_WORKSPACE_TYPES_H
#define AETHERMIND_BASE_WORKSPACE_TYPES_H

/// @file workspace_types.h
/// @brief Workspace requirement data types shared by backend preparation and
/// execution planning.
///
/// Pure data: `WorkspaceRequirement` only describes how much scratch memory a
/// step needs and how it may be reused. Planning (offset assignment) happens
/// in the runtime workspace header. The types live in the base layer so
/// execution contract headers can carry them without depending on runtime
/// headers.

#include "aethermind/base/macros.h"

#include <cstddef>

namespace aethermind {

/// @brief Scratch-memory lifetime scope for a workspace requirement.
enum class WorkspaceLifetime {
    /// No workspace is required.
    kNone = 0,
    /// Scratch space is needed only during one operator invocation.
    kPerOperator,
    /// Scratch space may be reused across operators within one model layer.
    kPerLayer,
    /// Scratch space is scoped to one token step, such as decode-time temporary buffers.
    kPerToken,
    /// Scratch space is scoped to one request sequence, such as prefill/decode shared buffers.
    kPerSequence,
    /// Workspace must remain valid for the owning runtime object's lifetime.
    kPersistent,
};

/// @brief Describes workspace requirements for a single runtime step or
/// operator.
///
/// The prepared backend kernel sets `bytes`, `alignment`, `lifetime`, and
/// `reusable`. The `offset` field is a planning result filled by
/// PlanWorkspaceRequirements().
struct WorkspaceRequirement {
    /// Number of scratch bytes this requirement needs.
    /// Zero-byte requirements consume no space but still receive an offset marker.
    size_t bytes = 0;

    /// Alignment constraint for this workspace slice.
    /// Must be a non-zero power of two. Default 64-byte cache-line alignment.
    size_t alignment = 64;

    /// Lifetime scope for this workspace requirement.
    WorkspaceLifetime lifetime = WorkspaceLifetime::kNone;

    /// Whether this workspace slice may be reused by later compatible requirements.
    bool reusable = true;

    /// Computed offset into the unified workspace (output field).
    /// Filled by PlanWorkspaceRequirements(). Measured from workspace base.
    size_t offset = 0;

    AM_NODISCARD bool empty() const noexcept {
        return bytes == 0;
    }
};

}// namespace aethermind

#endif// AETHERMIND_BASE_WORKSPACE_TYPES_H
