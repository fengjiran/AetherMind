#ifndef AETHERMIND_RUNTIME_WORKSPACE_PLANNER_H
#define AETHERMIND_RUNTIME_WORKSPACE_PLANNER_H

/// @file workspace_planner.h
/// @brief Runtime workspace planning types and helpers.
///
/// Workspace provides temporary scratch memory for operator and kernel execution.
/// Unlike model weights (long-lived) or activations (request-scoped), workspace
/// is runtime scratch storage that can be planned once and reused via
/// offset-based slicing.
///
/// Architecture:
/// - `WorkspaceRequirement`: Describes what a step or operator needs
///   (pure data type, defined in `base/workspace_types.h`)
/// - `PlanWorkspaceRequirements()`: Plans offsets for all requirements into a unified layout
/// - `WorkspaceBinding`: Actual slice handed to a kernel at execution time
///   (defined in `base/workspace_types.h`)
/// - `WorkspacePlanLayout`: Summary of the total workspace size and alignment needs
///
/// The planning phase is done before execution. The binding phase happens when
/// a runtime workspace arena provides the slice for one execution step.
///
/// @see WorkspaceArena, ExecutionContext, ExecutionPlanBuilder

#include "aethermind/base/alignment.h"
#include "aethermind/base/macros.h"
#include "aethermind/base/status.h"
#include "aethermind/base/workspace_types.h"

#include <cstddef>
#include <span>

namespace aethermind {

/// @brief Summary of a workspace plan's total memory needs.
///
/// Returned by PlanWorkspaceRequirements() after computing all offsets.
/// Used to allocate a WorkspaceArena of appropriate size and alignment.
struct WorkspacePlanLayout {
    /// Total bytes needed for the unified workspace.
    /// Sum of all (aligned) WorkspaceRequirement.bytes plus alignment padding.
    size_t total_bytes = 0;

    /// Maximum alignment required across all steps.
    /// The WorkspaceArena base pointer should satisfy this alignment.
    /// Used when allocating the underlying scratch buffer.
    size_t required_alignment = 1;
};

/// @brief Checks if an alignment value is valid for workspace planning.
///
/// Valid alignments are non-zero powers of two (1, 2, 4, 8, 16, ...).
///
/// @param alignment The alignment value to validate.
/// @return True if alignment is valid, false otherwise.
AM_NODISCARD constexpr bool IsValidWorkspaceAlignment(size_t alignment) noexcept {
    return IsValidAlignment(alignment);
}

/// @brief Aligns an offset up to the specified alignment boundary.
///
/// @param offset Current offset (typically end of previous allocation).
/// @param alignment Target alignment (non-zero power of two).
/// @return Aligned offset, kInvalidArgument for invalid alignment, or kOverflow
///         if `offset + alignment - 1` is not representable in size_t.
///         Diagnostic allocation failure returns kResourceExhausted with an
///         empty message.
///
/// @post On success: result >= offset and (result % alignment) == 0.
StatusOr<size_t> AlignWorkspaceOffset(size_t offset,
                                      size_t alignment) noexcept;

/// @brief Plans workspace offsets for a sequence of runtime requirements.
///
/// Given a list of WorkspaceRequirements, computes offsets so each workspace
/// slice is properly aligned within a unified buffer. This allows allocating one
/// contiguous scratch buffer and slicing it per requirement.
///
/// Validates all alignments before checking layout overflow. All validation
/// completes before assigning offsets; successful planning allocates no memory.
///
/// @param requirements List of workspace needs. `offset` field is filled.
/// @return WorkspacePlanLayout with total size and alignment, kInvalidArgument
///         for an invalid alignment, or kOverflow for unrepresentable offsets
///         or total size. Diagnostic allocation failure returns
///         kResourceExhausted with an empty message.
///
/// @pre Caller has exclusive access to requirements for the entire call.
/// @post On success: all offsets are assigned and total_bytes covers all steps
/// @post On failure: requirements.offset unchanged
///
/// Zero-byte semantics:
/// - `bytes == 0` steps receive `offset = total_bytes` but don't advance it.
/// - This allows "position markers" for steps that conceptually participate
///   but don't consume scratch space (e.g., metadata-only operations).
/// - Their alignment requirement is still tracked for `required_alignment`.
StatusOr<WorkspacePlanLayout> PlanWorkspaceRequirements(
        std::span<WorkspaceRequirement> requirements) noexcept;

} // namespace aethermind

#endif
