/// \file
/// Workspace planning and binding types for execution-time scratch memory.
///
/// Workspace provides temporary scratch memory for kernel execution.
/// Unlike model weights (long-lived) or activations (request-scoped),
/// workspace is execution-scoped: allocated once per inference request
/// and shared across all execution steps via offset-based slicing.
///
/// Architecture:
/// - `WorkspaceRequirement`: Describes what a single step needs (bytes + alignment)
/// - `PlanWorkspaceRequirements()`: Plans offsets for all steps into a unified layout
/// - `WorkspaceBinding`: Actual slice handed to a kernel at execution time
/// - `WorkspacePlanLayout`: Summary of the total workspace size and alignment needs
///
/// The planning phase (PlanWorkspaceRequirements) is done once per ExecutionPlan build.
/// The binding phase (via WorkspaceArena::Bind) happens at each step execution.
///
/// \see WorkspaceArena, RuntimeBindingContext, ExecutionPlanBuilder

#ifndef AETHERMIND_EXECUTION_WORKSPACE_TYPES_H
#define AETHERMIND_EXECUTION_WORKSPACE_TYPES_H

#include "aethermind/base/status.h"
#include "aethermind/utils/overflow_check.h"
#include "macros.h"

#include <algorithm>
#include <cstddef>
#include <span>

namespace aethermind {

/// Describes workspace requirements for a single execution step.
///
/// Used during execution plan building to compute unified workspace layout.
/// The `offset` field is an output: filled by PlanWorkspaceRequirements().
struct WorkspaceRequirement {
    /// Number of scratch bytes this step needs.
    /// Zero-byte requirements consume no space but still receive an offset marker.
    size_t bytes = 0;

    /// Alignment constraint for this step's workspace slice.
    /// Must be a non-zero power of two. Default 64-byte cache-line alignment.
    size_t alignment = 64;

    /// Computed offset into the unified workspace (output field).
    /// Filled by PlanWorkspaceRequirements(). Measured from workspace base.
    size_t offset = 0;
};

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

/// Summary of a workspace plan's total memory needs.
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

/// Checks if an alignment value is valid for workspace planning.
///
/// Valid alignments are non-zero powers of two (1, 2, 4, 8, 16, ...).
///
/// \param alignment The alignment value to validate.
/// \return true if alignment is valid, false otherwise.
AM_NODISCARD constexpr bool IsValidWorkspaceAlignment(size_t alignment) noexcept {
    return alignment != 0 && (alignment & (alignment - 1)) == 0;
}

/// Aligns an offset up to the specified alignment boundary.
///
/// Uses the classic bit-twiddling formula: `(offset + align - 1) & ~(align - 1)`.
/// This matches ammalloc's AlignUp pattern for consistency across the codebase.
///
/// \param offset Current offset (typically end of previous allocation).
/// \param alignment Target alignment (must be non-zero power of two).
/// \return Aligned offset, or error if alignment is invalid or overflow occurs.
///
/// \pre IsValidWorkspaceAlignment(alignment) == true
/// \post Result >= offset and (Result % alignment) == 0
///
/// Overflow handling:
/// - Checks `offset + alignment - 1` for overflow before masking.
/// - Returns kOutOfRange status if overflow would occur.
/// - In practice, workspace sizes (<100MB typical) never trigger this.
AM_NODISCARD inline StatusOr<size_t> AlignWorkspaceOffset(size_t offset,
                                                          size_t alignment) noexcept {
    if (!IsValidWorkspaceAlignment(alignment)) {
        return Status::InvalidArgument("Workspace alignment must be a non-zero power of two");
    }

    size_t t = 0;
    if (CheckOverflowAdd(offset, alignment - 1, &t)) {
        return Status(StatusCode::kOutOfRange, "Workspace offset alignment overflowed size_t");
    }

    return t & ~(alignment - 1);
}

/// Plans workspace offsets for a sequence of execution steps.
///
/// Given a list of WorkspaceRequirements, computes offsets so each step's
/// workspace slice is properly aligned within a unified buffer. This allows
/// allocating one contiguous scratch buffer and slicing it per-step.
///
/// Algorithm:
/// - Validate all alignments first (early exit if any invalid)
/// - Track running `total_bytes` as cumulative end position
/// - For each requirement:
///   - If bytes == 0: record current position but don't advance
///   - Otherwise: align current position, assign offset, advance by bytes
/// - Track max alignment for arena base pointer allocation
///
/// In-place modification:
/// - Fills `requirement.offset` for each input element
/// - If the function returns error, all offsets remain unchanged
///   (validation happens before any mutation)
///
/// \param requirements List of step workspace needs. `offset` field is filled.
/// \return WorkspacePlanLayout with total size and alignment, or error.
///
/// \pre All requirements.alignment must be valid powers of two
/// \post On success: all offsets are assigned and total_bytes covers all steps
/// \post On failure: requirements.offset unchanged
///
/// Zero-byte semantics:
/// - `bytes == 0` steps receive `offset = total_bytes` but don't advance it.
/// - This allows "position markers" for steps that conceptually participate
///   but don't consume scratch space (e.g., metadata-only operations).
/// - Their alignment requirement is still tracked for `required_alignment`.
AM_NODISCARD inline StatusOr<WorkspacePlanLayout> PlanWorkspaceRequirements(
        std::span<WorkspaceRequirement> requirements) noexcept {
    WorkspacePlanLayout layout;

    // Validate all alignments first to avoid partial modification on failure.
    for (const auto& [bytes, alignment, offset]: requirements) {
        if (!IsValidWorkspaceAlignment(alignment)) {
            return Status::InvalidArgument(
                    "Workspace requirement alignment must be a non-zero power of two");
        }
    }

    for (auto& [bytes, alignment, offset]: requirements) {
        layout.required_alignment = std::max(layout.required_alignment, alignment);

        if (bytes == 0) {
            offset = layout.total_bytes;
            continue;
        }

        const StatusOr<size_t> aligned_offset = AlignWorkspaceOffset(layout.total_bytes, alignment);
        if (!aligned_offset.ok()) {
            return aligned_offset.status();
        }

        offset = aligned_offset.value();

        size_t next_total = 0;
        if (CheckOverflowAdd(offset, bytes, &next_total)) {
            return Status(StatusCode::kOutOfRange,
                          "Workspace planning exceeded size_t capacity");
        }

        layout.total_bytes = next_total;
    }

    return layout;
}

}// namespace aethermind

#endif
