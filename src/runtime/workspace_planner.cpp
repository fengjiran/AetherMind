#include "aethermind/runtime/workspace_planner.h"
#include "aethermind/base/alignment.h"
#include "utils/overflow_check.h"

#include <algorithm>
#include <new>
#include <utility>

namespace aethermind {

StatusOr<WorkspacePlanLayout> PlanWorkspaceRequirements(
        std::span<WorkspaceRequirement> requirements) noexcept try {
    WorkspacePlanLayout layout;

    // Keep invalid alignment errors ahead of layout overflow errors.
    for (const WorkspaceRequirement& requirement: requirements) {
        if (!IsValidAlignment(requirement.alignment)) {
            return Status::InvalidArgument(
                    "Workspace requirement alignment must be a non-zero power of two");
        }
    }

    // Prove the entire layout before modifying any caller-owned offsets.
    for (const WorkspaceRequirement& requirement: requirements) {
        layout.required_alignment = std::max(layout.required_alignment, requirement.alignment);
        if (requirement.empty()) {
            continue;
        }

        StatusOr<size_t> aligned_offset = AlignOffsetUp(
                layout.total_bytes, requirement.alignment);
        if (!aligned_offset.ok()) {
            return std::move(aligned_offset).status();
        }

        size_t next_total = 0;
        if (CheckOverflowAdd(aligned_offset.value(), requirement.bytes, &next_total)) {
            return Status::Overflow(
                    "Workspace planning exceeded size_t capacity");
        }
        layout.total_bytes = next_total;
    }

    // Exclusive access keeps bytes/alignment stable. The checks above prove
    // these sums safe; keep the same checked path so future changes cannot
    // silently diverge between the validation and commit passes.
    size_t total_bytes = 0;
    for (WorkspaceRequirement& requirement: requirements) {
        if (!requirement.empty()) {
            AM_ASSIGN_OR_RETURN(const size_t aligned_offset,
                                AlignOffsetUp(total_bytes, requirement.alignment));
            total_bytes = aligned_offset;

            size_t next_total = 0;
            if (CheckOverflowAdd(total_bytes, requirement.bytes, &next_total)) {
                return Status::Overflow(
                        "Workspace planning exceeded size_t capacity");
            }
            requirement.offset = total_bytes;
            total_bytes = next_total;
        } else {
            requirement.offset = total_bytes;
        }
    }

    return layout;
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

} // namespace aethermind
