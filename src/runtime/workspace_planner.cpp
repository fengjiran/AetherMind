#include "aethermind/runtime/workspace_planner.h"
#include "utils/overflow_check.h"

#include <algorithm>
#include <new>
#include <utility>

namespace aethermind {

StatusOr<size_t> AlignWorkspaceOffset(size_t offset,
                                      size_t alignment) noexcept try {
    if (!IsValidWorkspaceAlignment(alignment)) {
        return Status::InvalidArgument(
                "Workspace alignment must be a non-zero power of two");
    }

    size_t rounded_offset = 0;
    if (CheckOverflowAdd(offset, alignment - 1, &rounded_offset)) {
        return Status::Overflow(
                "Workspace offset alignment overflowed size_t");
    }

    return rounded_offset & ~(alignment - 1);
} catch (const std::bad_alloc&) {
    // Error diagnostics may allocate; reporting OOM must not allocate again.
    return Status::ResourceExhausted({});
}

StatusOr<WorkspacePlanLayout> PlanWorkspaceRequirements(
        std::span<WorkspaceRequirement> requirements) noexcept try {
    WorkspacePlanLayout layout;

    // Keep invalid alignment errors ahead of layout overflow errors.
    for (const WorkspaceRequirement& requirement: requirements) {
        if (!IsValidWorkspaceAlignment(requirement.alignment)) {
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

        StatusOr<size_t> aligned_offset = AlignWorkspaceOffset(
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
    // these sums safe, so committing offsets cannot fail or allocate.
    size_t total_bytes = 0;
    for (WorkspaceRequirement& requirement: requirements) {
        if (!requirement.empty()) {
            total_bytes = (total_bytes + (requirement.alignment - 1)) & ~(requirement.alignment - 1);
        }
        requirement.offset = total_bytes;
        total_bytes += requirement.bytes;
    }

    return layout;
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

} // namespace aethermind
