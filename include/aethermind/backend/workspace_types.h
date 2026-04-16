#ifndef AETHERMIND_BACKEND_WORKSPACE_TYPES_H
#define AETHERMIND_BACKEND_WORKSPACE_TYPES_H

#include "aethermind/base/status.h"
#include "aethermind/utils/overflow_check.h"
#include "macros.h"

#include <algorithm>
#include <cstddef>
#include <span>

namespace aethermind {

struct WorkspaceRequirement {
    size_t bytes = 0;
    size_t alignment = 64;
    size_t offset = 0;
};

struct WorkspaceBinding {
    void* data = nullptr;
    size_t size = 0;
};

struct WorkspacePlanLayout {
    size_t total_bytes = 0;
    size_t required_alignment = 1;
};

AM_NODISCARD constexpr bool IsValidWorkspaceAlignment(size_t alignment) noexcept {
    return alignment != 0 && (alignment & (alignment - 1)) == 0;
}

AM_NODISCARD inline StatusOr<size_t> AlignWorkspaceOffset(size_t offset,
                                                          size_t alignment) noexcept {
    if (!IsValidWorkspaceAlignment(alignment)) {
        return Status::InvalidArgument("Workspace alignment must be a non-zero power of two");
    }

    const size_t mask = alignment - 1;
    const size_t remainder = offset & mask;
    if (remainder == 0) {
        return offset;
    }

    const size_t padding = alignment - remainder;
    size_t aligned_offset = 0;
    if (CheckOverflowAdd(offset, padding, &aligned_offset)) {
        return Status(StatusCode::kOutOfRange, "Workspace offset alignment overflowed size_t");
    }
    return aligned_offset;
}

AM_NODISCARD inline StatusOr<WorkspacePlanLayout> PlanWorkspaceRequirements(
        std::span<WorkspaceRequirement> requirements) noexcept {
    WorkspacePlanLayout layout;

    for (WorkspaceRequirement& requirement: requirements) {
        if (!IsValidWorkspaceAlignment(requirement.alignment)) {
            return Status::InvalidArgument("Workspace requirement alignment must be a non-zero power of two");
        }

        layout.required_alignment = std::max(layout.required_alignment, requirement.alignment);

        if (requirement.bytes == 0) {
            requirement.offset = layout.total_bytes;
            continue;
        }

        const StatusOr<size_t> aligned_offset =
                AlignWorkspaceOffset(layout.total_bytes, requirement.alignment);
        if (!aligned_offset.ok()) {
            return aligned_offset.status();
        }

        requirement.offset = aligned_offset.value();

        size_t next_total = 0;
        if (CheckOverflowAdd(requirement.offset, requirement.bytes, &next_total)) {
            return Status(StatusCode::kOutOfRange,
                          "Workspace planning exceeded size_t capacity");
        }

        layout.total_bytes = next_total;
    }

    return layout;
}

}// namespace aethermind

#endif
