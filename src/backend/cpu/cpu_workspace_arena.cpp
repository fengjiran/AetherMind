#include "aethermind/backend/cpu/cpu_workspace_arena.h"

#include <cstdint>

namespace aethermind {

CpuWorkspaceArena::CpuWorkspaceArena(void* base, size_t size) noexcept
    : base_(base), size_(size) {}

WorkspaceBinding CpuWorkspaceArena::Bind(const WorkspaceRequirement& requirement) noexcept {
    const size_t bytes = requirement.bytes;
    const size_t alignment = requirement.alignment;
    const size_t offset = requirement.offset;

    if (base_ == nullptr || bytes == 0 || alignment == 0 || offset >= size_) {
        return {};
    }

    if (bytes > size_ - offset) {
        return {};
    }

    uint8_t* ptr = static_cast<uint8_t*>(base_) + offset;

    if (reinterpret_cast<uintptr_t>(ptr) % alignment != 0) {
        return {};
    }

    return {.data = ptr, .size = bytes};
}

void CpuWorkspaceArena::Reset() noexcept {
}

void* CpuWorkspaceArena::base() const noexcept {
    return base_;
}

size_t CpuWorkspaceArena::size() const noexcept {
    return size_;
}

}// namespace aethermind
