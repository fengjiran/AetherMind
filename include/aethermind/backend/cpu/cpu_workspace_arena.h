#ifndef AETHERMIND_BACKEND_CPU_CPU_WORKSPACE_ARENA_H
#define AETHERMIND_BACKEND_CPU_CPU_WORKSPACE_ARENA_H

#include "aethermind/execution/workspace_arena.h"
#include "macros.h"

#include <cstddef>

namespace aethermind {

class CpuWorkspaceArena final : public WorkspaceArena {
public:
    CpuWorkspaceArena(void* base, size_t size) noexcept;
    WorkspaceBinding Bind(const WorkspaceRequirement& requirement) noexcept override;
    void Reset() noexcept override;
    AM_NODISCARD void* base() const noexcept;
    AM_NODISCARD size_t size() const noexcept;

private:
    void* base_ = nullptr;
    size_t size_ = 0;
};

}// namespace aethermind
#endif
