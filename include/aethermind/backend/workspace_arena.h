#ifndef AETHERMIND_BACKEND_WORKSPACE_ARENA_H
#define AETHERMIND_BACKEND_WORKSPACE_ARENA_H

#include <cstddef>

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

class WorkspaceArena {
public:
    virtual ~WorkspaceArena() = default;
    virtual WorkspaceBinding Bind(const WorkspaceRequirement& requirement) noexcept = 0;
    virtual void Reset() noexcept = 0;
};

}// namespace aethermind
#endif
