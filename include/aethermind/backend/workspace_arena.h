#ifndef AETHERMIND_BACKEND_WORKSPACE_ARENA_H
#define AETHERMIND_BACKEND_WORKSPACE_ARENA_H

#include "aethermind/backend/workspace_types.h"

namespace aethermind {

class WorkspaceArena {
public:
    virtual ~WorkspaceArena() = default;
    virtual WorkspaceBinding Bind(const WorkspaceRequirement& requirement) noexcept = 0;
    virtual void Reset() noexcept = 0;
};

}// namespace aethermind
#endif
