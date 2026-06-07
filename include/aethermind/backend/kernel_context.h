#ifndef AETHERMIND_BACKEND_KERNEL_CONTEXT_H
#define AETHERMIND_BACKEND_KERNEL_CONTEXT_H

#include "aethermind/backend/stream.h"
#include "aethermind/execution/workspace_arena.h"
#include "aethermind/runtime/workspace.h"
#include "device.h"

#include <cstddef>
#include <span>

namespace aethermind {

struct KernelContext {
    DeviceType device_type = DeviceType::kUndefined;
    Stream* stream = nullptr;
    WorkspaceArena* workspace = nullptr;
    WorkspaceBinding workspace_binding{};
    const void* packed_weights = nullptr;
    const void* kernel_params = nullptr;
    std::span<const std::byte> attrs{};
};

}// namespace aethermind

#endif
