#ifndef AETHERMIND_BACKEND_OP_KERNEL_CONTEXT_H
#define AETHERMIND_BACKEND_OP_KERNEL_CONTEXT_H

#include "aethermind/backend/backend_capabilities.h"
#include "aethermind/backend/stream.h"
#include "aethermind/backend/tracing_sink.h"
#include "aethermind/backend/workspace_arena.h"
#include "device.h"

#include <cstddef>
#include <span>

namespace aethermind {

struct BackendExecutionResources {
    void* opaque_backend_resources = nullptr;
};

struct OpKernelContext {
    Device device{};
    Stream* stream = nullptr;
    WorkspaceArena* workspace = nullptr;
    TracingSink* tracing = nullptr;
    const BackendCapabilities* caps = nullptr;
    const void* packed_params = nullptr;
    std::span<const std::byte> attrs{};
    const char* debug_name = nullptr;
    BackendExecutionResources backend_resources{};
};

}// namespace aethermind

#endif
