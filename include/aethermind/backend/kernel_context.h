#ifndef AETHERMIND_BACKEND_KERNEL_CONTEXT_H
#define AETHERMIND_BACKEND_KERNEL_CONTEXT_H

#include "aethermind/backend/backend_capabilities.h"
#include "aethermind/backend/kernel_selector.h"
#include "aethermind/backend/stream.h"
#include "aethermind/backend/tracing_sink.h"
#include "aethermind/execution/workspace_arena.h"
#include "aethermind/operators/op_type.h"
#include "aethermind/runtime/workspace.h"
#include "device.h"

#include <cstddef>
#include <span>

namespace aethermind {

struct BackendExecutionResources {
    void* opaque_backend_resources = nullptr;
};

struct KernelContext {
    OpType op_type = OpType::kUnknown;
    KernelSelector selector{};

    Device device{};
    Stream* stream = nullptr;
    WorkspaceArena* workspace = nullptr;
    WorkspaceBinding workspace_binding{};

    TracingSink* tracing = nullptr;
    const BackendCapabilities* caps = nullptr;

    const void* packed_params = nullptr;
    std::span<const std::byte> attrs{};
    const char* debug_name = nullptr;

    BackendExecutionResources backend_resources{};
};

}// namespace aethermind

#endif
