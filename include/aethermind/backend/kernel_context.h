#ifndef AETHERMIND_BACKEND_KERNEL_CONTEXT_H
#define AETHERMIND_BACKEND_KERNEL_CONTEXT_H

#include "aethermind/backend/stream.h"
#include "aethermind/base/device.h"
#include "aethermind/base/kv_cache_binding.h"
#include "aethermind/base/workspace.h"
#include "aethermind/base/workspace_arena.h"

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
    /// Per-invocation KV append binding. Borrowed storage is valid only while
    /// this synchronous kernel call is active and must never be cached.
    const KVCacheAppendBinding* kv_append = nullptr;
    /// Per-invocation KV read binding. Borrowed storage is valid only while
    /// this synchronous kernel call is active and must never be cached.
    const KVCacheReadBinding* kv_read = nullptr;
    std::span<const std::byte> attrs{};
};

} // namespace aethermind

#endif
