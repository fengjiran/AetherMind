#ifndef AETHERMIND_RUNTIME_RUNTIME_OPTIONS_H
#define AETHERMIND_RUNTIME_RUNTIME_OPTIONS_H

#include "aethermind/backend/cpu/cpu_capabilities.h"
#include "aethermind/dtypes/data_type.h"

#include <cstddef>

namespace aethermind {

struct AllocatorRuntimeOptions {
    bool enable_cpu = true;
    bool enable_cuda = false;
    bool enable_cann = false;
};

struct BackendRuntimeOptions {
    bool enable_cpu = true;
    /// No built-in GPU backend factories exist. Enabling one of these requires
    /// a custom factory for the same device in the RuntimeBuilder.
    bool enable_cuda = false;
    bool enable_cann = false;
    CpuFeaturePolicy cpu_feature_policy{};
};

struct ExecutionRuntimeOptions {
    /// Reserved execution policy; only the compatibility defaults are accepted.
    bool enable_default_stream = true;
    bool enable_graph_executor = false;
};

struct WorkspaceRuntimeOptions {
    /// Reserved; true is rejected because no workspace manager is assembled.
    bool enable_workspace_manager = false;
    /// Maximum request workspace payload; zero is unlimited. The limit applies
    /// to the Session-owned arena without requiring a workspace manager.
    size_t default_workspace_limit_bytes = 0;
};

struct TracingRuntimeOptions {
    /// Reserved tracing policies; enabling either is currently rejected.
    bool enable_profiling = false;
    bool enable_memory_profiling = false;
};

struct KVCacheRuntimeOptions {
    bool enable_manager = false;
    size_t num_layers = 0;
    size_t num_kv_heads = 0;
    size_t max_tokens = 0;
    size_t head_dim = 0;
    DataType kv_dtype{};
    size_t alignment = 64;
    /// Maximum combined K/V plane payload, including layout padding; zero is unlimited.
    size_t max_bytes = 0;
};

struct RuntimeOptions {
    AllocatorRuntimeOptions allocator;
    BackendRuntimeOptions backend;
    ExecutionRuntimeOptions execution;
    WorkspaceRuntimeOptions workspace;
    TracingRuntimeOptions tracing;
    KVCacheRuntimeOptions kv_cache;
};

} // namespace aethermind

#endif
