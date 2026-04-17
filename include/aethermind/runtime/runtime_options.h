#ifndef AETHERMIND_RUNTIME_RUNTIME_OPTIONS_H
#define AETHERMIND_RUNTIME_RUNTIME_OPTIONS_H

#include "data_type.h"

#include <cstddef>

namespace aethermind {

struct AllocatorRuntimeOptions {
    bool enable_cpu = true;
    bool enable_cuda = false;
    bool enable_cann = false;
};

struct BackendRuntimeOptions {
    bool enable_cpu = true;
    bool enable_cuda = false;
    bool enable_cann = false;
};

struct ExecutionRuntimeOptions {
    bool enable_default_stream = true;
    bool enable_graph_executor = false;
};

struct WorkspaceRuntimeOptions {
    bool enable_workspace_manager = false;
    size_t default_workspace_limit_bytes = 0;
};

struct TracingRuntimeOptions {
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
};

struct RuntimeOptions {
    AllocatorRuntimeOptions allocator;
    BackendRuntimeOptions backend;
    ExecutionRuntimeOptions execution;
    WorkspaceRuntimeOptions workspace;
    TracingRuntimeOptions tracing;
    KVCacheRuntimeOptions kv_cache;
};

}// namespace aethermind

#endif
