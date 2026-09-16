#ifndef AETHERMIND_BASE_KV_CACHE_BINDING_H
#define AETHERMIND_BASE_KV_CACHE_BINDING_H

/// @file kv_cache_binding.h
/// @brief Narrow per-invocation KV-cache storage contracts for backend kernels.

#include "aethermind/dtypes/data_type.h"

#include <cstddef>
#include <cstdint>

namespace aethermind {

/// @brief Physical storage description for one decoder layer's key/value planes.
///
/// The pointers borrow manager-owned KV arena storage. They are valid only for
/// the synchronous kernel invocation that receives the enclosing binding and
/// must never be retained in prepared parameters or across Execute() calls.
struct KVCacheLayerStorageBinding {
    std::byte* key_data = nullptr;
    std::byte* value_data = nullptr;
    DataType dtype{};
    uint32_t layer_index = 0;
    size_t num_kv_heads = 0;
    size_t head_dim = 0;
    size_t token_capacity = 0;
    size_t token_stride_bytes = 0;
    size_t head_stride_bytes = 0;
};

/// @brief Write window for one plan-local KV append transaction.
///
/// The update kernel writes exactly `[begin, end)`. `end` is also the local
/// visibility frontier; it is not committed until the whole plan succeeds.
struct KVCacheAppendBinding {
    KVCacheLayerStorageBinding storage{};
    size_t begin = 0;
    size_t end = 0;
};

/// @brief Read window for one plan-local KV transaction.
///
/// `[0, committed_end)` is visible across plans. `[committed_end, visible_end)`
/// may be read only by a later, validated step in this same synchronous plan.
struct KVCacheReadBinding {
    KVCacheLayerStorageBinding storage{};
    size_t committed_end = 0;
    size_t visible_end = 0;
};

} // namespace aethermind

#endif // AETHERMIND_BASE_KV_CACHE_BINDING_H
