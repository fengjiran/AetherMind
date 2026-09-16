#ifndef AETHERMIND_BACKEND_CPU_KERNELS_KVCACHE_UPDATE_KVCACHE_UPDATE_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_KVCACHE_UPDATE_KVCACHE_UPDATE_INTERNAL_H

/// @file kvcache_update_internal.h
/// @brief Prepared FP32 activation operands for the CPU KV cache update kernel.

#include "aethermind/base/kv_cache_binding.h"
#include "aethermind/base/status.h"

#include <cstdint>

namespace aethermind::cpu::detail {

/// @brief Cold-path-stable K/V activation views for one KVCacheUpdate step.
///
/// Cache pointers and append positions are intentionally absent: they are
/// per-invocation state supplied by KernelContext::kv_append.
struct KVCacheUpdateF32KernelArgs {
    const float* key_data = nullptr;
    const float* value_data = nullptr;
    int64_t seq_len = 0;
    int64_t hidden = 0;
    int64_t key_row_stride = 0;
    int64_t key_col_stride = 0;
    int64_t value_row_stride = 0;
    int64_t value_col_stride = 0;
};

/// @brief Runs the prevalidated FP32 cache update for one append window.
Status RunKVCacheUpdateF32Reference(
        const KVCacheUpdateF32KernelArgs& args,
        const KVCacheAppendBinding& append) noexcept;

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_KVCACHE_UPDATE_KVCACHE_UPDATE_INTERNAL_H
