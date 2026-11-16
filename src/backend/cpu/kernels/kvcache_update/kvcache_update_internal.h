#ifndef AETHERMIND_BACKEND_CPU_KERNELS_KVCACHE_UPDATE_KVCACHE_UPDATE_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_KVCACHE_UPDATE_KVCACHE_UPDATE_INTERNAL_H

/// @file kvcache_update_internal.h
/// @brief Prepared FP32 activation operands for the CPU KV cache update kernel.

#include "aethermind/base/kv_cache_binding.h"
#include "aethermind/base/status.h"
#include "aethermind/dtypes/data_type.h"

#include <cstddef>
#include <cstdint>

namespace aethermind::cpu::detail {

/// @brief Cold-path-stable K/V activation views for one KVCacheUpdate step.
///
/// Cache pointers and append positions are intentionally absent: they are
/// per-invocation state supplied by KernelContext::kv_append.
struct KVCacheUpdateA32KernelArgs {
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
        const KVCacheUpdateA32KernelArgs& args,
        const KVCacheAppendBinding& append) noexcept;

/// @brief Narrows F32 K/V activations to F16/BF16 and writes one append window.
///
/// The activation operands are the same F32 views as the FP32 path; only the
/// cache element dtype differs. `kv_dtype` must be scalar Float(16) or
/// BFloat(16) and match the append binding's storage dtype (validated by the
/// caller). Writes are byte-safe and touch only the logical head_dim.
Status RunKVCacheUpdateKV16Reference(
        const KVCacheUpdateA32KernelArgs& args,
        const KVCacheAppendBinding& append,
        DataType kv_dtype) noexcept;

/// @brief Cold-path-stable A16 K/V activation views for one append window.
///
/// Strides are element strides in 16-bit storage. The cache and the activation
/// share the kernel dtype, so the update is a write-through copy with no
/// rounding (frozen P4 placement).
struct KVCacheUpdateA16KernelArgs {
    const std::byte* key_data{};
    const std::byte* value_data{};
    DataType act_dtype{};
    int64_t seq_len{};
    int64_t hidden{};
    int64_t key_row_stride{};
    int64_t key_col_stride{};
    int64_t value_row_stride{};
    int64_t value_col_stride{};
};

/// @brief Copies one A16 append window into an A16 cache without rounding.
///
/// `act_dtype` must be scalar Float(16) or BFloat(16) and match the append
/// binding's storage dtype (validated by the caller). Writes are byte-safe and
/// touch only the logical head_dim.
Status RunKVCacheUpdateA16Reference(
        const KVCacheUpdateA16KernelArgs& args,
        const KVCacheAppendBinding& append) noexcept;

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_KVCACHE_UPDATE_KVCACHE_UPDATE_INTERNAL_H
