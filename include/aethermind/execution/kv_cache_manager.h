#ifndef AETHERMIND_EXECUTION_KV_CACHE_MANAGER_H
#define AETHERMIND_EXECUTION_KV_CACHE_MANAGER_H

#include "aethermind/execution/kv_cache_view.h"

namespace aethermind {

class KVCacheManager {
public:
    Status Init(size_t num_layers,
                size_t num_kv_heads,
                size_t max_tokens,
                size_t head_dim,
                DataType kv_dtype,
                size_t alignment = 64);

    AM_NODISCARD StatusOr<KVCacheView> ReserveForSession(size_t prompt_len,
                                                         size_t max_new_tokens) noexcept;
    Status ResetSession(KVCacheView& view) noexcept;
    Status ReleaseSession(KVCacheView& view) noexcept;

    AM_NODISCARD const KVCacheLayout& layout() const noexcept;
    AM_NODISCARD size_t capacity_tokens() const noexcept;
    AM_NODISCARD size_t total_bytes() const noexcept;
    AM_NODISCARD bool is_initialized() const noexcept;

private:
    Status AllocateStorage(size_t bytes_per_plane, size_t alignment);

    KVCacheLayout layout_{};
    KVCacheStorage storage_{};
    SessionKVSlot slot_{};
    size_t total_bytes_ = 0;
    bool initialized_ = false;
};

}// namespace aethermind

#endif
