#ifndef AETHERMIND_EXECUTION_KV_CACHE_VIEW_H
#define AETHERMIND_EXECUTION_KV_CACHE_VIEW_H

#include "aethermind/base/status.h"
#include "aethermind/memory/buffer.h"
#include "data_type.h"
#include "macros.h"

namespace aethermind {

struct KVLayoutContract {
    size_t num_layers = 0;
    size_t num_kv_heads = 0;
    size_t max_tokens = 0;
    size_t head_dim = 0;
    size_t head_dim_stride = 0;
    size_t token_stride = 0;
    size_t head_stride = 0;
    size_t layer_stride = 0;
    DataType kv_dtype{};
    size_t alignment = 64;

    AM_NODISCARD size_t ElementBytes() const noexcept;
    AM_NODISCARD Status Validate() const noexcept;
    AM_NODISCARD StatusOr<size_t> Offset(size_t layer_idx,
                                         size_t kv_head_idx,
                                         size_t seq_pos,
                                         size_t dim_idx) const noexcept;
    AM_NODISCARD StatusOr<size_t> BytesPerPlane() const noexcept;
};

struct KVCacheStorage {
    Buffer key_buffer{};
    Buffer value_buffer{};
    DataType kv_dtype{};
    size_t alignment = 64;

    AM_NODISCARD bool is_initialized() const noexcept;
};

struct SessionKVSlot {
    uint64_t generation = 0;
    bool in_use = false;
    size_t capacity_tokens = 0;
    size_t prompt_len = 0;
    size_t current_pos = 0;
};

class KVCacheView {
public:
    KVCacheView() = default;
    KVCacheView(const KVLayoutContract* layout,
                KVCacheStorage* storage,
                SessionKVSlot* slot) noexcept;

    AM_NODISCARD bool valid() const noexcept;
    AM_NODISCARD size_t max_tokens() const noexcept;
    AM_NODISCARD size_t current_pos() const noexcept;
    AM_NODISCARD size_t num_layers() const noexcept;
    AM_NODISCARD size_t num_kv_heads() const noexcept;
    AM_NODISCARD size_t head_dim() const noexcept;
    AM_NODISCARD size_t token_capacity() const noexcept;
    AM_NODISCARD size_t committed_tokens() const noexcept;

    AM_NODISCARD Status ValidateWrite(size_t layer_idx,
                                      size_t kv_head_idx,
                                      size_t seq_pos,
                                      size_t token_count) const noexcept;
    AM_NODISCARD Status ValidateRead(size_t layer_idx,
                                     size_t kv_head_idx,
                                     size_t seq_begin,
                                     size_t seq_end) const noexcept;

    AM_NODISCARD StatusOr<void*> MutableKeyData(size_t layer_idx,
                                                size_t kv_head_idx,
                                                size_t seq_pos,
                                                size_t dim_idx = 0) noexcept;
    AM_NODISCARD StatusOr<void*> MutableValueData(size_t layer_idx,
                                                  size_t kv_head_idx,
                                                  size_t seq_pos,
                                                  size_t dim_idx = 0) noexcept;
    AM_NODISCARD StatusOr<const void*> KeyData(size_t layer_idx,
                                               size_t kv_head_idx,
                                               size_t seq_pos,
                                               size_t dim_idx = 0) const noexcept;
    AM_NODISCARD StatusOr<const void*> ValueData(size_t layer_idx,
                                                 size_t kv_head_idx,
                                                 size_t seq_pos,
                                                 size_t dim_idx = 0) const noexcept;

    AM_NODISCARD Status CommitUntil(size_t new_pos) noexcept;
    void Invalidate() noexcept;

private:
    AM_NODISCARD bool IsSlotAlive() const noexcept;
    AM_NODISCARD Status ValidateBaseState() const noexcept;
    AM_NODISCARD StatusOr<size_t> Offset(size_t layer_idx,
                                         size_t kv_head_idx,
                                         size_t seq_pos,
                                         size_t dim_idx) const noexcept;

    const KVLayoutContract* layout_ = nullptr;
    KVCacheStorage* storage_ = nullptr;
    SessionKVSlot* slot_ = nullptr;
    uint64_t generation_ = 0;
};

}// namespace aethermind

#endif
