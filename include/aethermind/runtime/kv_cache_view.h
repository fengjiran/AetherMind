#ifndef AETHERMIND_RUNTIME_KV_CACHE_VIEW_H
#define AETHERMIND_RUNTIME_KV_CACHE_VIEW_H

/// @file kv_cache_view.h
/// @brief Borrowed views into KV cache storage with validation and offsets.

#include "aethermind/base/macros.h"
#include "aethermind/base/status.h"
#include "aethermind/dtypes/data_type.h"
#include "aethermind/memory/buffer.h"

namespace aethermind {

/// @brief Cache geometry and byte-offset computation for one key/value plane.
///
/// Each plane stores `num_layers * num_kv_heads * max_tokens * head_dim`
/// elements. `head_dim_stride` may exceed `head_dim` to allow alignment
/// padding between heads.
struct KVCacheLayout {
    size_t num_layers = 0;
    size_t num_kv_heads = 0;
    size_t max_tokens = 0;
    size_t head_dim = 0;
    /// Element stride within a head; >= head_dim when padding is used.
    size_t head_dim_stride = 0;
    /// Byte offset between consecutive sequence positions.
    size_t token_stride = 0;
    /// Byte offset between consecutive KV heads.
    size_t head_stride = 0;
    /// Byte offset between consecutive layers.
    size_t layer_stride = 0;
    DataType kv_dtype{};
    size_t alignment = 64;

    /// @brief Returns the element size of the KV dtype.
    AM_NODISCARD size_t ElementBytes() const noexcept;
    /// @brief Validates the layout geometry and alignment.
    Status Validate() const noexcept;
    /// @brief Computes the byte offset of an element within the plane.
    ///
    /// @param layer_idx Layer index.
    /// @param kv_head_idx KV head index.
    /// @param seq_pos Sequence position.
    /// @param dim_idx Dimension index within the head.
    /// @return The element's byte offset, or an error if an index or stride
    ///         computation is out of range or overflows.
    StatusOr<size_t> Offset(size_t layer_idx,
                            size_t kv_head_idx,
                            size_t seq_pos,
                            size_t dim_idx) const noexcept;
    /// @brief Returns the total bytes of one key/value plane.
    StatusOr<size_t> BytesPerPlane() const noexcept;
};

/// @brief Key and value backing buffers for one cache.
struct KVCacheStorage {
    Buffer key_buffer{};
    Buffer value_buffer{};
    DataType kv_dtype{};
    size_t alignment = 64;

    /// @brief Returns whether both backing buffers are initialized.
    AM_NODISCARD bool is_initialized() const noexcept;
};

/// @brief State of the active session reservation inside a KVCacheManager.
///
/// `generation` is bumped on every reserve/release so stale views can be
/// detected.
struct SessionKVSlot {
    uint64_t generation = 0;
    bool in_use = false;
    size_t capacity_tokens = 0;
    size_t prompt_len = 0;
    size_t current_pos = 0;
};

/// @brief Borrowed, validated view into manager-owned KV cache state.
///
/// Holds non-owning pointers to the manager's layout, storage, and slot.
/// The view is stale (invalid) after the session is released or the slot's
/// generation changes. Reads and writes target the shared storage in place;
/// no copies are made. Not thread-safe.
class KVCacheView {
public:
    KVCacheView() = default;
    /// @brief Binds the view to manager-owned state.
    ///
    /// @param layout Cache geometry, borrowed.
    /// @param storage Key/value buffers, borrowed.
    /// @param slot Active session slot, borrowed.
    KVCacheView(const KVCacheLayout* layout,
                KVCacheStorage* storage,
                SessionKVSlot* slot) noexcept;

    /// @brief Returns whether the view is bound to live, initialized state.
    AM_NODISCARD bool valid() const noexcept;
    /// @brief Returns the physical maximum token capacity.
    AM_NODISCARD size_t max_tokens() const noexcept;
    /// @brief Returns the current commit position (0 when invalid).
    AM_NODISCARD size_t current_pos() const noexcept;
    /// @brief Returns the layer count.
    AM_NODISCARD size_t num_layers() const noexcept;
    /// @brief Returns the KV head count.
    AM_NODISCARD size_t num_kv_heads() const noexcept;
    /// @brief Returns the head dimension.
    AM_NODISCARD size_t head_dim() const noexcept;
    /// @brief Returns the KV cache element dtype.
    AM_NODISCARD DataType kv_dtype() const noexcept;
    /// @brief Returns the reserved session token capacity.
    AM_NODISCARD size_t token_capacity() const noexcept;
    /// @brief Returns the number of committed tokens (== current_pos()).
    AM_NODISCARD size_t committed_tokens() const noexcept;

    /// @brief Validates a contiguous write of `token_count` tokens.
    ///
    /// @param layer_idx Layer index.
    /// @param kv_head_idx KV head index.
    /// @param seq_pos Starting sequence position.
    /// @param token_count Number of tokens written.
    /// @return Status::Ok() if the write fits the reserved and physical
    ///         capacity.
    Status ValidateWrite(size_t layer_idx,
                         size_t kv_head_idx,
                         size_t seq_pos,
                         size_t token_count) const noexcept;
    /// @brief Validates a read over the committed token range.
    ///
    /// @param layer_idx Layer index.
    /// @param kv_head_idx KV head index.
    /// @param seq_begin Inclusive read start.
    /// @param seq_end Exclusive read end.
    /// @return Status::Ok() if the range is committed and in bounds.
    Status ValidateRead(size_t layer_idx,
                        size_t kv_head_idx,
                        size_t seq_begin,
                        size_t seq_end) const noexcept;

    /// @brief Returns a mutable pointer to key data for one token element.
    ///
    /// @param layer_idx Layer index.
    /// @param kv_head_idx KV head index.
    /// @param seq_pos Sequence position.
    /// @param dim_idx Dimension index within the head.
    /// @return Pointer into the shared key buffer after write validation.
    StatusOr<void*> MutableKeyData(size_t layer_idx,
                                   size_t kv_head_idx,
                                   size_t seq_pos,
                                   size_t dim_idx = 0) noexcept;
    /// @brief Returns a mutable pointer to value data for one token element.
    ///
    /// @param layer_idx Layer index.
    /// @param kv_head_idx KV head index.
    /// @param seq_pos Sequence position.
    /// @param dim_idx Dimension index within the head.
    /// @return Pointer into the shared value buffer after write validation.
    StatusOr<void*> MutableValueData(size_t layer_idx,
                                     size_t kv_head_idx,
                                     size_t seq_pos,
                                     size_t dim_idx = 0) noexcept;
    /// @brief Returns a const pointer to key data for one token element.
    ///
    /// @param layer_idx Layer index.
    /// @param kv_head_idx KV head index.
    /// @param seq_pos Sequence position.
    /// @param dim_idx Dimension index within the head.
    /// @return Pointer into the shared key buffer after read validation.
    AM_NODISCARD StatusOr<const void*> KeyData(size_t layer_idx,
                                               size_t kv_head_idx,
                                               size_t seq_pos,
                                               size_t dim_idx = 0) const noexcept;
    /// @brief Returns a const pointer to value data for one token element.
    ///
    /// @param layer_idx Layer index.
    /// @param kv_head_idx KV head index.
    /// @param seq_pos Sequence position.
    /// @param dim_idx Dimension index within the head.
    /// @return Pointer into the shared value buffer after read validation.
    AM_NODISCARD StatusOr<const void*> ValueData(size_t layer_idx,
                                                 size_t kv_head_idx,
                                                 size_t seq_pos,
                                                 size_t dim_idx = 0) const noexcept;

    /// @brief Advances the commit watermark to `new_pos`.
    ///
    /// @param new_pos New commit position; must not move backwards or exceed
    ///                the reserved session capacity.
    /// @return Status::Ok() on success.
    Status CommitUntil(size_t new_pos) noexcept;
    /// @brief Unbinds the view from all manager state, making it invalid.
    void Invalidate() noexcept;

private:
    /// @brief Returns whether the bound slot is still live at this generation.
    AM_NODISCARD bool IsSlotAlive() const noexcept;
    /// @brief Validates that the view is bound to live, initialized state.
    Status ValidateBaseState() const noexcept;
    /// @brief Computes a validated element offset within the key/value plane.
    ///
    /// @param layer_idx Layer index.
    /// @param kv_head_idx KV head index.
    /// @param seq_pos Sequence position.
    /// @param dim_idx Dimension index within the head.
    /// @return The element's byte offset.
    StatusOr<size_t> Offset(size_t layer_idx,
                            size_t kv_head_idx,
                            size_t seq_pos,
                            size_t dim_idx) const noexcept;

    const KVCacheLayout* layout_ = nullptr;
    KVCacheStorage* storage_ = nullptr;
    SessionKVSlot* slot_ = nullptr;
    uint64_t generation_ = 0;
};

} // namespace aethermind

#endif
