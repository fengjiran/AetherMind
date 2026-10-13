#ifndef AETHERMIND_KV_CACHE_KV_CACHE_SESSION_VIEW_H
#define AETHERMIND_KV_CACHE_KV_CACHE_SESSION_VIEW_H

/// @file kv_cache/kv_cache_session_view.h
/// @brief Borrowed session view into KV cache storage with validation and offsets.

#include "aethermind/base/kv_cache_binding.h"
#include "aethermind/base/macros.h"
#include "aethermind/base/status.h"
#include "aethermind/dtypes/data_type.h"

#include <cstddef>
#include <cstdint>

namespace aethermind {

namespace detail {
class KVCacheCommitAccess;
}

struct KVCacheLayout;
struct KVCacheStorage;
struct KVCacheSessionState;

/// @brief Borrowed, validated view into manager-owned KV cache state.
///
/// Holds non-owning pointers to the manager's layout, storage, and session state.
/// The view is stale (invalid) after the session is released or its state
/// generation changes. Reads and writes target the shared storage in place;
/// no copies are made. The manager must outlive every view, including stale
/// copies, and must not be moved or assigned while they exist. Not thread-safe.
class KVCacheSessionView {
public:
    KVCacheSessionView() = default;
    /// @brief Binds the view to manager-owned state.
    ///
    /// @param layout Cache geometry, borrowed.
    /// @param storage Key/value buffers, borrowed.
    /// @param session_state Active session state, borrowed.
    KVCacheSessionView(const KVCacheLayout* layout,
                       KVCacheStorage* storage,
                       KVCacheSessionState* session_state) noexcept;

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
    /// @brief Returns the immutable prompt reservation length.
    AM_NODISCARD size_t prompt_len() const noexcept;
    /// @brief Returns whether the next append must be the full Prefill write.
    AM_NODISCARD bool awaiting_prefill() const noexcept;

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

    /// @brief Binds one layer for an append transaction.
    ///
    /// Requires `begin == current_pos()` and `end > begin`; a normal append
    /// exposes exactly its written range, so `end` is the sole visible frontier.
    AM_NODISCARD StatusOr<KVCacheAppendBinding> BindLayerForAppend(
            size_t layer_idx, size_t begin, size_t end) const noexcept;

    /// @brief Binds one layer for a validated read window.
    ///
    /// `committed_end` must equal the current global commit watermark. Callers
    /// may pass a larger `visible_end` only after their transaction has proved
    /// that this layer's append completed earlier in the same plan.
    AM_NODISCARD StatusOr<KVCacheReadBinding> BindLayerForRead(
            size_t layer_idx, size_t committed_end, size_t visible_end) const noexcept;

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

    /// @brief Unbinds the view from all manager state, making it invalid.
    void Invalidate() noexcept;

private:
    friend class KVCacheManager;
    friend class detail::KVCacheCommitAccess;

    /// @brief Returns whether the bound session state is live at this generation.
    AM_NODISCARD bool IsSessionAlive() const noexcept;
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
    AM_NODISCARD StatusOr<KVCacheLayerStorageBinding> BindLayerStorage(
            size_t layer_idx) const noexcept;
    /// Commits one fully completed append transaction through the private
    /// KVCacheCommitAccess capability.
    Status CommitAppend(size_t begin, size_t end) noexcept;

    const KVCacheLayout* layout_ = nullptr;
    KVCacheStorage* storage_ = nullptr;
    KVCacheSessionState* session_state_ = nullptr;
    uint64_t generation_ = 0;
};

} // namespace aethermind

#endif // AETHERMIND_KV_CACHE_KV_CACHE_SESSION_VIEW_H
