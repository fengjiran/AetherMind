#ifndef AETHERMIND_KV_CACHE_KV_CACHE_SESSION_VIEW_H
#define AETHERMIND_KV_CACHE_KV_CACHE_SESSION_VIEW_H

/// @file kv_cache_session_view.h
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
///
/// Committed state advances only through the source-private
/// detail::KVCacheCommitAccess capability after a complete plan transaction.
/// Element accessors are read-only; execution writes use append bindings.
/// For CPU plane storage, validation additionally requires each plane base to
/// be aligned to the layout alignment, so kernels observe absolutely aligned
/// token rows.
class KVCacheSessionView {
public:
    /// @brief Constructs an unbound view that reports `valid() == false`.
    KVCacheSessionView() = default;
    /// @brief Binds the view to manager-owned state.
    ///
    /// Captures the current session-state generation, so a later release or
    /// re-reservation makes this view stale.
    ///
    /// @param layout Cache geometry, borrowed.
    /// @param storage Key/value buffers, borrowed.
    /// @param session_state Active session state, borrowed.
    KVCacheSessionView(const KVCacheLayout* layout,
                       KVCacheStorage* storage,
                       KVCacheSessionState* session_state) noexcept;

    /// @brief Returns whether the view is bound to live, initialized state.
    ///
    /// @return True while bound to a live, initialized session; false when
    ///         unbound, stale, or released.
    AM_NODISCARD bool valid() const noexcept;
    /// @brief Returns the physical maximum token capacity.
    ///
    /// @return The physical token capacity, or 0 when the view is unbound.
    AM_NODISCARD size_t max_tokens() const noexcept;
    /// @brief Returns the current commit position (0 when invalid).
    ///
    /// @return The committed token count, or 0 when the view is invalid.
    AM_NODISCARD size_t current_pos() const noexcept;
    /// @brief Returns the layer count.
    ///
    /// @return The layer count, or 0 when the view is unbound.
    AM_NODISCARD size_t num_layers() const noexcept;
    /// @brief Returns the KV head count.
    ///
    /// @return The KV head count, or 0 when the view is unbound.
    AM_NODISCARD size_t num_kv_heads() const noexcept;
    /// @brief Returns the head dimension.
    ///
    /// @return The head dimension, or 0 when the view is unbound.
    AM_NODISCARD size_t head_dim() const noexcept;
    /// @brief Returns the KV cache element dtype.
    ///
    /// @return The element dtype, or an undefined `DataType` when unbound.
    AM_NODISCARD DataType kv_dtype() const noexcept;
    /// @brief Returns the reserved session token capacity.
    ///
    /// @return The reserved capacity, or 0 when the view is invalid.
    AM_NODISCARD size_t token_capacity() const noexcept;
    /// @brief Returns the number of committed tokens (== current_pos()).
    ///
    /// @return The committed token count, or 0 when the view is invalid.
    AM_NODISCARD size_t committed_tokens() const noexcept;
    /// @brief Returns the immutable prompt reservation length.
    ///
    /// @return The reserved prompt length, or 0 when the view is invalid.
    AM_NODISCARD size_t prompt_len() const noexcept;
    /// @brief Returns whether the next append must be the full Prefill write.
    ///
    /// @return True while bound and awaiting the initial Prefill append.
    AM_NODISCARD bool awaiting_prefill() const noexcept;

    /// @brief Validates a contiguous write of `token_count` tokens.
    ///
    /// @param layer_idx Layer index.
    /// @param kv_head_idx KV head index.
    /// @param seq_pos Starting sequence position.
    /// @param token_count Number of tokens written. Must be non-zero.
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
    /// @param seq_end Exclusive read end. Must not precede `seq_begin`.
    /// @return Status::Ok() if the range is committed and in bounds.
    Status ValidateRead(size_t layer_idx,
                        size_t kv_head_idx,
                        size_t seq_begin,
                        size_t seq_end) const noexcept;

    /// @brief Binds one layer for an append transaction.
    ///
    /// A normal append exposes exactly its written range, so `end` is the sole
    /// visible frontier until the plan transaction commits.
    ///
    /// @param layer_idx Layer index.
    /// @param begin First appended sequence position. Must equal the current
    ///        commit position.
    /// @param end Exclusive append end. Must be greater than `begin` and fit
    ///        the reserved and physical capacity.
    /// @return The layer append binding, or an error when the view is stale,
    ///         unbound, or the range is invalid.
    AM_NODISCARD StatusOr<KVCacheAppendBinding> BindLayerForAppend(
            size_t layer_idx, size_t begin, size_t end) const noexcept;

    /// @brief Binds one layer for a validated read window.
    ///
    /// Callers may pass a `visible_end` beyond `committed_end` only after their
    /// transaction has proved that this layer's append completed earlier in the
    /// same plan.
    ///
    /// @param layer_idx Layer index.
    /// @param committed_end Exclusive end of the range committed across plans.
    ///        Must equal the current global commit watermark.
    /// @param visible_end Exclusive end of the locally visible range. Must not
    ///        precede `committed_end`.
    /// @return The layer read binding, or an error when the view is stale,
    ///         unbound, or the window is invalid.
    AM_NODISCARD StatusOr<KVCacheReadBinding> BindLayerForRead(
            size_t layer_idx, size_t committed_end, size_t visible_end) const noexcept;

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
    ///
    /// @note Only detaches this view; manager-owned storage is not modified.
    void Invalidate() noexcept;

private:
    friend class KVCacheManager;
    friend class detail::KVCacheCommitAccess;

    /// @brief Returns whether the bound session state is live at this generation.
    ///
    /// @return True when the state is bound, its generation matches the view's,
    ///         and a session is active.
    AM_NODISCARD bool IsSessionAlive() const noexcept;

    /// @brief Validates that the view is bound to live, initialized state.
    ///
    /// @return Status::Ok() when all binding and session invariants hold.
    Status ValidateBaseState() const noexcept;

    /// @brief Builds the per-layer key/value storage binding for kernels.
    ///
    /// @pre ValidateBaseState() succeeded in the current public operation.
    /// @param layer_idx Layer index.
    /// @return The layer storage binding, or an index/arithmetic error.
    AM_NODISCARD StatusOr<KVCacheLayerStorageBinding> BindLayerStorageValidated(
            size_t layer_idx) const;

    /// @brief Commits one fully completed append transaction through the
    ///        private KVCacheCommitAccess capability.
    ///
    /// The first commit must complete the reserved Prefill range; each later
    /// commit must append exactly one token.
    ///
    /// @param begin First committed sequence position. Must equal the current
    ///        commit position.
    /// @param end Exclusive commit end. Must be greater than `begin`.
    /// @return Status::Ok() on success.
    Status CommitAppend(size_t begin, size_t end) const noexcept;

    // Borrowed manager state. The three pointers are set together on binding
    // and cleared together by Invalidate(); `generation_` records the session
    // generation at binding, so a stale view keeps its pointers but reports
    // `valid() == false` once the generation or lifecycle changes.
    const KVCacheLayout* layout_ = nullptr;
    KVCacheStorage* storage_ = nullptr;
    KVCacheSessionState* session_state_ = nullptr;
    uint64_t generation_ = 0;
};

} // namespace aethermind

#endif // AETHERMIND_KV_CACHE_KV_CACHE_SESSION_VIEW_H
