#ifndef AETHERMIND_EXECUTION_KV_CACHE_MANAGER_H
#define AETHERMIND_EXECUTION_KV_CACHE_MANAGER_H

/// @file kv_cache_manager.h
/// @brief Owner of the physical KV cache and the single session reservation.

#include "aethermind/execution/kv_cache_view.h"

namespace aethermind {

/// @brief Manages the physical KV cache storage and session lifecycle.
///
/// The manager holds one active session reservation at a time; a new
/// ReserveForSession fails while a session is in use. KVCacheView objects
/// borrow manager-owned state and become stale when the session is released.
class KVCacheManager {
public:
    /// @brief (Re)initializes the cache with the given geometry.
    ///
    /// @param num_layers Number of decoder layers.
    /// @param num_kv_heads Number of KV heads per layer.
    /// @param max_tokens Maximum physical token capacity.
    /// @param head_dim Head dimension of the key/value vectors.
    /// @param kv_dtype Element type stored in the cache.
    /// @param alignment Byte alignment for the backing buffers.
    /// @return Status::Ok() on success.
    /// @note Re-initializing releases the previous storage and any active
    ///       session reservation.
    Status Init(size_t num_layers,
                size_t num_kv_heads,
                size_t max_tokens,
                size_t head_dim,
                DataType kv_dtype,
                size_t alignment = 64);

    /// @brief Reserves the cache for a new session.
    ///
    /// @param prompt_len Prompt token count.
    /// @param max_new_tokens Tokens the session may still generate.
    /// @return A view over the reserved slot, or an error if the manager is
    ///         uninitialized, already reserved, or the request exceeds
    ///         physical capacity.
    AM_NODISCARD StatusOr<KVCacheView> ReserveForSession(size_t prompt_len,
                                                         size_t max_new_tokens) noexcept;
    /// @brief Rewinds a session's commit position to its prompt length.
    ///
    /// @param view Active session view to reset.
    /// @return Status::Ok() on success.
    Status ResetSession(KVCacheView& view) noexcept;
    /// @brief Releases the active reservation and invalidates the view.
    ///
    /// @param view Active session view; invalidated on success.
    /// @return Status::Ok() on success.
    Status ReleaseSession(KVCacheView& view) noexcept;

    /// @brief Returns the cache geometry.
    AM_NODISCARD const KVCacheLayout& layout() const noexcept;
    /// @brief Returns the maximum physical token capacity.
    AM_NODISCARD size_t capacity_tokens() const noexcept;
    /// @brief Returns the total allocated bytes across key and value planes.
    AM_NODISCARD size_t total_bytes() const noexcept;
    /// @brief Returns whether the manager has been initialized.
    AM_NODISCARD bool is_initialized() const noexcept;

private:
    /// @brief Allocates the key and value backing buffers for one plane.
    ///
    /// @param bytes_per_plane Bytes per key/value plane.
    /// @param alignment Byte alignment for both buffers.
    /// @return Status::Ok() on success.
    Status AllocateStorage(size_t bytes_per_plane, size_t alignment);

    KVCacheLayout layout_{};
    KVCacheStorage storage_{};
    SessionKVSlot slot_{};
    size_t total_bytes_ = 0;
    bool initialized_ = false;
};

}// namespace aethermind

#endif
