#ifndef AETHERMIND_KV_CACHE_KV_CACHE_MANAGER_H
#define AETHERMIND_KV_CACHE_KV_CACHE_MANAGER_H

/// @file kv_cache_manager.h
/// @brief Owner of the physical KV cache and the single session reservation.

#include "aethermind/kv_cache/detail/kv_cache_state.h"
#include "aethermind/kv_cache/kv_cache_layout.h"
#include "aethermind/kv_cache/kv_cache_session_view.h"

namespace aethermind {

class Allocator;

/// @brief Manages the physical KV cache storage and session lifecycle.
///
/// The manager holds one active session reservation at a time; a new
/// ReserveForSession fails while a session is in use. KVCacheSessionView objects
/// borrow manager-owned state and become stale when the session is released.
/// The manager must outlive all borrowed views, including stale copies, and
/// must not be moved or assigned while any such views exist. It may be moved
/// during resource assembly before views are borrowed. Access to the manager
/// and its views must be serialized by the caller.
///
/// Init, ReserveForSession, ResetSession, and ReleaseSession must not be
/// interleaved with Execute() on this manager's views, including through kernel
/// reentry. Execution owns append-transaction commit until it returns.
class KVCacheManager {
public:
    KVCacheManager() = default;
    KVCacheManager(const KVCacheManager&) = delete;
    KVCacheManager& operator=(const KVCacheManager&) = delete;
    KVCacheManager(KVCacheManager&&) noexcept = default;
    KVCacheManager& operator=(KVCacheManager&&) noexcept = default;

    /// @brief (Re)initializes the cache with the given geometry.
    ///
    /// Both key and value planes follow `allocator` storage policy. It is borrowed only
    /// for the duration of this call; the manager keeps no reference to it, and
    /// the resulting Buffers stay valid after the allocator is destroyed.
    ///
    /// @param allocator Source of the key and value plane storage.
    /// @param num_layers Number of decoder layers.
    /// @param num_kv_heads Number of KV heads per layer.
    /// @param max_tokens Maximum physical token capacity.
    /// @param head_dim Head dimension of the key/value vectors.
    /// @param kv_dtype Element type stored in the cache.
    /// @param alignment Minimum byte alignment for the backing buffers, and the
    ///        row-layout guarantee recorded in the layout.
    /// @return Status::Ok() on success.
    /// @note Re-initializing releases the previous storage and any active
    ///       session reservation, even if initialization fails. All previous
    ///       views remain stale after subsequent reservations.
    Status Init(Allocator& allocator,
                size_t num_layers,
                size_t num_kv_heads,
                size_t max_tokens,
                size_t head_dim,
                DataType kv_dtype,
                size_t alignment = 64);

    /// @brief Reserves the cache for a new session.
    ///
    /// @param prompt_len Prompt token count.
    /// @param future_kv_appends Maximum number of KV append positions after the
    ///        prompt. A Prefill-generated first token is not yet in KV, so a
    ///        Generate limit of N tokens passes N - 1 here.
    /// @return A view over the reserved slot, or an error if the manager is
    ///         uninitialized, already reserved, or the request exceeds
    ///         physical capacity.
    StatusOr<KVCacheSessionView> ReserveForSession(size_t prompt_len,
                                                   size_t future_kv_appends) noexcept;
    /// @brief Clears the committed position while retaining its reservation.
    ///
    /// @param view Active session view belonging to this manager.
    /// @return Status::Ok() on success. The next append must run Prefill again;
    ///         this does not preserve prompt KV for repeated Decode.
    Status ResetSession(KVCacheSessionView& view) noexcept;
    /// @brief Releases the active reservation and invalidates the view.
    ///
    /// @param view Active session view belonging to this manager; invalidated
    ///             on success.
    /// @return Status::Ok() on success.
    Status ReleaseSession(KVCacheSessionView& view) noexcept;

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
    /// Both planes are held in locals until both allocations succeed, so a
    /// failure releases the first one.
    ///
    /// @param allocator Source of the plane storage.
    /// @param bytes_per_plane Bytes per key/value plane.
    /// @param alignment Minimum byte alignment for both buffers.
    /// @return Status::Ok() on success.
    Status AllocateStorage(Allocator& allocator,
                           size_t bytes_per_plane,
                           size_t alignment);

    KVCacheLayout layout_{};
    KVCacheStorage storage_{};
    KVCacheSessionState session_state_{};
    size_t total_bytes_ = 0;
    bool initialized_ = false;
};

} // namespace aethermind

#endif // AETHERMIND_KV_CACHE_KV_CACHE_MANAGER_H
