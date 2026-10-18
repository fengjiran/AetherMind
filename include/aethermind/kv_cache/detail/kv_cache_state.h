#ifndef AETHERMIND_KV_CACHE_DETAIL_KV_CACHE_STATE_H
#define AETHERMIND_KV_CACHE_DETAIL_KV_CACHE_STATE_H

/// @file kv_cache_state.h
/// @brief Manager-owned KV storage and single-session state.

#include "aethermind/base/macros.h"
#include "aethermind/kv_cache/kv_cache_layout.h"
#include "aethermind/memory/buffer.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace aethermind {

/// @brief Physical layout and its key/value backing buffers for one cache.
struct KVCacheStorage {
    KVCacheLayout layout{};
    Buffer key_buffer{};
    Buffer value_buffer{};

    /// @brief Returns whether both backing buffers are initialized.
    AM_NODISCARD bool is_initialized() const noexcept {
        return key_buffer.is_initialized() && value_buffer.is_initialized();
    }
};

// The manager relies on non-throwing move assignment to publish initialized storage.
static_assert(std::is_nothrow_move_assignable_v<KVCacheStorage>);

/// @brief State of the active session reservation inside a KVCacheManager.
///
/// `generation` is retained across re-initialization and advances on reserve
/// and release. At its maximum value, release still succeeds but no further
/// reservations are allowed, so stale views can never become live again.
/// Reset retains the reservation and generation, so all of its views observe
/// the reset together. Production mutations belong to the manager and the
/// private completed-append commit path.
///
/// Idle has zero capacity, prompt length, and committed tokens. Active states
/// satisfy 0 < prompt_len <= capacity_tokens <= the physical token capacity.
/// AwaitingPrefill has zero committed tokens; PrefillCommitted satisfies
/// prompt_len <= committed_tokens <= capacity_tokens.
struct KVCacheSessionState {
    enum class Lifecycle : uint8_t {
        kIdle,
        kAwaitingPrefill,
        kPrefillCommitted,
    };

    uint64_t generation = 0;
    size_t capacity_tokens = 0;
    size_t prompt_len = 0;
    size_t committed_tokens = 0;
    Lifecycle lifecycle = Lifecycle::kIdle;
};

} // namespace aethermind

#endif // AETHERMIND_KV_CACHE_DETAIL_KV_CACHE_STATE_H
