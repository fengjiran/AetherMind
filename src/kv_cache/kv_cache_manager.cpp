#include "aethermind/kv_cache/kv_cache_manager.h"
#include "aethermind/memory/allocator.h"
#include "utils/overflow_check.h"

#include <limits>
#include <new>
#include <type_traits>
#include <utility>

namespace aethermind {

Status KVCacheManager::Init(Allocator& allocator,
                            size_t num_layers,
                            size_t num_kv_heads,
                            size_t max_tokens,
                            size_t head_dim,
                            DataType kv_dtype,
                            size_t alignment) try {
    layout_ = {};
    storage_ = {};
    // Preserve the epoch so a later reserve cannot revive a view borrowed
    // before this re-initialization, including when initialization fails.
    session_state_ = KVCacheSessionState{.generation = session_state_.generation};
    total_bytes_ = 0;
    initialized_ = false;

    KVCacheLayout candidate_layout{
            .num_layers = num_layers,
            .num_kv_heads = num_kv_heads,
            .max_tokens = max_tokens,
            .head_dim = head_dim,
            .kv_dtype = kv_dtype,
            .alignment = alignment,
    };

    const size_t element_bytes = candidate_layout.ElementBytes();
    if (element_bytes == 0) {
        return Status::InvalidArgument("KV dtype must have non-zero element bytes");
    }

    // Pad padded_head_dim up so every token row starts at an alignment
    // boundary; head/layer strides derived from token_stride then stay
    // aligned too. alignment and element_bytes are powers of two, so
    // alignment / element_bytes is exact whenever element_bytes <= alignment;
    // otherwise the element size already guarantees token_stride alignment.
    size_t dims_per_alignment = alignment > element_bytes
                                        ? alignment / element_bytes
                                        : 1;
    candidate_layout.padded_head_dim =
            dims_per_alignment > 1
                    ? (head_dim + dims_per_alignment - 1) / dims_per_alignment *
                              dims_per_alignment
                    : head_dim;

    if (CheckOverflowMul(candidate_layout.padded_head_dim, element_bytes, &candidate_layout.token_stride) ||
        CheckOverflowMul(candidate_layout.max_tokens, candidate_layout.token_stride, &candidate_layout.head_stride) ||
        CheckOverflowMul(candidate_layout.num_kv_heads, candidate_layout.head_stride, &candidate_layout.layer_stride)) {
        return Status::Overflow("KV layout stride computation overflowed size_t");
    }

    AM_RETURN_IF_ERROR(candidate_layout.Validate());

    const StatusOr<size_t> bytes_per_plane = candidate_layout.BytesPerPlane();
    if (!bytes_per_plane.ok()) {
        return bytes_per_plane.status();
    }

    size_t total_bytes = 0;
    if (CheckOverflowMul(bytes_per_plane.value(), size_t{2}, &total_bytes)) {
        return Status::Overflow("KV total bytes overflowed size_t");
    }

    const AllocationRequest request{.nbytes = bytes_per_plane.value(),
                                    .alignment = alignment};
    AM_ASSIGN_OR_RETURN(Buffer key_buffer, allocator.TryAllocate(request));
    AM_ASSIGN_OR_RETURN(Buffer value_buffer, allocator.TryAllocate(request));
    KVCacheStorage candidate_storage{.key_buffer = std::move(key_buffer),
                                     .value_buffer = std::move(value_buffer),
                                     .kv_dtype = kv_dtype};

    // Publish only after both allocations succeed. These transfers cannot
    // throw, so no failure can expose a partially initialized manager.
    static_assert(std::is_nothrow_move_assignable_v<KVCacheLayout>);
    static_assert(std::is_nothrow_move_assignable_v<KVCacheStorage>);
    layout_ = std::move(candidate_layout);
    storage_ = std::move(candidate_storage);
    total_bytes_ = total_bytes;
    initialized_ = true;
    return Status::Ok();
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

StatusOr<KVCacheSessionView> KVCacheManager::ReserveForSession(
        size_t prompt_len,
        size_t future_kv_appends) noexcept try {
    if (!initialized_) {
        return Status::FailedPrecondition("KVCacheManager is not initialized");
    }

    if (session_state_.lifecycle != KVCacheSessionState::Lifecycle::kIdle) {
        return Status::FailedPrecondition(
                "KVCacheManager already has an active session reservation");
    }

    if (prompt_len == 0) {
        return Status::InvalidArgument(
                "KV session reservation requires a non-empty prompt");
    }

    size_t requested_tokens = 0;
    if (CheckOverflowAdd(prompt_len, future_kv_appends, &requested_tokens)) {
        return Status::Overflow("KV session reservation overflowed size_t");
    }

    if (requested_tokens == 0) {
        return Status::InvalidArgument(
                "KV session reservation must request at least one token");
    }

    if (requested_tokens > layout_.max_tokens) {
        return Status::OutOfRange(
                "KV session reservation exceeds physical KV capacity");
    }

    if (session_state_.generation == std::numeric_limits<uint64_t>::max()) {
        return Status::Overflow("KV session generation is exhausted");
    }

    ++session_state_.generation;
    session_state_.capacity_tokens = requested_tokens;
    session_state_.prompt_len = prompt_len;
    session_state_.committed_tokens = 0;
    session_state_.lifecycle = KVCacheSessionState::Lifecycle::kAwaitingPrefill;
    return KVCacheSessionView(&layout_, &storage_, &session_state_);
} catch (const std::bad_alloc&) {
    // Error diagnostics may allocate; reporting OOM must not allocate again.
    return Status::ResourceExhausted({});
}

Status KVCacheManager::ResetSession(const KVCacheSessionView& view) noexcept try {
    if (!initialized_) {
        return Status::FailedPrecondition("KVCacheManager is not initialized");
    }

    if (view.layout_ != &layout_ || view.storage_ != &storage_ ||
        view.session_state_ != &session_state_) {
        return Status::FailedPrecondition(
                "KVCacheSessionView does not belong to this manager");
    }

    if (!view.valid()) {
        return Status::FailedPrecondition(
                "Cannot reset an invalid or stale KVCacheSessionView");
    }

    session_state_.committed_tokens = 0;
    session_state_.lifecycle = KVCacheSessionState::Lifecycle::kAwaitingPrefill;
    return Status::Ok();
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

Status KVCacheManager::ReleaseSession(KVCacheSessionView& view) noexcept try {
    if (!initialized_) {
        return Status::FailedPrecondition("KVCacheManager is not initialized");
    }

    if (view.layout_ != &layout_ || view.storage_ != &storage_ ||
        view.session_state_ != &session_state_) {
        return Status::FailedPrecondition(
                "KVCacheSessionView does not belong to this manager");
    }

    if (!view.valid()) {
        return Status::FailedPrecondition(
                "Cannot release an invalid or stale KVCacheSessionView");
    }

    session_state_.capacity_tokens = 0;
    session_state_.prompt_len = 0;
    session_state_.committed_tokens = 0;
    session_state_.lifecycle = KVCacheSessionState::Lifecycle::kIdle;
    // Cleanup must succeed even for the last representable reservation.
    // Leaving the epoch saturated makes subsequent reservations fail safely.
    if (session_state_.generation != std::numeric_limits<uint64_t>::max()) {
        ++session_state_.generation;
    }
    view.Invalidate();
    return Status::Ok();
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

const KVCacheLayout& KVCacheManager::layout() const noexcept { return layout_; }
size_t KVCacheManager::capacity_tokens() const noexcept { return layout_.max_tokens; }
size_t KVCacheManager::total_bytes() const noexcept { return total_bytes_; }
bool KVCacheManager::is_initialized() const noexcept { return initialized_; }

} // namespace aethermind
