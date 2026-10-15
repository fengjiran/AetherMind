#include "aethermind/kv_cache/kv_cache_manager.h"
#include "aethermind/memory/allocator.h"
#include "utils/overflow_check.h"

#include <limits>
#include <new>

namespace aethermind {

Status KVCacheManager::AllocateStorage(Allocator& allocator,
                                       size_t bytes_per_plane,
                                       size_t alignment) {
    const AllocationRequest request{.nbytes = bytes_per_plane,
                                    .alignment = alignment};

    // Both planes stay in locals until both succeed, so a second-allocation
    // failure releases the first buffer.
    const StatusOr<Buffer> key_buffer = allocator.TryAllocate(request);
    if (!key_buffer.ok()) {
        return key_buffer.status();
    }

    const StatusOr<Buffer> value_buffer = allocator.TryAllocate(request);
    if (!value_buffer.ok()) {
        return value_buffer.status();
    }

    storage_.key_buffer = key_buffer.value();
    storage_.value_buffer = value_buffer.value();
    storage_.kv_dtype = layout_.kv_dtype;
    return Status::Ok();
}

Status KVCacheManager::Init(Allocator& allocator,
                            size_t num_layers,
                            size_t num_kv_heads,
                            size_t max_tokens,
                            size_t head_dim,
                            DataType kv_dtype,
                            size_t alignment) {
    layout_ = {};
    storage_ = {};
    // Preserve the epoch so a later reserve cannot revive a view borrowed
    // before this re-initialization, including when initialization fails.
    session_state_ = KVCacheSessionState{.generation = session_state_.generation};
    total_bytes_ = 0;
    initialized_ = false;

    layout_.num_layers = num_layers;
    layout_.num_kv_heads = num_kv_heads;
    layout_.max_tokens = max_tokens;
    layout_.head_dim = head_dim;
    layout_.kv_dtype = kv_dtype;
    layout_.alignment = alignment;

    const size_t element_bytes = layout_.ElementBytes();
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
    layout_.padded_head_dim =
            dims_per_alignment > 1
                    ? (head_dim + dims_per_alignment - 1) / dims_per_alignment *
                              dims_per_alignment
                    : head_dim;

    if (CheckOverflowMul(layout_.padded_head_dim, element_bytes, &layout_.token_stride) ||
        CheckOverflowMul(layout_.max_tokens, layout_.token_stride, &layout_.head_stride) ||
        CheckOverflowMul(layout_.num_kv_heads, layout_.head_stride, &layout_.layer_stride)) {
        return Status::Overflow("KV layout stride computation overflowed size_t");
    }

    AM_RETURN_IF_ERROR(layout_.Validate());

    const StatusOr<size_t> bytes_per_plane = layout_.BytesPerPlane();
    if (!bytes_per_plane.ok()) {
        return bytes_per_plane.status();
    }

    size_t total_bytes = 0;
    if (CheckOverflowMul(bytes_per_plane.value(), size_t{2}, &total_bytes)) {
        return Status::Overflow("KV total bytes overflowed size_t");
    }

    AM_RETURN_IF_ERROR(AllocateStorage(allocator, bytes_per_plane.value(), alignment));

    total_bytes_ = total_bytes;
    initialized_ = true;
    return Status::Ok();
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
        return Status::OutOfRange("KV session reservation exceeds physical KV capacity");
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

Status KVCacheManager::ResetSession(KVCacheSessionView& view) noexcept try {
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
