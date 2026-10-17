#include "aethermind/kv_cache/kv_cache_session_view.h"

#include "aethermind/kv_cache/detail/kv_cache_state.h"
#include "aethermind/kv_cache/kv_cache_layout.h"
#include "utils/overflow_check.h"

#include <cstdint>
#include <limits>
#include <new>
#include <string>

namespace aethermind {
namespace {

Status ValidateIndexRange(size_t index, size_t upper_bound, const char* label) {
    if (index >= upper_bound) {
        return Status::OutOfRange(
                std::string(label) + " index out of range");
    }
    return Status::Ok();
}

} // namespace

KVCacheSessionView::KVCacheSessionView(KVCacheStorage* storage,
                                       KVCacheSessionState* session_state) noexcept
    : storage_(storage), session_state_(session_state),
      generation_(session_state != nullptr ? session_state->generation : 0) {}

bool KVCacheSessionView::IsSessionAlive() const noexcept {
    if (session_state_ == nullptr || session_state_->generation != generation_) {
        return false;
    }

    switch (session_state_->lifecycle) {
        case KVCacheSessionState::Lifecycle::kAwaitingPrefill:
        case KVCacheSessionState::Lifecycle::kPrefillCommitted:
            return true;
        case KVCacheSessionState::Lifecycle::kIdle:
            return false;
    }
    return false;
}

bool KVCacheSessionView::valid() const noexcept {
    return storage_ != nullptr &&
           storage_->is_initialized() && IsSessionAlive();
}

size_t KVCacheSessionView::max_tokens() const noexcept {
    return storage_ != nullptr ? storage_->layout.max_tokens : 0;
}

size_t KVCacheSessionView::current_pos() const noexcept {
    return valid() ? session_state_->committed_tokens : 0;
}

size_t KVCacheSessionView::num_layers() const noexcept {
    return storage_ != nullptr ? storage_->layout.num_layers : 0;
}

size_t KVCacheSessionView::num_kv_heads() const noexcept {
    return storage_ != nullptr ? storage_->layout.num_kv_heads : 0;
}

size_t KVCacheSessionView::head_dim() const noexcept {
    return storage_ != nullptr ? storage_->layout.head_dim : 0;
}

DataType KVCacheSessionView::kv_dtype() const noexcept {
    return storage_ != nullptr ? storage_->layout.kv_dtype : DataType{};
}

size_t KVCacheSessionView::token_capacity() const noexcept {
    return valid() ? session_state_->capacity_tokens : 0;
}

size_t KVCacheSessionView::committed_tokens() const noexcept {
    return current_pos();
}

size_t KVCacheSessionView::prompt_len() const noexcept {
    return valid() ? session_state_->prompt_len : 0;
}

bool KVCacheSessionView::awaiting_prefill() const noexcept {
    return valid() && session_state_->lifecycle ==
                              KVCacheSessionState::Lifecycle::kAwaitingPrefill;
}

Status KVCacheSessionView::ValidateBaseState() const noexcept try {
    if (storage_ == nullptr || session_state_ == nullptr) {
        return Status::FailedPrecondition(
                "KVCacheSessionView is not bound to manager-owned state");
    }

    if (!storage_->is_initialized()) {
        return Status::FailedPrecondition(
                "KVCacheSessionView storage is not initialized");
    }

    if (session_state_->generation != generation_) {
        return Status::FailedPrecondition(
                "KVCacheSessionView is stale or released");
    }

    AM_RETURN_IF_ERROR(storage_->layout.Validate());

    if (session_state_->prompt_len == 0 ||
        session_state_->prompt_len > session_state_->capacity_tokens ||
        session_state_->capacity_tokens > storage_->layout.max_tokens) {
        return Status::FailedPrecondition(
                "KVCacheSessionView has invalid session capacity");
    }

    switch (session_state_->lifecycle) {
        case KVCacheSessionState::Lifecycle::kAwaitingPrefill:
            if (session_state_->committed_tokens != 0) {
                return Status::FailedPrecondition("KVCacheSessionView awaiting Prefill "
                                                  "must have zero committed tokens");
            }
            break;
        case KVCacheSessionState::Lifecycle::kPrefillCommitted:
            if (session_state_->committed_tokens < session_state_->prompt_len ||
                session_state_->committed_tokens > session_state_->capacity_tokens) {
                return Status::FailedPrecondition("KVCacheSessionView committed tokens "
                                                  "violate the session reservation");
            }
            break;
        case KVCacheSessionState::Lifecycle::kIdle:
            return Status::FailedPrecondition(
                    "KVCacheSessionView is not reserved");
        default:
            return Status::FailedPrecondition(
                    "KVCacheSessionView has unknown lifecycle state");
    }

    // The layout must fit the backing buffers; a mismatched pair would let
    // validated offsets escape the allocation.
    const auto bytes_per_plane = storage_->layout.BytesPerPlaneValidated();
    if (!bytes_per_plane.ok()) {
        return bytes_per_plane.status();
    }

    if (*bytes_per_plane > storage_->key_buffer.nbytes() ||
        *bytes_per_plane > storage_->value_buffer.nbytes()) {
        return Status::FailedPrecondition(
                "KVCacheSessionView layout exceeds the backing buffer size");
    }

    // CPU kernels consume the bound base pointers directly, so absolute token
    // row alignment also requires each plane base to satisfy the layout
    // alignment; the layout alone only constrains row offsets relative to the
    // plane base. Non-CPU planes have no host-address row-alignment semantics.
    if ((storage_->key_buffer.device().is_cpu() &&
         reinterpret_cast<uintptr_t>(storage_->key_buffer.data()) % storage_->layout.alignment != 0) ||
        (storage_->value_buffer.device().is_cpu() &&
         reinterpret_cast<uintptr_t>(storage_->value_buffer.data()) % storage_->layout.alignment != 0)) {
        return Status::FailedPrecondition(
                "KVCacheSessionView CPU plane base breaks the layout row alignment");
    }

    return Status::Ok();
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

Status KVCacheSessionView::ValidateWrite(size_t layer_idx,
                                         size_t kv_head_idx,
                                         size_t seq_pos,
                                         size_t token_count) const noexcept try {
    AM_RETURN_IF_ERROR(ValidateBaseState());
    AM_RETURN_IF_ERROR(ValidateIndexRange(
            layer_idx, storage_->layout.num_layers, "layer"));
    AM_RETURN_IF_ERROR(ValidateIndexRange(
            kv_head_idx, storage_->layout.num_kv_heads, "kv_head"));

    if (token_count == 0) {
        return Status::InvalidArgument(
                "KV write token_count must be non-zero");
    }

    size_t seq_end = 0;
    if (CheckOverflowAdd(seq_pos, token_count, &seq_end)) {
        return Status::Overflow("KV write range overflowed size_t");
    }

    if (seq_end > session_state_->capacity_tokens) {
        return Status::OutOfRange(
                "KV write exceeds reserved session token capacity");
    }

    if (seq_end > storage_->layout.max_tokens) {
        return Status::OutOfRange(
                "KV write exceeds physical KV capacity");
    }

    return Status::Ok();
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

Status KVCacheSessionView::ValidateRead(size_t layer_idx,
                                        size_t kv_head_idx,
                                        size_t seq_begin,
                                        size_t seq_end) const noexcept try {
    AM_RETURN_IF_ERROR(ValidateBaseState());
    AM_RETURN_IF_ERROR(ValidateIndexRange(
            layer_idx, storage_->layout.num_layers, "layer"));
    AM_RETURN_IF_ERROR(ValidateIndexRange(
            kv_head_idx, storage_->layout.num_kv_heads, "kv_head"));

    if (seq_begin > seq_end) {
        return Status::InvalidArgument("KV read seq_begin must be <= seq_end");
    }

    if (seq_end > session_state_->committed_tokens) {
        return Status::OutOfRange("KV read exceeds committed token range");
    }
    return Status::Ok();
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

StatusOr<KVCacheLayerStorageBinding> KVCacheSessionView::BindLayerStorageValidated(
        size_t layer_idx) const {
    AM_RETURN_IF_ERROR(ValidateIndexRange(
            layer_idx, storage_->layout.num_layers, "layer"));

    if (layer_idx > std::numeric_limits<uint32_t>::max()) {
        return Status::OutOfRange(
                "KV layer index exceeds kernel binding range");
    }

    const auto layer_offset = storage_->layout.OffsetValidated(
            layer_idx, 0, 0, 0);
    if (!layer_offset.ok()) {
        return layer_offset.status();
    }

    auto* const key_base = static_cast<std::byte*>(storage_->key_buffer.mutable_data());
    auto* const value_base = static_cast<std::byte*>(storage_->value_buffer.mutable_data());
    if (key_base == nullptr || value_base == nullptr) {
        return Status::FailedPrecondition(
                "KVCacheSessionView storage has a null backing pointer");
    }

    return KVCacheLayerStorageBinding{
            .key_data = key_base + *layer_offset,
            .value_data = value_base + *layer_offset,
            .dtype = storage_->layout.kv_dtype,
            .layer_index = static_cast<uint32_t>(layer_idx),
            .num_kv_heads = storage_->layout.num_kv_heads,
            .head_dim = storage_->layout.head_dim,
            .token_capacity = session_state_->capacity_tokens,
            .token_stride_bytes = storage_->layout.token_stride,
            .head_stride_bytes = storage_->layout.head_stride,
    };
}

StatusOr<KVCacheAppendBinding> KVCacheSessionView::BindLayerForAppend(
        size_t layer_idx, size_t begin, size_t end) const noexcept try {
    AM_RETURN_IF_ERROR(ValidateBaseState());
    if (begin != session_state_->committed_tokens) {
        return Status::InvalidArgument(
                "KV append begin must equal the committed token position");
    }

    if (end <= begin) {
        return Status::InvalidArgument(
                "KV append end must be greater than begin");
    }

    if (end > session_state_->capacity_tokens) {
        return Status::OutOfRange(
                "KV append exceeds reserved session token capacity");
    }

    if (end > storage_->layout.max_tokens) {
        return Status::OutOfRange(
                "KV append exceeds physical KV capacity");
    }

    AM_ASSIGN_OR_RETURN(KVCacheLayerStorageBinding storage, BindLayerStorageValidated(layer_idx));
    return KVCacheAppendBinding{.storage = storage, .begin = begin, .end = end};
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

StatusOr<KVCacheReadBinding> KVCacheSessionView::BindLayerForRead(
        size_t layer_idx, size_t committed_end, size_t visible_end) const noexcept try {
    AM_RETURN_IF_ERROR(ValidateBaseState());
    if (committed_end != session_state_->committed_tokens) {
        return Status::InvalidArgument(
                "KV read committed_end must equal the current commit position");
    }

    if (visible_end < committed_end) {
        return Status::InvalidArgument(
                "KV read visible_end must not precede committed_end");
    }

    if (visible_end > session_state_->capacity_tokens) {
        return Status::OutOfRange(
                "KV read visible range exceeds reserved session token capacity");
    }

    if (visible_end > storage_->layout.max_tokens) {
        return Status::OutOfRange(
                "KV read visible range exceeds physical KV capacity");
    }

    AM_ASSIGN_OR_RETURN(KVCacheLayerStorageBinding storage, BindLayerStorageValidated(layer_idx));
    return KVCacheReadBinding{
            .storage = storage,
            .committed_end = committed_end,
            .visible_end = visible_end,
    };
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

StatusOr<const void*> KVCacheSessionView::KeyData(size_t layer_idx,
                                                  size_t kv_head_idx,
                                                  size_t seq_pos,
                                                  size_t dim_idx) const noexcept try {
    AM_RETURN_IF_ERROR(ValidateRead(layer_idx, kv_head_idx, seq_pos, seq_pos + 1));
    const StatusOr<size_t> offset = storage_->layout.OffsetValidated(layer_idx, kv_head_idx, seq_pos, dim_idx);
    if (!offset.ok()) {
        return offset.status();
    }
    return static_cast<const std::byte*>(storage_->key_buffer.data()) + offset.value();
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

StatusOr<const void*> KVCacheSessionView::ValueData(size_t layer_idx,
                                                    size_t kv_head_idx,
                                                    size_t seq_pos,
                                                    size_t dim_idx) const noexcept try {
    AM_RETURN_IF_ERROR(ValidateRead(layer_idx, kv_head_idx, seq_pos, seq_pos + 1));
    const StatusOr<size_t> offset = storage_->layout.OffsetValidated(layer_idx, kv_head_idx, seq_pos, dim_idx);
    if (!offset.ok()) {
        return offset.status();
    }
    return static_cast<const std::byte*>(storage_->value_buffer.data()) + offset.value();
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

Status KVCacheSessionView::CommitAppend(size_t begin, size_t end) const noexcept try {
    AM_RETURN_IF_ERROR(ValidateBaseState());
    if (begin != session_state_->committed_tokens) {
        return Status::InvalidArgument(
                "KV append commit must begin at the committed position");
    }

    if (end <= begin) {
        return Status::InvalidArgument(
                "KV append commit range must be non-empty");
    }

    if (end > session_state_->capacity_tokens) {
        return Status::OutOfRange(
                "KV append commit exceeds reserved session capacity");
    }

    if (end > storage_->layout.max_tokens) {
        return Status::OutOfRange(
                "KV append commit exceeds physical KV capacity");
    }

    KVCacheSessionState::Lifecycle next_lifecycle{};
    switch (session_state_->lifecycle) {
        case KVCacheSessionState::Lifecycle::kAwaitingPrefill:
            if (begin != 0 || end != session_state_->prompt_len) {
                return Status::FailedPrecondition(
                        "KV initial commit must complete the reserved Prefill range");
            }

            next_lifecycle = KVCacheSessionState::Lifecycle::kPrefillCommitted;
            break;
        case KVCacheSessionState::Lifecycle::kPrefillCommitted:
            if (end - begin != 1) {
                return Status::InvalidArgument(
                        "KV Decode commit must append exactly one token");
            }
            next_lifecycle = KVCacheSessionState::Lifecycle::kPrefillCommitted;
            break;
        case KVCacheSessionState::Lifecycle::kIdle:
            return Status::FailedPrecondition(
                    "KV append commit requires an active session");
        default:
            return Status::FailedPrecondition(
                    "KV append commit has unknown lifecycle state");
    }

    session_state_->committed_tokens = end;
    session_state_->lifecycle = next_lifecycle;
    return Status::Ok();
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

void KVCacheSessionView::Invalidate() noexcept {
    storage_ = nullptr;
    session_state_ = nullptr;
    generation_ = 0;
}


} // namespace aethermind
