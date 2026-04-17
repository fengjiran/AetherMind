#include "aethermind/execution/kv_cache_view.h"

namespace aethermind {
namespace {

Status ValidateIndexRange(size_t index, size_t upper_bound, const char* label) noexcept {
    if (index >= upper_bound) {
        return Status(StatusCode::kOutOfRange,
                      std::string(label) + " index out of range");
    }
    return Status::Ok();
}

}// namespace

size_t KVLayoutContract::ElementBytes() const noexcept {
    return static_cast<size_t>(kv_dtype.nbytes());
}

Status KVLayoutContract::Validate() const noexcept {
    if (num_layers == 0 || num_kv_heads == 0 || max_tokens == 0 || head_dim == 0) {
        return Status::InvalidArgument("KV layout dimensions must be non-zero");
    }
    if (!IsValidWorkspaceAlignment(alignment)) {
        return Status::InvalidArgument("KV layout alignment must be a non-zero power of two");
    }
    if (kv_dtype.bits() == 0 || kv_dtype.lanes() == 0) {
        return Status::InvalidArgument("KV layout dtype must be initialized");
    }
    if (head_dim_stride < head_dim) {
        return Status::InvalidArgument("KV head_dim_stride must be >= head_dim");
    }
    return Status::Ok();
}

StatusOr<size_t> KVLayoutContract::Offset(size_t layer_idx,
                                          size_t kv_head_idx,
                                          size_t seq_pos,
                                          size_t dim_idx) const noexcept {
    AM_RETURN_IF_ERROR(Validate());
    AM_RETURN_IF_ERROR(ValidateIndexRange(layer_idx, num_layers, "layer"));
    AM_RETURN_IF_ERROR(ValidateIndexRange(kv_head_idx, num_kv_heads, "kv_head"));
    AM_RETURN_IF_ERROR(ValidateIndexRange(seq_pos, max_tokens, "seq_pos"));
    AM_RETURN_IF_ERROR(ValidateIndexRange(dim_idx, head_dim, "dim"));

    size_t offset = 0;
    if (CheckOverflowMul(layer_idx, layer_stride, &offset)) {
        return Status(StatusCode::kOutOfRange, "KV layer offset overflowed size_t");
    }

    size_t head_offset = 0;
    if (CheckOverflowMul(kv_head_idx, head_stride, &head_offset) ||
        CheckOverflowAdd(offset, head_offset, &offset)) {
        return Status(StatusCode::kOutOfRange, "KV head offset overflowed size_t");
    }

    size_t token_offset = 0;
    if (CheckOverflowMul(seq_pos, token_stride, &token_offset) ||
        CheckOverflowAdd(offset, token_offset, &offset)) {
        return Status(StatusCode::kOutOfRange, "KV token offset overflowed size_t");
    }

    size_t dim_offset = 0;
    if (CheckOverflowMul(dim_idx, ElementBytes(), &dim_offset) ||
        CheckOverflowAdd(offset, dim_offset, &offset)) {
        return Status(StatusCode::kOutOfRange, "KV dim offset overflowed size_t");
    }

    return offset;
}

StatusOr<size_t> KVLayoutContract::BytesPerPlane() const noexcept {
    AM_RETURN_IF_ERROR(Validate());
    size_t bytes = 0;
    if (CheckOverflowMul(num_layers, layer_stride, &bytes)) {
        return Status(StatusCode::kOutOfRange, "KV bytes-per-plane overflowed size_t");
    }
    return bytes;
}

bool KVCacheStorage::is_initialized() const noexcept {
    return key_buffer.is_initialized() && value_buffer.is_initialized();
}

KVCacheView::KVCacheView(const KVLayoutContract* layout,
                         KVCacheStorage* storage,
                         SessionKVSlot* slot) noexcept
    : layout_(layout), storage_(storage), slot_(slot), generation_(slot != nullptr ? slot->generation : 0) {}

bool KVCacheView::IsSlotAlive() const noexcept {
    return slot_ != nullptr && slot_->in_use && slot_->generation == generation_;
}

bool KVCacheView::valid() const noexcept {
    return layout_ != nullptr && storage_ != nullptr && storage_->is_initialized() && IsSlotAlive();
}

size_t KVCacheView::max_tokens() const noexcept { return layout_ != nullptr ? layout_->max_tokens : 0; }
size_t KVCacheView::current_pos() const noexcept { return valid() ? slot_->current_pos : 0; }
size_t KVCacheView::num_layers() const noexcept { return layout_ != nullptr ? layout_->num_layers : 0; }
size_t KVCacheView::num_kv_heads() const noexcept { return layout_ != nullptr ? layout_->num_kv_heads : 0; }
size_t KVCacheView::head_dim() const noexcept { return layout_ != nullptr ? layout_->head_dim : 0; }
size_t KVCacheView::token_capacity() const noexcept { return valid() ? slot_->capacity_tokens : 0; }
size_t KVCacheView::committed_tokens() const noexcept { return current_pos(); }

Status KVCacheView::ValidateBaseState() const noexcept {
    if (layout_ == nullptr || storage_ == nullptr || slot_ == nullptr) {
        return Status(StatusCode::kFailedPrecondition, "KVCacheView is not bound to manager-owned state");
    }
    if (!storage_->is_initialized()) {
        return Status(StatusCode::kFailedPrecondition, "KVCacheView storage is not initialized");
    }
    if (!IsSlotAlive()) {
        return Status(StatusCode::kFailedPrecondition, "KVCacheView is stale or released");
    }
    return layout_->Validate();
}

Status KVCacheView::ValidateWrite(size_t layer_idx,
                                  size_t kv_head_idx,
                                  size_t seq_pos,
                                  size_t token_count) const noexcept {
    AM_RETURN_IF_ERROR(ValidateBaseState());
    AM_RETURN_IF_ERROR(ValidateIndexRange(layer_idx, layout_->num_layers, "layer"));
    AM_RETURN_IF_ERROR(ValidateIndexRange(kv_head_idx, layout_->num_kv_heads, "kv_head"));
    if (token_count == 0) {
        return Status::InvalidArgument("KV write token_count must be non-zero");
    }
    size_t seq_end = 0;
    if (CheckOverflowAdd(seq_pos, token_count, &seq_end)) {
        return Status(StatusCode::kOutOfRange, "KV write range overflowed size_t");
    }
    if (seq_end > slot_->capacity_tokens) {
        return Status(StatusCode::kOutOfRange, "KV write exceeds reserved session token capacity");
    }
    if (seq_end > layout_->max_tokens) {
        return Status(StatusCode::kOutOfRange, "KV write exceeds physical KV capacity");
    }
    return Status::Ok();
}

Status KVCacheView::ValidateRead(size_t layer_idx,
                                 size_t kv_head_idx,
                                 size_t seq_begin,
                                 size_t seq_end) const noexcept {
    AM_RETURN_IF_ERROR(ValidateBaseState());
    AM_RETURN_IF_ERROR(ValidateIndexRange(layer_idx, layout_->num_layers, "layer"));
    AM_RETURN_IF_ERROR(ValidateIndexRange(kv_head_idx, layout_->num_kv_heads, "kv_head"));
    if (seq_begin > seq_end) {
        return Status::InvalidArgument("KV read seq_begin must be <= seq_end");
    }
    if (seq_end > slot_->current_pos) {
        return Status(StatusCode::kOutOfRange, "KV read exceeds committed token range");
    }
    return Status::Ok();
}

StatusOr<size_t> KVCacheView::Offset(size_t layer_idx,
                                     size_t kv_head_idx,
                                     size_t seq_pos,
                                     size_t dim_idx) const noexcept {
    AM_RETURN_IF_ERROR(ValidateBaseState());
    return layout_->Offset(layer_idx, kv_head_idx, seq_pos, dim_idx);
}

StatusOr<void*> KVCacheView::MutableKeyData(size_t layer_idx,
                                            size_t kv_head_idx,
                                            size_t seq_pos,
                                            size_t dim_idx) noexcept {
    AM_RETURN_IF_ERROR(ValidateWrite(layer_idx, kv_head_idx, seq_pos, 1));
    const StatusOr<size_t> offset = Offset(layer_idx, kv_head_idx, seq_pos, dim_idx);
    if (!offset.ok()) {
        return offset.status();
    }
    return static_cast<std::byte*>(storage_->key_buffer.mutable_data()) + offset.value();
}

StatusOr<void*> KVCacheView::MutableValueData(size_t layer_idx,
                                              size_t kv_head_idx,
                                              size_t seq_pos,
                                              size_t dim_idx) noexcept {
    AM_RETURN_IF_ERROR(ValidateWrite(layer_idx, kv_head_idx, seq_pos, 1));
    const StatusOr<size_t> offset = Offset(layer_idx, kv_head_idx, seq_pos, dim_idx);
    if (!offset.ok()) {
        return offset.status();
    }
    return static_cast<std::byte*>(storage_->value_buffer.mutable_data()) + offset.value();
}

StatusOr<const void*> KVCacheView::KeyData(size_t layer_idx,
                                           size_t kv_head_idx,
                                           size_t seq_pos,
                                           size_t dim_idx) const noexcept {
    AM_RETURN_IF_ERROR(ValidateRead(layer_idx, kv_head_idx, seq_pos, seq_pos + 1));
    const StatusOr<size_t> offset = Offset(layer_idx, kv_head_idx, seq_pos, dim_idx);
    if (!offset.ok()) {
        return offset.status();
    }
    return static_cast<const std::byte*>(storage_->key_buffer.data()) + offset.value();
}

StatusOr<const void*> KVCacheView::ValueData(size_t layer_idx,
                                             size_t kv_head_idx,
                                             size_t seq_pos,
                                             size_t dim_idx) const noexcept {
    AM_RETURN_IF_ERROR(ValidateRead(layer_idx, kv_head_idx, seq_pos, seq_pos + 1));
    const StatusOr<size_t> offset = Offset(layer_idx, kv_head_idx, seq_pos, dim_idx);
    if (!offset.ok()) {
        return offset.status();
    }
    return static_cast<const std::byte*>(storage_->value_buffer.data()) + offset.value();
}

Status KVCacheView::CommitUntil(size_t new_pos) noexcept {
    AM_RETURN_IF_ERROR(ValidateBaseState());
    if (new_pos < slot_->current_pos) {
        return Status::InvalidArgument("KV commit position cannot move backwards");
    }
    if (new_pos > slot_->capacity_tokens) {
        return Status(StatusCode::kOutOfRange, "KV commit exceeds reserved session token capacity");
    }
    slot_->current_pos = new_pos;
    return Status::Ok();
}

void KVCacheView::Invalidate() noexcept {
    layout_ = nullptr;
    storage_ = nullptr;
    slot_ = nullptr;
    generation_ = 0;
}

}// namespace aethermind
