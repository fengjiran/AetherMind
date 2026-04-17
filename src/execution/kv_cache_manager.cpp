#include "aethermind/execution/kv_cache_manager.h"

#include "aethermind/backend/workspace_types.h"

#include <cstdlib>

namespace aethermind {
namespace {

void FreeAlignedMemory(void*, void* ptr) noexcept {
    std::free(ptr);
}

StatusOr<Buffer> AllocateAlignedCpuBuffer(size_t nbytes, size_t alignment) noexcept {
    if (!IsValidWorkspaceAlignment(alignment)) {
        return Status::InvalidArgument("KV buffer alignment must be a non-zero power of two");
    }

    void* ptr = nullptr;
    const int rc = posix_memalign(&ptr, alignment, nbytes == 0 ? 1 : nbytes);
    if (rc != 0 || ptr == nullptr) {
        return Status(StatusCode::kResourceExhausted, "Failed to allocate aligned KV buffer");
    }

    return Buffer{nbytes,
                  MemoryHandle(ptr, nullptr, &FreeAlignedMemory, Device::CPU(), alignment)};
}

}// namespace

Status KVCacheManager::AllocateStorage(size_t bytes_per_plane, size_t alignment) {
    const StatusOr<Buffer> key_buffer = AllocateAlignedCpuBuffer(bytes_per_plane, alignment);
    if (!key_buffer.ok()) {
        return key_buffer.status();
    }

    const StatusOr<Buffer> value_buffer = AllocateAlignedCpuBuffer(bytes_per_plane, alignment);
    if (!value_buffer.ok()) {
        return value_buffer.status();
    }

    storage_.key_buffer = key_buffer.value();
    storage_.value_buffer = value_buffer.value();
    storage_.kv_dtype = layout_.kv_dtype;
    storage_.alignment = alignment;
    return Status::Ok();
}

Status KVCacheManager::Init(size_t num_layers,
                            size_t num_kv_heads,
                            size_t max_tokens,
                            size_t head_dim,
                            DataType kv_dtype,
                            size_t alignment) {
    layout_ = {};
    storage_ = {};
    slot_ = {};
    total_bytes_ = 0;
    initialized_ = false;

    layout_.num_layers = num_layers;
    layout_.num_kv_heads = num_kv_heads;
    layout_.max_tokens = max_tokens;
    layout_.head_dim = head_dim;
    layout_.head_dim_stride = head_dim;
    layout_.kv_dtype = kv_dtype;
    layout_.alignment = alignment;

    const size_t element_bytes = layout_.ElementBytes();
    if (element_bytes == 0) {
        return Status::InvalidArgument("KV dtype must have non-zero element bytes");
    }

    if (CheckOverflowMul(layout_.head_dim_stride, element_bytes, &layout_.token_stride) ||
        CheckOverflowMul(layout_.max_tokens, layout_.token_stride, &layout_.head_stride) ||
        CheckOverflowMul(layout_.num_kv_heads, layout_.head_stride, &layout_.layer_stride)) {
        return Status(StatusCode::kOutOfRange, "KV layout stride computation overflowed size_t");
    }

    AM_RETURN_IF_ERROR(layout_.Validate());

    const StatusOr<size_t> bytes_per_plane = layout_.BytesPerPlane();
    if (!bytes_per_plane.ok()) {
        return bytes_per_plane.status();
    }

    size_t total_bytes = 0;
    if (CheckOverflowMul(bytes_per_plane.value(), size_t{2}, &total_bytes)) {
        return Status(StatusCode::kOutOfRange, "KV total bytes overflowed size_t");
    }

    AM_RETURN_IF_ERROR(AllocateStorage(bytes_per_plane.value(), alignment));

    total_bytes_ = total_bytes;
    initialized_ = true;
    return Status::Ok();
}

StatusOr<KVCacheView> KVCacheManager::ReserveForSession(size_t prompt_len,
                                                        size_t max_new_tokens) noexcept {
    if (!initialized_) {
        return Status(StatusCode::kFailedPrecondition, "KVCacheManager is not initialized");
    }
    if (slot_.in_use) {
        return Status(StatusCode::kFailedPrecondition, "KVCacheManager already has an active session reservation");
    }

    size_t requested_tokens = 0;
    if (CheckOverflowAdd(prompt_len, max_new_tokens, &requested_tokens)) {
        return Status(StatusCode::kOutOfRange, "KV session reservation overflowed size_t");
    }
    if (requested_tokens == 0) {
        return Status::InvalidArgument("KV session reservation must request at least one token");
    }
    if (requested_tokens > layout_.max_tokens) {
        return Status(StatusCode::kOutOfRange, "KV session reservation exceeds physical KV capacity");
    }

    ++slot_.generation;
    slot_.in_use = true;
    slot_.capacity_tokens = requested_tokens;
    slot_.prompt_len = prompt_len;
    slot_.current_pos = prompt_len;

    return KVCacheView(&layout_, &storage_, &slot_);
}

Status KVCacheManager::ResetSession(KVCacheView& view) noexcept {
    if (!initialized_) {
        return Status(StatusCode::kFailedPrecondition, "KVCacheManager is not initialized");
    }
    if (!view.valid()) {
        return Status(StatusCode::kFailedPrecondition, "Cannot reset an invalid or stale KVCacheView");
    }

    slot_.current_pos = slot_.prompt_len;
    return Status::Ok();
}

Status KVCacheManager::ReleaseSession(KVCacheView& view) noexcept {
    if (!initialized_) {
        return Status(StatusCode::kFailedPrecondition, "KVCacheManager is not initialized");
    }
    if (!view.valid()) {
        return Status(StatusCode::kFailedPrecondition, "Cannot release an invalid or stale KVCacheView");
    }

    slot_.in_use = false;
    slot_.capacity_tokens = 0;
    slot_.prompt_len = 0;
    slot_.current_pos = 0;
    ++slot_.generation;
    view.Invalidate();
    return Status::Ok();
}

const KVLayoutContract& KVCacheManager::layout() const noexcept { return layout_; }
size_t KVCacheManager::capacity_tokens() const noexcept { return layout_.max_tokens; }
size_t KVCacheManager::total_bytes() const noexcept { return total_bytes_; }
bool KVCacheManager::is_initialized() const noexcept { return initialized_; }

}// namespace aethermind
