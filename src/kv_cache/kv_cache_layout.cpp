#include "aethermind/kv_cache/kv_cache_layout.h"
#include "aethermind/base/alignment.h"
#include "utils/overflow_check.h"

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

size_t KVCacheLayout::ElementBytes() const noexcept {
    return static_cast<size_t>(kv_dtype.nbytes());
}

Status KVCacheLayout::Validate() const noexcept try {
    if (num_layers == 0 || num_kv_heads == 0 || max_tokens == 0 || head_dim == 0) {
        return Status::InvalidArgument("KV layout dimensions must be non-zero");
    }

    if (!IsValidAlignment(alignment)) {
        return Status::InvalidArgument(
                "KV layout alignment must be a non-zero power of two");
    }

    if (kv_dtype.bits() == 0 || kv_dtype.lanes() == 0) {
        return Status::InvalidArgument("KV layout dtype must be initialized");
    }

    if (padded_head_dim < head_dim) {
        return Status::InvalidArgument("KV padded_head_dim must be >= head_dim");
    }

    const size_t element_bytes = ElementBytes();
    if (element_bytes == 0) {
        return Status::InvalidArgument("KV dtype must have non-zero element bytes");
    }

    // Stride consistency: a token row must cover the head without overlap,
    // and token/head/layer strides must keep every row and plane start
    // aligned to the layout alignment (the manager pads padded_head_dim so
    // token_stride is an alignment multiple; hand-built layouts are checked
    // here). head/layer strides are integer multiples of token_stride by
    // construction, so token_stride alignment implies their alignment.
    if (token_stride % element_bytes != 0) {
        return Status::InvalidArgument(
                "KV token_stride must be a multiple of the element size");
    }

    if (token_stride % alignment != 0) {
        return Status::InvalidArgument(
                "KV token_stride must be aligned to the layout alignment");
    }

    size_t min_token_stride = 0;
    if (CheckOverflowMul(padded_head_dim, element_bytes, &min_token_stride)) {
        return Status::Overflow("KV token_stride computation overflowed size_t");
    }

    if (token_stride < min_token_stride) {
        return Status::InvalidArgument(
                "KV token_stride must cover the full head row");
    }

    size_t min_head_stride = 0;
    if (CheckOverflowMul(token_stride, max_tokens, &min_head_stride)) {
        return Status::Overflow("KV head_stride computation overflowed size_t");
    }

    if (head_stride < min_head_stride) {
        return Status::InvalidArgument(
                "KV head_stride must cover all tokens of a head");
    }

    size_t min_layer_stride = 0;
    if (CheckOverflowMul(head_stride, num_kv_heads, &min_layer_stride)) {
        return Status::Overflow("KV layer_stride computation overflowed size_t");
    }

    if (layer_stride < min_layer_stride) {
        return Status::InvalidArgument(
                "KV layer_stride must cover all heads of a layer");
    }
    return Status::Ok();
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

StatusOr<size_t> KVCacheLayout::Offset(size_t layer_idx,
                                       size_t kv_head_idx,
                                       size_t seq_pos,
                                       size_t dim_idx) const noexcept try {
    AM_RETURN_IF_ERROR(Validate());
    return OffsetValidated(layer_idx, kv_head_idx, seq_pos, dim_idx);
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

StatusOr<size_t> KVCacheLayout::OffsetValidated(size_t layer_idx,
                                                size_t kv_head_idx,
                                                size_t seq_pos,
                                                size_t dim_idx) const {
    AM_RETURN_IF_ERROR(ValidateIndexRange(layer_idx, num_layers, "layer"));
    AM_RETURN_IF_ERROR(ValidateIndexRange(kv_head_idx, num_kv_heads, "kv_head"));
    AM_RETURN_IF_ERROR(ValidateIndexRange(seq_pos, max_tokens, "seq_pos"));
    AM_RETURN_IF_ERROR(ValidateIndexRange(dim_idx, head_dim, "dim"));

    size_t offset = 0;
    if (CheckOverflowMul(layer_idx, layer_stride, &offset)) {
        return Status::Overflow("KV layer offset overflowed size_t");
    }

    size_t head_offset = 0;
    if (CheckOverflowMul(kv_head_idx, head_stride, &head_offset) ||
        CheckOverflowAdd(offset, head_offset, &offset)) {
        return Status::Overflow("KV head offset overflowed size_t");
    }

    size_t token_offset = 0;
    if (CheckOverflowMul(seq_pos, token_stride, &token_offset) ||
        CheckOverflowAdd(offset, token_offset, &offset)) {
        return Status::Overflow("KV token offset overflowed size_t");
    }

    size_t dim_offset = 0;
    if (CheckOverflowMul(dim_idx, ElementBytes(), &dim_offset) ||
        CheckOverflowAdd(offset, dim_offset, &offset)) {
        return Status::Overflow("KV dim offset overflowed size_t");
    }

    return offset;
}

StatusOr<size_t> KVCacheLayout::BytesPerPlane() const noexcept try {
    AM_RETURN_IF_ERROR(Validate());
    return BytesPerPlaneValidated();
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

StatusOr<size_t> KVCacheLayout::BytesPerPlaneValidated() const {
    size_t bytes = 0;
    if (CheckOverflowMul(num_layers, layer_stride, &bytes)) {
        return Status::Overflow("KV bytes-per-plane overflowed size_t");
    }
    return bytes;
}


} // namespace aethermind
