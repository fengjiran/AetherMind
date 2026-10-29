#include "aethermind/kv_cache/kv_cache_layout.h"
#include "aethermind/base/alignment.h"
#include "utils/overflow_check.h"

#include <limits>
#include <new>
#include <numeric>
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

Status ValidateGeometry(const KVCacheLayout& layout) {
    if (layout.num_layers == 0 || layout.num_kv_heads == 0 ||
        layout.max_tokens == 0 || layout.head_dim == 0) {
        return Status::InvalidArgument("KV layout dimensions must be non-zero");
    }

    if (!IsValidAlignment(layout.alignment)) {
        return Status::InvalidArgument(
                "KV layout alignment must be a non-zero power of two");
    }

    if (layout.ElementBytes() == 0) {
        return Status::InvalidArgument(
                "KV layout dtype must have a fixed non-zero element size");
    }
    return Status::Ok();
}

} // namespace

size_t KVCacheLayout::ElementBytes() const noexcept {
    // nbytes()/lanes() require a fixed-length lane encoding. Reject scalable
    // and reserved encodings before either assertion-bearing dtype query.
    if (kv_dtype.IsUndefined() || kv_dtype.bits() == 0 || kv_dtype.raw_lanes() == 0 ||
        kv_dtype.raw_lanes() > std::numeric_limits<int16_t>::max()) {
        return 0;
    }
    return static_cast<size_t>(kv_dtype.nbytes());
}

Status KVCacheLayout::Validate() const noexcept try {
    AM_RETURN_IF_ERROR(ValidateGeometry(*this));

    if (padded_head_dim < head_dim) {
        return Status::InvalidArgument("KV padded_head_dim must be >= head_dim");
    }

    const size_t element_bytes = ElementBytes();

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

    if (size_t head_offset = 0;
        CheckOverflowMul(kv_head_idx, head_stride, &head_offset) ||
        CheckOverflowAdd(offset, head_offset, &offset)) {
        return Status::Overflow("KV head offset overflowed size_t");
    }

    if (size_t token_offset = 0;
        CheckOverflowMul(seq_pos, token_stride, &token_offset) ||
        CheckOverflowAdd(offset, token_offset, &offset)) {
        return Status::Overflow("KV token offset overflowed size_t");
    }

    if (size_t dim_offset = 0;
        CheckOverflowMul(dim_idx, ElementBytes(), &dim_offset) ||
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

StatusOr<size_t> KVCacheLayout::TotalBytes() const noexcept try {
    AM_ASSIGN_OR_RETURN(const size_t plane_bytes, BytesPerPlane());
    size_t bytes = 0;
    if (CheckOverflowMul(plane_bytes, size_t{2}, &bytes)) {
        return Status::Overflow("KV total bytes overflowed size_t");
    }
    return bytes;
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

StatusOr<KVCacheLayout> PlanKVCacheLayout(size_t num_layers,
                                          size_t num_kv_heads,
                                          size_t capacity_tokens,
                                          size_t head_dim,
                                          DataType dtype,
                                          size_t alignment) noexcept try {
    KVCacheLayout layout{
            .num_layers = num_layers,
            .num_kv_heads = num_kv_heads,
            .max_tokens = capacity_tokens,
            .head_dim = head_dim,
            .kv_dtype = dtype,
            .alignment = alignment,
    };
    AM_RETURN_IF_ERROR(ValidateGeometry(layout));

    const size_t element_bytes = layout.ElementBytes();
    // This element-count grain is a power of two because alignment is one.
    // It makes row bytes divisible by alignment without splitting an element.
    const size_t elements_per_alignment =
            alignment / std::gcd(alignment, element_bytes);
    AM_ASSIGN_OR_RETURN(layout.padded_head_dim,
                        AlignOffsetUp(head_dim, elements_per_alignment));

    if (CheckOverflowMul(layout.padded_head_dim, element_bytes, &layout.token_stride) ||
        CheckOverflowMul(capacity_tokens, layout.token_stride, &layout.head_stride) ||
        CheckOverflowMul(num_kv_heads, layout.head_stride, &layout.layer_stride)) {
        return Status::Overflow("KV layout stride computation overflowed size_t");
    }

    if (const auto total_bytes = layout.TotalBytes(); !total_bytes.ok()) {
        return total_bytes.status();
    }
    return layout;
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

} // namespace aethermind
