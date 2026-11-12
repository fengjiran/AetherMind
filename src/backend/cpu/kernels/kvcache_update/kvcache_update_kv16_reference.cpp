#include "aethermind/dtypes/bfloat16.h"
#include "aethermind/dtypes/half.h"
#include "kvcache_update_internal.h"

#include <cstring>

namespace aethermind::cpu::detail {
namespace {

// Scalar narrowing follows the frozen conversion contract (fixed RNE, gradual
// underflow, signed zero, overflow to signed infinity, quiet NaN). The batch
// converters are not used here: cache rows are strided and may be unaligned.
uint16_t NarrowElement(float value, DataType kv_dtype) noexcept {
    if (kv_dtype == DataType::Float(16)) {
        return Half(value).bits();
    }
    return BFloat16(value).x;
}

} // namespace

Status RunKVCacheUpdateKV16Reference(const KVCacheUpdateF32KernelArgs& args,
                                     const KVCacheAppendBinding& append,
                                     DataType kv_dtype) noexcept {
    const KVCacheLayerStorageBinding& storage = append.storage;
    const size_t token_count = append.end - append.begin;
    const size_t head_dim = storage.head_dim;

    for (size_t head = 0; head < storage.num_kv_heads; ++head) {
        std::byte* const key_head = storage.key_data + head * storage.head_stride_bytes;
        std::byte* const value_head = storage.value_data + head * storage.head_stride_bytes;
        const size_t src_head_offset = head * head_dim;

        for (size_t token = 0; token < token_count; ++token) {
            const auto src_row = static_cast<int64_t>(token);
            const float* const key_src =
                    args.key_data + src_row * args.key_row_stride +
                    static_cast<int64_t>(src_head_offset) * args.key_col_stride;
            const float* const value_src =
                    args.value_data + src_row * args.value_row_stride +
                    static_cast<int64_t>(src_head_offset) * args.value_col_stride;
            std::byte* const key_dst =
                    key_head + (append.begin + token) * storage.token_stride_bytes;
            std::byte* const value_dst =
                    value_head + (append.begin + token) * storage.token_stride_bytes;

            // Only the logical head_dim is written; padding, committed history
            // and out-of-window bytes are untouched. Two-byte stores are
            // byte-safe and impose no float alignment on the cache.
            for (size_t dim = 0; dim < head_dim; ++dim) {
                const uint16_t key_bits = NarrowElement(
                        key_src[static_cast<int64_t>(dim) * args.key_col_stride], kv_dtype);
                const uint16_t value_bits = NarrowElement(
                        value_src[static_cast<int64_t>(dim) * args.value_col_stride], kv_dtype);
                std::memcpy(key_dst + dim * 2, &key_bits, sizeof(uint16_t));
                std::memcpy(value_dst + dim * 2, &value_bits, sizeof(uint16_t));
            }
        }
    }
    return Status::Ok();
}

} // namespace aethermind::cpu::detail