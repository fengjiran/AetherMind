#include "kvcache_update_internal.h"

#include <cstring>

namespace aethermind::cpu::detail {

Status RunKVCacheUpdateA16Reference(const KVCacheUpdateA16KernelArgs& args,
                                    const KVCacheAppendBinding& append) noexcept {
    const KVCacheLayerStorageBinding& storage = append.storage;
    const size_t token_count = append.end - append.begin;
    const size_t head_dim = storage.head_dim;
    constexpr size_t kElementBytes = sizeof(uint16_t);

    for (size_t head = 0; head < storage.num_kv_heads; ++head) {
        std::byte* const key_head = storage.key_data + head * storage.head_stride_bytes;
        std::byte* const value_head = storage.value_data + head * storage.head_stride_bytes;
        const size_t src_head_offset = head * head_dim;

        for (size_t token = 0; token < token_count; ++token) {
            const auto src_row = static_cast<int64_t>(token);
            const std::byte* const key_src =
                    args.key_data +
                    static_cast<size_t>(src_row * args.key_row_stride +
                                        static_cast<int64_t>(src_head_offset) *
                                                args.key_col_stride) *
                            kElementBytes;
            const std::byte* const value_src =
                    args.value_data +
                    static_cast<size_t>(src_row * args.value_row_stride +
                                        static_cast<int64_t>(src_head_offset) *
                                                args.value_col_stride) *
                            kElementBytes;
            std::byte* const key_dst =
                    key_head + (append.begin + token) * storage.token_stride_bytes;
            std::byte* const value_dst =
                    value_head + (append.begin + token) * storage.token_stride_bytes;

            // Same-dtype write-through: each element is copied verbatim, so the
            // update introduces no additional rounding. Only the logical
            // head_dim is written; padding and committed history stay intact.
            for (size_t dim = 0; dim < head_dim; ++dim) {
                const size_t src_offset =
                        static_cast<size_t>(static_cast<int64_t>(dim) * args.key_col_stride) *
                        kElementBytes;
                std::memcpy(key_dst + dim * kElementBytes,
                            key_src + src_offset, kElementBytes);
                std::memcpy(value_dst + dim * kElementBytes,
                            value_src + static_cast<size_t>(
                                                static_cast<int64_t>(dim) *
                                                args.value_col_stride) *
                                                kElementBytes,
                            kElementBytes);
            }
        }
    }
    return Status::Ok();
}

} // namespace aethermind::cpu::detail
