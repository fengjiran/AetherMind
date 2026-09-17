#include "kvcache_update_internal.h"

#include <cstring>

namespace aethermind::cpu::detail {

Status RunKVCacheUpdateF32Reference(const KVCacheUpdateF32KernelArgs& args,
                                    const KVCacheAppendBinding& append) noexcept {
    const auto& storage = append.storage;
    const size_t token_count = append.end - append.begin;
    const size_t head_dim = storage.head_dim;

    for (size_t head = 0; head < storage.num_kv_heads; ++head) {
        auto* const key_head = reinterpret_cast<float*>(
                storage.key_data + head * storage.head_stride_bytes);
        auto* const value_head = reinterpret_cast<float*>(
                storage.value_data + head * storage.head_stride_bytes);
        const size_t src_head_offset = head * head_dim;

        for (size_t token = 0; token < token_count; ++token) {
            const auto src_row = static_cast<int64_t>(token);
            const auto* const key_src =
                    args.key_data + src_row * args.key_row_stride +
                    static_cast<int64_t>(src_head_offset) * args.key_col_stride;
            const auto* const value_src =
                    args.value_data + src_row * args.value_row_stride +
                    static_cast<int64_t>(src_head_offset) * args.value_col_stride;
            auto* const key_dst =
                    reinterpret_cast<float*>(reinterpret_cast<std::byte*>(key_head) +
                                             (append.begin + token) * storage.token_stride_bytes);
            auto* const value_dst =
                    reinterpret_cast<float*>(reinterpret_cast<std::byte*>(value_head) +
                                             (append.begin + token) * storage.token_stride_bytes);

            if (args.key_col_stride == 1) {
                std::memcpy(key_dst, key_src, head_dim * sizeof(float));
            } else {
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    key_dst[dim] = key_src[static_cast<int64_t>(dim) * args.key_col_stride];
                }
            }

            if (args.value_col_stride == 1) {
                std::memcpy(value_dst, value_src, head_dim * sizeof(float));
            } else {
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    value_dst[dim] = value_src[static_cast<int64_t>(dim) * args.value_col_stride];
                }
            }
        }
    }
    return Status::Ok();
}

} // namespace aethermind::cpu::detail
