#ifndef AETHERMIND_BACKEND_CPU_KERNELS_ATTENTION_ATTENTION_DECODE_AVX2_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_ATTENTION_ATTENTION_DECODE_AVX2_INTERNAL_H

#include "attention_pair_avx2_internal.h"

namespace aethermind::cpu::detail {
namespace {

constexpr size_t kMinGroupedDecodeContext = 16;

bool UseGroupedDecode(const auto& args, const KVCacheReadBinding& read) noexcept {
    // Shape and the invocation frontier identify an append of one token after
    // committed history; ordinary Prefill (including one token) stays per-head.
    return args.seq_len == 1 && read.query_end >= kMinGroupedDecodeContext &&
           read.query_begin == read.committed_end &&
           args.num_q_heads > args.num_kv_heads;
}

template<KVElement Element, bool A16 = false, typename Args>
Status RunGroupedDecode(const Args& args, const KVCacheReadBinding& read) noexcept {
    const size_t dims = static_cast<size_t>(args.head_dim);
    const size_t group_size = static_cast<size_t>(args.num_q_heads / args.num_kv_heads);
    const auto& storage = read.storage;
    for (size_t kv_head = 0; kv_head < static_cast<size_t>(args.num_kv_heads); ++kv_head) {
        const auto* keys = storage.key_data + kv_head * storage.head_stride_bytes;
        const auto* values = storage.value_data + kv_head * storage.head_stride_bytes;
        for (size_t i = 0; i < group_size; i += 2) {
            const size_t head_offset = (kv_head * group_size + i) * dims;
            const bool paired = group_size - i >= 2;
            if constexpr (A16) {
                // Reuse bounded scratch for at most two heads, independent of context.
                float widened[2][kMaxA16HeadDim];
                float accum[2][kMaxA16HeadDim];
                const auto* q = args.query + head_offset * kElementBytes<Element>;
                WidenQueryHead<Element>(q, widened[0], dims);
                if (paired) {
                    WidenQueryHead<Element>(q + dims * kElementBytes<Element>, widened[1], dims);
                    AccumulateHeadPair<Element>(widened[0], widened[1], accum[0], accum[1],
                                                dims, args.scale, keys, values, storage.token_stride_bytes, read.query_end);
                } else {
                    AccumulateHead<Element>(widened[0], accum[0], dims, args.scale,
                                            keys, values, storage.token_stride_bytes, read.query_end);
                }
                auto* out = args.output + head_offset * kElementBytes<Element>;
                for (size_t h = 0; h < (paired ? 2U : 1U); ++h) {
                    for (size_t d = 0; d < dims; ++d) {
                        StoreA16Element(out + (h * dims + d) * kElementBytes<Element>, accum[h][d], args.act_dtype);
                    }
                }
            } else {
                const float* q = args.query + head_offset;
                float* out = args.output + head_offset;
                if (paired) {
                    AccumulateHeadPair<Element>(q, q + dims, out, out + dims, dims, args.scale,
                                                keys, values, storage.token_stride_bytes, read.query_end);
                } else {
                    AccumulateHead<Element>(q, out, dims, args.scale,
                                            keys, values, storage.token_stride_bytes, read.query_end);
                }
            }
        }
    }
    return Status::Ok();
}

} // namespace
} // namespace aethermind::cpu::detail
#endif
