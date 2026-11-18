#ifndef AETHERMIND_BACKEND_CPU_KERNELS_ATTENTION_ATTENTION_PREFILL_AVX2_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_ATTENTION_ATTENTION_PREFILL_AVX2_INTERNAL_H

#include "attention_pair_avx2_internal.h"

namespace aethermind::cpu::detail {
namespace {

// Short Prefill retains the per-head driver; the dispatch floor is covered by
// the local A/B matrix and is independent of the invocation's KV capacity.
constexpr int64_t kMinPairedPrefillRows = 32;

template<KVElement Element, bool A16 = false, typename Args>
Status RunPairedPrefill(const Args& args, const KVCacheReadBinding& read) noexcept {
    const size_t rows = static_cast<size_t>(args.seq_len);
    const size_t dims = static_cast<size_t>(args.head_dim);
    const size_t group_size = static_cast<size_t>(args.num_q_heads / args.num_kv_heads);
    const auto& storage = read.storage;
    constexpr size_t act_bytes = A16 ? kElementBytes<Element> : sizeof(float);
    size_t row = 0;
    for (; rows - row >= 2; row += 2) {
        const size_t key_end = read.query_begin + row + 1;
        for (size_t kv_head = 0; kv_head < static_cast<size_t>(args.num_kv_heads); ++kv_head) {
            const size_t kv_offset = kv_head * storage.head_stride_bytes;
            for (size_t i = 0; i < group_size; ++i) {
                const size_t head_offset = (kv_head * group_size + i) * dims;
                const size_t query_offset = row * args.query_row_stride + head_offset;
                const size_t output_offset = row * args.output_row_stride + head_offset;
                if constexpr (A16) {
                    float widened[2][kMaxA16HeadDim];
                    float accum[2][kMaxA16HeadDim];
                    const auto* q0 = args.query + query_offset * act_bytes;
                    const auto* q1 = q0 + args.query_row_stride * act_bytes;
                    WidenQueryHead<Element>(q0, widened[0], dims);
                    WidenQueryHead<Element>(q1, widened[1], dims);
                    AccumulateHeadPair<Element, true>(
                            widened[0], widened[1], accum[0], accum[1], dims, args.scale,
                            storage.key_data + kv_offset, storage.value_data + kv_offset,
                            storage.token_stride_bytes, key_end);
                    auto* out0 = args.output + output_offset * act_bytes;
                    auto* out1 = out0 + args.output_row_stride * act_bytes;
                    for (size_t d = 0; d < dims; ++d) {
                        StoreA16Element(out0 + d * act_bytes, accum[0][d], args.act_dtype);
                        StoreA16Element(out1 + d * act_bytes, accum[1][d], args.act_dtype);
                    }
                } else {
                    const auto* q0 = args.query + query_offset;
                    auto* out0 = args.output + output_offset;
                    AccumulateHeadPair<Element, true>(
                            q0, q0 + args.query_row_stride, out0, out0 + args.output_row_stride, dims, args.scale,
                            storage.key_data + kv_offset, storage.value_data + kv_offset,
                            storage.token_stride_bytes, key_end);
                }
            }
        }
    }
    if (row == rows) {
        return Status::Ok();
    }
    return RunContiguousHeads<Element, A16>(args, read, row);
}

} // namespace
} // namespace aethermind::cpu::detail
#endif
