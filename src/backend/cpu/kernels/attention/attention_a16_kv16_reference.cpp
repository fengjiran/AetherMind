#include "aethermind/backend/cpu/kernels/common/a16_utils.h"
#include "aethermind/dtypes/bfloat16.h"
#include "aethermind/dtypes/half.h"
#include "attention_internal.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>

// This reference TU is compiled with -ffp-contract=off (see src/CMakeLists.txt):
// every dot product and weighted-V accumulation is an independent rounded F32
// product followed by a rounded F32 add, in ascending head_dim / token order.
//
// The 16-bit output cannot hold the F32 value accumulation the KV16 reference
// keeps in its F32 output buffer. Head dimensions are therefore processed in a
// fixed-size stack block: logits are recomputed per block, which trades a few
// extra QK dot products for zero heap allocation and exactly one output
// narrowing.

namespace aethermind::cpu::detail {
namespace {

/// Head-dimension block accumulated in F32 before one narrowing store.
constexpr int64_t kDimBlock = 64;

// Byte-safe widening of one stored 16-bit element; no live Half/BFloat16
// object is required at the storage address.
float WidenElement(const std::byte* src, DataType dtype) noexcept {
    uint16_t bits = 0;
    std::memcpy(&bits, src, sizeof(uint16_t));
    if (dtype == DataType::Float(16)) {
        return Half(bits, Half::from_bits());
    }
    return BFloat16(bits, BFloat16::from_bits());
}

float DotProductA16(const std::byte* query,
                    int64_t query_col_stride,
                    const std::byte* key,
                    int64_t head_dim,
                    DataType dtype) noexcept {
    const size_t element_bytes = A16ElementBytes(dtype);
    float sum = 0.0F;
    for (int64_t dim = 0; dim < head_dim; ++dim) {
        const float key_value =
                WidenElement(key + static_cast<size_t>(dim) * element_bytes, dtype);
        const float query_value =
                WidenElement(query + static_cast<size_t>(dim * query_col_stride) * element_bytes,
                             dtype);
        const float product = query_value * key_value;
        sum = sum + product;
    }
    return sum;
}

} // namespace

Status RunAttentionA16Reference(const AttentionA16KernelArgs& args,
                                const KVCacheReadBinding& read) noexcept {
    const KVCacheLayerStorageBinding& storage = read.storage;
    const auto group_size = static_cast<size_t>(args.num_q_heads / args.num_kv_heads);
    const auto head_dim = static_cast<size_t>(args.head_dim);
    const size_t element_bytes = A16ElementBytes(args.act_dtype);

    for (size_t q_row = 0; q_row < static_cast<size_t>(args.seq_len); ++q_row) {
        const size_t key_end = read.query_begin + q_row + 1;
        const std::byte* const q_row_data =
                args.query +
                static_cast<size_t>(static_cast<int64_t>(q_row) * args.query_row_stride) *
                        element_bytes;
        std::byte* const output_row_data =
                args.output +
                static_cast<size_t>(static_cast<int64_t>(q_row) * args.output_row_stride) *
                        element_bytes;

        for (size_t q_head = 0; q_head < static_cast<size_t>(args.num_q_heads); ++q_head) {
            const size_t kv_head = q_head / group_size;
            const auto head_offset = static_cast<int64_t>(q_head) * args.head_dim;
            const std::byte* const q_head_data =
                    q_row_data +
                    static_cast<size_t>(head_offset * args.query_col_stride) * element_bytes;
            std::byte* const output_head_data =
                    output_row_data +
                    static_cast<size_t>(head_offset * args.output_col_stride) * element_bytes;
            const std::byte* const key_head_data =
                    storage.key_data + kv_head * storage.head_stride_bytes;
            const std::byte* const value_head_data =
                    storage.value_data + kv_head * storage.head_stride_bytes;

            float max_logit = -std::numeric_limits<float>::infinity();
            for (size_t token = 0; token < key_end; ++token) {
                const std::byte* const key =
                        key_head_data + token * storage.token_stride_bytes;
                const float logit = DotProductA16(q_head_data, args.query_col_stride,
                                                  key, args.head_dim, args.act_dtype) *
                                    args.scale;
                max_logit = std::max(max_logit, logit);
            }

            float denominator = 0.0F;
            for (size_t token = 0; token < key_end; ++token) {
                const std::byte* const key =
                        key_head_data + token * storage.token_stride_bytes;
                const float logit = DotProductA16(q_head_data, args.query_col_stride,
                                                  key, args.head_dim, args.act_dtype) *
                                    args.scale;
                denominator = denominator + std::exp(logit - max_logit);
            }

            for (size_t dim_begin = 0; dim_begin < head_dim; dim_begin += kDimBlock) {
                const size_t dims = std::min<size_t>(
                        static_cast<size_t>(kDimBlock), head_dim - dim_begin);
                float accumulator[kDimBlock] = {};
                for (size_t token = 0; token < key_end; ++token) {
                    const std::byte* const key =
                            key_head_data + token * storage.token_stride_bytes;
                    const std::byte* const value =
                            value_head_data + token * storage.token_stride_bytes;
                    const float logit = DotProductA16(q_head_data, args.query_col_stride,
                                                      key, args.head_dim, args.act_dtype) *
                                        args.scale;
                    const float weight = std::exp(logit - max_logit);
                    for (size_t offset = 0; offset < dims; ++offset) {
                        const float value_element = WidenElement(
                                value + (dim_begin + offset) * element_bytes, args.act_dtype);
                        const float product = weight * value_element;
                        accumulator[offset] = accumulator[offset] + product;
                    }
                }

                for (size_t offset = 0; offset < dims; ++offset) {
                    const auto dim = static_cast<int64_t>(dim_begin + offset);
                    StoreA16Element(
                            output_head_data +
                                    static_cast<size_t>(dim * args.output_col_stride) *
                                            element_bytes,
                            accumulator[offset] / denominator,
                            args.act_dtype);
                }
            }
        }
    }
    return Status::Ok();
}

} // namespace aethermind::cpu::detail
