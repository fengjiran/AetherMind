#include "attention_internal.h"

#include "aethermind/dtypes/bfloat16.h"
#include "aethermind/dtypes/half.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>

// This reference TU is compiled with -ffp-contract=off (see src/CMakeLists.txt):
// every dot product and weighted-V accumulation is an independent rounded F32
// product followed by a rounded F32 add, in ascending head_dim / token order.

namespace aethermind::cpu::detail {
namespace {

// Byte-safe widening of one cache element; no float alignment is required on
// the cache. Finite F16/BF16 values widen exactly to F32.
float WidenElement(const std::byte* src, DataType kv_dtype) noexcept {
    uint16_t bits = 0;
    std::memcpy(&bits, src, sizeof(uint16_t));
    if (kv_dtype == DataType::Float(16)) {
        return Half(bits, Half::from_bits());
    }
    return aethermind::detail::bf16_to_fp32_value(bits);
}

float DotProductKV16(const float* query,
                     int64_t query_col_stride,
                     const std::byte* key,
                     int64_t head_dim,
                     DataType kv_dtype) noexcept {
    float sum = 0.0F;
    for (int64_t dim = 0; dim < head_dim; ++dim) {
        const float key_value = WidenElement(key + dim * 2, kv_dtype);
        const float product = query[dim * query_col_stride] * key_value;
        sum = sum + product;
    }
    return sum;
}

} // namespace

Status RunAttentionKV16Reference(const AttentionF32KernelArgs& args,
                                 const KVCacheReadBinding& read,
                                 DataType kv_dtype) noexcept {
    const KVCacheLayerStorageBinding& storage = read.storage;
    const auto group_size = static_cast<size_t>(args.num_q_heads / args.num_kv_heads);
    const auto head_dim = static_cast<size_t>(args.head_dim);

    for (size_t q_row = 0; q_row < static_cast<size_t>(args.seq_len); ++q_row) {
        const size_t key_end = read.query_begin + q_row + 1;
        const float* const q_row_data = args.query + q_row * args.query_row_stride;
        float* const output_row_data = args.output + q_row * args.output_row_stride;

        for (size_t q_head = 0; q_head < static_cast<size_t>(args.num_q_heads); ++q_head) {
            const size_t kv_head = q_head / group_size;
            const auto head_offset = q_head * args.head_dim;
            const float* const q_head_data = q_row_data + head_offset * args.query_col_stride;
            float* const output_head_data =
                    output_row_data + head_offset * args.output_col_stride;
            const std::byte* const key_head_data =
                    storage.key_data + kv_head * storage.head_stride_bytes;
            const std::byte* const value_head_data =
                    storage.value_data + kv_head * storage.head_stride_bytes;

            float max_logit = -std::numeric_limits<float>::infinity();
            for (size_t token = 0; token < key_end; ++token) {
                const std::byte* const key =
                        key_head_data + token * storage.token_stride_bytes;
                const float logit = DotProductKV16(q_head_data, args.query_col_stride,
                                                   key, args.head_dim, kv_dtype) *
                                    args.scale;
                max_logit = std::max(max_logit, logit);
            }

            for (size_t dim = 0; dim < head_dim; ++dim) {
                output_head_data[static_cast<int64_t>(dim) * args.output_col_stride] = 0.0F;
            }

            float denominator = 0.0F;
            for (size_t token = 0; token < key_end; ++token) {
                const std::byte* const key =
                        key_head_data + token * storage.token_stride_bytes;
                const std::byte* const value =
                        value_head_data + token * storage.token_stride_bytes;
                const float logit = DotProductKV16(q_head_data, args.query_col_stride,
                                                   key, args.head_dim, kv_dtype) *
                                    args.scale;
                const float weight = std::exp(logit - max_logit);
                denominator = denominator + weight;
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    const float value_element = WidenElement(value + dim * 2, kv_dtype);
                    const float product = weight * value_element;
                    float* const out =
                            output_head_data + static_cast<int64_t>(dim) * args.output_col_stride;
                    *out = *out + product;
                }
            }

            for (size_t dim = 0; dim < head_dim; ++dim) {
                output_head_data[static_cast<int64_t>(dim) * args.output_col_stride] /= denominator;
            }
        }
    }
    return Status::Ok();
}

} // namespace aethermind::cpu::detail