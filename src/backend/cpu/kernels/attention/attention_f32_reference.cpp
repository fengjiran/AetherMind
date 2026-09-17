#include "attention_internal.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace aethermind::cpu::detail {
namespace {

float DotProduct(const float* query,
                 int64_t query_col_stride,
                 const float* key,
                 int64_t head_dim) noexcept {
    float sum = 0.0F;
    for (int64_t dim = 0; dim < head_dim; ++dim) {
        sum += query[dim * query_col_stride] * key[dim];
    }
    return sum;
}

} // namespace

Status RunAttentionF32Reference(const AttentionF32KernelArgs& args,
                                const KVCacheReadBinding& read) noexcept {
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
            const auto* const key_head_data =
                    storage.key_data + kv_head * storage.head_stride_bytes;
            const auto* const value_head_data =
                    storage.value_data + kv_head * storage.head_stride_bytes;

            float max_logit = -std::numeric_limits<float>::infinity();
            for (size_t token = 0; token < key_end; ++token) {
                const auto* const key = reinterpret_cast<const float*>(
                        key_head_data + token * storage.token_stride_bytes);
                const float logit = DotProduct(q_head_data, args.query_col_stride,
                                               key, args.head_dim) *
                                    args.scale;
                max_logit = std::max(max_logit, logit);
            }

            for (size_t dim = 0; dim < head_dim; ++dim) {
                output_head_data[static_cast<int64_t>(dim) * args.output_col_stride] = 0.0F;
            }

            float denominator = 0.0f;
            for (size_t token = 0; token < key_end; ++token) {
                const auto* const key = reinterpret_cast<const float*>(
                        key_head_data + token * storage.token_stride_bytes);
                const auto* const value = reinterpret_cast<const float*>(
                        value_head_data + token * storage.token_stride_bytes);
                const float logit = DotProduct(q_head_data, args.query_col_stride,
                                               key, args.head_dim) *
                                    args.scale;
                const float weight = std::exp(logit - max_logit);
                denominator += weight;
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    output_head_data[static_cast<int64_t>(dim) * args.output_col_stride] +=
                            weight * value[dim];
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
