#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/dtypes/bfloat16.h"
#include "aethermind/dtypes/half.h"
#include "aethermind/operators/op_params.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

namespace {

using namespace aethermind;

KernelSelector MakeSelector(DataType act_dtype) {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = act_dtype,
            .output_dtype = act_dtype,
            .weight_dtype = act_dtype,
            .state_dtype = act_dtype,
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kBoth,
    };
}

StatusOr<ResolvedKernel> PrepareKernel(DataType act_dtype, int64_t num_q_heads,
                                       int64_t num_kv_heads, int64_t head_dim) {
    CpuBackend backend;
    return backend.PrepareKernel(OpType::kAttention, MakeSelector(act_dtype),
                                 OpParams{AttentionParams{.num_q_heads = num_q_heads,
                                                          .num_kv_heads = num_kv_heads,
                                                          .head_dim = head_dim}});
}

struct PreparedParams {
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> storage{};
};

Status RunEntry(const ResolvedKernel& kernel, const TensorView& query,
                const MutableTensorView& output, const KVCacheReadBinding& read) noexcept {
    PreparedParams prepared;
    const std::array<TensorView, 1> inputs = {query};
    const std::array<MutableTensorView, 1> outputs = {output};
    AM_RETURN_IF_ERROR(kernel.params_builder(
            KernelParamsBuildContext{.inputs = inputs, .outputs = outputs, .attrs = kernel.attrs},
            prepared.storage.data()));
    return kernel.fn(KernelContext{.kernel_params = prepared.storage.data(),
                                   .kv_read = &read,
                                   .attrs = kernel.attrs});
}

void WriteA16(std::byte* destination, float value, DataType dtype, size_t index) {
    const uint16_t bits = dtype == DataType::Float(16) ? Half(value).bits() : BFloat16(value).x;
    std::memcpy(destination + index * 2, &bits, sizeof(bits));
}

float ReadA16(const std::byte* source, DataType dtype, size_t index) {
    uint16_t bits = 0;
    std::memcpy(&bits, source + index * 2, sizeof(bits));
    return dtype == DataType::Float(16) ? static_cast<float>(Half(bits, Half::from_bits()))
                                        : static_cast<float>(BFloat16(bits, BFloat16::from_bits()));
}

KVCacheReadBinding MakeRead(std::byte* key, std::byte* value, DataType dtype,
                            size_t heads, size_t capacity, size_t head_dim,
                            size_t token_stride_bytes, size_t head_stride_bytes,
                            size_t committed_end, size_t visible_end,
                            size_t query_begin, size_t query_end) {
    return KVCacheReadBinding{
            .storage = {.key_data = key,
                        .value_data = value,
                        .dtype = dtype,
                        .num_kv_heads = heads,
                        .head_dim = head_dim,
                        .token_capacity = capacity,
                        .token_stride_bytes = token_stride_bytes,
                        .head_stride_bytes = head_stride_bytes},
            .committed_end = committed_end,
            .visible_end = visible_end,
            .query_begin = query_begin,
            .query_end = query_end,
    };
}

// Independent double-precision causal attention oracle. All inputs are exactly
// representable in the 16-bit dtype, so widening reproduces the source floats;
// the oracle never calls the kernel's widening or narrowing helpers.
std::vector<float> OracleAttention(const std::vector<float>& query,
                                   const std::vector<float>& key,
                                   const std::vector<float>& value,
                                   int64_t num_q_heads, int64_t num_kv_heads,
                                   int64_t head_dim, int64_t seq_len,
                                   int64_t query_begin) {
    const int64_t group = num_q_heads / num_kv_heads;
    const int64_t cache_tokens = query_begin + seq_len;
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    std::vector<float> output(static_cast<size_t>(seq_len * num_q_heads * head_dim), 0.0F);
    auto kv_index = [&](int64_t kv_head, int64_t token, int64_t dim) {
        return static_cast<size_t>((kv_head * cache_tokens + token) * head_dim + dim);
    };
    for (int64_t q_row = 0; q_row < seq_len; ++q_row) {
        const int64_t key_end = query_begin + q_row + 1;
        for (int64_t q_head = 0; q_head < num_q_heads; ++q_head) {
            const int64_t kv_head = q_head / group;
            const int64_t q_base = (q_row * num_q_heads + q_head) * head_dim;
            double max_logit = -std::numeric_limits<double>::infinity();
            for (int64_t token = 0; token < key_end; ++token) {
                double dot = 0.0;
                for (int64_t dim = 0; dim < head_dim; ++dim) {
                    dot += static_cast<double>(query[static_cast<size_t>(q_base + dim)]) *
                           static_cast<double>(key[kv_index(kv_head, token, dim)]);
                }
                max_logit = std::max(max_logit, dot * scale);
            }
            double denominator = 0.0;
            std::vector<double> acc(static_cast<size_t>(head_dim), 0.0);
            for (int64_t token = 0; token < key_end; ++token) {
                double dot = 0.0;
                for (int64_t dim = 0; dim < head_dim; ++dim) {
                    dot += static_cast<double>(query[static_cast<size_t>(q_base + dim)]) *
                           static_cast<double>(key[kv_index(kv_head, token, dim)]);
                }
                const double weight = std::exp(dot * scale - max_logit);
                denominator += weight;
                for (int64_t dim = 0; dim < head_dim; ++dim) {
                    acc[static_cast<size_t>(dim)] +=
                            weight * static_cast<double>(value[kv_index(kv_head, token, dim)]);
                }
            }
            for (int64_t dim = 0; dim < head_dim; ++dim) {
                output[static_cast<size_t>(q_base + dim)] =
                        static_cast<float>(acc[static_cast<size_t>(dim)] / denominator);
            }
        }
    }
    return output;
}

void ExpectMatchesOracle(DataType dtype, int64_t num_q_heads, int64_t num_kv_heads,
                         int64_t head_dim, int64_t seq_len, int64_t query_begin,
                         const std::vector<float>& query, const std::vector<float>& key,
                         const std::vector<float>& value) {
    const auto kernel = PrepareKernel(dtype, num_q_heads, num_kv_heads, head_dim);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();

    const int64_t cache_tokens = query_begin + seq_len;
    const size_t kv_elems = static_cast<size_t>(num_kv_heads * cache_tokens * head_dim);
    std::vector<std::byte> key_cache(kv_elems * 2);
    std::vector<std::byte> value_cache(kv_elems * 2);
    for (size_t index = 0; index < kv_elems; ++index) {
        WriteA16(key_cache.data(), key[index], dtype, index);
        WriteA16(value_cache.data(), value[index], dtype, index);
    }

    const int64_t hidden = num_q_heads * head_dim;
    const std::array<int64_t, 2> shape = {seq_len, hidden};
    const std::array<int64_t, 2> strides = {hidden, 1};
    std::vector<std::byte> query_bytes(static_cast<size_t>(seq_len * hidden) * 2);
    for (size_t index = 0; index < query.size(); ++index) {
        WriteA16(query_bytes.data(), query[index], dtype, index);
    }
    std::vector<std::byte> output_bytes(static_cast<size_t>(seq_len * hidden) * 2);

    const size_t token_stride = static_cast<size_t>(head_dim) * 2;
    const size_t head_stride = static_cast<size_t>(cache_tokens) * token_stride;
    const auto read = MakeRead(key_cache.data(), value_cache.data(), dtype,
                               static_cast<size_t>(num_kv_heads), static_cast<size_t>(cache_tokens),
                               static_cast<size_t>(head_dim), token_stride, head_stride,
                               static_cast<size_t>(cache_tokens),
                               static_cast<size_t>(cache_tokens),
                               static_cast<size_t>(query_begin),
                               static_cast<size_t>(cache_tokens));
    ASSERT_TRUE(RunEntry(*kernel,
                         TensorView(query_bytes.data(), dtype, shape, strides),
                         MutableTensorView(output_bytes.data(), dtype, shape, strides),
                         read)
                        .ok());

    const auto expected = OracleAttention(query, key, value, num_q_heads, num_kv_heads,
                                          head_dim, seq_len, query_begin);
    const double tolerance = dtype == DataType::Float(16) ? 4.9e-4 : 3.9e-3;
    ASSERT_EQ(output_bytes.size(), expected.size() * 2);
    for (size_t index = 0; index < expected.size(); ++index) {
        const double actual = ReadA16(output_bytes.data(), dtype, index);
        EXPECT_NEAR(actual, expected[index],
                    tolerance * std::abs(expected[index]) + 2.0e-7)
                << "output index " << index;
    }
}

TEST(CPUKernelAttentionA16, ResolvesF16AndBF16References) {
    const auto f16 = PrepareKernel(DataType::Float(16), 1, 1, 2);
    ASSERT_TRUE(f16.ok()) << f16.status().ToString();
    EXPECT_EQ(std::string_view{f16->name}, "cpu::attention_a16_f16_reference");
    const auto bf16 = PrepareKernel(DataType::BFloat(16), 1, 1, 2);
    ASSERT_TRUE(bf16.ok()) << bf16.status().ToString();
    EXPECT_EQ(std::string_view{bf16->name}, "cpu::attention_a16_bf16_reference");
}

TEST(CPUKernelAttentionA16, Float16PrefillMatchesIndependentOracle) {
    ExpectMatchesOracle(DataType::Float(16), 1, 1, 2, 2, 0,
                        {1.0F, 0.0F, 0.0F, 1.0F},
                        {1.0F, 0.0F, 0.0F, 1.0F},
                        {10.0F, 0.0F, 0.0F, 20.0F});
}

TEST(CPUKernelAttentionA16, BFloat16PrefillMatchesIndependentOracle) {
    ExpectMatchesOracle(DataType::BFloat(16), 1, 1, 2, 2, 0,
                        {1.0F, 0.0F, 0.0F, 1.0F},
                        {1.0F, 0.0F, 0.0F, 1.0F},
                        {10.0F, 0.0F, 0.0F, 20.0F});
}

TEST(CPUKernelAttentionA16, GroupedQueryHeadsMapToKVHeads) {
    ExpectMatchesOracle(DataType::Float(16), 4, 2, 1, 1, 0,
                        {1.0F, 1.0F, 1.0F, 1.0F},
                        {1.0F, 1.0F},
                        {10.0F, 20.0F});
}

TEST(CPUKernelAttentionA16, DecodeUsesHistoryAndCurrentToken) {
    // query_begin=1: the single query row reads tokens [0, 2).
    ExpectMatchesOracle(DataType::BFloat(16), 1, 1, 1, 1, 1,
                        {1.0F},
                        {0.0F, 1.0F},
                        {2.0F, 6.0F});
}

TEST(CPUKernelAttentionA16, HeadDimWiderThanAccumulatorBlockMatchesOracle) {
    // head_dim 4 with a two-token prefill exercises the block loop boundary.
    ExpectMatchesOracle(DataType::Float(16), 1, 1, 4, 2, 0,
                        {1.0F, 0.5F, -1.0F, 2.0F,
                         0.5F, 1.0F, 2.0F, -1.0F},
                        {1.0F, 0.5F, -1.0F, 2.0F,
                         2.0F, -0.5F, 1.0F, 0.5F},
                        {4.0F, 8.0F, -2.0F, 1.0F,
                         1.0F, 2.0F, 4.0F, 16.0F});
}

TEST(CPUKernelAttentionA16, RejectsStorageDTypeMismatch) {
    const auto kernel = PrepareKernel(DataType::Float(16), 1, 1, 1);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    constexpr int64_t shape[2] = {1, 1};
    constexpr int64_t strides[2] = {1, 1};
    std::array<std::byte, 2> query{};
    std::array<std::byte, 2> output{};
    std::array<std::byte, 2> key_cache{};
    std::array<std::byte, 2> value_cache{};
    const auto read = MakeRead(key_cache.data(), value_cache.data(), DataType::BFloat(16),
                               1, 1, 1, 2, 2, 1, 1, 0, 1);
    EXPECT_EQ(RunEntry(*kernel,
                       TensorView(query.data(), DataType::Float(16), shape, strides),
                       MutableTensorView(output.data(), DataType::Float(16), shape, strides),
                       read),
              StatusCode::kInvalidArgument);
}

} // namespace
