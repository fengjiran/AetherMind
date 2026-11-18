#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/cpu/cpu_info.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/dtypes/bfloat16.h"
#include "aethermind/dtypes/half.h"
#include "aethermind/operators/op_params.h"
#include "inference/test_malloc_interposer.h"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

namespace {

using namespace aethermind;

struct Precision {
    DataType activation;
    DataType cache;
};

const std::array kPrecisions{
        Precision{DataType::Float32(), DataType::Float32()},
        Precision{DataType::Float32(), DataType::Float(16)},
        Precision{DataType::Float32(), DataType::BFloat(16)},
        Precision{DataType::BFloat(16), DataType::BFloat(16)},
};

StatusOr<ResolvedKernel> Resolve(const CpuBackend& backend, Precision precision, int64_t dim,
                                 ExecPhase phase = ExecPhase::kPrefill, int64_t group = 2, int64_t kv_heads = 2) {
    return backend.PrepareKernel(
            OpType::kAttention,
            {.device_type = DeviceType::kCPU,
             .act_dtype = precision.activation,
             .output_dtype = precision.activation,
             .weight_dtype = precision.activation,
             .state_dtype = precision.cache,
             .weight_format = WeightFormat::kPlain,
             .phase = phase},
            AttentionParams{.num_q_heads = group * kv_heads, .num_kv_heads = kv_heads, .head_dim = dim});
}

void Write(std::byte* dst, DataType dtype, float value) {
    if (dtype == DataType::Float32()) {
        std::memcpy(dst, &value, sizeof(value));
    } else {
        const uint16_t bits = dtype == DataType::Float(16) ? Half(value).bits() : BFloat16(value).x;
        std::memcpy(dst, &bits, sizeof(bits));
    }
}

bool IsNaN(const std::byte* src, DataType dtype) {
    if (dtype == DataType::Float32()) {
        float value;
        std::memcpy(&value, src, sizeof(value));
        return std::isnan(value);
    }
    uint16_t bits;
    std::memcpy(&bits, src, sizeof(bits));
    return std::isnan(static_cast<float>(BFloat16(bits, BFloat16::from_bits())));
}

Status Execute(const ResolvedKernel& kernel, const TensorView& q, const MutableTensorView& out,
               const KVCacheReadBinding& read, test::MallocCallCounts* counts = nullptr) {
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> params{};
    const std::array inputs{q};
    const std::array outputs{out};
    AM_RETURN_IF_ERROR(kernel.params_builder(
            {.inputs = inputs, .outputs = outputs, .attrs = kernel.attrs}, params.data()));
    if (counts != nullptr) {
        test::BeginMallocCallCounting();
    }
    const auto status = kernel.fn({.kernel_params = params.data(), .kv_read = &read, .attrs = kernel.attrs});
    if (counts != nullptr) {
        *counts = test::EndMallocCallCounting();
    }
    return status;
}

enum class Input { kFinite,
                   kInfiniteValue,
                   kNaNQuery,
                   kMaskedDiagonalValue,
                   kOverflowingScore };

void CompareWithReference(Precision p, int64_t dim, int64_t col_stride, int64_t rows,
                          size_t query_begin, Input input = Input::kFinite, int64_t group = 2,
                          int64_t kv_heads = 2) {
    const CpuBackend scalar(CpuFeaturePolicy{.disabled_features = CpuFeatureSet::From({CpuFeature::kAvx2})});
    const CpuBackend cpu;
    const auto phase = rows == 1 ? ExecPhase::kDecode : ExecPhase::kPrefill;
    const auto reference = Resolve(scalar, p, dim, phase, group, kv_heads);
    const auto candidate = Resolve(cpu, p, dim, phase, group, kv_heads);
    ASSERT_TRUE(reference.ok()) << reference.status().ToString();
    ASSERT_TRUE(candidate.ok()) << candidate.status().ToString();

    const int64_t width = group * kv_heads * dim;
    const int64_t query_stride = width * col_stride + 3;
    const int64_t output_stride = width * col_stride + 5;
    const size_t q_bytes = p.activation.nbytes();
    const size_t kv_bytes = p.cache.nbytes();
    const size_t capacity = query_begin + rows + 1;
    const size_t token_stride = (dim + 3) * kv_bytes;
    const size_t head_stride = (capacity + 1) * token_stride;
    std::vector<std::byte> queries(rows * query_stride * q_bytes);
    std::vector<std::byte> scalar_out(rows * output_stride * q_bytes, std::byte{0xcd});
    std::vector<std::byte> simd_out = scalar_out;
    std::vector<std::byte> keys(kv_heads * head_stride);
    std::vector<std::byte> values(keys.size());
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < width; ++col) {
            const float q = input == Input::kOverflowingScore
                                    ? 2.0e19F
                                    : static_cast<float>((row * 13 + col * 7) % 17 - 8) / 8.0F;
            Write(queries.data() + (row * query_stride + col * col_stride) * q_bytes, p.activation, q);
        }
    }
    if (input == Input::kNaNQuery) {
        // Only query head 0 is poisoned; its partner shares exactly the same KV.
        Write(queries.data(), p.activation, std::numeric_limits<float>::quiet_NaN());
    }
    for (size_t head = 0; head < static_cast<size_t>(kv_heads); ++head) {
        for (size_t token = 0; token < capacity; ++token) {
            for (size_t d = 0; d < static_cast<size_t>(dim); ++d) {
                const size_t offset = head * head_stride + token * token_stride + d * kv_bytes;
                const float key = input == Input::kOverflowingScore
                                          ? 1.0e19F
                                          : static_cast<float>(static_cast<int>((d + token * 3 + head) % 19) - 9) / 8.0F;
                float value = static_cast<float>((d * 3 + token + head * 9) % 23) / 16.0F;
                if (input == Input::kMaskedDiagonalValue && token == query_begin + 1) {
                    value = std::numeric_limits<float>::quiet_NaN();
                } else if (token == capacity - 1 || (input == Input::kInfiniteValue && token == 0)) {
                    value = std::numeric_limits<float>::infinity();
                }
                Write(keys.data() + offset, p.cache, key);
                Write(values.data() + offset, p.cache, value);
            }
        }
    }

    const size_t query_end = query_begin + rows;
    KVCacheReadBinding read{
            .storage = {.key_data = keys.data(), .value_data = values.data(), .dtype = p.cache, .num_kv_heads = static_cast<size_t>(kv_heads), .head_dim = static_cast<size_t>(dim), .token_capacity = capacity, .token_stride_bytes = token_stride, .head_stride_bytes = head_stride},
            .committed_end = query_begin,
            .visible_end = query_end,
            .query_begin = query_begin,
            .query_end = query_end};
    const std::array<int64_t, 2> shape{rows, width};
    const std::array query_strides{query_stride, col_stride};
    const std::array output_strides{output_stride, col_stride};
    const TensorView query(queries.data(), p.activation, shape, query_strides);
    const MutableTensorView ref_output(scalar_out.data(), p.activation, shape, output_strides);
    const MutableTensorView simd_output(simd_out.data(), p.activation, shape, output_strides);

    // Reject an invalid invocation before touching the prepared output.
    --read.visible_end;
    EXPECT_EQ(Execute(*candidate, query, simd_output, read), StatusCode::kInvalidArgument);
    ASSERT_EQ(scalar_out, simd_out);
    read.visible_end = capacity;
    auto reference_read = read;
    reference_read.visible_end = query_end;
    ASSERT_TRUE(Execute(*reference, query, ref_output, reference_read).ok());
    test::MallocCallCounts counts{};
    const bool check_allocations = test::MallocInterposerAvailable();
    ASSERT_TRUE(Execute(*candidate, query, simd_output, read, check_allocations ? &counts : nullptr).ok());
    if (check_allocations) {
        EXPECT_EQ(counts.malloc_calls + counts.calloc_calls + counts.realloc_calls + counts.free_calls +
                          counts.aligned_alloc_calls + counts.posix_memalign_calls + counts.memalign_calls,
                  0U);
    }
    if (input == Input::kNaNQuery) {
        for (int64_t col = 0; col < width; ++col) {
            const auto offset = col * col_stride * q_bytes;
            const bool poisoned = col < dim;
            EXPECT_EQ(IsNaN(scalar_out.data() + offset, p.activation), poisoned);
            EXPECT_EQ(IsNaN(simd_out.data() + offset, p.activation), poisoned);
            if (poisoned) {
                // Compare classification, not the implementation's NaN payload.
                std::memcpy(simd_out.data() + offset, scalar_out.data() + offset, q_bytes);
            }
        }
    }
    if (input == Input::kMaskedDiagonalValue) {
        for (int64_t row = 0; row < rows; ++row) {
            for (int64_t col = 0; col < width; ++col) {
                const auto offset = (row * output_stride + col * col_stride) * q_bytes;
                EXPECT_EQ(IsNaN(scalar_out.data() + offset, p.activation), row != 0);
                EXPECT_EQ(IsNaN(simd_out.data() + offset, p.activation), row != 0);
                if (row != 0) {
                    std::memcpy(simd_out.data() + offset, scalar_out.data() + offset, q_bytes);
                }
            }
        }
    }
    if (input == Input::kOverflowingScore) {
        ASSERT_TRUE(IsNaN(scalar_out.data(), p.activation));
        for (int64_t row = 0; row < rows; ++row) {
            for (int64_t col = 0; col < width; ++col) {
                const auto offset = (row * output_stride + col * col_stride) * q_bytes;
                EXPECT_TRUE(IsNaN(simd_out.data() + offset, p.activation));
            }
        }
    } else {
        // Full byte comparison also checks the untouched row and column padding.
        EXPECT_EQ(std::memcmp(scalar_out.data(), simd_out.data(), scalar_out.size()), 0);
    }
}

TEST(CPUKernelAttentionSimd, FeaturePolicySelectsReferenceAndF16cIsIndependent) {
    const auto capabilities = aethermind::cpu::DetectCpuCapabilities();
    ASSERT_TRUE(capabilities.ok());
    const CpuBackend cpu(*capabilities);
    const CpuBackend no_avx2(CpuFeaturePolicy{.disabled_features = CpuFeatureSet::From({CpuFeature::kAvx2})});
    const CpuBackend no_f16c(CpuFeaturePolicy{.disabled_features = CpuFeatureSet::From({CpuFeature::kF16c})});
    for (const auto p: kPrecisions) {
        const auto native = Resolve(cpu, p, 128);
        const auto scalar = Resolve(no_avx2, p, 128);
        const auto restricted = Resolve(no_f16c, p, 128);
        ASSERT_TRUE(native.ok());
        ASSERT_TRUE(scalar.ok());
        ASSERT_TRUE(restricted.ok());
        EXPECT_TRUE(scalar->name.ends_with("_reference"));
        constexpr bool compiled_avx2 =
#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
                true;
#else
                false;
#endif
        constexpr bool compiled_f16 =
#if defined(GEMM_HAS_A32_F16_AVX2_FMA_KERNEL)
                true;
#else
                false;
#endif
        const bool f16 = p.cache == DataType::Float(16);
        const auto features = capabilities->effective_features;
        const bool expect_simd = compiled_avx2 &&
                                 features.ContainsAll(CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma})) &&
                                 (!f16 || (compiled_f16 && features.Contains(CpuFeature::kF16c)));
        EXPECT_EQ(native->name.ends_with("_avx2"), expect_simd);
        EXPECT_EQ(restricted->name, f16 ? scalar->name : native->name);
    }
}

TEST(CPUKernelAttentionSimd, MatchesReferenceForPaddedCausalPrefillAndLongDecode) {
    for (const auto p: kPrecisions) {
        CompareWithReference(p, 17, 1, 4, 5);
        CompareWithReference(p, 128, 1, 1, 2047);
    }
}

TEST(CPUKernelAttentionSimd, PrefillPairUsesItsOwnCausalDiagonalWithOddRowTail) {
    for (const auto p: kPrecisions) {
        CompareWithReference(p, 65, 1, 31, 17, Input::kMaskedDiagonalValue, 1, 2);
        CompareWithReference(p, 65, 1, 32, 17, Input::kMaskedDiagonalValue, 1, 2);
        CompareWithReference(p, 65, 1, 33, 17, Input::kMaskedDiagonalValue, 1, 2);
        CompareWithReference(p, 128, 1, 33, 0, Input::kFinite, 3, 2);
        CompareWithReference(p, 128, 1, 33, 2015, Input::kFinite);
    }
}

TEST(CPUKernelAttentionSimd, DecodeGroupPairsAndOddTailDoNotCrossKvHeads) {
    for (const auto p: kPrecisions) {
        CompareWithReference(p, 65, 1, 1, 255, Input::kFinite, 3, 2);
        CompareWithReference(p, 128, 1, 1, 255, Input::kFinite, 5, 1);
        CompareWithReference(p, 128, 1, 1, 255, Input::kFinite, 1, 2);
    }
}

TEST(CPUKernelAttentionSimd, ContextBoundaryAndStridedDecodeMatchReference) {
    for (const auto p: kPrecisions) {
        CompareWithReference(p, 64, 1, 1, 14);
        CompareWithReference(p, 64, 1, 1, 15);
        CompareWithReference(p, 256, 1, 1, 127);
        CompareWithReference(p, 128, 2, 1, 255);
    }
}

TEST(CPUKernelAttentionSimd, FallsBackForStridedColumnsAndWideA16Heads) {
    for (const auto p: kPrecisions) {
        CompareWithReference(p, 19, 2, 3, 0);
    }
    CompareWithReference(kPrecisions.back(), 257, 1, 1, 7);
}

TEST(CPUKernelAttentionSimd, SharedKvDoesNotPropagateNaNToPartnerQuery) {
    for (const auto p: kPrecisions) {
        CompareWithReference(p, 128, 1, 1, 127, Input::kNaNQuery);
    }
}

TEST(CPUKernelAttentionSimd, PreservesInfinityAndOverflowClassification) {
    for (const auto p: kPrecisions) {
        CompareWithReference(p, 65, 1, 1, 255, Input::kInfiniteValue);
        if (p.cache != DataType::Float(16)) {
            CompareWithReference(p, 65, 1, 1, 255, Input::kOverflowingScore);
        }
    }
}

} // namespace
