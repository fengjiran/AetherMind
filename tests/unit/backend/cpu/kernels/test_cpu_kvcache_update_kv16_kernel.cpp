#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/dtypes/bfloat16.h"
#include "aethermind/dtypes/half.h"
#include "aethermind/operators/op_params.h"
#include "backend/cpu/kernels/kvcache_update/kvcache_update_internal.h"
#include "inference/test_malloc_interposer.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#include <gtest/gtest.h>

namespace {

using namespace aethermind;

KernelSelector MakeSelector(DataType state_dtype) {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .state_dtype = state_dtype,
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kBoth,
    };
}

StatusOr<ResolvedKernel> PrepareKernel(DataType state_dtype) {
    CpuBackend backend;
    return backend.PrepareKernel(OpType::kKVCacheUpdate, MakeSelector(state_dtype),
                                 OpParams{KVCacheUpdateParams{}});
}

struct PreparedParams {
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> storage{};
};

Status RunEntry(const ResolvedKernel& kernel, const TensorView& key, const TensorView& value,
                const KVCacheAppendBinding& append) noexcept {
    PreparedParams prepared;
    const std::array<TensorView, 2> inputs = {key, value};
    AM_RETURN_IF_ERROR(kernel.params_builder(
            KernelParamsBuildContext{.inputs = inputs, .outputs = {}, .attrs = kernel.attrs},
            prepared.storage.data()));
    return kernel.fn(KernelContext{.kernel_params = prepared.storage.data(),
                                   .kv_append = &append,
                                   .attrs = kernel.attrs});
}

KVCacheAppendBinding MakeAppend(std::byte* key, std::byte* value, DataType dtype,
                                size_t heads, size_t capacity, size_t head_dim,
                                size_t token_stride_bytes, size_t head_stride_bytes,
                                size_t begin, size_t end) {
    return KVCacheAppendBinding{
            .storage = {.key_data = key,
                        .value_data = value,
                        .dtype = dtype,
                        .num_kv_heads = heads,
                        .head_dim = head_dim,
                        .token_capacity = capacity,
                        .token_stride_bytes = token_stride_bytes,
                        .head_stride_bytes = head_stride_bytes},
            .begin = begin,
            .end = end,
    };
}

uint16_t ReadBits(const std::byte* cache, size_t index) {
    uint16_t bits = 0;
    std::memcpy(&bits, cache + index * 2, sizeof(bits));
    return bits;
}

// Independent expected 16-bit patterns for exactly-representable and special
// F32 values (not produced by the kernel's conversion helpers).
TEST(CPUKernelKVCacheUpdateKV16, ResolvesF16AndBF16References) {
    const auto f16 = PrepareKernel(DataType::Float(16));
    ASSERT_TRUE(f16.ok()) << f16.status().ToString();
    EXPECT_EQ(std::string_view{f16->name}, "cpu::kvcache_update_f16_reference");
    const auto bf16 = PrepareKernel(DataType::BFloat(16));
    ASSERT_TRUE(bf16.ok()) << bf16.status().ToString();
    EXPECT_EQ(std::string_view{bf16->name}, "cpu::kvcache_update_bf16_reference");
}

TEST(CPUKernelKVCacheUpdateKV16, WritesKnownFloat16Bits) {
    constexpr int64_t shape[2] = {1, 4};
    constexpr int64_t strides[2] = {4, 1};
    const std::array<float, 4> act = {1.0F, 0.5F, -2.0F, 0.0F};
    std::array<std::byte, 8> key_cache{};
    std::array<std::byte, 8> value_cache{};

    const auto kernel = PrepareKernel(DataType::Float(16));
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    ASSERT_TRUE(RunEntry(*kernel,
                         TensorView(act.data(), DataType::Float32(), shape, strides),
                         TensorView(act.data(), DataType::Float32(), shape, strides),
                         MakeAppend(key_cache.data(), value_cache.data(), DataType::Float(16),
                                    1, 1, 4, 8, 8, 0, 1))
                        .ok());

    const std::array<uint16_t, 4> expected = {0x3C00, 0x3800, 0xC000, 0x0000};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(ReadBits(key_cache.data(), i), expected[i]) << "key index " << i;
        EXPECT_EQ(ReadBits(value_cache.data(), i), expected[i]) << "value index " << i;
    }
}

TEST(CPUKernelKVCacheUpdateKV16, WritesKnownBFloat16Bits) {
    constexpr int64_t shape[2] = {1, 4};
    constexpr int64_t strides[2] = {4, 1};
    const std::array<float, 4> act = {1.0F, 0.5F, -2.0F, -0.0F};
    std::array<std::byte, 8> key_cache{};
    std::array<std::byte, 8> value_cache{};

    const auto kernel = PrepareKernel(DataType::BFloat(16));
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    ASSERT_TRUE(RunEntry(*kernel,
                         TensorView(act.data(), DataType::Float32(), shape, strides),
                         TensorView(act.data(), DataType::Float32(), shape, strides),
                         MakeAppend(key_cache.data(), value_cache.data(), DataType::BFloat(16),
                                    1, 1, 4, 8, 8, 0, 1))
                        .ok());

    const std::array<uint16_t, 4> expected = {0x3F80, 0x3F00, 0xC000, 0x8000};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(ReadBits(key_cache.data(), i), expected[i]) << "key index " << i;
        EXPECT_EQ(ReadBits(value_cache.data(), i), expected[i]) << "value index " << i;
    }
}

TEST(CPUKernelKVCacheUpdateKV16, NarrowRoundsToNearestEvenAndHandlesSpecials) {
    constexpr int64_t shape[2] = {1, 4};
    constexpr int64_t strides[2] = {4, 1};
    // 1.00048828125 is the F32 midpoint between F16 1.0 (0x3C00) and the next
    // F16 1.0009765625 (0x3C01); ties-to-even rounds down to 0x3C00.
    // 70000 overflows F16 to +Inf; a NaN stays a quiet NaN.
    const std::array<float, 4> act = {1.00048828125F, 70000.0F, -70000.0F,
                                      std::numeric_limits<float>::quiet_NaN()};
    std::array<std::byte, 8> key_cache{};
    std::array<std::byte, 8> value_cache{};

    const auto kernel = PrepareKernel(DataType::Float(16));
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    ASSERT_TRUE(RunEntry(*kernel,
                         TensorView(act.data(), DataType::Float32(), shape, strides),
                         TensorView(act.data(), DataType::Float32(), shape, strides),
                         MakeAppend(key_cache.data(), value_cache.data(), DataType::Float(16),
                                    1, 1, 4, 8, 8, 0, 1))
                        .ok());

    EXPECT_EQ(ReadBits(key_cache.data(), 0), 0x3C00);
    EXPECT_EQ(ReadBits(key_cache.data(), 1), 0x7C00);
    EXPECT_EQ(ReadBits(key_cache.data(), 2), 0xFC00);
    const uint16_t nan_bits = ReadBits(key_cache.data(), 3);
    EXPECT_EQ(nan_bits & 0x7C00, 0x7C00);
    EXPECT_NE(nan_bits & 0x03FF, 0U);
    EXPECT_NE(nan_bits & 0x0200, 0U);
}

TEST(CPUKernelKVCacheUpdateKV16, LeavesPaddingAndHistoryUntouched) {
    constexpr int64_t shape[2] = {1, 2};
    constexpr int64_t strides[2] = {2, 1};
    const std::array<float, 2> act = {1.0F, 2.0F};
    // Two tokens of capacity, head_dim 2 padded to a 4-element (8-byte) row.
    std::array<std::byte, 16> key_cache{};
    std::array<std::byte, 16> value_cache{};
    std::memset(key_cache.data(), 0xAB, key_cache.size());
    std::memset(value_cache.data(), 0xAB, value_cache.size());

    const auto kernel = PrepareKernel(DataType::Float(16));
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    // Append only token 1; token 0 (history) and the padding must stay 0xAB.
    ASSERT_TRUE(RunEntry(*kernel,
                         TensorView(act.data(), DataType::Float32(), shape, strides),
                         TensorView(act.data(), DataType::Float32(), shape, strides),
                         MakeAppend(key_cache.data(), value_cache.data(), DataType::Float(16),
                                    1, 2, 2, 8, 16, 1, 2))
                        .ok());

    EXPECT_EQ(ReadBits(key_cache.data(), 4), 0x3C00);
    EXPECT_EQ(ReadBits(key_cache.data(), 5), 0x4000);
    for (size_t i: {0U, 1U, 2U, 3U, 6U, 7U}) {
        EXPECT_EQ(ReadBits(key_cache.data(), i), 0xABAB) << "untouched index " << i;
        EXPECT_EQ(ReadBits(value_cache.data(), i), 0xABAB) << "untouched index " << i;
    }
}

TEST(CPUKernelKVCacheUpdateKV16, ReadsStridedActivation) {
    constexpr int64_t shape[2] = {1, 2};
    constexpr int64_t strides[2] = {4, 2};
    const std::array<float, 4> act = {1.0F, 99.0F, 2.0F, 99.0F};
    std::array<std::byte, 4> key_cache{};
    std::array<std::byte, 4> value_cache{};

    const auto kernel = PrepareKernel(DataType::Float(16));
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    ASSERT_TRUE(RunEntry(*kernel,
                         TensorView(act.data(), DataType::Float32(), shape, strides),
                         TensorView(act.data(), DataType::Float32(), shape, strides),
                         MakeAppend(key_cache.data(), value_cache.data(), DataType::Float(16),
                                    1, 1, 2, 4, 4, 0, 1))
                        .ok());
    EXPECT_EQ(ReadBits(key_cache.data(), 0), 0x3C00);
    EXPECT_EQ(ReadBits(key_cache.data(), 1), 0x4000);
}

TEST(CPUKernelKVCacheUpdateKV16, RejectsStorageDTypeMismatch) {
    constexpr int64_t shape[2] = {1, 2};
    constexpr int64_t strides[2] = {2, 1};
    const std::array<float, 2> act = {1.0F, 2.0F};
    std::array<std::byte, 4> key_cache{};
    std::array<std::byte, 4> value_cache{};

    const auto kernel = PrepareKernel(DataType::Float(16));
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    // F16 kernel, but the binding claims BF16 storage.
    const Status status = RunEntry(*kernel,
                                   TensorView(act.data(), DataType::Float32(), shape, strides),
                                   TensorView(act.data(), DataType::Float32(), shape, strides),
                                   MakeAppend(key_cache.data(), value_cache.data(),
                                              DataType::BFloat(16), 1, 1, 2, 4, 4, 0, 1));
    EXPECT_EQ(status, StatusCode::kInvalidArgument);
}

TEST(CPUKernelKVCacheUpdateKV16, RejectsOverlappingCacheDestinations) {
    constexpr int64_t shape[2] = {1, 2};
    constexpr int64_t strides[2] = {2, 1};
    const std::array<float, 2> act = {1.0F, 2.0F};
    std::array<std::byte, 8> shared{};

    const auto kernel = PrepareKernel(DataType::Float(16));
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    // Key and value destinations alias the same bytes: disjointness unprovable.
    const Status status = RunEntry(*kernel,
                                   TensorView(act.data(), DataType::Float32(), shape, strides),
                                   TensorView(act.data(), DataType::Float32(), shape, strides),
                                   MakeAppend(shared.data(), shared.data(), DataType::Float(16),
                                              1, 1, 2, 4, 4, 0, 1));
    EXPECT_EQ(status, StatusCode::kInvalidArgument);
}

TEST(CPUKernelKVCacheUpdateKV16, PreparedAppendIsZeroAllocation) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }
    constexpr int64_t shape[2] = {1, 2};
    constexpr int64_t strides[2] = {2, 1};
    const std::array<float, 2> act = {1.0F, 2.0F};
    std::array<std::byte, 4> key_cache{};
    std::array<std::byte, 4> value_cache{};

    const auto kernel = PrepareKernel(DataType::Float(16));
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    const auto append = MakeAppend(key_cache.data(), value_cache.data(), DataType::Float(16),
                                   1, 1, 2, 4, 4, 0, 1);
    const TensorView key_view(act.data(), DataType::Float32(), shape, strides);
    const TensorView value_view(act.data(), DataType::Float32(), shape, strides);

    test::BeginMallocCallCounting();
    const Status status = RunEntry(*kernel, key_view, value_view, append);
    const auto counts = test::EndMallocCallCounting();
    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_EQ(counts.malloc_calls, 0U);
    EXPECT_EQ(counts.calloc_calls, 0U);
    EXPECT_EQ(counts.realloc_calls, 0U);
    EXPECT_EQ(counts.free_calls, 0U);
}

} // namespace