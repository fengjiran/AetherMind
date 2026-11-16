#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/operators/op_params.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#include <gtest/gtest.h>

namespace {

using namespace aethermind;

KernelSelector MakeSelector(DataType act_dtype) {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = act_dtype,
            .weight_dtype = act_dtype,
            .state_dtype = act_dtype,
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kBoth,
    };
}

StatusOr<ResolvedKernel> PrepareKernel(DataType act_dtype) {
    CpuBackend backend;
    return backend.PrepareKernel(OpType::kKVCacheUpdate, MakeSelector(act_dtype),
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

void WriteBits(std::byte* destination, uint16_t bits, size_t index) {
    std::memcpy(destination + index * 2, &bits, sizeof(bits));
}

TEST(CPUKernelKVCacheUpdateA16, ResolvesF16AndBF16References) {
    const auto f16 = PrepareKernel(DataType::Float(16));
    ASSERT_TRUE(f16.ok()) << f16.status().ToString();
    EXPECT_EQ(std::string_view{f16->name}, "cpu::kvcache_update_a16_f16_reference");
    const auto bf16 = PrepareKernel(DataType::BFloat(16));
    ASSERT_TRUE(bf16.ok()) << bf16.status().ToString();
    EXPECT_EQ(std::string_view{bf16->name}, "cpu::kvcache_update_a16_bf16_reference");
}

TEST(CPUKernelKVCacheUpdateA16, CopiesStoredBitsWithoutRounding) {
    constexpr int64_t shape[2] = {1, 4};
    constexpr int64_t strides[2] = {4, 1};
    // Patterns that a widen/narrow round trip would alter (subnormal, midpoint,
    // signed zero) prove the update is a verbatim write-through.
    constexpr std::array<uint16_t, 4> key_patterns{0x0001, 0x3C01, 0x8000, 0x7BFF};
    constexpr std::array<uint16_t, 4> value_patterns{0x4000, 0x3D00, 0xFC00, 0x0000};
    std::array<std::byte, 8> key_activation{};
    std::array<std::byte, 8> value_activation{};
    std::array<std::byte, 8> key_cache{};
    std::array<std::byte, 8> value_cache{};
    for (size_t index = 0; index < key_patterns.size(); ++index) {
        WriteBits(key_activation.data(), key_patterns[index], index);
        WriteBits(value_activation.data(), value_patterns[index], index);
    }

    const auto kernel = PrepareKernel(DataType::Float(16));
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    ASSERT_TRUE(RunEntry(*kernel,
                         TensorView(key_activation.data(), DataType::Float(16),
                                    shape, strides),
                         TensorView(value_activation.data(), DataType::Float(16),
                                    shape, strides),
                         MakeAppend(key_cache.data(), value_cache.data(), DataType::Float(16),
                                    1, 1, 4, 8, 8, 0, 1))
                        .ok());

    for (size_t index = 0; index < key_patterns.size(); ++index) {
        EXPECT_EQ(ReadBits(key_cache.data(), index), key_patterns[index]) << "key " << index;
        EXPECT_EQ(ReadBits(value_cache.data(), index), value_patterns[index]) << "value " << index;
    }
}

TEST(CPUKernelKVCacheUpdateA16, LeavesPaddingAndHistoryUntouched) {
    constexpr int64_t shape[2] = {1, 2};
    constexpr int64_t strides[2] = {2, 1};
    constexpr std::array<uint16_t, 2> patterns{0x3F80, 0x4000};
    std::array<std::byte, 4> activation{};
    WriteBits(activation.data(), patterns[0], 0);
    WriteBits(activation.data(), patterns[1], 1);
    // Two tokens of capacity, head_dim 2 padded to a 4-element (8-byte) row.
    std::array<std::byte, 16> key_cache{};
    std::array<std::byte, 16> value_cache{};
    std::memset(key_cache.data(), 0xAB, key_cache.size());
    std::memset(value_cache.data(), 0xAB, value_cache.size());

    const auto kernel = PrepareKernel(DataType::BFloat(16));
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    ASSERT_TRUE(RunEntry(*kernel,
                         TensorView(activation.data(), DataType::BFloat(16),
                                    shape, strides),
                         TensorView(activation.data(), DataType::BFloat(16),
                                    shape, strides),
                         MakeAppend(key_cache.data(), value_cache.data(), DataType::BFloat(16),
                                    1, 2, 2, 8, 16, 1, 2))
                        .ok());

    EXPECT_EQ(ReadBits(key_cache.data(), 4), patterns[0]);
    EXPECT_EQ(ReadBits(key_cache.data(), 5), patterns[1]);
    for (size_t index: {0U, 1U, 2U, 3U, 6U, 7U}) {
        EXPECT_EQ(ReadBits(key_cache.data(), index), 0xABAB) << "untouched index " << index;
        EXPECT_EQ(ReadBits(value_cache.data(), index), 0xABAB) << "untouched index " << index;
    }
}

TEST(CPUKernelKVCacheUpdateA16, ReadsStridedActivation) {
    constexpr int64_t shape[2] = {1, 2};
    constexpr int64_t strides[2] = {4, 2};
    constexpr std::array<uint16_t, 4> patterns{0x3F80, 0x7C00, 0x4000, 0x7C00};
    std::array<std::byte, 8> activation{};
    for (size_t index = 0; index < patterns.size(); ++index) {
        WriteBits(activation.data(), patterns[index], index);
    }
    std::array<std::byte, 4> key_cache{};
    std::array<std::byte, 4> value_cache{};

    const auto kernel = PrepareKernel(DataType::Float(16));
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    ASSERT_TRUE(RunEntry(*kernel,
                         TensorView(activation.data(), DataType::Float(16), shape, strides),
                         TensorView(activation.data(), DataType::Float(16), shape, strides),
                         MakeAppend(key_cache.data(), value_cache.data(), DataType::Float(16),
                                    1, 1, 2, 4, 4, 0, 1))
                        .ok());
    EXPECT_EQ(ReadBits(key_cache.data(), 0), patterns[0]);
    EXPECT_EQ(ReadBits(key_cache.data(), 1), patterns[2]);
}

TEST(CPUKernelKVCacheUpdateA16, RejectsStorageDTypeMismatch) {
    constexpr int64_t shape[2] = {1, 2};
    constexpr int64_t strides[2] = {2, 1};
    std::array<std::byte, 4> activation{};
    std::array<std::byte, 4> key_cache{};
    std::array<std::byte, 4> value_cache{};

    const auto kernel = PrepareKernel(DataType::Float(16));
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    EXPECT_EQ(RunEntry(*kernel,
                       TensorView(activation.data(), DataType::Float(16), shape, strides),
                       TensorView(activation.data(), DataType::Float(16), shape, strides),
                       MakeAppend(key_cache.data(), value_cache.data(), DataType::BFloat(16),
                                  1, 1, 2, 4, 4, 0, 1)),
              StatusCode::kInvalidArgument);
}

TEST(CPUKernelKVCacheUpdateA16, RejectsOverlappingCacheDestinations) {
    constexpr int64_t shape[2] = {1, 2};
    constexpr int64_t strides[2] = {2, 1};
    std::array<std::byte, 4> activation{};
    std::array<std::byte, 8> shared{};

    const auto kernel = PrepareKernel(DataType::Float(16));
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    EXPECT_EQ(RunEntry(*kernel,
                       TensorView(activation.data(), DataType::Float(16), shape, strides),
                       TensorView(activation.data(), DataType::Float(16), shape, strides),
                       MakeAppend(shared.data(), shared.data(), DataType::Float(16),
                                  1, 1, 2, 4, 4, 0, 1)),
              StatusCode::kInvalidArgument);
}

} // namespace
