#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/cpu/cpu_info.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_types.h"
#include "inference/test_malloc_interposer.h"
#include "test_utils/test_allocator.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <vector>

namespace {
using namespace aethermind;

struct EmbeddingW16Views {
    TensorView tokens;
    TensorView weight;
    MutableTensorView output;
};

using ParamsStorage = std::array<std::byte, kMaxKernelParamsSize>;

DataType WeightDType(bool bfloat) {
    return bfloat ? DataType::BFloat(16) : DataType::Float(16);
}

StatusOr<ResolvedKernel> ResolveEmbeddingW16(bool bfloat, ExecPhase phase = ExecPhase::kBoth) {
    const CpuBackend backend(CpuCapabilities{});
    return backend.PrepareKernel(
            OpType::kEmbedding,
            {.device_type = DeviceType::kCPU,
             .act_dtype = DataType::Float32(),
             .weight_dtype = WeightDType(bfloat),
             .weight_format = WeightFormat::kPlain,
             .phase = phase},
            EmbeddingParams{.output_dtype = DataType::Float32()});
}

Status BuildEmbeddingW16(const ResolvedKernel& kernel,
                         const EmbeddingW16Views& views,
                         ParamsStorage& storage) {
    const std::array<TensorView, 2> inputs{views.tokens, views.weight};
    const std::array<MutableTensorView, 1> outputs{views.output};
    return kernel.params_builder(
            {.inputs = inputs, .outputs = outputs, .attrs = kernel.attrs}, storage.data());
}

Status RunEmbeddingW16(bool bfloat, const EmbeddingW16Views& views) {
    auto kernel = ResolveEmbeddingW16(bfloat);
    if (!kernel.ok()) return kernel.status();
    alignas(std::max_align_t) ParamsStorage storage{};
    AM_RETURN_IF_ERROR(BuildEmbeddingW16(*kernel, views, storage));
    return kernel->fn({.device_type = DeviceType::kCPU,
                       .kernel_params = storage.data(),
                       .attrs = kernel->attrs});
}

uint32_t WideningOracle(uint16_t bits, bool bfloat) {
    const unsigned mantissa_bits = bfloat ? 7 : 10;
    const unsigned exponent_mask = bfloat ? 0xFFU : 0x1FU;
    const unsigned exponent = (bits >> mantissa_bits) & exponent_mask;
    const unsigned fraction = bits & ((1U << mantissa_bits) - 1U);
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000U) << 16;
    if (exponent == exponent_mask) {
        return sign | 0x7F800000U |
               (fraction == 0 ? 0U : (fraction << (23 - mantissa_bits)) | 0x00400000U);
    }
    const int bias = bfloat ? 127 : 15;
    const double significand = exponent == 0 ? fraction : (1U << mantissa_bits) + fraction;
    const int scale = static_cast<int>(exponent == 0 ? 1 : exponent) - bias -
                      static_cast<int>(mantissa_bits);
    const float value = static_cast<float>(std::ldexp(significand, scale));
    return sign | std::bit_cast<uint32_t>(value);
}

class EmbeddingW16Kernel : public ::testing::TestWithParam<bool> {};

TEST_P(EmbeddingW16Kernel, SelectedRowWideningMatchesEveryBitPatternAndHonorsFeaturePolicy) {
    const CpuBackend backend;
    const KernelSelector selector{.device_type = DeviceType::kCPU,
                                  .act_dtype = DataType::Float32(),
                                  .weight_dtype = WeightDType(GetParam()),
                                  .weight_format = WeightFormat::kPlain,
                                  .phase = ExecPhase::kBoth};
    auto selected = backend.PrepareKernel(OpType::kEmbedding, selector,
                                          EmbeddingParams{.output_dtype = DataType::Float32()});
    ASSERT_TRUE(selected.ok()) << selected.status().ToString();
    CpuFeaturePolicy policy;
    policy.disabled_features.Enable(CpuFeature::kAvx);
    const CpuBackend restricted(policy);
    auto scalar = restricted.PrepareKernel(OpType::kEmbedding, selector,
                                           EmbeddingParams{.output_dtype = DataType::Float32()});
    ASSERT_TRUE(scalar.ok()) << scalar.status().ToString();
    EXPECT_EQ(scalar->name, GetParam() ? "cpu::embedding_f32_bf16_reference"
                                       : "cpu::embedding_f32_f16_reference");

    auto capabilities = cpu::DetectCpuCapabilities();
    ASSERT_TRUE(capabilities.ok());
    bool vector_eligible = false;
#if defined(FP16_HAS_F16C_CONVERSION)
    if (!GetParam()) {
        vector_eligible = capabilities->effective_features.Contains(CpuFeature::kAvx) &&
                          capabilities->effective_features.Contains(CpuFeature::kF16c);
    }
#endif
#if defined(BF16_HAS_AVX2_CONVERSION)
    if (GetParam()) {
        vector_eligible = capabilities->effective_features.Contains(CpuFeature::kAvx) &&
                          capabilities->effective_features.Contains(CpuFeature::kAvx2);
    }
#endif
    EXPECT_EQ(selected->name != scalar->name, vector_eligible);

    constexpr int64_t hidden = 17;
    constexpr int64_t rows = (65536 + hidden - 1) / hidden;
    constexpr size_t elements = static_cast<size_t>(rows * hidden);
    std::vector<int64_t> tokens(rows);
    std::vector<std::byte> weight(elements * 2 + 1);
    std::vector<uint32_t> actual(elements), expected(elements);
    for (size_t index = 0; index < elements; ++index) {
        const auto bits = static_cast<uint16_t>(index);
        std::memcpy(weight.data() + 1 + index * 2, &bits, sizeof(bits));
        expected[index] = WideningOracle(bits, GetParam());
    }
    for (int64_t row = 0; row < rows; ++row) tokens[static_cast<size_t>(row)] = row;
    const std::array<int64_t, 1> token_shape{rows}, token_stride{1};
    const std::array<int64_t, 2> weight_shape{rows, hidden}, output_shape{rows, hidden};
    const std::array<int64_t, 2> strides{hidden, 1};
    alignas(std::max_align_t) ParamsStorage params{};
    ASSERT_TRUE(BuildEmbeddingW16(*selected,
                                  {TensorView{tokens.data(), DataType::Int(64), token_shape, token_stride},
                                   TensorView{weight.data() + 1, WeightDType(GetParam()), weight_shape, strides},
                                   MutableTensorView{actual.data(), DataType::Float32(), output_shape, strides}},
                                  params)
                        .ok());
    ASSERT_TRUE(selected->fn({.kernel_params = params.data(), .attrs = selected->attrs}).ok());
    EXPECT_EQ(actual, expected);
}

TEST_P(EmbeddingW16Kernel, ResolvesBothExecutionPhasesWithNoCpuFeatures) {
    for (const auto phase: {ExecPhase::kBoth, ExecPhase::kPrefill, ExecPhase::kDecode}) {
        const auto kernel = ResolveEmbeddingW16(GetParam(), phase);
        ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
        EXPECT_EQ(kernel->name, GetParam() ? "cpu::embedding_f32_bf16_reference"
                                           : "cpu::embedding_f32_f16_reference");
        EXPECT_NE(kernel->params_builder, nullptr);
        EXPECT_LE(kernel->params_size, kMaxKernelParamsSize);
    }
}

TEST_P(EmbeddingW16Kernel, RequiresExplicitFloat32OutputParams) {
    const CpuBackend backend(CpuCapabilities{});
    const KernelSelector selector{.device_type = DeviceType::kCPU,
                                  .act_dtype = DataType::Float32(),
                                  .weight_dtype = WeightDType(GetParam()),
                                  .weight_format = WeightFormat::kPlain,
                                  .phase = ExecPhase::kBoth};
    EXPECT_EQ(backend.PrepareKernel(OpType::kEmbedding, selector, EmbeddingParams{}).status(),
              StatusCode::kInvalidArgument);
    EXPECT_EQ(backend.PrepareKernel(OpType::kEmbedding, selector, LinearParams{}).status(),
              StatusCode::kInvalidArgument);
    EXPECT_EQ(backend.PrepareKernel(OpType::kEmbedding, selector,
                                    EmbeddingParams{.output_dtype = DataType::Float(16)})
                      .status(),
              StatusCode::kInvalidArgument);
}

TEST_P(EmbeddingW16Kernel, WidensUnalignedSelectedRowsWithSpecialValuesAndTails) {
    constexpr std::array<uint16_t, 15> patterns{
            0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x007F, 0x3C00, 0x3F80,
            0x7BFF, 0x7F7F, 0x7C00, 0x7F80, 0xFC01, 0x7F81, 0xFFFF};
    constexpr std::array<int64_t, 3> tokens{2, 0, 2};
    constexpr std::array<int64_t, 1> token_shape{3}, token_stride{1};
    for (const int64_t hidden: {1, 7, 8, 9, 15, 16, 17, 31}) {
        const size_t elements = static_cast<size_t>(3 * hidden);
        std::vector<std::byte> weight(1 + elements * 2 + 7, std::byte{0xA5});
        std::vector<uint32_t> output(elements + 2, 0xDEADBEEFU);
        for (size_t index = 0; index < elements; ++index) {
            const uint16_t bits = patterns[index % patterns.size()];
            std::memcpy(weight.data() + 1 + index * 2, &bits, sizeof(bits));
        }
        const auto original = weight;
        const std::array<int64_t, 2> weight_shape{3, hidden}, weight_stride{hidden, 1};
        const std::array<int64_t, 2> output_shape{3, hidden}, output_stride{hidden, 1};
        const auto status = RunEmbeddingW16(
                GetParam(),
                {TensorView(tokens.data(), DataType::Int(64), token_shape, token_stride),
                 TensorView(weight.data() + 1, WeightDType(GetParam()), weight_shape, weight_stride),
                 MutableTensorView(output.data() + 1, DataType::Float32(), output_shape, output_stride)});
        ASSERT_TRUE(status.ok()) << status.ToString();
        for (size_t row = 0; row < tokens.size(); ++row) {
            for (size_t column = 0; column < static_cast<size_t>(hidden); ++column) {
                const size_t source_index = static_cast<size_t>(tokens[row]) * hidden + column;
                EXPECT_EQ(output[1 + row * hidden + column],
                          WideningOracle(patterns[source_index % patterns.size()], GetParam()));
            }
        }
        EXPECT_EQ(output.front(), 0xDEADBEEFU);
        EXPECT_EQ(output.back(), 0xDEADBEEFU);
        EXPECT_EQ(weight, original);
    }
}

TEST_P(EmbeddingW16Kernel, SupportsAllTokenTypesAndMultipleTokenAxes) {
    constexpr std::array<int64_t, 3> ids64{2, 0, 1};
    constexpr std::array<int32_t, 3> ids32{2, 0, 1};
    constexpr std::array<uint32_t, 3> ids_unsigned{2, 0, 1};
    constexpr std::array<uint16_t, 6> weight{0, 0x8000, 1, 2, 0x3C00, 0x3F80};
    constexpr std::array<int64_t, 2> token_shape{1, 3}, token_stride{3, 1};
    constexpr std::array<int64_t, 2> weight_shape{3, 2}, weight_stride{2, 1};
    constexpr std::array<int64_t, 3> output_shape{1, 3, 2}, output_stride{6, 2, 1};
    const std::array<TensorView, 3> token_views{
            TensorView(ids64.data(), DataType::Int(64), token_shape, token_stride),
            TensorView(ids32.data(), DataType::Int(32), token_shape, token_stride),
            TensorView(ids_unsigned.data(), DataType::UInt(32), token_shape, token_stride)};
    for (const auto& tokens: token_views) {
        std::array<uint32_t, 6> output{};
        const auto status = RunEmbeddingW16(
                GetParam(), {tokens,
                             TensorView(weight.data(), WeightDType(GetParam()), weight_shape, weight_stride),
                             MutableTensorView(output.data(), DataType::Float32(), output_shape, output_stride)});
        ASSERT_TRUE(status.ok()) << status.ToString();
        for (size_t row = 0; row < ids64.size(); ++row) {
            for (size_t column = 0; column < 2; ++column) {
                EXPECT_EQ(output[row * 2 + column], WideningOracle(weight[ids64[row] * 2 + column], GetParam()));
            }
        }
    }
}

TEST_P(EmbeddingW16Kernel, AcceptsEmptyTokensWithNullInputAndOutput) {
    constexpr std::array<uint16_t, 2> weight{0, 1};
    constexpr std::array<int64_t, 1> token_shape{0}, token_stride{1};
    constexpr std::array<int64_t, 2> weight_shape{1, 2}, weight_stride{2, 1};
    constexpr std::array<int64_t, 2> output_shape{0, 2}, output_stride{2, 1};
    EXPECT_TRUE(RunEmbeddingW16(
                        GetParam(),
                        {TensorView(nullptr, DataType::Int(64), token_shape, token_stride),
                         TensorView(weight.data(), WeightDType(GetParam()), weight_shape, weight_stride),
                         MutableTensorView(nullptr, DataType::Float32(), output_shape, output_stride)})
                        .ok());
}

TEST_P(EmbeddingW16Kernel, OutOfRangeTokenPreservesPreviouslyCompletedRows) {
    constexpr std::array<int64_t, 3> tokens{1, -1, 0};
    constexpr std::array<uint16_t, 4> weight{0, 1, 0x3C00, 0x3F80};
    constexpr std::array<int64_t, 1> token_shape{3}, token_stride{1};
    constexpr std::array<int64_t, 2> weight_shape{2, 2}, weight_stride{2, 1};
    constexpr std::array<int64_t, 2> output_shape{3, 2}, output_stride{2, 1};
    std::array<uint32_t, 6> output;
    output.fill(0xDEADBEEFU);
    const auto status = RunEmbeddingW16(
            GetParam(), {TensorView(tokens.data(), DataType::Int(64), token_shape, token_stride),
                         TensorView(weight.data(), WeightDType(GetParam()), weight_shape, weight_stride),
                         MutableTensorView(output.data(), DataType::Float32(), output_shape, output_stride)});
    EXPECT_EQ(status, StatusCode::kOutOfRange);
    EXPECT_EQ(output[0], WideningOracle(weight[2], GetParam()));
    EXPECT_EQ(output[1], WideningOracle(weight[3], GetParam()));
    EXPECT_TRUE(std::all_of(output.begin() + 2, output.end(), [](uint32_t bits) { return bits == 0xDEADBEEFU; }));
}

TEST_P(EmbeddingW16Kernel, RejectsUnsignedTokenAboveVocabularyWithoutWriting) {
    constexpr uint32_t token = std::numeric_limits<uint32_t>::max();
    constexpr std::array<uint16_t, 2> weight{0, 1};
    constexpr std::array<int64_t, 1> token_shape{1}, token_stride{1};
    constexpr std::array<int64_t, 2> weight_shape{1, 2}, weight_stride{2, 1};
    constexpr std::array<int64_t, 2> output_shape{1, 2}, output_stride{2, 1};
    std::array<uint32_t, 2> output{0xDEADBEEFU, 0xDEADBEEFU};
    const auto status = RunEmbeddingW16(
            GetParam(), {TensorView(&token, DataType::UInt(32), token_shape, token_stride),
                         TensorView(weight.data(), WeightDType(GetParam()), weight_shape, weight_stride),
                         MutableTensorView(output.data(), DataType::Float32(), output_shape, output_stride)});
    EXPECT_EQ(status, StatusCode::kOutOfRange);
    EXPECT_EQ(output, (std::array<uint32_t, 2>{0xDEADBEEFU, 0xDEADBEEFU}));
}

TEST_P(EmbeddingW16Kernel, RejectsMismatchedWeightDtypeAndInvalidOutputGeometry) {
    constexpr int64_t token = 0;
    constexpr std::array<uint16_t, 4> weight{};
    constexpr std::array<int64_t, 1> token_shape{1}, token_stride{1};
    constexpr std::array<int64_t, 2> weight_shape{2, 2}, weight_stride{2, 1};
    constexpr std::array<int64_t, 2> output_shape{1, 2}, output_stride{2, 1};
    constexpr std::array<int64_t, 2> wrong_output_shape{1, 3}, wrong_output_stride{3, 1};
    constexpr std::array<int64_t, 2> strided_weight_stride{4, 2};
    std::array<float, 6> output{};
    const TensorView tokens(&token, DataType::Int(64), token_shape, token_stride);
    const MutableTensorView output_view(output.data(), DataType::Float32(), output_shape, output_stride);
    EXPECT_EQ(RunEmbeddingW16(GetParam(),
                              {tokens, TensorView(weight.data(), WeightDType(!GetParam()), weight_shape, weight_stride), output_view}),
              StatusCode::kInvalidArgument);
    EXPECT_EQ(RunEmbeddingW16(GetParam(),
                              {tokens, TensorView(weight.data(), WeightDType(GetParam()), weight_shape, strided_weight_stride), output_view}),
              StatusCode::kInvalidArgument);
    EXPECT_EQ(RunEmbeddingW16(GetParam(),
                              {tokens, TensorView(weight.data(), WeightDType(GetParam()), weight_shape, weight_stride),
                               MutableTensorView(output.data(), DataType::Float32(), wrong_output_shape, wrong_output_stride)}),
              StatusCode::kInvalidArgument);
}

TEST_P(EmbeddingW16Kernel, RejectsOutputOverlappingWeightOrTokens) {
    constexpr int64_t token = 0;
    constexpr std::array<uint16_t, 4> weight{};
    constexpr std::array<int64_t, 1> token_shape{1}, token_stride{1};
    constexpr std::array<int64_t, 2> weight_shape{2, 2}, weight_stride{2, 1};
    constexpr std::array<int64_t, 2> output_shape{1, 2}, output_stride{2, 1};
    alignas(std::max_align_t) std::array<std::byte, 32> storage{};
    EXPECT_EQ(RunEmbeddingW16(GetParam(),
                              {TensorView(&token, DataType::Int(64), token_shape, token_stride),
                               TensorView(storage.data(), WeightDType(GetParam()), weight_shape, weight_stride),
                               MutableTensorView(storage.data() + 2, DataType::Float32(), output_shape, output_stride)}),
              StatusCode::kInvalidArgument);
    EXPECT_EQ(RunEmbeddingW16(GetParam(),
                              {TensorView(storage.data(), DataType::Int(64), token_shape, token_stride),
                               TensorView(weight.data(), WeightDType(GetParam()), weight_shape, weight_stride),
                               MutableTensorView(storage.data() + 4, DataType::Float32(), output_shape, output_stride)}),
              StatusCode::kInvalidArgument);
}

TEST_P(EmbeddingW16Kernel, AcceptsDisjointWeightAndOutputWithinOneAllocation) {
    constexpr int64_t token = 1;
    constexpr std::array<uint16_t, 4> weight{0, 1, 0x3C00, 0x3F80};
    constexpr std::array<int64_t, 1> token_shape{1}, token_stride{1};
    constexpr std::array<int64_t, 2> weight_shape{2, 2}, weight_stride{2, 1};
    constexpr std::array<int64_t, 2> output_shape{1, 2}, output_stride{2, 1};
    alignas(std::max_align_t) std::array<std::byte, 16> storage{};
    std::memcpy(storage.data(), weight.data(), sizeof(weight));
    const auto status = RunEmbeddingW16(
            GetParam(), {TensorView(&token, DataType::Int(64), token_shape, token_stride),
                         TensorView(storage.data(), WeightDType(GetParam()), weight_shape, weight_stride),
                         MutableTensorView(storage.data() + 8, DataType::Float32(), output_shape, output_stride)});
    ASSERT_TRUE(status.ok()) << status.ToString();
    std::array<uint32_t, 2> output{};
    std::memcpy(output.data(), storage.data() + 8, sizeof(output));
    EXPECT_EQ(output[0], WideningOracle(weight[2], GetParam()));
    EXPECT_EQ(output[1], WideningOracle(weight[3], GetParam()));
}

TEST_P(EmbeddingW16Kernel, RejectsAddressRangeOverflowBeforeReadingOrWriting) {
    constexpr int64_t token = 0;
    constexpr std::array<int64_t, 1> token_shape{1}, token_stride{1};
    constexpr std::array<int64_t, 2> weight_shape{1, 2}, weight_stride{2, 1};
    constexpr std::array<int64_t, 2> output_shape{1, 2}, output_stride{2, 1};
    std::array<uint32_t, 2> output{0xDEADBEEFU, 0xDEADBEEFU};
    const void* invalid_weight = reinterpret_cast<const void*>(std::numeric_limits<uintptr_t>::max() - 1);
    const auto status = RunEmbeddingW16(
            GetParam(), {TensorView(&token, DataType::Int(64), token_shape, token_stride),
                         TensorView(invalid_weight, WeightDType(GetParam()), weight_shape, weight_stride),
                         MutableTensorView(output.data(), DataType::Float32(), output_shape, output_stride)});
    EXPECT_EQ(status, StatusCode::kInvalidArgument);
    EXPECT_EQ(output, (std::array<uint32_t, 2>{0xDEADBEEFU, 0xDEADBEEFU}));
}

TEST_P(EmbeddingW16Kernel, PreparedExecutionAllocatesNoStorage) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "malloc interposer is unavailable for this build";
    }
    constexpr std::array<int64_t, 3> tokens{2, 0, 1};
    constexpr std::array<uint16_t, 6> weight{0, 1, 2, 3, 4, 5};
    constexpr std::array<int64_t, 1> token_shape{3}, token_stride{1};
    constexpr std::array<int64_t, 2> weight_shape{3, 2}, weight_stride{2, 1};
    constexpr std::array<int64_t, 2> output_shape{3, 2}, output_stride{2, 1};
    std::array<float, 6> output{};
    auto kernel = ResolveEmbeddingW16(GetParam());
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    alignas(std::max_align_t) ParamsStorage params{};
    ASSERT_TRUE(BuildEmbeddingW16(*kernel,
                                  {TensorView(tokens.data(), DataType::Int(64), token_shape, token_stride),
                                   TensorView(weight.data(), WeightDType(GetParam()), weight_shape, weight_stride),
                                   MutableTensorView(output.data(), DataType::Float32(), output_shape, output_stride)},
                                  params)
                        .ok());
    const KernelContext context{.device_type = DeviceType::kCPU,
                                .kernel_params = params.data(),
                                .attrs = kernel->attrs};
    ASSERT_TRUE(kernel->fn(context).ok());
    bool successful = true;
    test::BeginMallocCallCounting();
    for (size_t iteration = 0; iteration < 32; ++iteration) {
        successful = kernel->fn(context).ok() && successful;
    }
    const auto calls = test::EndMallocCallCounting();
    EXPECT_TRUE(successful);
    EXPECT_EQ(calls.malloc_calls + calls.calloc_calls + calls.realloc_calls +
                      calls.aligned_alloc_calls + calls.posix_memalign_calls + calls.memalign_calls,
              0U);
    EXPECT_EQ(calls.free_calls, 0U);
}

INSTANTIATE_TEST_SUITE_P(WeightTypes, EmbeddingW16Kernel, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                             return info.param ? "Bfloat16" : "Float16";
                         });

TEST_P(EmbeddingW16Kernel, PackedIdentityGathersWidenedRowsAndRejectsFp32Artifact) {
    const bool bfloat = GetParam();
    constexpr int64_t token_ids[3] = {2, 0, 3};
    constexpr int64_t token_shape[1] = {3};
    constexpr int64_t token_strides[1] = {1};
    constexpr int64_t weight_shape[2] = {4, 3};
    constexpr int64_t weight_strides[2] = {3, 1};
    constexpr int64_t output_shape[2] = {3, 3};
    constexpr int64_t output_strides[2] = {3, 1};
    constexpr std::array<uint16_t, 12> half_patterns{
            0x3C00, 0x4000, 0x4200, 0x4400, 0x3800, 0x0001,
            0x3DC0, 0x3E00, 0x3F00, 0x4000, 0x4100, 0x4200};
    constexpr std::array<uint16_t, 12> bfloat_patterns{
            0x3F80, 0x4000, 0x4040, 0x4080, 0x3F00, 0x0001,
            0x3FC0, 0x3E80, 0x3F00, 0x4000, 0x4020, 0x4040};
    const auto& patterns = bfloat ? bfloat_patterns : half_patterns;
    std::array<std::byte, 24> weight_bytes{};
    for (size_t index = 0; index < patterns.size(); ++index) {
        const uint16_t bits = patterns[index];
        std::memcpy(weight_bytes.data() + index * sizeof(bits), &bits, sizeof(bits));
    }

    const TensorView weight_view(weight_bytes.data(), WeightDType(bfloat),
                                 weight_shape, weight_strides);
    const KernelSelector selector{.device_type = DeviceType::kCPU,
                                  .act_dtype = DataType::Float32(),
                                  .weight_dtype = WeightDType(bfloat),
                                  .weight_format = WeightFormat::kPacked,
                                  .phase = ExecPhase::kBoth};
    CpuBackend backend(CpuCapabilities{});
    const auto recipe = backend.GetPackingRecipe(OpType::kEmbedding, selector);
    ASSERT_TRUE(recipe.ok()) << recipe.status().ToString();
    auto packed = backend.PackWeights(
            OpType::kEmbedding, std::array<TensorView, 1>{weight_view},
            selector, *recipe, test::TestCpuAllocator());
    ASSERT_TRUE(packed.ok()) << packed.status().ToString();
    EXPECT_EQ((*packed)->logical_dtype(), WeightDType(bfloat));
    EXPECT_EQ((*packed)->storage().nbytes(), weight_bytes.size());

    const auto kernel = backend.PrepareKernel(
            OpType::kEmbedding, selector,
            EmbeddingParams{.output_dtype = DataType::Float32()});
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    EXPECT_EQ(kernel->name, bfloat ? "cpu::embedding_f32_bf16_packed_identity_reference"
                                   : "cpu::embedding_f32_f16_packed_identity_reference");
    const PackedWeightView packed_view{
            .data = (*packed)->storage().data(),
            .nbytes = (*packed)->storage().nbytes(),
            .logical_dtype = (*packed)->logical_dtype(),
            .logical_shape = (*packed)->logical_shape(),
            .recipe_layout = (*packed)->recipe().layout,
            .recipe_alignment = (*packed)->recipe().alignment,
            .alignment = (*packed)->storage().alignment(),
    };

    std::array<uint32_t, 9> output{};
    const TensorView tokens(token_ids, DataType::Int(64), token_shape, token_strides);
    const MutableTensorView output_view(output.data(), DataType::Float32(),
                                        output_shape, output_strides);
    alignas(std::max_align_t) ParamsStorage storage{};
    ASSERT_TRUE(kernel->params_builder(
                              KernelParamsBuildContext{
                                      .inputs = std::array<TensorView, 1>{tokens},
                                      .outputs = std::array<MutableTensorView, 1>{output_view},
                                      .attrs = kernel->attrs,
                                      .packed_weight = packed_view,
                              },
                              storage.data())
                        .ok());
    ASSERT_TRUE(kernel->fn({.kernel_params = storage.data(), .attrs = kernel->attrs}).ok());
    for (size_t row = 0; row < 3; ++row) {
        for (size_t column = 0; column < 3; ++column) {
            const size_t source = static_cast<size_t>(token_ids[row]) * 3 + column;
            EXPECT_EQ(output[row * 3 + column], WideningOracle(patterns[source], bfloat));
        }
    }

    alignas(64) std::array<float, 12> fp32_storage{};
    const PackedWeightView wrong_dtype{
            .data = fp32_storage.data(),
            .nbytes = fp32_storage.size() * sizeof(float),
            .logical_dtype = DataType::Float32(),
            .logical_shape = (*packed)->logical_shape(),
            .recipe_layout = (*packed)->recipe().layout,
            .recipe_alignment = (*packed)->recipe().alignment,
            .alignment = (*packed)->storage().alignment(),
    };
    EXPECT_EQ(kernel->params_builder(
                      KernelParamsBuildContext{
                              .inputs = std::array<TensorView, 1>{tokens},
                              .outputs = std::array<MutableTensorView, 1>{output_view},
                              .attrs = kernel->attrs,
                              .packed_weight = wrong_dtype,
                      },
                      storage.data()),
              StatusCode::kInvalidArgument);
}

} // namespace
