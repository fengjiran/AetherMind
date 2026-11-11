#include "aethermind/memory/cpu_allocator.h"
#include "aethermind/model/loaded_model.h"
#include "aethermind/model/weight/raw_weight_fp32_conversion.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

namespace {
using namespace aethermind;

struct SourceStorage final : RawStorage {
    explicit SourceStorage(size_t bytes) : data(bytes) {}
    std::vector<std::byte> data;
    bool known = true;
    RawStorageKind kind = RawStorageKind::kExternal;
    AM_NODISCARD std::optional<RawStorageFootprint> footprint() const noexcept override {
        if (!known) {
            return std::nullopt;
        }
        return RawStorageFootprint{kind, data.size()};
    }
};

struct AllocationStats {
    size_t calls = 0;
    size_t live = 0;
    size_t released = 0;
    size_t fail_on_call = std::numeric_limits<size_t>::max();
};

class TrackingAllocator final : public Allocator {
public:
    explicit TrackingAllocator(std::shared_ptr<AllocationStats> stats)
        : stats_(std::move(stats)), cpu_(Device::CPU()) {}

    AM_NODISCARD Device device() const noexcept override {
        return Device::CPU();
    }

protected:
    StatusOr<MemoryHandle> AllocateStorage(const AllocationRequest& request) override {
        if (const size_t call = stats_->calls++; call == stats_->fail_on_call) {
            return Status::ResourceExhausted("test allocation failure");
        }

        AM_ASSIGN_OR_RETURN(auto buffer, cpu_.TryAllocate(request));

        struct Owner {
            Buffer buffer;
            std::shared_ptr<AllocationStats> stats;
        };

        auto owner = std::make_unique<Owner>(Owner{std::move(buffer), stats_});
        void* data = owner->buffer.mutable_data();
        const size_t alignment = owner->buffer.alignment();
        ++stats_->live;
        return MemoryHandle(data, owner.release(), [](void* context, void*) noexcept {
            auto* owned = static_cast<Owner*>(context);
            --owned->stats->live;
            ++owned->stats->released;
            delete owned; }, Device::CPU(), alignment);
    }

private:
    std::shared_ptr<AllocationStats> stats_;
    CPUAllocator cpu_;
};

RawWeightView MakeView(const std::shared_ptr<SourceStorage>& source,
                       DataType dtype, size_t count, size_t offset = 0) {
    return {.data = source->data.data() + offset,
            .bytes = count * dtype.nbytes(),
            .dtype = dtype,
            .shape = {static_cast<int64_t>(count)},
            .storage = source};
}

ResolvedModelWeights MakeWeights(const RawWeightView& view) {
    ResolvedModelWeights weights;
    weights.embed_tokens = view;
    weights.final_norm = view;
    return weights;
}

HfModelConfig MakeConfig(DataType hint = DataType::Float(16)) {
    HfModelConfig config;
    config.weight_dtype_hint = hint;
    config.weight_dtype_hint_name = "source dtype hint";
    return config;
}

uint32_t ReadBits(const RawWeightView& view, size_t index) {
    uint32_t value;
    std::memcpy(&value, view.data + index * sizeof(float), sizeof(value));
    return value;
}

uint32_t WidenOracle(uint16_t bits, bool bfloat) {
    const int fraction_bits = bfloat ? 7 : 10;
    const int exponent_bits = bfloat ? 8 : 5;
    const int bias = bfloat ? 127 : 15;
    const uint32_t fraction_mask = (1U << fraction_bits) - 1U;
    const uint32_t exponent_mask = (1U << exponent_bits) - 1U;
    const uint32_t fraction = bits & fraction_mask;
    const uint32_t exponent = (bits >> fraction_bits) & exponent_mask;
    const uint32_t sign = (bits & 0x8000U) == 0 ? 0 : 0x80000000U;
    if (exponent == exponent_mask) {
        return sign | 0x7f800000U | (fraction << (23 - fraction_bits)) |
               (fraction == 0 ? 0U : 0x00400000U);
    }
    // The finite oracle reconstructs the exact dyadic value in double; no
    // production Half/BFloat16 helper or narrowing operation supplies expected.
    const double magnitude = exponent == 0
                                     ? std::ldexp(static_cast<double>(fraction), 1 - bias - fraction_bits)
                                     : std::ldexp(1.0 + std::ldexp(static_cast<double>(fraction), -fraction_bits),
                                                  static_cast<int>(exponent) - bias);
    const auto value = static_cast<float>(sign == 0 ? magnitude : -magnitude);
    return std::bit_cast<uint32_t>(value);
}

void CheckEveryPattern(DataType dtype, bool bfloat) {
    auto source = std::make_shared<SourceStorage>(65536 * sizeof(uint16_t));
    for (uint32_t bits = 0; bits < 65536; ++bits) {
        const auto value = static_cast<uint16_t>(bits);
        std::memcpy(source->data.data() + bits * sizeof(value), &value, sizeof(value));
    }
    LoadedModel model(MakeConfig(dtype), MakeWeights(MakeView(source, dtype, 65536)));
    auto stats = std::make_shared<AllocationStats>();
    TrackingAllocator allocator(stats);
    auto converted = ConvertRawModelWeightsToFp32(model, allocator);
    ASSERT_TRUE(converted.ok()) << converted.status().ToString();
    const auto& output = (*converted)->GetResolvedWeights().embed_tokens;
    EXPECT_EQ(output.dtype, DataType::Float32());
    EXPECT_EQ(output.bytes, 65536 * sizeof(float));
    for (uint32_t bits = 0; bits < 65536; ++bits) {
        ASSERT_EQ(ReadBits(output, bits), WidenOracle(static_cast<uint16_t>(bits), bfloat))
                << "source bits " << bits;
    }
    EXPECT_EQ(stats->calls, 1U);
    EXPECT_EQ(output.storage, (*converted)->GetResolvedWeights().final_norm.storage);
}

TEST(RawWeightFp32Conversion, Float16EveryBitPatternMatchesIndependentOracle) {
    CheckEveryPattern(DataType::Float(16), false);
}

TEST(RawWeightFp32Conversion, BFloat16EveryBitPatternMatchesIndependentOracle) {
    CheckEveryPattern(DataType::BFloat(16), true);
}

class RawWeightFp32BatchConversion : public ::testing::Test {
protected:
    void SetUp() override { Reset(); }
    void TearDown() override { Reset(); }
    static void Widen(const std::byte* source, std::byte* destination, size_t count) noexcept {
        ++calls_;
        ConvertFp16ToFp32Scalar(source, destination, count);
    }
    static void WidenBfloat(const std::byte* source, std::byte* destination, size_t count) noexcept {
        ++bfloat_calls_;
        bfloat_elements_ += count;
        ConvertBf16ToFp32Scalar(source, destination, count);
    }
    static inline size_t calls_ = 0;
    static inline size_t bfloat_calls_ = 0;
    static inline size_t bfloat_elements_ = 0;

private:
    static void Reset() {
        calls_ = 0;
        bfloat_calls_ = 0;
        bfloat_elements_ = 0;
    }
};

TEST_F(RawWeightFp32BatchConversion, InvokesPreparedCallbackOncePerUniqueUnalignedSource) {
    auto source = std::make_shared<SourceStorage>(35);
    for (uint16_t index = 0; index < 17; ++index) {
        std::memcpy(source->data.data() + 1 + index * sizeof(index), &index, sizeof(index));
    }
    const auto original = source->data;
    LoadedModel model(MakeConfig(), MakeWeights(MakeView(source, DataType::Float(16), 17, 1)));
    CPUAllocator allocator(Device::CPU());
    auto result = ConvertRawModelWeightsToFp32(model, allocator, {.fp16_to_fp32 = Widen});
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(calls_, 1U);
    EXPECT_EQ(source->data, original);
    const auto& weights = (*result)->GetResolvedWeights();
    EXPECT_EQ(weights.embed_tokens.storage, weights.final_norm.storage);
    for (uint16_t index = 0; index < 17; ++index) {
        EXPECT_EQ(ReadBits(weights.embed_tokens, index), WidenOracle(index, false));
    }
}

TEST_F(RawWeightFp32BatchConversion, NullCallbackAndBudgetFailurePrecedeAllocationAndConversion) {
    auto source = std::make_shared<SourceStorage>(16);
    LoadedModel model(MakeConfig(), MakeWeights(MakeView(source, DataType::Float(16), 8)));
    auto stats = std::make_shared<AllocationStats>();
    TrackingAllocator allocator(stats);
    EXPECT_EQ(ConvertRawModelWeightsToFp32(model, allocator, {.fp16_to_fp32 = nullptr}).status().code(),
              StatusCode::kInvalidArgument);
    EXPECT_EQ(ConvertRawModelWeightsToFp32(model, allocator,
                                           {.max_owned_weight_bytes = 1, .fp16_to_fp32 = Widen})
                      .status()
                      .code(),
              StatusCode::kResourceExhausted);
    EXPECT_EQ(stats->calls, 0U);
    EXPECT_EQ(calls_, 0U);
}

TEST_F(RawWeightFp32BatchConversion, Bfloat16DoesNotInvokeFp16Callback) {
    auto source = std::make_shared<SourceStorage>(16);
    LoadedModel model(MakeConfig(DataType::BFloat(16)), MakeWeights(MakeView(source, DataType::BFloat(16), 8)));
    CPUAllocator allocator(Device::CPU());
    EXPECT_TRUE(ConvertRawModelWeightsToFp32(model, allocator, {.fp16_to_fp32 = Widen}).ok());
    EXPECT_EQ(calls_, 0U);
}

TEST_F(RawWeightFp32BatchConversion, InvokesBfloatCallbackOncePerUniqueUnalignedSource) {
    const std::vector<uint16_t> values{0x0000, 0x8000, 0x0001, 0x007f, 0x0080,
                                       0x3f80, 0x7f7f, 0x7f80, 0xff80, 0x7f81, 0xffa5};
    auto source = std::make_shared<SourceStorage>(1 + values.size() * sizeof(uint16_t));
    std::memcpy(source->data.data() + 1, values.data(), values.size() * sizeof(uint16_t));
    const auto original = source->data;
    auto input = MakeWeights(MakeView(source, DataType::BFloat(16), values.size(), 1));
    input.lm_head = input.embed_tokens;
    LoadedModel model(MakeConfig(DataType::BFloat(16)), std::move(input));
    auto stats = std::make_shared<AllocationStats>();
    TrackingAllocator allocator(stats);
    auto result = ConvertRawModelWeightsToFp32(
            model, allocator, {.fp16_to_fp32 = Widen, .bf16_to_fp32 = WidenBfloat});
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(calls_, 0U);
    EXPECT_EQ(bfloat_calls_, 1U);
    EXPECT_EQ(bfloat_elements_, values.size());
    EXPECT_EQ(stats->calls, 1U);
    EXPECT_EQ(source->data, original);
    const auto& weights = (*result)->GetResolvedWeights();
    ASSERT_TRUE(weights.lm_head.has_value());
    EXPECT_EQ(weights.embed_tokens.storage, weights.final_norm.storage);
    EXPECT_EQ(weights.embed_tokens.storage, weights.lm_head->storage);
    for (size_t index = 0; index < values.size(); ++index) {
        EXPECT_EQ(ReadBits(weights.embed_tokens, index), WidenOracle(values[index], true));
    }
}

TEST_F(RawWeightFp32BatchConversion, MixedDtypesUseIndependentCallbacksAndRetainFloat32View) {
    auto source = std::make_shared<SourceStorage>(32);
    const uint16_t half_one = 0x3c00;
    const uint16_t bfloat_one = 0x3f80;
    std::memcpy(source->data.data(), &half_one, sizeof(half_one));
    std::memcpy(source->data.data() + 8, &bfloat_one, sizeof(bfloat_one));
    auto weights = MakeWeights(MakeView(source, DataType::Float(16), 1));
    weights.final_norm = MakeView(source, DataType::BFloat(16), 1, 8);
    weights.lm_head = MakeView(source, DataType::Float32(), 1, 16);
    const auto original_fp32 = *weights.lm_head;
    LoadedModel model(MakeConfig(), std::move(weights));
    CPUAllocator allocator(Device::CPU());
    auto result = ConvertRawModelWeightsToFp32(
            model, allocator, {.fp16_to_fp32 = Widen, .bf16_to_fp32 = WidenBfloat});
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(calls_, 1U);
    EXPECT_EQ(bfloat_calls_, 1U);
    EXPECT_EQ(bfloat_elements_, 1U);
    const auto& output = (*result)->GetResolvedWeights();
    EXPECT_EQ(ReadBits(output.embed_tokens, 0), 0x3f800000U);
    EXPECT_EQ(ReadBits(output.final_norm, 0), 0x3f800000U);
    ASSERT_TRUE(output.lm_head.has_value());
    EXPECT_EQ(output.lm_head->data, original_fp32.data);
    EXPECT_EQ(output.lm_head->storage, original_fp32.storage);
}

TEST_F(RawWeightFp32BatchConversion, BfloatNullCallbackAndBudgetFailurePrecedeAllocationAndConversion) {
    auto source = std::make_shared<SourceStorage>(16);
    LoadedModel model(MakeConfig(DataType::BFloat(16)), MakeWeights(MakeView(source, DataType::BFloat(16), 8)));
    auto stats = std::make_shared<AllocationStats>();
    TrackingAllocator allocator(stats);
    EXPECT_EQ(ConvertRawModelWeightsToFp32(model, allocator, {.bf16_to_fp32 = nullptr}).status().code(),
              StatusCode::kInvalidArgument);
    EXPECT_EQ(ConvertRawModelWeightsToFp32(
                      model, allocator,
                      {.max_owned_weight_bytes = 1, .bf16_to_fp32 = WidenBfloat})
                      .status()
                      .code(),
              StatusCode::kResourceExhausted);
    EXPECT_EQ(stats->calls, 0U);
    EXPECT_EQ(bfloat_calls_, 0U);
}

TEST_F(RawWeightFp32BatchConversion, BfloatConvertedOwnerOutlivesInputAndAllocator) {
    std::unique_ptr<LoadedModel> converted;
    std::weak_ptr<const RawStorage> source_weak;
    auto stats = std::make_shared<AllocationStats>();
    {
        auto source = std::make_shared<SourceStorage>(2);
        source_weak = source;
        const uint16_t one = 0x3f80;
        std::memcpy(source->data.data(), &one, sizeof(one));
        LoadedModel model(MakeConfig(DataType::BFloat(16)), MakeWeights(MakeView(source, DataType::BFloat(16), 1)));
        TrackingAllocator allocator(stats);
        auto result = ConvertRawModelWeightsToFp32(model, allocator, {.bf16_to_fp32 = WidenBfloat});
        ASSERT_TRUE(result.ok()) << result.status().ToString();
        converted = std::move(*result);
    }
    EXPECT_TRUE(source_weak.expired());
    EXPECT_EQ(stats->live, 1U);
    EXPECT_EQ(bfloat_calls_, 1U);
    EXPECT_EQ(ReadBits(converted->GetResolvedWeights().embed_tokens, 0), 0x3f800000U);
    converted.reset();
    EXPECT_EQ(stats->live, 0U);
    EXPECT_EQ(stats->released, 1U);
}

TEST(RawWeightFp32Conversion, AcceptsUnalignedSourceWithoutModifyingBytes) {
    auto source = std::make_shared<SourceStorage>(9);
    const std::vector<uint16_t> values{0x0001, 0x8000, 0x7c01, 0xfc00};
    std::memcpy(source->data.data() + 1, values.data(), 8);
    const auto original = source->data;
    LoadedModel model(MakeConfig(), MakeWeights(MakeView(source, DataType::Float(16), 4, 1)));
    CPUAllocator allocator(Device::CPU());
    auto converted = ConvertRawModelWeightsToFp32(model, allocator);
    ASSERT_TRUE(converted.ok()) << converted.status().ToString();
    for (size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ(ReadBits((*converted)->GetResolvedWeights().embed_tokens, i), WidenOracle(values[i], false));
    }
    EXPECT_EQ(source->data, original);
    EXPECT_EQ(model.GetResolvedWeights().embed_tokens.dtype, DataType::Float(16));
}

TEST(RawWeightFp32Conversion, PreservesTieAndDistinguishesLogicalShapes) {
    auto source = std::make_shared<SourceStorage>(8);
    auto weights = MakeWeights(MakeView(source, DataType::Float(16), 4));
    weights.final_norm.shape = {2, 2};
    weights.lm_head = weights.embed_tokens;
    LoadedModel model(MakeConfig(), std::move(weights));
    auto stats = std::make_shared<AllocationStats>();
    TrackingAllocator allocator(stats);
    auto converted = ConvertRawModelWeightsToFp32(model, allocator);
    ASSERT_TRUE(converted.ok()) << converted.status().ToString();
    const auto& result = (*converted)->GetResolvedWeights();
    ASSERT_TRUE(result.lm_head.has_value());
    EXPECT_EQ(result.embed_tokens.data, result.lm_head->data);
    EXPECT_EQ(result.embed_tokens.storage, result.lm_head->storage);
    EXPECT_NE(result.embed_tokens.storage, result.final_norm.storage);
    EXPECT_EQ(stats->calls, 2U);
    const auto& usage = *(*converted)->GetWeightFp32ConversionUsage();
    EXPECT_EQ(usage.converted_source_bytes, 16U);
    EXPECT_EQ(usage.converted_weight_bytes, 32U);
    EXPECT_EQ(usage.source_weights.referenced_view_bytes, 8U);
    EXPECT_EQ(usage.converted_weights.owned_bytes, 32U);
}

TEST(RawWeightFp32Conversion, DoesNotMergeEqualContentFromDifferentOwners) {
    auto first = std::make_shared<SourceStorage>(8), second = std::make_shared<SourceStorage>(8);
    auto weights = MakeWeights(MakeView(first, DataType::Float(16), 4));
    weights.final_norm = MakeView(second, DataType::Float(16), 4);
    LoadedModel model(MakeConfig(), std::move(weights));
    CPUAllocator allocator(Device::CPU());
    auto converted = ConvertRawModelWeightsToFp32(model, allocator);
    ASSERT_TRUE(converted.ok()) << converted.status().ToString();
    const auto& result = (*converted)->GetResolvedWeights();
    EXPECT_NE(result.embed_tokens.data, result.final_norm.data);
    EXPECT_NE(result.embed_tokens.storage, result.final_norm.storage);
}

TEST(RawWeightFp32Conversion, Float32SharesSourceAndPreservesHintAndOptionalRole) {
    auto source = std::make_shared<SourceStorage>(16);
    auto weights = MakeWeights(MakeView(source, DataType::Float32(), 4));
    LoadedModel model(MakeConfig(DataType::BFloat(16)), weights);
    auto stats = std::make_shared<AllocationStats>();
    TrackingAllocator allocator(stats);
    auto converted = ConvertRawModelWeightsToFp32(model, allocator);
    ASSERT_TRUE(converted.ok()) << converted.status().ToString();
    EXPECT_EQ(stats->calls, 0U);
    EXPECT_EQ((*converted)->GetResolvedWeights().embed_tokens.storage, source);
    EXPECT_EQ((*converted)->GetResolvedWeights().embed_tokens.data, weights.embed_tokens.data);
    EXPECT_FALSE((*converted)->GetResolvedWeights().lm_head.has_value());
    EXPECT_EQ((*converted)->GetConfig().weight_dtype_hint, DataType::BFloat(16));
    EXPECT_EQ((*converted)->GetConfig().weight_dtype_hint_name, "source dtype hint");
    EXPECT_EQ((*converted)->GetWeightFp32ConversionUsage()->converted_weight_bytes, 0U);
}

TEST(RawWeightFp32Conversion, ConvertedOwnerOutlivesInputAndAllocator) {
    std::unique_ptr<LoadedModel> converted;
    std::weak_ptr<const RawStorage> source_weak;
    auto stats = std::make_shared<AllocationStats>();
    {
        auto source = std::make_shared<SourceStorage>(2);
        source_weak = source;
        const uint16_t one = 0x3c00;
        std::memcpy(source->data.data(), &one, sizeof(one));
        LoadedModel model(MakeConfig(), MakeWeights(MakeView(source, DataType::Float(16), 1)));
        TrackingAllocator allocator(stats);
        auto result = ConvertRawModelWeightsToFp32(model, allocator);
        ASSERT_TRUE(result.ok()) << result.status().ToString();
        converted = std::move(*result);
    }
    EXPECT_TRUE(source_weak.expired());
    EXPECT_EQ(stats->live, 1U);
    EXPECT_EQ(ReadBits(converted->GetResolvedWeights().embed_tokens, 0), 0x3f800000U);
    converted.reset();
    EXPECT_EQ(stats->live, 0U);
    EXPECT_EQ(stats->released, 1U);
}

TEST(RawWeightFp32Conversion, EnforcesFinalOwnedBudgetBeforeAnyPayloadAllocation) {
    auto source = std::make_shared<SourceStorage>(8);
    source->kind = RawStorageKind::kOwned;
    LoadedModel model(MakeConfig(), MakeWeights(MakeView(source, DataType::Float(16), 4)));
    auto stats = std::make_shared<AllocationStats>();
    TrackingAllocator allocator(stats);
    auto rejected = ConvertRawModelWeightsToFp32(model, allocator, {.max_owned_weight_bytes = 15});
    EXPECT_EQ(rejected.status().code(), StatusCode::kResourceExhausted);
    EXPECT_EQ(stats->calls, 0U);
    auto accepted = ConvertRawModelWeightsToFp32(model, allocator, {.max_owned_weight_bytes = 16});
    ASSERT_TRUE(accepted.ok()) << accepted.status().ToString();
    const auto& usage = *(*accepted)->GetWeightFp32ConversionUsage();
    EXPECT_EQ(usage.peak_owned_weight_bytes, 24U);
    EXPECT_EQ(usage.converted_weights.owned_bytes, 16U);
    EXPECT_TRUE(usage.peak_owned_weight_bytes_complete);
    EXPECT_TRUE(ConvertRawModelWeightsToFp32(model, allocator).ok());
}

TEST(RawWeightFp32Conversion, CountsRetainedFloat32BackingAlongsideConvertedPayload) {
    auto source = std::make_shared<SourceStorage>(24);
    source->kind = RawStorageKind::kOwned;
    auto weights = MakeWeights(MakeView(source, DataType::Float(16), 4));
    weights.final_norm = MakeView(source, DataType::Float32(), 4, 8);
    LoadedModel model(MakeConfig(), std::move(weights));
    auto stats = std::make_shared<AllocationStats>();
    TrackingAllocator allocator(stats);
    EXPECT_EQ(ConvertRawModelWeightsToFp32(model, allocator, {.max_owned_weight_bytes = 39}).status().code(),
              StatusCode::kResourceExhausted);
    EXPECT_EQ(stats->calls, 0U);
    auto converted = ConvertRawModelWeightsToFp32(model, allocator, {.max_owned_weight_bytes = 40});
    ASSERT_TRUE(converted.ok()) << converted.status().ToString();
    EXPECT_EQ((*converted)->GetWeightFp32ConversionUsage()->converted_weights.owned_bytes, 40U);
}

TEST(RawWeightFp32Conversion, UnknownSourceIsReportedAndFiniteRetainedCapFailsClosed) {
    auto source = std::make_shared<SourceStorage>(8);
    source->known = false;
    LoadedModel reduced(MakeConfig(), MakeWeights(MakeView(source, DataType::Float(16), 4)));
    auto stats = std::make_shared<AllocationStats>();
    TrackingAllocator allocator(stats);
    auto converted = ConvertRawModelWeightsToFp32(reduced, allocator, {.max_owned_weight_bytes = 16});
    ASSERT_TRUE(converted.ok()) << converted.status().ToString();
    EXPECT_FALSE((*converted)->GetWeightFp32ConversionUsage()->peak_owned_weight_bytes_complete);
    EXPECT_TRUE((*converted)->GetWeightFp32ConversionUsage()->converted_weights.backing_usage_complete);
    LoadedModel fp32(MakeConfig(), MakeWeights(MakeView(source, DataType::Float32(), 2)));
    const auto calls = stats->calls;
    EXPECT_EQ(ConvertRawModelWeightsToFp32(fp32, allocator, {.max_owned_weight_bytes = 16}).status().code(),
              StatusCode::kFailedPrecondition);
    EXPECT_EQ(stats->calls, calls);
    EXPECT_TRUE(ConvertRawModelWeightsToFp32(fp32, allocator).ok());
}

TEST(RawWeightFp32Conversion, PartialAllocationFailureReleasesCandidatesAndAllowsRetry) {
    auto source = std::make_shared<SourceStorage>(8);
    auto weights = MakeWeights(MakeView(source, DataType::Float(16), 2));
    weights.final_norm = MakeView(source, DataType::Float(16), 2, 4);
    LoadedModel model(MakeConfig(), std::move(weights));
    auto stats = std::make_shared<AllocationStats>();
    stats->fail_on_call = 1;
    TrackingAllocator allocator(stats);
    auto result = ConvertRawModelWeightsToFp32(model, allocator);
    EXPECT_EQ(result.status().code(), StatusCode::kResourceExhausted);
    EXPECT_EQ(result.status().message(), "test allocation failure");
    EXPECT_EQ(stats->live, 0U);
    EXPECT_EQ(stats->released, 1U);
    EXPECT_EQ(model.GetResolvedWeights().embed_tokens.dtype, DataType::Float(16));
    stats->fail_on_call = std::numeric_limits<size_t>::max();
    EXPECT_TRUE(ConvertRawModelWeightsToFp32(model, allocator).ok());
    EXPECT_EQ(stats->live, 0U);
}

TEST(RawWeightFp32Conversion, ValidatesEveryLayerBeforePayloadAllocation) {
    auto source = std::make_shared<SourceStorage>(8);
    auto weights = MakeWeights(MakeView(source, DataType::Float(16), 4));
    weights.layers.emplace_back();
    LoadedModel model(MakeConfig(), std::move(weights));
    auto stats = std::make_shared<AllocationStats>();
    TrackingAllocator allocator(stats);
    EXPECT_FALSE(ConvertRawModelWeightsToFp32(model, allocator).ok());
    EXPECT_EQ(stats->calls, 0U);
}

TEST(RawWeightFp32Conversion, RejectsMalformedShapeSizeDTypeAndContiguityBeforeAllocation) {
    auto source = std::make_shared<SourceStorage>(8);
    const auto valid = MakeView(source, DataType::Float(16), 4);
    auto stats = std::make_shared<AllocationStats>();
    TrackingAllocator allocator(stats);
    const auto reject = [&](RawWeightView bad) {
        auto weights = MakeWeights(valid);
        weights.final_norm = std::move(bad);
        LoadedModel model(MakeConfig(), std::move(weights));
        EXPECT_FALSE(ConvertRawModelWeightsToFp32(model, allocator).ok());
        EXPECT_EQ(stats->calls, 0U);
    };
    auto bad = valid;
    bad.shape = {-1};
    reject(bad);
    bad = valid;
    bad.bytes = 7;
    reject(bad);
    bad = valid;
    bad.dtype = DataType::Int(16);
    reject(bad);
    bad = valid;
    bad.dtype = DataType::Float(16, 2);
    reject(bad);
    for (const auto dtype: {DataType::ScalableFloat(16, 2),
                            DataType(DLDataTypeCode::kBFloat, 16, 2, true),
                            DataType::ScalableFloat(32, 2),
                            DataType::Float(16, 65535), DataType{}}) {
        bad = valid;
        bad.dtype = dtype;
        reject(bad);
    }
    bad = valid;
    bad.is_contiguous = false;
    reject(bad);
    bad = valid;
    bad.storage.reset();
    reject(bad);
    bad = valid;
    bad.shape = {std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max()};
    reject(bad);
}

TEST(RawWeightFp32Conversion, RejectsTargetSizeOverflowBeforeAllocationOrPayloadRead) {
    auto source = std::make_shared<SourceStorage>(1);
    constexpr size_t count = std::numeric_limits<size_t>::max() / sizeof(float) + 1;
    RawWeightView huge{.data = reinterpret_cast<const std::byte*>(64),
                       .bytes = count * sizeof(uint16_t),
                       .dtype = DataType::Float(16),
                       .shape = {static_cast<int64_t>(count)},
                       .storage = source};
    LoadedModel model(MakeConfig(), MakeWeights(huge));
    auto stats = std::make_shared<AllocationStats>();
    TrackingAllocator allocator(stats);
    EXPECT_EQ(ConvertRawModelWeightsToFp32(model, allocator).status().code(), StatusCode::kOverflow);
    EXPECT_EQ(stats->calls, 0U);
}

TEST(RawWeightFp32Conversion, ZeroElementWeightsDoNotReadSourcePayload) {
    auto source = std::make_shared<SourceStorage>(1);
    RawWeightView empty{.data = nullptr,
                        .bytes = 0,
                        .dtype = DataType::Float(16),
                        .shape = {0},
                        .storage = source};
    LoadedModel model(MakeConfig(), MakeWeights(empty));
    CPUAllocator allocator(Device::CPU());
    auto converted = ConvertRawModelWeightsToFp32(model, allocator, {.max_owned_weight_bytes = 1});
    ASSERT_TRUE(converted.ok()) << converted.status().ToString();
    const auto& output = (*converted)->GetResolvedWeights().embed_tokens;
    EXPECT_EQ(output.dtype, DataType::Float32());
    EXPECT_EQ(output.bytes, 0U);
    EXPECT_EQ(output.shape, std::vector<int64_t>{0});
    EXPECT_TRUE(ValidateRawWeightView(output).ok());
    EXPECT_EQ((*converted)->GetWeightFp32ConversionUsage()->converted_weight_bytes, 0U);
}

struct NormConversionFixture {
    std::shared_ptr<SourceStorage> matrices;
    std::shared_ptr<SourceStorage> reduced_norm;
    std::shared_ptr<SourceStorage> fp32_norm;
    ResolvedModelWeights weights;
};

NormConversionFixture MakeNormConversionFixture(DataType dtype, bool shared_backing = false) {
    NormConversionFixture fixture;
    fixture.matrices = std::make_shared<SourceStorage>(shared_backing ? 12 : 8);
    fixture.reduced_norm = shared_backing ? fixture.matrices : std::make_shared<SourceStorage>(4);
    fixture.fp32_norm = std::make_shared<SourceStorage>(8);
    for (const auto& owner: {fixture.matrices, fixture.reduced_norm, fixture.fp32_norm}) {
        owner->kind = RawStorageKind::kOwned;
    }
    const std::array<uint16_t, 2> norm_bits = dtype == DataType::BFloat(16)
                                                      ? std::array<uint16_t, 2>{0x3f80, 0xc000}
                                                      : std::array<uint16_t, 2>{0x3c00, 0xc000};
    const size_t norm_offset = shared_backing ? 8 : 0;
    std::memcpy(fixture.reduced_norm->data.data() + norm_offset, norm_bits.data(), sizeof(norm_bits));
    auto matrix = MakeView(fixture.matrices, dtype, 4);
    matrix.shape = {2, 2};
    fixture.weights.embed_tokens = matrix;
    fixture.weights.lm_head = matrix;
    fixture.weights.final_norm = MakeView(fixture.reduced_norm, dtype, 2, norm_offset);
    fixture.weights.layers.resize(1);
    auto& layer = fixture.weights.layers[0];
    layer.norm.input_rmsnorm = fixture.weights.final_norm;
    layer.norm.post_attn_rmsnorm = MakeView(fixture.fp32_norm, DataType::Float32(), 2);
    layer.attn = {matrix, matrix, matrix, matrix};
    layer.mlp = {matrix, matrix, matrix};
    return fixture;
}

void ExpectSameView(const RawWeightView& result, const RawWeightView& input) {
    EXPECT_EQ(result.data, input.data);
    EXPECT_EQ(result.bytes, input.bytes);
    EXPECT_EQ(result.dtype, input.dtype);
    EXPECT_EQ(result.shape, input.shape);
    EXPECT_EQ(result.storage, input.storage);
    EXPECT_EQ(result.is_contiguous, input.is_contiguous);
}

void ExpectMatricesUnchanged(const ResolvedModelWeights& result, const ResolvedModelWeights& input) {
    ExpectSameView(result.embed_tokens, input.embed_tokens);
    ASSERT_TRUE(result.lm_head.has_value());
    ASSERT_TRUE(input.lm_head.has_value());
    ExpectSameView(*result.lm_head, *input.lm_head);
    ASSERT_EQ(result.layers.size(), input.layers.size());
    for (size_t index = 0; index < result.layers.size(); ++index) {
        const auto& result_layer = result.layers[index];
        const auto& input_layer = input.layers[index];
        for (const auto& [actual, expected]: {
                     std::pair{&result_layer.attn.q_proj, &input_layer.attn.q_proj},
                     std::pair{&result_layer.attn.k_proj, &input_layer.attn.k_proj},
                     std::pair{&result_layer.attn.v_proj, &input_layer.attn.v_proj},
                     std::pair{&result_layer.attn.o_proj, &input_layer.attn.o_proj},
                     std::pair{&result_layer.mlp.gate_proj, &input_layer.mlp.gate_proj},
                     std::pair{&result_layer.mlp.up_proj, &input_layer.mlp.up_proj},
                     std::pair{&result_layer.mlp.down_proj, &input_layer.mlp.down_proj}}) {
            ExpectSameView(*actual, *expected);
        }
    }
}

class RawWeightNormFp32Conversion : public ::testing::TestWithParam<bool> {
protected:
    DataType dtype() const { return GetParam() ? DataType::BFloat(16) : DataType::Float(16); }
    void SetUp() override { fp16_calls_ = bf16_calls_ = 0; }
    void TearDown() override { fp16_calls_ = bf16_calls_ = 0; }
    RawWeightFp32ConversionOptions options(size_t cap = 0) const {
        return {.max_owned_weight_bytes = cap,
                .fp16_to_fp32 = WidenFp16,
                .bf16_to_fp32 = WidenBf16,
                .scope = RawWeightFp32ConversionScope::kNormWeightsOnly};
    }
    size_t calls() const { return fp16_calls_ + bf16_calls_; }
    static inline size_t fp16_calls_ = 0;
    static inline size_t bf16_calls_ = 0;

private:
    static void WidenFp16(const std::byte* src, std::byte* dst, size_t count) noexcept {
        ++fp16_calls_;
        ConvertFp16ToFp32Scalar(src, dst, count);
    }
    static void WidenBf16(const std::byte* src, std::byte* dst, size_t count) noexcept {
        ++bf16_calls_;
        ConvertBf16ToFp32Scalar(src, dst, count);
    }
};

TEST_P(RawWeightNormFp32Conversion, ConvertsOnlyUniqueNormSourceAndPreservesEveryMatrixView) {
    const auto fixture = MakeNormConversionFixture(dtype());
    LoadedModel model(MakeConfig(dtype()), fixture.weights);
    auto stats = std::make_shared<AllocationStats>();
    TrackingAllocator allocator(stats);
    auto result = ConvertRawModelWeightsToFp32(model, allocator, options());
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    const auto& weights = (*result)->GetResolvedWeights();
    ExpectMatricesUnchanged(weights, fixture.weights);
    EXPECT_EQ(weights.embed_tokens.dtype.nbytes(), 2);
    EXPECT_EQ(weights.embed_tokens.storage, weights.lm_head->storage);
    EXPECT_EQ(weights.final_norm.dtype, DataType::Float32());
    EXPECT_EQ(weights.final_norm.bytes, 8U);
    EXPECT_EQ(weights.final_norm.shape, fixture.weights.final_norm.shape);
    EXPECT_EQ(weights.final_norm.storage, weights.layers[0].norm.input_rmsnorm.storage);
    EXPECT_NE(weights.final_norm.storage, fixture.reduced_norm);
    EXPECT_EQ(ReadBits(weights.final_norm, 0), 0x3f800000U);
    EXPECT_EQ(ReadBits(weights.final_norm, 1), 0xc0000000U);
    ExpectSameView(weights.layers[0].norm.post_attn_rmsnorm, fixture.weights.layers[0].norm.post_attn_rmsnorm);
    EXPECT_EQ(fp16_calls_, GetParam() ? 0U : 1U);
    EXPECT_EQ(bf16_calls_, GetParam() ? 1U : 0U);
    EXPECT_EQ(stats->calls, 1U);
    EXPECT_EQ(model.GetConfig().weight_dtype_hint, dtype());
    ExpectSameView(model.GetResolvedWeights().final_norm, fixture.weights.final_norm);
}

TEST_P(RawWeightNormFp32Conversion, EnforcesFinalBudget24AndReportsPeak28BeforeAnyPayloadAllocation) {
    const auto fixture = MakeNormConversionFixture(dtype());
    LoadedModel model(MakeConfig(dtype()), fixture.weights);
    auto stats = std::make_shared<AllocationStats>();
    TrackingAllocator allocator(stats);
    EXPECT_EQ(ConvertRawModelWeightsToFp32(model, allocator, options(23)).status().code(),
              StatusCode::kResourceExhausted);
    EXPECT_EQ(stats->calls, 0U);
    EXPECT_EQ(calls(), 0U);
    auto result = ConvertRawModelWeightsToFp32(model, allocator, options(24));
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    const auto& usage = *(*result)->GetWeightFp32ConversionUsage();
    EXPECT_EQ(usage.source_weights.owned_bytes, 20U);
    EXPECT_EQ(usage.converted_source_bytes, 4U);
    EXPECT_EQ(usage.converted_weight_bytes, 8U);
    EXPECT_EQ(usage.converted_weights.owned_bytes, 24U);
    EXPECT_EQ(usage.peak_owned_weight_bytes, 28U);
    EXPECT_TRUE(usage.peak_owned_weight_bytes_complete);
    EXPECT_EQ(stats->calls, 1U);
}

TEST_P(RawWeightNormFp32Conversion, CountsSharedBackingOnceWithFinal28AndPeak28) {
    const auto fixture = MakeNormConversionFixture(dtype(), true);
    LoadedModel model(MakeConfig(dtype()), fixture.weights);
    auto stats = std::make_shared<AllocationStats>();
    TrackingAllocator allocator(stats);
    EXPECT_EQ(ConvertRawModelWeightsToFp32(model, allocator, options(27)).status().code(),
              StatusCode::kResourceExhausted);
    EXPECT_EQ(stats->calls, 0U);
    EXPECT_EQ(calls(), 0U);
    auto result = ConvertRawModelWeightsToFp32(model, allocator, options(28));
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    const auto& usage = *(*result)->GetWeightFp32ConversionUsage();
    EXPECT_EQ(usage.source_weights.owned_bytes, 20U);
    EXPECT_EQ(usage.converted_source_bytes, 4U);
    EXPECT_EQ(usage.converted_weight_bytes, 8U);
    EXPECT_EQ(usage.converted_weights.owned_bytes, 28U);
    EXPECT_EQ(usage.converted_weights.referenced_view_bytes, 24U);
    EXPECT_EQ(usage.peak_owned_weight_bytes, 28U);
    EXPECT_EQ(calls(), 1U);
    ExpectMatricesUnchanged((*result)->GetResolvedWeights(), fixture.weights);
}

TEST_P(RawWeightNormFp32Conversion, AllowsIdenticalSourceAcrossConvertedNormAndRetainedMatrixRoles) {
    auto fixture = MakeNormConversionFixture(dtype());
    fixture.weights.final_norm = fixture.weights.embed_tokens;
    fixture.weights.layers[0].norm.input_rmsnorm = fixture.weights.embed_tokens;
    LoadedModel model(MakeConfig(dtype()), fixture.weights);
    CPUAllocator allocator(Device::CPU());
    auto result = ConvertRawModelWeightsToFp32(model, allocator, options(32));
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    const auto& weights = (*result)->GetResolvedWeights();
    ExpectMatricesUnchanged(weights, fixture.weights);
    EXPECT_EQ(weights.final_norm.dtype, DataType::Float32());
    EXPECT_EQ(weights.final_norm.bytes, 16U);
    EXPECT_NE(weights.final_norm.storage, weights.embed_tokens.storage);
    EXPECT_EQ(weights.final_norm.storage, weights.layers[0].norm.input_rmsnorm.storage);
    EXPECT_EQ((*result)->GetWeightFp32ConversionUsage()->converted_weights.owned_bytes, 32U);
    EXPECT_EQ(calls(), 1U);
}

TEST_P(RawWeightNormFp32Conversion, DoesNotMergeDistinctNormShapesOrOwners) {
    for (const bool different_owner: {false, true}) {
        auto fixture = MakeNormConversionFixture(dtype());
        if (different_owner) {
            auto other = std::make_shared<SourceStorage>(4);
            other->kind = RawStorageKind::kOwned;
            other->data = fixture.reduced_norm->data;
            fixture.weights.layers[0].norm.input_rmsnorm = MakeView(other, dtype(), 2);
        } else {
            fixture.weights.layers[0].norm.input_rmsnorm.shape = {1, 2};
        }
        LoadedModel model(MakeConfig(dtype()), fixture.weights);
        auto stats = std::make_shared<AllocationStats>();
        TrackingAllocator allocator(stats);
        auto result = ConvertRawModelWeightsToFp32(model, allocator, options());
        ASSERT_TRUE(result.ok()) << result.status().ToString();
        const auto& weights = (*result)->GetResolvedWeights();
        EXPECT_NE(weights.final_norm.storage, weights.layers[0].norm.input_rmsnorm.storage);
        EXPECT_EQ(stats->calls, 2U);
        EXPECT_EQ((*result)->GetWeightFp32ConversionUsage()->converted_source_bytes, 8U);
    }
}

TEST_P(RawWeightNormFp32Conversion, LeavesExistingFloat32NormsSharedWithoutAllocation) {
    auto fixture = MakeNormConversionFixture(dtype());
    fixture.weights.final_norm = fixture.weights.layers[0].norm.post_attn_rmsnorm;
    fixture.weights.layers[0].norm.input_rmsnorm = fixture.weights.final_norm;
    LoadedModel model(MakeConfig(dtype()), fixture.weights);
    auto stats = std::make_shared<AllocationStats>();
    TrackingAllocator allocator(stats);
    auto result = ConvertRawModelWeightsToFp32(model, allocator, options(16));
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ExpectSameView((*result)->GetResolvedWeights().final_norm, fixture.weights.final_norm);
    ExpectMatricesUnchanged((*result)->GetResolvedWeights(), fixture.weights);
    EXPECT_EQ(stats->calls, 0U);
    EXPECT_EQ(calls(), 0U);
    EXPECT_EQ((*result)->GetWeightFp32ConversionUsage()->converted_weight_bytes, 0U);
}

TEST_P(RawWeightNormFp32Conversion, UnknownRetainedW16BackingFailsFiniteCapBeforeAllocation) {
    auto fixture = MakeNormConversionFixture(dtype());
    fixture.matrices->known = false;
    LoadedModel model(MakeConfig(dtype()), fixture.weights);
    auto stats = std::make_shared<AllocationStats>();
    TrackingAllocator allocator(stats);
    EXPECT_EQ(ConvertRawModelWeightsToFp32(model, allocator, options(24)).status().code(),
              StatusCode::kFailedPrecondition);
    EXPECT_EQ(stats->calls, 0U);
    EXPECT_EQ(calls(), 0U);
    auto unlimited = ConvertRawModelWeightsToFp32(model, allocator, options());
    ASSERT_TRUE(unlimited.ok()) << unlimited.status().ToString();
    EXPECT_FALSE((*unlimited)->GetWeightFp32ConversionUsage()->converted_weights.backing_usage_complete);
}

TEST_P(RawWeightNormFp32Conversion, UnknownSourceOnlyBackingKeepsPeakIncompleteAndFinalComplete) {
    auto fixture = MakeNormConversionFixture(dtype());
    fixture.reduced_norm->known = false;
    LoadedModel model(MakeConfig(dtype()), fixture.weights);
    CPUAllocator allocator(Device::CPU());
    auto result = ConvertRawModelWeightsToFp32(model, allocator, options(24));
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    const auto& usage = *(*result)->GetWeightFp32ConversionUsage();
    EXPECT_FALSE(usage.peak_owned_weight_bytes_complete);
    EXPECT_TRUE(usage.converted_weights.backing_usage_complete);
    EXPECT_EQ(usage.converted_weights.owned_bytes, 24U);
}

TEST_P(RawWeightNormFp32Conversion, InvalidScopeCallbacksAndUnselectedMetadataPrecedeAllocation) {
    const auto fixture = MakeNormConversionFixture(dtype());
    LoadedModel model(MakeConfig(dtype()), fixture.weights);
    auto stats = std::make_shared<AllocationStats>();
    TrackingAllocator allocator(stats);
    for (size_t index = 0; index < 3; ++index) {
        auto invalid = options();
        if (index == 0) {
            invalid.scope = static_cast<RawWeightFp32ConversionScope>(-1);
        } else if (index == 1) {
            invalid.fp16_to_fp32 = nullptr;
        } else {
            invalid.bf16_to_fp32 = nullptr;
        }
        EXPECT_EQ(ConvertRawModelWeightsToFp32(model, allocator, invalid).status().code(),
                  StatusCode::kInvalidArgument);
    }
    for (size_t index = 0; index < 5; ++index) {
        auto malformed = fixture.weights;
        auto& matrix = malformed.layers[0].attn.q_proj;
        switch (index) {
            case 0:
                --matrix.bytes;
                break;
            case 1:
                matrix.shape = {-1};
                break;
            case 2:
                matrix.is_contiguous = false;
                break;
            case 3:
                matrix.storage.reset();
                break;
            case 4:
                matrix.dtype = DataType::Int(16);
                break;
        }
        LoadedModel bad_model(MakeConfig(dtype()), std::move(malformed));
        EXPECT_FALSE(ConvertRawModelWeightsToFp32(bad_model, allocator, options()).ok());
    }
    EXPECT_EQ(stats->calls, 0U);
    EXPECT_EQ(calls(), 0U);
}

TEST_P(RawWeightNormFp32Conversion, PartialNormAllocationFailureReleasesCandidatesAndAllowsRetry) {
    auto fixture = MakeNormConversionFixture(dtype());
    fixture.weights.layers[0].norm.input_rmsnorm.shape = {1, 2};
    LoadedModel model(MakeConfig(dtype()), fixture.weights);
    auto stats = std::make_shared<AllocationStats>();
    stats->fail_on_call = 1;
    TrackingAllocator allocator(stats);
    auto failed = ConvertRawModelWeightsToFp32(model, allocator, options());
    EXPECT_EQ(failed.status().code(), StatusCode::kResourceExhausted);
    EXPECT_EQ(stats->live, 0U);
    EXPECT_EQ(stats->released, 1U);
    EXPECT_EQ(calls(), 1U);
    ExpectSameView(model.GetResolvedWeights().final_norm, fixture.weights.final_norm);
    ExpectMatricesUnchanged(model.GetResolvedWeights(), fixture.weights);
    stats->fail_on_call = std::numeric_limits<size_t>::max();
    {
        auto retry = ConvertRawModelWeightsToFp32(model, allocator, options());
        ASSERT_TRUE(retry.ok()) << retry.status().ToString();
        EXPECT_EQ(stats->live, 2U);
    }
    EXPECT_EQ(stats->live, 0U);
    EXPECT_EQ(stats->released, 3U);
}

TEST_P(RawWeightNormFp32Conversion, OutputOutlivesInputAndAllocatorWhileKeepingOnlyRequiredSourceOwners) {
    for (const bool shared_backing: {false, true}) {
        std::unique_ptr<LoadedModel> output;
        std::weak_ptr<const RawStorage> norm_source;
        std::weak_ptr<const RawStorage> matrix_source;
        auto stats = std::make_shared<AllocationStats>();
        {
            const auto fixture = MakeNormConversionFixture(dtype(), shared_backing);
            norm_source = fixture.reduced_norm;
            matrix_source = fixture.matrices;
            LoadedModel model(MakeConfig(dtype()), fixture.weights);
            TrackingAllocator allocator(stats);
            auto result = ConvertRawModelWeightsToFp32(model, allocator, options());
            ASSERT_TRUE(result.ok()) << result.status().ToString();
            output = std::move(*result);
        }
        EXPECT_EQ(norm_source.expired(), !shared_backing);
        EXPECT_FALSE(matrix_source.expired());
        EXPECT_EQ(stats->live, 1U);
        EXPECT_EQ(ReadBits(output->GetResolvedWeights().final_norm, 0), 0x3f800000U);
        EXPECT_EQ(output->GetResolvedWeights().embed_tokens.dtype, dtype());
        output.reset();
        EXPECT_TRUE(norm_source.expired());
        EXPECT_TRUE(matrix_source.expired());
        EXPECT_EQ(stats->live, 0U);
        EXPECT_EQ(stats->released, 1U);
    }
}

INSTANTIATE_TEST_SUITE_P(ReducedDtypes, RawWeightNormFp32Conversion,
                         ::testing::Bool(), [](const auto& info) { return info.param ? "Bfloat16" : "Fp16"; });

} // namespace
