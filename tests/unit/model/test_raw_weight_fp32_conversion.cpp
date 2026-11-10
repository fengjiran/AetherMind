#include "aethermind/memory/cpu_allocator.h"
#include "aethermind/model/loaded_model.h"
#include "aethermind/model/weight/raw_weight_fp32_conversion.h"

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
    void SetUp() override { calls_ = 0; }
    void TearDown() override { calls_ = 0; }
    static void Widen(const std::byte* source, std::byte* destination, size_t count) noexcept {
        ++calls_;
        ConvertFp16ToFp32Scalar(source, destination, count);
    }
    static inline size_t calls_ = 0;
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

} // namespace
