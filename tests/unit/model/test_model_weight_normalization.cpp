#include "aethermind/memory/cpu_allocator.h"
#include "aethermind/model/loaded_model.h"
#include "aethermind/model/weight/model_weight_normalization.h"

#include <bit>
#include <cmath>
#include <cstdint>
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
    std::optional<RawStorageFootprint> footprint() const noexcept override {
        if (!known) return std::nullopt;
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
    Device device() const noexcept override { return Device::CPU(); }

protected:
    StatusOr<MemoryHandle> AllocateStorage(const AllocationRequest& request) override {
        const size_t call = stats_->calls++;
        if (call == stats_->fail_on_call) return Status::ResourceExhausted("test allocation failure");
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
    const float value = static_cast<float>(sign == 0 ? magnitude : -magnitude);
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
    auto normalized = NormalizeLoadedModelWeightsToFp32(model, allocator);
    ASSERT_TRUE(normalized.ok()) << normalized.status().ToString();
    const auto& output = (*normalized)->GetResolvedWeights().embed_tokens;
    EXPECT_EQ(output.dtype, DataType::Float32());
    EXPECT_EQ(output.bytes, 65536 * sizeof(float));
    for (uint32_t bits = 0; bits < 65536; ++bits) {
        ASSERT_EQ(ReadBits(output, bits), WidenOracle(static_cast<uint16_t>(bits), bfloat))
                << "source bits " << bits;
    }
    EXPECT_EQ(stats->calls, 1U);
    EXPECT_EQ(output.storage, (*normalized)->GetResolvedWeights().final_norm.storage);
}

TEST(ModelWeightNormalization, Float16EveryBitPatternMatchesIndependentOracle) {
    CheckEveryPattern(DataType::Float(16), false);
}

TEST(ModelWeightNormalization, BFloat16EveryBitPatternMatchesIndependentOracle) {
    CheckEveryPattern(DataType::BFloat(16), true);
}

TEST(ModelWeightNormalization, AcceptsUnalignedSourceWithoutModifyingBytes) {
    auto source = std::make_shared<SourceStorage>(9);
    const std::vector<uint16_t> values{0x0001, 0x8000, 0x7c01, 0xfc00};
    std::memcpy(source->data.data() + 1, values.data(), 8);
    const auto original = source->data;
    LoadedModel model(MakeConfig(), MakeWeights(MakeView(source, DataType::Float(16), 4, 1)));
    CPUAllocator allocator(Device::CPU());
    auto normalized = NormalizeLoadedModelWeightsToFp32(model, allocator);
    ASSERT_TRUE(normalized.ok()) << normalized.status().ToString();
    for (size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ(ReadBits((*normalized)->GetResolvedWeights().embed_tokens, i), WidenOracle(values[i], false));
    }
    EXPECT_EQ(source->data, original);
    EXPECT_EQ(model.GetResolvedWeights().embed_tokens.dtype, DataType::Float(16));
}

TEST(ModelWeightNormalization, PreservesTieAndDistinguishesLogicalShapes) {
    auto source = std::make_shared<SourceStorage>(8);
    auto weights = MakeWeights(MakeView(source, DataType::Float(16), 4));
    weights.final_norm.shape = {2, 2};
    weights.lm_head = weights.embed_tokens;
    LoadedModel model(MakeConfig(), std::move(weights));
    auto stats = std::make_shared<AllocationStats>();
    TrackingAllocator allocator(stats);
    auto normalized = NormalizeLoadedModelWeightsToFp32(model, allocator);
    ASSERT_TRUE(normalized.ok()) << normalized.status().ToString();
    const auto& result = (*normalized)->GetResolvedWeights();
    ASSERT_TRUE(result.lm_head.has_value());
    EXPECT_EQ(result.embed_tokens.data, result.lm_head->data);
    EXPECT_EQ(result.embed_tokens.storage, result.lm_head->storage);
    EXPECT_NE(result.embed_tokens.storage, result.final_norm.storage);
    EXPECT_EQ(stats->calls, 2U);
    const auto& usage = *(*normalized)->GetWeightNormalizationUsage();
    EXPECT_EQ(usage.converted_source_bytes, 16U);
    EXPECT_EQ(usage.converted_weight_bytes, 32U);
    EXPECT_EQ(usage.source_weights.logical_view_bytes, 8U);
    EXPECT_EQ(usage.normalized_weights.owned_bytes, 32U);
}

TEST(ModelWeightNormalization, DoesNotMergeEqualContentFromDifferentOwners) {
    auto first = std::make_shared<SourceStorage>(8), second = std::make_shared<SourceStorage>(8);
    auto weights = MakeWeights(MakeView(first, DataType::Float(16), 4));
    weights.final_norm = MakeView(second, DataType::Float(16), 4);
    LoadedModel model(MakeConfig(), std::move(weights));
    CPUAllocator allocator(Device::CPU());
    auto normalized = NormalizeLoadedModelWeightsToFp32(model, allocator);
    ASSERT_TRUE(normalized.ok()) << normalized.status().ToString();
    const auto& result = (*normalized)->GetResolvedWeights();
    EXPECT_NE(result.embed_tokens.data, result.final_norm.data);
    EXPECT_NE(result.embed_tokens.storage, result.final_norm.storage);
}

TEST(ModelWeightNormalization, Float32SharesSourceAndPreservesHintAndOptionalRole) {
    auto source = std::make_shared<SourceStorage>(16);
    auto weights = MakeWeights(MakeView(source, DataType::Float32(), 4));
    LoadedModel model(MakeConfig(DataType::BFloat(16)), weights);
    auto stats = std::make_shared<AllocationStats>();
    TrackingAllocator allocator(stats);
    auto normalized = NormalizeLoadedModelWeightsToFp32(model, allocator);
    ASSERT_TRUE(normalized.ok()) << normalized.status().ToString();
    EXPECT_EQ(stats->calls, 0U);
    EXPECT_EQ((*normalized)->GetResolvedWeights().embed_tokens.storage, source);
    EXPECT_EQ((*normalized)->GetResolvedWeights().embed_tokens.data, weights.embed_tokens.data);
    EXPECT_FALSE((*normalized)->GetResolvedWeights().lm_head.has_value());
    EXPECT_EQ((*normalized)->GetConfig().weight_dtype_hint, DataType::BFloat(16));
    EXPECT_EQ((*normalized)->GetConfig().weight_dtype_hint_name, "source dtype hint");
    EXPECT_EQ((*normalized)->GetWeightNormalizationUsage()->converted_weight_bytes, 0U);
}

TEST(ModelWeightNormalization, ConvertedOwnerOutlivesInputAndAllocator) {
    std::unique_ptr<LoadedModel> normalized;
    std::weak_ptr<const RawStorage> source_weak;
    auto stats = std::make_shared<AllocationStats>();
    {
        auto source = std::make_shared<SourceStorage>(2);
        source_weak = source;
        const uint16_t one = 0x3c00;
        std::memcpy(source->data.data(), &one, sizeof(one));
        LoadedModel model(MakeConfig(), MakeWeights(MakeView(source, DataType::Float(16), 1)));
        TrackingAllocator allocator(stats);
        auto result = NormalizeLoadedModelWeightsToFp32(model, allocator);
        ASSERT_TRUE(result.ok()) << result.status().ToString();
        normalized = std::move(*result);
    }
    EXPECT_TRUE(source_weak.expired());
    EXPECT_EQ(stats->live, 1U);
    EXPECT_EQ(ReadBits(normalized->GetResolvedWeights().embed_tokens, 0), 0x3f800000U);
    normalized.reset();
    EXPECT_EQ(stats->live, 0U);
    EXPECT_EQ(stats->released, 1U);
}

TEST(ModelWeightNormalization, EnforcesFinalOwnedBudgetBeforeAnyPayloadAllocation) {
    auto source = std::make_shared<SourceStorage>(8);
    source->kind = RawStorageKind::kOwned;
    LoadedModel model(MakeConfig(), MakeWeights(MakeView(source, DataType::Float(16), 4)));
    auto stats = std::make_shared<AllocationStats>();
    TrackingAllocator allocator(stats);
    auto rejected = NormalizeLoadedModelWeightsToFp32(model, allocator, {.max_owned_weight_bytes = 15});
    EXPECT_EQ(rejected.status().code(), StatusCode::kResourceExhausted);
    EXPECT_EQ(stats->calls, 0U);
    auto accepted = NormalizeLoadedModelWeightsToFp32(model, allocator, {.max_owned_weight_bytes = 16});
    ASSERT_TRUE(accepted.ok()) << accepted.status().ToString();
    const auto& usage = *(*accepted)->GetWeightNormalizationUsage();
    EXPECT_EQ(usage.peak_owned_weight_bytes, 24U);
    EXPECT_EQ(usage.normalized_weights.owned_bytes, 16U);
    EXPECT_TRUE(usage.peak_owned_weight_bytes_complete);
    EXPECT_TRUE(NormalizeLoadedModelWeightsToFp32(model, allocator).ok());
}

TEST(ModelWeightNormalization, CountsRetainedFloat32BackingAlongsideConvertedPayload) {
    auto source = std::make_shared<SourceStorage>(24);
    source->kind = RawStorageKind::kOwned;
    auto weights = MakeWeights(MakeView(source, DataType::Float(16), 4));
    weights.final_norm = MakeView(source, DataType::Float32(), 4, 8);
    LoadedModel model(MakeConfig(), std::move(weights));
    auto stats = std::make_shared<AllocationStats>();
    TrackingAllocator allocator(stats);
    EXPECT_EQ(NormalizeLoadedModelWeightsToFp32(model, allocator, {.max_owned_weight_bytes = 39}).status().code(),
              StatusCode::kResourceExhausted);
    EXPECT_EQ(stats->calls, 0U);
    auto normalized = NormalizeLoadedModelWeightsToFp32(model, allocator, {.max_owned_weight_bytes = 40});
    ASSERT_TRUE(normalized.ok()) << normalized.status().ToString();
    EXPECT_EQ((*normalized)->GetWeightNormalizationUsage()->normalized_weights.owned_bytes, 40U);
}

TEST(ModelWeightNormalization, UnknownSourceIsReportedAndFiniteRetainedCapFailsClosed) {
    auto source = std::make_shared<SourceStorage>(8);
    source->known = false;
    LoadedModel reduced(MakeConfig(), MakeWeights(MakeView(source, DataType::Float(16), 4)));
    auto stats = std::make_shared<AllocationStats>();
    TrackingAllocator allocator(stats);
    auto normalized = NormalizeLoadedModelWeightsToFp32(reduced, allocator, {.max_owned_weight_bytes = 16});
    ASSERT_TRUE(normalized.ok()) << normalized.status().ToString();
    EXPECT_FALSE((*normalized)->GetWeightNormalizationUsage()->peak_owned_weight_bytes_complete);
    EXPECT_TRUE((*normalized)->GetWeightNormalizationUsage()->normalized_weights.backing_complete);
    LoadedModel fp32(MakeConfig(), MakeWeights(MakeView(source, DataType::Float32(), 2)));
    const auto calls = stats->calls;
    EXPECT_EQ(NormalizeLoadedModelWeightsToFp32(fp32, allocator, {.max_owned_weight_bytes = 16}).status().code(),
              StatusCode::kFailedPrecondition);
    EXPECT_EQ(stats->calls, calls);
    EXPECT_TRUE(NormalizeLoadedModelWeightsToFp32(fp32, allocator).ok());
}

TEST(ModelWeightNormalization, PartialAllocationFailureReleasesCandidatesAndAllowsRetry) {
    auto source = std::make_shared<SourceStorage>(8);
    auto weights = MakeWeights(MakeView(source, DataType::Float(16), 2));
    weights.final_norm = MakeView(source, DataType::Float(16), 2, 4);
    LoadedModel model(MakeConfig(), std::move(weights));
    auto stats = std::make_shared<AllocationStats>();
    stats->fail_on_call = 1;
    TrackingAllocator allocator(stats);
    auto result = NormalizeLoadedModelWeightsToFp32(model, allocator);
    EXPECT_EQ(result.status().code(), StatusCode::kResourceExhausted);
    EXPECT_EQ(result.status().message(), "test allocation failure");
    EXPECT_EQ(stats->live, 0U);
    EXPECT_EQ(stats->released, 1U);
    EXPECT_EQ(model.GetResolvedWeights().embed_tokens.dtype, DataType::Float(16));
    stats->fail_on_call = std::numeric_limits<size_t>::max();
    EXPECT_TRUE(NormalizeLoadedModelWeightsToFp32(model, allocator).ok());
    EXPECT_EQ(stats->live, 0U);
}

TEST(ModelWeightNormalization, ValidatesEveryLayerBeforePayloadAllocation) {
    auto source = std::make_shared<SourceStorage>(8);
    auto weights = MakeWeights(MakeView(source, DataType::Float(16), 4));
    weights.layers.emplace_back();
    LoadedModel model(MakeConfig(), std::move(weights));
    auto stats = std::make_shared<AllocationStats>();
    TrackingAllocator allocator(stats);
    EXPECT_FALSE(NormalizeLoadedModelWeightsToFp32(model, allocator).ok());
    EXPECT_EQ(stats->calls, 0U);
}

TEST(ModelWeightNormalization, RejectsMalformedShapeSizeDTypeAndContiguityBeforeAllocation) {
    auto source = std::make_shared<SourceStorage>(8);
    const auto valid = MakeView(source, DataType::Float(16), 4);
    auto stats = std::make_shared<AllocationStats>();
    TrackingAllocator allocator(stats);
    const auto reject = [&](RawWeightView bad) {
        auto weights = MakeWeights(valid);
        weights.final_norm = std::move(bad);
        LoadedModel model(MakeConfig(), std::move(weights));
        EXPECT_FALSE(NormalizeLoadedModelWeightsToFp32(model, allocator).ok());
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

TEST(ModelWeightNormalization, RejectsTargetSizeOverflowBeforeAllocationOrPayloadRead) {
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
    EXPECT_EQ(NormalizeLoadedModelWeightsToFp32(model, allocator).status().code(), StatusCode::kOverflow);
    EXPECT_EQ(stats->calls, 0U);
}

TEST(ModelWeightNormalization, ZeroElementWeightsDoNotReadSourcePayload) {
    auto source = std::make_shared<SourceStorage>(1);
    RawWeightView empty{.data = nullptr,
                        .bytes = 0,
                        .dtype = DataType::Float(16),
                        .shape = {0},
                        .storage = source};
    LoadedModel model(MakeConfig(), MakeWeights(empty));
    CPUAllocator allocator(Device::CPU());
    auto normalized = NormalizeLoadedModelWeightsToFp32(model, allocator, {.max_owned_weight_bytes = 1});
    ASSERT_TRUE(normalized.ok()) << normalized.status().ToString();
    const auto& output = (*normalized)->GetResolvedWeights().embed_tokens;
    EXPECT_EQ(output.dtype, DataType::Float32());
    EXPECT_EQ(output.bytes, 0U);
    EXPECT_EQ(output.shape, std::vector<int64_t>{0});
    EXPECT_TRUE(ValidateRawWeightView(output).ok());
    EXPECT_EQ((*normalized)->GetWeightNormalizationUsage()->converted_weight_bytes, 0U);
}

} // namespace
