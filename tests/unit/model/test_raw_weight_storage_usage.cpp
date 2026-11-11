#include "aethermind/model/loaded_model.h"
#include "aethermind/model/model_loader.h"
#include "aethermind/model/resolved_model_weights.h"
#include "aethermind/model/weight/raw_weight_storage_usage.h"
#include "inference/test_malloc_interposer.h"

#include <array>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <utility>

namespace {
using namespace aethermind;

struct ResourceRawStorage final : RawStorage {
    std::array<std::byte, 64> data{};
    std::optional<RawStorageFootprint> info;

    AM_NODISCARD std::optional<RawStorageFootprint> footprint() const noexcept override {
        return info;
    }
};

RawWeightView View(const std::shared_ptr<ResourceRawStorage>& owner,
                   size_t offset, size_t count,
                   DataType dtype = DataType::UInt(8)) {
    return {.data = owner->data.data() + offset,
            .bytes = count * static_cast<size_t>(dtype.nbytes()),
            .dtype = dtype,
            .shape = {static_cast<int64_t>(count)},
            .storage = owner};
}

TEST(RawWeightStorageUsage, EmptyAndPartialGraphsAreSupported) {
    auto empty = ComputeRawWeightStorageUsage({});
    ASSERT_TRUE(empty.ok());
    EXPECT_TRUE(empty->backing_usage_complete);
    EXPECT_EQ(empty->owned_bytes, 0U);
    auto owner = std::make_shared<ResourceRawStorage>();
    owner->info = RawStorageFootprint{RawStorageKind::kOwned, 64};
    ResolvedModelWeights weights;
    weights.final_norm = View(owner, 0, 16);
    auto usage = ComputeRawWeightStorageUsage(weights);
    ASSERT_TRUE(usage.ok());
    EXPECT_EQ(usage->owned_bytes, 64U);
    EXPECT_EQ(usage->referenced_view_bytes, 16U);
}

TEST(RawWeightStorageUsage, MergesTiedAndOverlappingRangesPerOwner) {
    auto owner = std::make_shared<ResourceRawStorage>();
    owner->info = RawStorageFootprint{RawStorageKind::kOwned, 64};
    ResolvedModelWeights weights;
    weights.embed_tokens = View(owner, 0, 16);
    weights.lm_head = View(owner, 8, 16);
    weights.final_norm = View(owner, 32, 8);
    auto usage = ComputeRawWeightStorageUsage(weights);
    ASSERT_TRUE(usage.ok());
    EXPECT_EQ(usage->owned_bytes, 64U);
    EXPECT_EQ(usage->referenced_view_bytes, 32U);
    EXPECT_TRUE(usage->backing_usage_complete);
}

TEST(RawWeightStorageUsage, ExplicitViewsMergeMixedDTypesWithinTheSameOwner) {
    auto owner = std::make_shared<ResourceRawStorage>();
    owner->info = RawStorageFootprint{RawStorageKind::kOwned, 64};
    const auto fp16 = View(owner, 0, 4, DataType::Float(16));
    const auto fp32 = View(owner, 4, 2, DataType::Float32());
    const std::array<const RawWeightView*, 3> selected{&fp16, &fp32, &fp16};
    const auto usage = ComputeRawWeightViewStorageUsage(selected);
    ASSERT_TRUE(usage.ok()) << usage.status().ToString();
    EXPECT_EQ(usage->owned_bytes, 64U);
    EXPECT_EQ(usage->referenced_view_bytes, 12U);
    EXPECT_TRUE(usage->backing_usage_complete);
}

TEST(RawWeightStorageUsage, ExplicitViewsPermitEmptySelectionAndDefaultEmptyViews) {
    const auto empty = ComputeRawWeightViewStorageUsage({});
    ASSERT_TRUE(empty.ok()) << empty.status().ToString();
    EXPECT_EQ(empty->owned_bytes, 0U);
    EXPECT_TRUE(empty->backing_usage_complete);
    const RawWeightView unused;
    const std::array<const RawWeightView*, 1> selected{&unused};
    const auto usage = ComputeRawWeightViewStorageUsage(selected);
    ASSERT_TRUE(usage.ok()) << usage.status().ToString();
    EXPECT_EQ(usage->referenced_view_bytes, 0U);
    EXPECT_TRUE(usage->backing_usage_complete);
}

TEST(RawWeightStorageUsage, ExplicitViewsRejectNullPointersAndMalformedMetadata) {
    const std::array<const RawWeightView*, 1> null_view{nullptr};
    EXPECT_EQ(ComputeRawWeightViewStorageUsage(null_view).status().code(),
              StatusCode::kInvalidArgument);
    auto owner = std::make_shared<ResourceRawStorage>();
    auto malformed = View(owner, 0, 4, DataType::Float(16));
    --malformed.bytes;
    const std::array<const RawWeightView*, 1> selected{&malformed};
    EXPECT_EQ(ComputeRawWeightViewStorageUsage(selected).status().code(),
              StatusCode::kInvalidArgument);
}

TEST(RawWeightStorageUsage, FiltersDTypeAcrossGlobalAndLayerRoles) {
    auto owned = std::make_shared<ResourceRawStorage>();
    owned->info = RawStorageFootprint{RawStorageKind::kOwned, 64};
    auto mapped = std::make_shared<ResourceRawStorage>();
    mapped->info = RawStorageFootprint{RawStorageKind::kMapped, 64};
    auto external = std::make_shared<ResourceRawStorage>();
    external->info = RawStorageFootprint{RawStorageKind::kExternal, 64};
    auto unknown = std::make_shared<ResourceRawStorage>();

    ResolvedModelWeights weights;
    weights.embed_tokens = View(unknown, 0, 4, DataType::Float(16));
    weights.final_norm = View(owned, 0, 1, DataType::Float32());
    weights.lm_head = View(mapped, 0, 2, DataType::Float32());
    weights.layers.resize(1);
    weights.layers[0].attn.q_proj = View(owned, 8, 3, DataType::Float32());
    weights.layers[0].norm.post_attn_rmsnorm = View(external, 0, 2, DataType::Float32());
    weights.layers[0].mlp.down_proj = View(unknown, 8, 1, DataType::Float(16));

    const auto usage = ComputeRawWeightStorageUsage(
            weights, DataType::Float32());

    ASSERT_TRUE(usage.ok()) << usage.status().ToString();
    EXPECT_EQ(usage->owned_bytes, 64U);
    EXPECT_EQ(usage->mapped_bytes, 64U);
    EXPECT_EQ(usage->external_bytes, 64U);
    EXPECT_EQ(usage->referenced_view_bytes, 32U);
    EXPECT_TRUE(usage->backing_usage_complete);
}

TEST(RawWeightStorageUsage, FilteredViewsKeepCompleteSharedBackingAndMergeRanges) {
    auto owner = std::make_shared<ResourceRawStorage>();
    owner->info = RawStorageFootprint{RawStorageKind::kOwned, 64};
    ResolvedModelWeights weights;
    weights.embed_tokens = View(owner, 0, 4, DataType::Float(16));
    weights.final_norm = View(owner, 16, 4, DataType::Float32());
    weights.lm_head = View(owner, 24, 4, DataType::Float32());

    const auto usage = ComputeRawWeightStorageUsage(
            weights, DataType::Float32());

    ASSERT_TRUE(usage.ok()) << usage.status().ToString();
    EXPECT_EQ(usage->owned_bytes, 64U);
    EXPECT_EQ(usage->referenced_view_bytes, 24U);
    EXPECT_TRUE(usage->backing_usage_complete);
}

TEST(RawWeightStorageUsage, SupportsNonFloatDTypeFilters) {
    auto selected = std::make_shared<ResourceRawStorage>();
    selected->info = RawStorageFootprint{RawStorageKind::kExternal, 64};
    auto excluded = std::make_shared<ResourceRawStorage>();
    excluded->info = RawStorageFootprint{RawStorageKind::kOwned, 64};
    ResolvedModelWeights weights;
    weights.embed_tokens = View(selected, 0, 5);
    weights.final_norm = View(excluded, 0, 4, DataType::Float32());

    const auto usage = ComputeRawWeightStorageUsage(
            weights, DataType::UInt(8));

    ASSERT_TRUE(usage.ok()) << usage.status().ToString();
    EXPECT_EQ(usage->owned_bytes, 0U);
    EXPECT_EQ(usage->external_bytes, 64U);
    EXPECT_EQ(usage->referenced_view_bytes, 5U);
    EXPECT_TRUE(usage->backing_usage_complete);
}

TEST(RawWeightStorageUsage, DTypeFilterMatchesBitWidthAndLaneCount) {
    auto owner = std::make_shared<ResourceRawStorage>();
    owner->info = RawStorageFootprint{RawStorageKind::kOwned, 64};
    ResolvedModelWeights weights;
    weights.embed_tokens = View(owner, 0, 1, DataType::Float32());
    weights.final_norm = View(owner, 8, 1, DataType::Float(32, 2));
    weights.lm_head = View(owner, 16, 1, DataType::Float(16));

    for (const auto& [dtype, bytes]: {
                 std::pair{DataType::Float32(), 4U},
                 std::pair{DataType::Float(32, 2), 8U},
                 std::pair{DataType::Float(16), 2U}}) {
        const auto usage = ComputeRawWeightStorageUsage(weights, dtype);

        ASSERT_TRUE(usage.ok()) << usage.status().ToString();
        EXPECT_EQ(usage->owned_bytes, 64U);
        EXPECT_EQ(usage->referenced_view_bytes, bytes);
    }
}

TEST(RawWeightStorageUsage, FilteredUsageReportsSelectedUnknownBacking) {
    auto unknown = std::make_shared<ResourceRawStorage>();
    ResolvedModelWeights weights;
    weights.final_norm = View(unknown, 0, 4, DataType::Float32());

    const auto usage = ComputeRawWeightStorageUsage(
            weights, DataType::Float32());

    ASSERT_TRUE(usage.ok()) << usage.status().ToString();
    EXPECT_EQ(usage->owned_bytes, 0U);
    EXPECT_EQ(usage->referenced_view_bytes, 16U);
    EXPECT_FALSE(usage->backing_usage_complete);
}

TEST(RawWeightStorageUsage, FilteredUsageWithoutMatchesIsCompleteAndEmpty) {
    auto unknown = std::make_shared<ResourceRawStorage>();
    ResolvedModelWeights weights;
    weights.embed_tokens = View(unknown, 0, 4, DataType::Float(16));
    weights.lm_head.emplace();

    const auto usage = ComputeRawWeightStorageUsage(
            weights, DataType::Float32());

    ASSERT_TRUE(usage.ok()) << usage.status().ToString();
    EXPECT_EQ(usage->owned_bytes, 0U);
    EXPECT_EQ(usage->mapped_bytes, 0U);
    EXPECT_EQ(usage->external_bytes, 0U);
    EXPECT_EQ(usage->referenced_view_bytes, 0U);
    EXPECT_TRUE(usage->backing_usage_complete);
}

TEST(RawWeightStorageUsage, FilteredZeroElementViewRetainsBacking) {
    auto owner = std::make_shared<ResourceRawStorage>();
    owner->info = RawStorageFootprint{RawStorageKind::kOwned, 64};
    ResolvedModelWeights weights;
    weights.final_norm = View(owner, 0, 0, DataType::Float32());

    const auto usage = ComputeRawWeightStorageUsage(
            weights, DataType::Float32());

    ASSERT_TRUE(usage.ok()) << usage.status().ToString();
    EXPECT_EQ(usage->owned_bytes, 64U);
    EXPECT_EQ(usage->referenced_view_bytes, 0U);
    EXPECT_TRUE(usage->backing_usage_complete);
}

TEST(RawWeightStorageUsage, FilteredUsageRejectsMalformedUnselectedViews) {
    auto owner = std::make_shared<ResourceRawStorage>();
    const auto valid = View(owner, 0, 4, DataType::Float(16));
    std::array<RawWeightView, 4> invalid{valid, valid, valid, valid};
    invalid[0].bytes = 7;
    invalid[1].shape = {-1};
    invalid[2].storage.reset();
    invalid[3].is_contiguous = false;
    for (const auto& view: invalid) {
        ResolvedModelWeights weights;
        weights.final_norm = view;

        EXPECT_EQ(ComputeRawWeightStorageUsage(weights, DataType::Float32()).status(),
                  StatusCode::kInvalidArgument);
    }
}

TEST(RawWeightStorageUsage, FilteredUsageRejectsNonFixedLaneViewsBeforeDTypeComparison) {
    auto owner = std::make_shared<ResourceRawStorage>();
    for (const auto dtype: {DataType::ScalableFloat(32, 2),
                            DataType::ScalableFloat(16, 2),
                            DataType::Float(32, 65535),
                            DataType::Float(32, 65536)}) {
        SCOPED_TRACE(dtype.raw_lanes());
        ResolvedModelWeights weights;
        weights.final_norm = View(owner, 0, 4, DataType::Float32());
        weights.final_norm.dtype = dtype;

        EXPECT_EQ(ComputeRawWeightStorageUsage(weights, DataType::Float32()).status(),
                  StatusCode::kInvalidArgument);
    }
}

TEST(RawWeightStorageUsage, RejectsInvalidDTypeFiltersForEmptyWeights) {
    for (const auto dtype: {DataType{}, DataType::ScalableFloat(32, 2),
                            DataType::Float(32, 65535),
                            DataType::Float(32, 65536)}) {
        SCOPED_TRACE(dtype.raw_lanes());

        EXPECT_EQ(ComputeRawWeightStorageUsage({}, dtype).status().code(),
                  StatusCode::kInvalidArgument);
    }
}

TEST(RawWeightStorageUsage, UnknownBackingIsExplicitAndKnownCategoriesRemainPartial) {
    auto first = std::make_shared<ResourceRawStorage>(),
         second = std::make_shared<ResourceRawStorage>();
    second->info = RawStorageFootprint{RawStorageKind::kExternal, 64};
    ResolvedModelWeights weights;
    weights.embed_tokens = View(first, 0, 16);
    weights.final_norm = View(second, 0, 8);
    auto usage = ComputeRawWeightStorageUsage(weights);
    ASSERT_TRUE(usage.ok());
    EXPECT_FALSE(usage->backing_usage_complete);
    EXPECT_EQ(usage->external_bytes, 64U);
    EXPECT_EQ(usage->referenced_view_bytes, 24U);
}

TEST(RawWeightStorageUsage, RejectsMalformedAndUnderreportedBacking) {
    ResolvedModelWeights weights;
    weights.final_norm.bytes = 1;
    EXPECT_EQ(ComputeRawWeightStorageUsage(weights).status().code(),
              StatusCode::kInvalidArgument);
    auto owner = std::make_shared<ResourceRawStorage>();
    owner->info = RawStorageFootprint{RawStorageKind::kOwned, 8};
    weights.final_norm = View(owner, 0, 16);
    EXPECT_EQ(ComputeRawWeightStorageUsage(weights).status().code(),
              StatusCode::kInvalidArgument);
}

TEST(RawWeightStorageUsage, RejectsConflictingSharedOwnerFootprints) {
    struct Pair {
        ResourceRawStorage a, b;
    };

    auto pair = std::make_shared<Pair>();
    pair->a.info = RawStorageFootprint{RawStorageKind::kOwned, 64};
    pair->b.info = RawStorageFootprint{RawStorageKind::kOwned, 32};
    auto a = std::shared_ptr<ResourceRawStorage>(pair, &pair->a),
         b = std::shared_ptr<ResourceRawStorage>(pair, &pair->b);
    ResolvedModelWeights weights;
    weights.embed_tokens = View(a, 0, 8);
    weights.final_norm = View(b, 0, 8);
    EXPECT_EQ(ComputeRawWeightStorageUsage(weights).status(),
              StatusCode::kInvalidArgument);
    EXPECT_EQ(ComputeRawWeightStorageUsage(weights, DataType::UInt(8)).status(),
              StatusCode::kInvalidArgument);
}

TEST(RawWeightStorageUsage, RejectsBackingTotalOverflowWithoutReadingPayload) {
    auto a = std::make_shared<ResourceRawStorage>(),
         b = std::make_shared<ResourceRawStorage>();
    a->info = RawStorageFootprint{RawStorageKind::kOwned, std::numeric_limits<size_t>::max()};
    b->info = RawStorageFootprint{RawStorageKind::kOwned, 1};
    ResolvedModelWeights weights;
    weights.embed_tokens = View(a, 0, 1);
    weights.final_norm = View(b, 0, 1);
    EXPECT_EQ(ComputeRawWeightStorageUsage(weights).status(), StatusCode::kOverflow);
}

TEST(RawWeightStorageUsage, HfMappingCountsCompleteFileOnce) {
    const auto dir = std::filesystem::path(AETHERMIND_TEST_MODELS_DIR) / "tiny-random-LlamaForCausalLM";
    auto loaded = ModelLoader::Load(dir);
    ASSERT_TRUE(loaded.ok()) << loaded.status().ToString();
    auto usage = ComputeRawWeightStorageUsage((*loaded)->GetResolvedWeights());
    ASSERT_TRUE(usage.ok());
    EXPECT_TRUE(usage->backing_usage_complete);
    EXPECT_EQ(usage->mapped_bytes, std::filesystem::file_size(dir / "model.safetensors"));
    EXPECT_EQ(usage->owned_bytes, 0U);
    EXPECT_GT(usage->referenced_view_bytes, 0U);
}

TEST(RawWeightStorageUsageDiagnosticFailure, MetadataFailureReturnsResourceExhausted) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }

    const auto owner = std::make_shared<ResourceRawStorage>();
    ResolvedModelWeights weights;
    weights.embed_tokens = View(owner, 0, 16);
    test::ScopedMallocFailure failure(0);
    auto result = ComputeRawWeightStorageUsage(weights);
    const auto observed = failure.Stop();
    EXPECT_TRUE(observed.allocation_failed);
    EXPECT_EQ(result.status().code(), StatusCode::kResourceExhausted);
    EXPECT_TRUE(result.status().message().empty());
}

TEST(RawWeightStorageUsageDiagnosticFailure, FilteredMetadataFailureReturnsResourceExhausted) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }

    const auto owner = std::make_shared<ResourceRawStorage>();
    ResolvedModelWeights weights;
    weights.embed_tokens = View(owner, 0, 4, DataType::Float32());
    const auto dtype_filter = DataType::Float32();
    static_assert(noexcept(ComputeRawWeightStorageUsage(weights, dtype_filter)));

    test::ScopedMallocFailure failure(0);
    const auto result = ComputeRawWeightStorageUsage(
            weights, dtype_filter);
    const auto observed = failure.Stop();

    EXPECT_TRUE(observed.allocation_failed);
    EXPECT_EQ(result.status().code(), StatusCode::kResourceExhausted);
    EXPECT_TRUE(result.status().message().empty());
}

TEST(RawWeightStorageUsageDiagnosticFailure, ExplicitViewMetadataFailureReturnsResourceExhausted) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }

    const auto owner = std::make_shared<ResourceRawStorage>();
    const auto view = View(owner, 0, 4, DataType::Float32());
    const std::array<const RawWeightView*, 1> selected{&view};
    static_assert(noexcept(ComputeRawWeightViewStorageUsage(selected)));
    test::ScopedMallocFailure failure(0);
    const auto result = ComputeRawWeightViewStorageUsage(selected);
    const auto observed = failure.Stop();

    EXPECT_TRUE(observed.allocation_failed);
    EXPECT_EQ(result.status().code(), StatusCode::kResourceExhausted);
    EXPECT_TRUE(result.status().message().empty());
}
} // namespace
