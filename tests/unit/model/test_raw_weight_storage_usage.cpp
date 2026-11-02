#include "aethermind/model/loaded_model.h"
#include "aethermind/model/model_loader.h"
#include "aethermind/model/resolved_model_weights.h"
#include "aethermind/model/weight/raw_weight_storage_usage.h"

#include "inference/test_malloc_interposer.h"
#include <array>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>

namespace {
using namespace aethermind;
struct ResourceRawStorage final : RawStorage {
    std::array<std::byte, 64> data{};
    std::optional<RawStorageFootprint> info;
    std::optional<RawStorageFootprint> footprint() const noexcept override { return info; }
};
RawWeightView View(const std::shared_ptr<ResourceRawStorage>& owner, size_t offset, size_t count) {
    return {.data = owner->data.data() + offset, .bytes = count, .dtype = DataType::UInt(8), .shape = {static_cast<int64_t>(count)}, .storage = owner};
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
    auto summary = ComputeRawWeightStorageUsage(weights);
    ASSERT_TRUE(summary.ok());
    EXPECT_EQ(summary->owned_bytes, 64U);
    EXPECT_EQ(summary->referenced_view_bytes, 16U);
}
TEST(RawWeightStorageUsage, MergesTiedAndOverlappingRangesPerOwner) {
    auto owner = std::make_shared<ResourceRawStorage>();
    owner->info = RawStorageFootprint{RawStorageKind::kOwned, 64};
    ResolvedModelWeights weights;
    weights.embed_tokens = View(owner, 0, 16);
    weights.lm_head = View(owner, 8, 16);
    weights.final_norm = View(owner, 32, 8);
    auto summary = ComputeRawWeightStorageUsage(weights);
    ASSERT_TRUE(summary.ok());
    EXPECT_EQ(summary->owned_bytes, 64U);
    EXPECT_EQ(summary->referenced_view_bytes, 32U);
    EXPECT_TRUE(summary->backing_usage_complete);
}
TEST(RawWeightStorageUsage, UnknownBackingIsExplicitAndKnownCategoriesRemainPartial) {
    auto first = std::make_shared<ResourceRawStorage>(), second = std::make_shared<ResourceRawStorage>();
    second->info = RawStorageFootprint{RawStorageKind::kExternal, 64};
    ResolvedModelWeights weights;
    weights.embed_tokens = View(first, 0, 16);
    weights.final_norm = View(second, 0, 8);
    auto summary = ComputeRawWeightStorageUsage(weights);
    ASSERT_TRUE(summary.ok());
    EXPECT_FALSE(summary->backing_usage_complete);
    EXPECT_EQ(summary->external_bytes, 64U);
    EXPECT_EQ(summary->referenced_view_bytes, 24U);
}
TEST(RawWeightStorageUsage, RejectsMalformedAndUnderreportedBacking) {
    ResolvedModelWeights weights;
    weights.final_norm.bytes = 1;
    EXPECT_EQ(ComputeRawWeightStorageUsage(weights).status().code(), StatusCode::kInvalidArgument);
    auto owner = std::make_shared<ResourceRawStorage>();
    owner->info = RawStorageFootprint{RawStorageKind::kOwned, 8};
    weights.final_norm = View(owner, 0, 16);
    EXPECT_EQ(ComputeRawWeightStorageUsage(weights).status().code(), StatusCode::kInvalidArgument);
}
TEST(RawWeightStorageUsage, RejectsConflictingSharedOwnerFootprints) {
    struct Pair {
        ResourceRawStorage a, b;
    };
    auto pair = std::make_shared<Pair>();
    pair->a.info = RawStorageFootprint{RawStorageKind::kOwned, 64};
    pair->b.info = RawStorageFootprint{RawStorageKind::kOwned, 32};
    auto a = std::shared_ptr<ResourceRawStorage>(pair, &pair->a), b = std::shared_ptr<ResourceRawStorage>(pair, &pair->b);
    ResolvedModelWeights weights;
    weights.embed_tokens = View(a, 0, 8);
    weights.final_norm = View(b, 0, 8);
    EXPECT_EQ(ComputeRawWeightStorageUsage(weights).status().code(), StatusCode::kInvalidArgument);
}
TEST(RawWeightStorageUsage, RejectsBackingTotalOverflowWithoutReadingPayload) {
    auto a = std::make_shared<ResourceRawStorage>(), b = std::make_shared<ResourceRawStorage>();
    a->info = RawStorageFootprint{RawStorageKind::kOwned, std::numeric_limits<size_t>::max()};
    b->info = RawStorageFootprint{RawStorageKind::kOwned, 1};
    ResolvedModelWeights weights;
    weights.embed_tokens = View(a, 0, 1);
    weights.final_norm = View(b, 0, 1);
    EXPECT_EQ(ComputeRawWeightStorageUsage(weights).status().code(), StatusCode::kOverflow);
}
TEST(RawWeightStorageUsage, HfMappingCountsCompleteFileOnce) {
    const auto dir = std::filesystem::path(AETHERMIND_TEST_MODELS_DIR) / "tiny-random-LlamaForCausalLM";
    auto loaded = ModelLoader::Load(dir);
    ASSERT_TRUE(loaded.ok()) << loaded.status().ToString();
    auto summary = ComputeRawWeightStorageUsage((*loaded)->GetResolvedWeights());
    ASSERT_TRUE(summary.ok());
    EXPECT_TRUE(summary->backing_usage_complete);
    EXPECT_EQ(summary->mapped_bytes, std::filesystem::file_size(dir / "model.safetensors"));
    EXPECT_EQ(summary->owned_bytes, 0U);
    EXPECT_GT(summary->referenced_view_bytes, 0U);
}
TEST(RawWeightStorageUsageDiagnosticFailure, MetadataFailureReturnsResourceExhausted) {
    if (!test::MallocInterposerAvailable()) GTEST_SKIP() << "Requires the glibc malloc interposer";
    auto owner = std::make_shared<ResourceRawStorage>();
    ResolvedModelWeights weights;
    weights.embed_tokens = View(owner, 0, 16);
    test::ScopedMallocFailure failure(0);
    auto result = ComputeRawWeightStorageUsage(weights);
    const auto observed = failure.Stop();
    EXPECT_TRUE(observed.allocation_failed);
    EXPECT_EQ(result.status().code(), StatusCode::kResourceExhausted);
    EXPECT_TRUE(result.status().message().empty());
}
} // namespace
