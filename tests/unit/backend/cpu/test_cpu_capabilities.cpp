#include "aethermind/backend/cpu/cpu_capabilities.h"

#include <gtest/gtest.h>
#include <string>
#include <string_view>

namespace {

using namespace aethermind;

TEST(CpuCapabilities, DefaultSnapshotHasNoDetectedFeatures) {
    const CpuCapabilities capabilities;
    EXPECT_EQ(capabilities.architecture, CpuArchitecture::kUnknown);
    EXPECT_TRUE(capabilities.hardware_features.empty());
    EXPECT_TRUE(capabilities.usable_features.empty());
    EXPECT_TRUE(capabilities.effective_features.empty());
    EXPECT_EQ(capabilities.sve_vector_bytes, 0U);
}

TEST(CpuCapabilities, ToStringArchitectureNamesAllValues) {
    EXPECT_STREQ(ToString(CpuArchitecture::kUnknown), "Unknown");
    EXPECT_STREQ(ToString(CpuArchitecture::kX86_64), "x86_64");
    EXPECT_STREQ(ToString(CpuArchitecture::kAArch64), "AArch64");
}

TEST(CpuCapabilities, ToStringFeatureTableCoversEveryEnumValue) {
    // Every feature must render a name distinct from the out-of-range
    // fallback; adding a feature without extending the name table (in
    // cpu_capabilities.cpp) fails here instead of silently dumping "Unknown".
    for (int index = 0; index < static_cast<int>(CpuFeature::kCount); ++index) {
        const auto feature = static_cast<CpuFeature>(index);
        EXPECT_STRNE(ToString(feature), "Unknown") << "feature index " << index;
        EXPECT_NE(std::string_view{ToString(feature)}, std::string_view{})
                << "feature index " << index;
    }
}

TEST(CpuCapabilities, ToStringFeatureFallsBackForOutOfRangeValue) {
    EXPECT_STREQ(ToString(CpuFeature::kCount), "Unknown");
}

TEST(CpuCapabilities, ToStringFeatureSetRendersNamesInEnumOrder) {
    EXPECT_EQ(ToString(CpuFeatureSet::From({CpuFeature::kFma, CpuFeature::kAvx2})),
              std::string("{AVX2, FMA}"));
    EXPECT_EQ(ToString(CpuFeatureSet::From({CpuFeature::kFma})),
              std::string("{FMA}"));
}

TEST(CpuCapabilities, ToStringFeatureSetRendersEmptySet) {
    EXPECT_EQ(ToString(CpuFeatureSet{}), std::string("{}"));
}

TEST(CpuCapabilities, IntersectComputesSharedAndEmptyResults) {
    constexpr auto shared = CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma})
                                    .Intersect(CpuFeatureSet::From(
                                            {CpuFeature::kFma, CpuFeature::kF16c}));
    EXPECT_TRUE(shared.Contains(CpuFeature::kFma));
    EXPECT_FALSE(shared.Contains(CpuFeature::kAvx2));
    EXPECT_FALSE(shared.Contains(CpuFeature::kF16c));

    const auto disjoint = CpuFeatureSet::From({CpuFeature::kAvx2})
                                  .Intersect(CpuFeatureSet::From({CpuFeature::kFma}));
    EXPECT_TRUE(disjoint.empty());
}

TEST(CpuCapabilities, IntersectWithEmptySetIsEmpty) {
    constexpr auto result = CpuFeatureSet::From({CpuFeature::kAvx2})
                                    .Intersect(CpuFeatureSet{});
    EXPECT_TRUE(result.empty());
}

TEST(CpuCapabilities, DifferenceRemovesOnlyRequestedFeatures) {
    constexpr auto result = CpuFeatureSet::From(
                                    {CpuFeature::kAvx2, CpuFeature::kFma, CpuFeature::kF16c})
                                    .Difference(CpuFeatureSet::From({CpuFeature::kFma}));
    EXPECT_TRUE(result.Contains(CpuFeature::kAvx2));
    EXPECT_TRUE(result.Contains(CpuFeature::kF16c));
    EXPECT_FALSE(result.Contains(CpuFeature::kFma));
}

TEST(CpuCapabilities, DifferenceWithDisjointSetIsIdentity) {
    constexpr auto left = CpuFeatureSet::From({CpuFeature::kAvx2});
    EXPECT_EQ(left.Difference(CpuFeatureSet::From({CpuFeature::kFma})), left);
}

} // namespace
