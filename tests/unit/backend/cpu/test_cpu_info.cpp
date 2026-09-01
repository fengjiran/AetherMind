#include "aethermind/backend/cpu/cpu_info.h"

#include <gtest/gtest.h>

namespace aethermind::cpu {
namespace {

TEST(CpuInfo, DetectionReportsAConsistentSnapshot) {
    const auto detected = DetectCpuCapabilities();
    ASSERT_TRUE(detected.ok()) << detected.status().ToString();

    EXPECT_TRUE(detected->hardware_features.ContainsAll(detected->usable_features));
    EXPECT_TRUE(detected->usable_features.ContainsAll(detected->effective_features));

#if defined(__x86_64__) || defined(_M_X64)
    EXPECT_EQ(detected->architecture, CpuArchitecture::kX86_64);
#elif defined(__aarch64__)
    EXPECT_EQ(detected->architecture, CpuArchitecture::kAArch64);
    EXPECT_TRUE(detected->hardware_features.Contains(CpuFeature::kNeon));
#endif
}

TEST(CpuInfo, PolicyOnlyRestrictsUsableFeatures) {
    const CpuFeatureSet x86_features = CpuFeatureSet::From(
            {CpuFeature::kAvx2, CpuFeature::kFma});
    const CpuCapabilities snapshot{
            .architecture = CpuArchitecture::kX86_64,
            .hardware_features = x86_features,
            .usable_features = x86_features,
    };
    const auto applied = ApplyCpuFeaturePolicy(
            snapshot,
            CpuFeaturePolicy{
                    .disabled_features = CpuFeatureSet::From({CpuFeature::kFma}),
                    .required_features = CpuFeatureSet::From({CpuFeature::kAvx2}),
            });

    ASSERT_TRUE(applied.ok()) << applied.status().ToString();
    EXPECT_TRUE(applied->effective_features.Contains(CpuFeature::kAvx2));
    EXPECT_FALSE(applied->effective_features.Contains(CpuFeature::kFma));
    EXPECT_TRUE(applied->hardware_features.Contains(CpuFeature::kFma));
}

TEST(CpuInfo, FeatureRequirementsPreserveIndependentConjunctions) {
    const CpuFeatureSet available = CpuFeatureSet::From({CpuFeature::kAvx2});
    const CpuFeatureSet requirements = CpuFeatureSet::From(
            {CpuFeature::kAvx2, CpuFeature::kFma});

    EXPECT_FALSE(available.ContainsAll(requirements));
    EXPECT_TRUE(available.ContainsAll(CpuFeatureSet::From({CpuFeature::kAvx2})));
}

TEST(CpuInfo, PolicyRejectsDisabledRequiredFeature) {
    const CpuFeatureSet features = CpuFeatureSet::From({CpuFeature::kAvx2});
    const CpuCapabilities snapshot{
            .architecture = CpuArchitecture::kX86_64,
            .hardware_features = features,
            .usable_features = features,
    };
    const auto applied = ApplyCpuFeaturePolicy(
            snapshot,
            CpuFeaturePolicy{
                    .disabled_features = features,
                    .required_features = features,
            });

    EXPECT_EQ(applied.status().code(), StatusCode::kFailedPrecondition);
}

TEST(CpuInfo, PolicyRejectsInvalidSnapshot) {
    const CpuCapabilities snapshot{
            .architecture = CpuArchitecture::kX86_64,
            .usable_features = CpuFeatureSet::From({CpuFeature::kAvx2}),
    };

    const auto applied = ApplyCpuFeaturePolicy(snapshot, {});

    EXPECT_EQ(applied.status().code(), StatusCode::kInvalidArgument);
}

} // namespace
} // namespace aethermind::cpu
