#include "aethermind/backend/cpu/cpu_capabilities.h"

#include <gtest/gtest.h>

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

} // namespace
