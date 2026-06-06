#include "aethermind/backend/cpu/cpu_info.h"

#include <gtest/gtest.h>

namespace aethermind::cpu {
namespace {

TEST(CpuInfo, GetCpuFeaturesReturnsStableSingleton) {
    const CpuFeatures& first = GetCpuFeatures();
    const CpuFeatures& second = GetCpuFeatures();

    EXPECT_EQ(&first, &second);

    EXPECT_TRUE(first.has_avx2);
    EXPECT_TRUE(first.has_vnni);
    EXPECT_FALSE(first.has_avx512f);
}

}// namespace
}// namespace aethermind::cpu
