#include "aethermind/backend/cpu/cpu_info.h"

#include <gtest/gtest.h>

namespace aethermind::cpu {
namespace {

TEST(CpuInfo, GetCpuFeaturesReturnsStableSingleton) {
    const CpuFeatures& first = GetCpuFeatures();
    const CpuFeatures& second = GetCpuFeatures();

    EXPECT_EQ(&first, &second);

    // Sanity-check detection against the architecture's baseline ISA, plus
    // hardware-independent invariants. VNNI always requires its base ISA:
    // AVX512-VNNI needs AVX-512F and AVX-VNNI needs AVX2. Concrete VNNI /
    // AVX-512F expectations are hardware-dependent (no Intel Mac exposes
    // either) and therefore not asserted.
#if defined(__x86_64__) || defined(_M_X64)
    EXPECT_TRUE(first.has_avx2);
    EXPECT_FALSE(first.has_vnni && !first.has_avx512f && !first.has_avx2);
#elif defined(__aarch64__)
    EXPECT_TRUE(first.has_neon);
#endif
}

}// namespace
}// namespace aethermind::cpu
