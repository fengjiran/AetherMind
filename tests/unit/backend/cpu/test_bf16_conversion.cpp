#include "aethermind/backend/cpu/bf16_conversion.h"
#include "aethermind/backend/cpu/cpu_backend.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <span>

namespace {

using namespace aethermind;

TEST(CpuBf16Conversion, SelectsPortableScalarForBothDirections) {
    for (const auto features: {CpuFeatureSet{},
                               CpuFeatureSet::From({CpuFeature::kAvx, CpuFeature::kAvx2, CpuFeature::kAvx512Bf16}),
                               CpuFeatureSet::From({CpuFeature::kNeon, CpuFeature::kBf16})}) {
        const auto functions = cpu::SelectBf16Conversions(features);
        EXPECT_EQ(functions.to_fp32, ConvertBf16ToFp32Scalar);
        EXPECT_EQ(functions.from_fp32, ConvertFp32ToBf16Scalar);
    }
}

TEST(CpuBf16Conversion, BackendTableOutlivesBackendWithNoEffectiveFeatures) {
    Bf16ConversionFuncs functions;
    {
        CpuBackend backend(CpuCapabilities{});
        functions = backend.GetHostBf16Conversions();
        EXPECT_EQ(functions.to_fp32, ConvertBf16ToFp32Scalar);
        EXPECT_EQ(functions.from_fp32, ConvertFp32ToBf16Scalar);
    }
    const std::array<uint16_t, 4> input{0x0001, 0x8000, 0x3f80, 0xff81};
    std::array<uint32_t, 4> widened{};
    ASSERT_TRUE(ConvertBf16ToFp32(std::as_bytes(std::span(input)),
                                  std::as_writable_bytes(std::span(widened)), functions)
                        .ok());
    EXPECT_EQ(widened, (std::array<uint32_t, 4>{0x00010000U, 0x80000000U, 0x3f800000U, 0xffc10000U}));
    std::array<uint16_t, 4> narrowed{};
    ASSERT_TRUE(ConvertFp32ToBf16(std::as_bytes(std::span(widened)),
                                  std::as_writable_bytes(std::span(narrowed)), functions)
                        .ok());
    EXPECT_EQ(narrowed, (std::array<uint16_t, 4>{0x0001, 0x8000, 0x3f80, 0xffc1}));
}

} // namespace
