/// @file test_bf16_conversion.cpp
/// @brief Bfloat16/binary32 conversion, rounding, and NaN contracts.
#include "aethermind/dtypes/bfloat16.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>

namespace {
using namespace aethermind;
using namespace aethermind::detail;

TEST(BFloat16ToFP32Test, QuietsSignalingNaNs) {
    // Single-value widening must agree with the batch contract: payload
    // preserved, quiet bit forced.
    const BFloat16 snan{0x7F85, BFloat16::from_bits()};
    EXPECT_EQ(std::bit_cast<uint32_t>(static_cast<float>(snan)), 0x7FC50000U);
    const BFloat16 negative_snan{0xFF85, BFloat16::from_bits()};
    EXPECT_EQ(std::bit_cast<uint32_t>(static_cast<float>(negative_snan)), 0xFFC50000U);

    // Quiet NaNs widen with their payload untouched.
    const BFloat16 qnan{0x7FE5, BFloat16::from_bits()};
    EXPECT_EQ(std::bit_cast<uint32_t>(static_cast<float>(qnan)), 0x7FE50000U);
}

TEST(BFloat16FromFP32Test, RoundToNearestEvenTies) {
    // At exponent 0 the bfloat16 ULP is 2^-7; ties must break to the even mantissa.
    EXPECT_EQ(bf16_from_fp32_value(1.00390625f), 0x3F80U); // 1 + 2^-8: even lower
    EXPECT_EQ(bf16_from_fp32_value(1.01171875f), 0x3F82U); // 1 + 3*2^-8: even upper

    // At exponent 1 the bfloat16 ULP is 2^-6.
    EXPECT_EQ(bf16_from_fp32_value(2.0078125f), 0x4000U); // 2 + 2^-7: even lower
    EXPECT_EQ(bf16_from_fp32_value(2.0234375f), 0x4002U); // 2 + 3*2^-7: even upper

    // Sign is preserved on both tie directions.
    EXPECT_EQ(bf16_from_fp32_value(-1.00390625f), 0xBF80U);
    EXPECT_EQ(bf16_from_fp32_value(-1.01171875f), 0xBF82U);
}

TEST(BFloat16FromFP32Test, RoundsInsteadOfTruncating) {
    // Truncation maps both values to 0x3F80; round-to-nearest must distinguish them.
    EXPECT_EQ(bf16_from_fp32_value(1.005859375f), 0x3F81U); // 1 + 0.75 ULP
    EXPECT_EQ(bf16_from_fp32_value(1.001953125f), 0x3F80U); // 1 + 0.25 ULP
    EXPECT_EQ(bf16_from_fp32_value(-1.005859375f), 0xBF81U);
}

TEST(BFloat16FromFP32Test, OverflowBoundary) {
    // The largest finite bfloat16 value is 0x7F7F; the next step is infinity.
    constexpr float max_float = std::numeric_limits<float>::max();
    EXPECT_EQ(bf16_from_fp32_value(max_float), 0x7F80U);
    EXPECT_EQ(bf16_from_fp32_value(-max_float), 0xFF80U);

    // Exactly halfway between 0x7F7F and 2^128 breaks to the even (upper) encoding.
    const float tie = std::ldexp(511.0f, 119);
    EXPECT_EQ(bf16_from_fp32_value(tie), 0x7F80U);
    EXPECT_EQ(bf16_from_fp32_value(-tie), 0xFF80U);

    // Values just below the midpoint still round to the largest finite value.
    EXPECT_EQ(bf16_from_fp32_value(std::ldexp(510.0f, 119)), 0x7F7FU);
    EXPECT_EQ(bf16_from_fp32_value(-std::ldexp(510.0f, 119)), 0xFF7FU);
}

TEST(BFloat16FromFP32Test, NaNPreservesSignAndTrimmedPayload) {
    struct ConversionCase {
        uint32_t source;
        uint16_t expected;
    };
    constexpr ConversionCase cases[] = {
            {0x7F800001U, 0x7FC0U}, // payload truncates away; force a quiet payload
            {0x7F810000U, 0x7FC1U},
            {0x7F8F0000U, 0x7FCFU},
            {0x7FFFFFFFU, 0x7FFFU},
            {0xFF800001U, 0xFFC0U},
            {0xFFC12345U, 0xFFC1U},
    };
    for (const auto& [source, expected]: cases) {
        const BFloat16 value(std::bit_cast<float>(source));
        EXPECT_EQ(value.x, expected) << "source bits=" << source;
    }
}

TEST(BFloat16FromFP32Test, RoundTrip) {
    constexpr float test_values[] = {
            1.0f, 2.0f, 0.5f, 3.14159f, 1e38f, 1e-38f,
            -1.0f, -2.0f, -0.5f, -3.14159f, -1e38f, -1e-38f};

    for (float f: test_values) {
        if (std::isnan(f) || std::isinf(f)) {
            continue; // NaN and infinity cannot be compared exactly.
        }

        BFloat16 bf16 = f;
        float f_roundtrip = static_cast<float>(bf16);
        BFloat16 bf16_roundtrip = f_roundtrip;

        // BF16 -> FP32 -> BF16 must reproduce identical encodings.
        EXPECT_EQ(bf16.x, bf16_roundtrip.x) << "Original: " << f << ", Roundtrip: " << f_roundtrip;
    }
}

} // namespace
