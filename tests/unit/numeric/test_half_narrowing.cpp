/// @file
/// Fixed-RNE binary32 to binary16 conversion, including gradual underflow.
#include "aethermind/dtypes/half.h"

#include <gtest/gtest.h>

#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace {
using aethermind::Half;

// Reconstruct finite positive binary16 values independently of production
// widening. All intermediate dyadic values are exact in double and float.
float PositiveHalfValue(uint16_t bits) {
    const uint32_t exponent = bits >> 10;
    const uint32_t fraction = bits & 0x03FFU;
    const double value = exponent == 0
                                 ? std::ldexp(static_cast<double>(fraction), -24)
                                 : std::ldexp(static_cast<double>(0x0400U + fraction),
                                              static_cast<int>(exponent) - 25);
    return static_cast<float>(value);
}

void ExpectNarrowingForBothSigns(float positive, uint16_t expected) {
    for (const uint32_t sign: {0U, 0x80000000U}) {
        const uint32_t bits = std::bit_cast<uint32_t>(positive) | sign;
        const float value = std::bit_cast<float>(bits);
        EXPECT_EQ(Half(value).bits(), expected | (sign >> 16)) << "source bits=" << bits;
    }
}

TEST(HalfNarrowing, CreatesSubnormalResults) {
    ExpectNarrowingForBothSigns(0x1p-24f, 0x0001);
    ExpectNarrowingForBothSigns(0x1p-23f, 0x0002);
    ExpectNarrowingForBothSigns(0x1.ff8p-15f, 0x03FF);
}

TEST(HalfNarrowing, EveryFiniteHalfValueNarrowsExactly) {
    for (uint32_t bits = 0; bits < 0x7C00U; ++bits) {
        const auto expected = static_cast<uint16_t>(bits);
        ExpectNarrowingForBothSigns(PositiveHalfValue(expected), expected);
    }
}

TEST(HalfNarrowing, EveryFiniteMidpointRoundsToNearestEven) {
    for (uint32_t lower = 0; lower < 0x7BFFU; ++lower) {
        const double a = PositiveHalfValue(static_cast<uint16_t>(lower));
        const double b = PositiveHalfValue(static_cast<uint16_t>(lower + 1U));
        // Adjacent binary16 midpoints are exactly representable in binary32.
        const float midpoint = static_cast<float>((a + b) * 0.5);
        const auto even = static_cast<uint16_t>(lower + (lower & 1U));

        ExpectNarrowingForBothSigns(std::nextafter(midpoint, 0.0f), static_cast<uint16_t>(lower));
        ExpectNarrowingForBothSigns(midpoint, even);
        ExpectNarrowingForBothSigns(std::nextafter(midpoint, std::numeric_limits<float>::infinity()),
                                    static_cast<uint16_t>(lower + 1U));
    }
}

TEST(HalfNarrowing, OverflowMidpointRoundsToInfinity) {
    constexpr float midpoint = 65520.0f;
    ExpectNarrowingForBothSigns(std::nextafter(midpoint, 0.0f), 0x7BFF);
    ExpectNarrowingForBothSigns(midpoint, 0x7C00);
    ExpectNarrowingForBothSigns(std::nextafter(midpoint, std::numeric_limits<float>::infinity()), 0x7C00);
}

TEST(HalfNarrowing, NaNsQuietAndPreserveRetainedPayload) {
    for (uint32_t retained = 0; retained < 0x0400U; ++retained) {
        for (const uint32_t tail: {0U, 1U, 0x1FFFU}) {
            const uint32_t fraction = (retained << 13) | tail;
            if (fraction == 0) {
                continue;
            }
            const float value = std::bit_cast<float>(0x7F800000U | fraction);
            ExpectNarrowingForBothSigns(value, static_cast<uint16_t>(0x7E00U | (retained & 0x01FFU)));
        }
    }
}

TEST(HalfNarrowing, ArithmeticPreservesSubnormalResults) {
    const Half normal{0x0400, Half::from_bits()};
    const Half below{0x03FF, Half::from_bits()};
    EXPECT_EQ((normal - below).bits(), 0x0001U);
    EXPECT_EQ((normal * Half(0.5f)).bits(), 0x0200U);

    Half compound = normal;
    compound /= Half(1024.0f);
    EXPECT_EQ(compound.bits(), 0x0001U);
}

struct ConversionCase {
    uint32_t source;
    uint16_t expected;
};

constexpr ConversionCase kEnvironmentCases[] = {
        {0x00000000U, 0x0000}, // zero
        {0x00000001U, 0x0000}, // binary32 subnormal
        {0x33000000U, 0x0000}, // zero/subnormal midpoint
        {0x33000001U, 0x0001}, // immediately above that midpoint
        {0x33800000U, 0x0001}, // smallest binary16 subnormal
        {0x33C00000U, 0x0002}, // tie between first and second subnormals
        {0x387FE000U, 0x0400}, // subnormal/normal midpoint
        {0x3F801000U, 0x3C00}, // normal tie, even lower
        {0x3F803000U, 0x3C02}, // normal tie, even upper
        {0x477FF000U, 0x7C00}, // overflow midpoint
        {0x7F7FFFFFU, 0x7C00}, // largest binary32 finite value
        {0x7F800000U, 0x7C00}, // infinity
        {0x7F800001U, 0x7E00}, // signaling NaN, retained payload is zero
        {0x7FC00000U, 0x7E00}, // quiet NaN
};

class HalfNarrowingEnvironment : public ::testing::Test {
protected:
    void SetUp() override {
        saved_ = std::feholdexcept(&environment_) == 0;
        ASSERT_TRUE(saved_);
    }

    void TearDown() override {
        if (saved_) {
            EXPECT_EQ(std::fesetenv(&environment_), 0);
        }
    }

private:
    std::fenv_t environment_{};
    bool saved_ = false;
};

TEST_F(HalfNarrowingEnvironment, FixedRoundingAcrossAllModes) {
    for (const int mode: {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        ASSERT_EQ(std::fesetround(mode), 0);
        for (const auto& [bits, expected]: kEnvironmentCases) {
            ExpectNarrowingForBothSigns(std::bit_cast<float>(bits), expected);
        }
        EXPECT_EQ(std::fegetround(), mode);
    }
}

TEST_F(HalfNarrowingEnvironment, ConversionDoesNotRaiseFloatingPointExceptions) {
    for (const auto& [bits, expected]: kEnvironmentCases) {
        ExpectNarrowingForBothSigns(std::bit_cast<float>(bits), expected);
    }
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), 0);
}

} // namespace
