#include "aethermind/dtypes/bfloat16.h"
#include "aethermind/dtypes/half.h"

#include <gtest/gtest.h>
#include <type_traits>

namespace {

using aethermind::BFloat16;
using aethermind::Half;

template<typename T>
uint16_t RawBits(const T& value) {
    if constexpr (std::is_same_v<T, Half>) {
        return value.bits();
    } else {
        return value.x;
    }
}

template<typename T>
class ReducedFloatNegation : public ::testing::Test {};

using ReducedFloatTypes = ::testing::Types<Half, BFloat16>;
TYPED_TEST_SUITE(ReducedFloatNegation, ReducedFloatTypes);

TYPED_TEST(ReducedFloatNegation, PreservesSubnormalMagnitude) {
    const TypeParam positive{0x0001, TypeParam::from_bits()};
    const TypeParam negative{0x8001, TypeParam::from_bits()};

    EXPECT_EQ(RawBits(-positive), 0x8001U);
    EXPECT_EQ(RawBits(-negative), 0x0001U);
}

TYPED_TEST(ReducedFloatNegation, FlipsSignedZero) {
    const TypeParam positive{0x0000, TypeParam::from_bits()};
    const TypeParam negative{0x8000, TypeParam::from_bits()};

    EXPECT_EQ(RawBits(-positive), 0x8000U);
    EXPECT_EQ(RawBits(-negative), 0x0000U);
}

TYPED_TEST(ReducedFloatNegation, FlipsInfinitySign) {
    constexpr uint16_t positive_bits = std::is_same_v<TypeParam, Half> ? 0x7c00 : 0x7f80;
    constexpr uint16_t negative_bits = std::is_same_v<TypeParam, Half> ? 0xfc00 : 0xff80;
    const TypeParam positive{positive_bits, TypeParam::from_bits()};
    const TypeParam negative{negative_bits, TypeParam::from_bits()};

    EXPECT_EQ(RawBits(-positive), negative_bits);
    EXPECT_EQ(RawBits(-negative), positive_bits);
}

TYPED_TEST(ReducedFloatNegation, PreservesNaNPayloadAndSignalingState) {
    constexpr uint16_t signaling = std::is_same_v<TypeParam, Half> ? 0x7c15 : 0x7f95;
    constexpr uint16_t quiet = std::is_same_v<TypeParam, Half> ? 0x7e55 : 0x7fd5;
    for (const uint16_t bits: {signaling, quiet}) {
        const auto negative_bits = static_cast<uint16_t>(bits + 0x8000U);
        const TypeParam positive{bits, TypeParam::from_bits()};
        const TypeParam negative{negative_bits, TypeParam::from_bits()};

        EXPECT_EQ(RawBits(-positive), negative_bits);
        EXPECT_EQ(RawBits(-negative), bits);
    }
}

TYPED_TEST(ReducedFloatNegation, PreservesMagnitudeAndIsInvolutiveForEveryBitPattern) {
    static_assert(std::is_same_v<decltype(-TypeParam{}), TypeParam>);
    for (uint32_t bits = 0; bits < 0x10000U; ++bits) {
        const TypeParam source{static_cast<uint16_t>(bits), TypeParam::from_bits()};
        const TypeParam negative = -source;
        const uint16_t negative_bits = RawBits(negative);

        EXPECT_EQ(negative_bits & 0x7fffU, bits & 0x7fffU) << "source bits=" << bits;
        EXPECT_NE(negative_bits & 0x8000U, bits & 0x8000U) << "source bits=" << bits;
        EXPECT_EQ(RawBits(-negative), bits) << "source bits=" << bits;
        EXPECT_EQ(RawBits(source), bits) << "source bits=" << bits;
    }
}

} // namespace
