//
// Created by 赵丹 on 2025/9/9.
//
#include "utils/floating_point_utils.h"

#include <gtest/gtest.h>

using namespace aethermind::details;

namespace {

TEST(FloatingPointUtilsTest, Fp32FromBitsBasic) {
    // Test zero
    EXPECT_EQ(fp32_from_bits(0x00000000), 0.0f);

    // Test one
    EXPECT_EQ(fp32_from_bits(0x3F800000), 1.0f);

    // Test negative zero
    EXPECT_EQ(fp32_from_bits(0x80000000), -0.0f);

    // Test negative one
    EXPECT_EQ(fp32_from_bits(0xBF800000), -1.0f);

    EXPECT_EQ(fp32_from_bits(0xC15A0000), -13.625f);
}

TEST(FloatingPointUtilsTest, Fp32FromBitsSpecialValues) {
    // Test positive infinity
    EXPECT_EQ(fp32_from_bits(0x7F800000), std::numeric_limits<float>::infinity());

    // Test negative infinity
    EXPECT_EQ(fp32_from_bits(0xFF800000), -std::numeric_limits<float>::infinity());

    // Test quiet NaN
    EXPECT_TRUE(std::isnan(fp32_from_bits(0x7FC00000)));

    // Test signaling NaN
    EXPECT_TRUE(std::isnan(fp32_from_bits(0x7F800001)));
}

TEST(FloatingPointUtilsTest, Fp32FromBitsDenormal) {
    // Test smallest denormal
    EXPECT_FLOAT_EQ(fp32_from_bits(0x00000001), std::numeric_limits<float>::denorm_min());

    // Test largest denormal
    EXPECT_FLOAT_EQ(fp32_from_bits(0x007FFFFF), 1.1754942e-38f);
}


TEST(FloatingPointUtilsTest, Fp32FromBitsRandomValues) {
    // Test random positive value
    EXPECT_FLOAT_EQ(fp32_from_bits(0x40490FDB), 3.1415927f);

    // Test random negative value
    EXPECT_FLOAT_EQ(fp32_from_bits(0xC0490FDB), -3.1415927f);

    // Test subnormal value
    EXPECT_FLOAT_EQ(fp32_from_bits(0x00012345), 1.0449e-40f);
}

}// namespace