//
// Created by 赵丹 on 2025/9/17.
//
#include "utils/float8_e5m2.h"

#include <bitset>
#include <cmath>
#include <gtest/gtest.h>

using namespace aethermind;
using namespace aethermind::details;

namespace {

TEST(Float8E5M2Test, SpecialValues) {
    // Test zero values
    EXPECT_EQ(fp8e5m2_from_fp32_value(0.0f), 0x00);
    EXPECT_EQ(fp8e5m2_from_fp32_value(-0.0f), 0x80);
    EXPECT_EQ(fp8e5m2_to_fp32_value(0x00), 0.0f);
    EXPECT_EQ(fp8e5m2_to_fp32_value(0x80), -0.0f);

    // Test infinity
    EXPECT_EQ(fp8e5m2_from_fp32_value(INFINITY), 0x7C);
    EXPECT_EQ(fp8e5m2_from_fp32_value(-INFINITY), 0xFC);
    EXPECT_TRUE(isinf(fp8e5m2_to_fp32_value(0x7C)));
    EXPECT_TRUE(isinf(fp8e5m2_to_fp32_value(0xFC)));

    // Test NaN
    EXPECT_EQ(fp8e5m2_from_fp32_value(NAN), 0x7E);// Canonical NaN
    EXPECT_TRUE(isnan(fp8e5m2_to_fp32_value(0x7F)));
}

TEST(Float8E5M2Test, NormalRange) {
    // Test positive normal numbers
    EXPECT_EQ(fp8e5m2_from_fp32_value(1.0f), 0x3C);
    EXPECT_FLOAT_EQ(fp8e5m2_to_fp32_value(0x3C), 1.0f);

    EXPECT_EQ(fp8e5m2_from_fp32_value(2.0f), 0x40);
    EXPECT_FLOAT_EQ(fp8e5m2_to_fp32_value(0x40), 2.0f);

    // Test negative normal numbers
    EXPECT_EQ(fp8e5m2_from_fp32_value(-1.0f), 0xBC);
    EXPECT_FLOAT_EQ(fp8e5m2_to_fp32_value(0xBC), -1.0f);

    // Test subnormal numbers
    // EXPECT_EQ(fp8e5m2_from_fp32_value(1.0f / 8388608), 0x00);// smallest positive subnormal
    // EXPECT_FLOAT_EQ(fp8e5m2_to_fp32_value(0x01), 1.0f / 8388608);

    // Test rounding
    EXPECT_EQ(fp8e5m2_from_fp32_value(1.1f), 0x3C);// should round to 1.0 or 1.25
}

TEST(Float8E5M2Test, OverflowAndUnderflow) {
    // Test overflow (clamp to infinity)
    EXPECT_EQ(fp8e5m2_from_fp32_value(65504.0f), 0x7C); // max normal fp8e5m2
    EXPECT_EQ(fp8e5m2_from_fp32_value(100000.0f), 0x7C);// should clamp to infinity

    // Test underflow (flush to zero)
    EXPECT_EQ(fp8e5m2_from_fp32_value(1.0e-10f), 0x00);// should flush to zero
}

TEST(Float8E5M2Test, DenormalHandling) {
    // Test denormal inputs
    EXPECT_EQ(fp8e5m2_from_fp32_value(1.0f / 65536), 0x01); // should be represented as normal
    EXPECT_EQ(fp8e5m2_from_fp32_value(1.0f / 131072), 0x00);// should flush to zero
}

TEST(Float8E5M2Test, RoundingModes) {
    // Test rounding to nearest even
    EXPECT_EQ(fp8e5m2_from_fp32_value(1.5f), 0x3E);  // 1.5 should round to 1.5
    EXPECT_EQ(fp8e5m2_from_fp32_value(1.25f), 0x3D); // 1.25 should round to 1.25
    EXPECT_EQ(fp8e5m2_from_fp32_value(1.375f), 0x3E);// 1.375 should round to 1.25 (nearest even)
}

}// namespace