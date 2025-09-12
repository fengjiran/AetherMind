//
// Created by 赵丹 on 2025/9/12.
//
#include "utils/float8_e4m3fn.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace aethermind;
using namespace aethermind::details;

namespace {

TEST(Float8E4M3FNTest, ZeroValue) {
    EXPECT_EQ(0.0f, fp8e4m3fn_to_fp32_value(0x00));
}

TEST(Float8E4M3FNTest, OneValue) {
    EXPECT_EQ(1.0f, fp8e4m3fn_to_fp32_value(0x38));
}

TEST(Float8E4M3FNTest, NegativeOneValue) {
    EXPECT_EQ(-1.0f, fp8e4m3fn_to_fp32_value(0xB8));
}

TEST(Float8E4M3FNTest, MaxPositiveValue) {
    EXPECT_FLOAT_EQ(448.0f, fp8e4m3fn_to_fp32_value(0x7E));
}

TEST(Float8E4M3FNTest, MinPositiveNormalValue) {
    EXPECT_FLOAT_EQ(0.0625f, fp8e4m3fn_to_fp32_value(0x08));
}

TEST(Float8E4M3FNTest, MinPositiveSubnormalValue) {
    EXPECT_FLOAT_EQ(0.0009765625f, fp8e4m3fn_to_fp32_value(0x01));
}

TEST(Float8E4M3FNTest, MaxNegativeValue) {
    EXPECT_FLOAT_EQ(-448.0f, fp8e4m3fn_to_fp32_value(0xFF));
}

TEST(Float8E4M3FNTest, NaNValue) {
    EXPECT_TRUE(std::isnan(fp8e4m3fn_to_fp32_value(0x80)));
}

TEST(Float8E4M3FNTest, RandomPositiveValue) {
    EXPECT_FLOAT_EQ(1.5f, fp8e4m3fn_to_fp32_value(0x3E));
}

TEST(Float8E4M3FNTest, RandomNegativeValue) {
    EXPECT_FLOAT_EQ(-1.5f, fp8e4m3fn_to_fp32_value(0xBE));
}

TEST(Fp8E4M3FnToFp32Test, ZeroValue) {
    EXPECT_EQ(0.0f, fp8e4m3fn_to_fp32_value(0x00));
}

TEST(Float8E4M3FNTest, DenormalizedValues) {
    // Smallest denormalized value
    EXPECT_FLOAT_EQ(0.0f, fp8e4m3fn_to_fp32_value(0x01));

    // Largest denormalized value
    EXPECT_NEAR(0.21875f, fp8e4m3fn_to_fp32_value(0x07), 0.0001f);
}

TEST(Float8E4M3FNTest, NormalizedValues) {
    // Smallest normalized value
    EXPECT_FLOAT_EQ(0.25f, fp8e4m3fn_to_fp32_value(0x08));

    // Value with exponent bias 0
    EXPECT_NEAR(1.0f, fp8e4m3fn_to_fp32_value(0x20), 0.0001f);

    // Largest normalized value
    EXPECT_NEAR(448.0f, fp8e4m3fn_to_fp32_value(0x7F), 0.1f);
}

TEST(Float8E4M3FNTest, SpecialValues) {
    // Positive infinity (if supported)
    // EXPECT_TRUE(std::isinf(fp8e4m3fn_to_fp32_value(0x7F)));

    // NaN (if supported)
    // EXPECT_TRUE(std::isnan(fp8e4m3fn_to_fp32_value(0x80)));

    // Sign bit cases (if supported)
    // EXPECT_LT(fp8e4m3fn_to_fp32_value(0x80), 0.0f);
}

TEST(Float8E4M3FNTest, RoundingCases) {
    // Test values that might trigger rounding
    EXPECT_NEAR(0.5f, fp8e4m3fn_to_fp32_value(0x10), 0.0001f);
    EXPECT_NEAR(2.0f, fp8e4m3fn_to_fp32_value(0x28), 0.0001f);
    EXPECT_NEAR(3.5f, fp8e4m3fn_to_fp32_value(0x36), 0.0001f);
}

}// namespace