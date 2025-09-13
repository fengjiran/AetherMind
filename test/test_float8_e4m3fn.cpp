//
// Created by 赵丹 on 2025/9/12.
//
#include "utils/float8_e4m3fn.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace aethermind;
using namespace aethermind::details;

namespace {

TEST(Float8E4M3FNToFloat32Test, ZeroValue) {
    EXPECT_EQ(0.0f, fp8e4m3fn_to_fp32_value(0x00));
}

TEST(Float8E4M3FNToFloat32Test, OneValue) {
    EXPECT_EQ(1.0f, fp8e4m3fn_to_fp32_value(0x38));
}

TEST(Float8E4M3FNToFloat32Test, NegativeOneValue) {
    EXPECT_EQ(-1.0f, fp8e4m3fn_to_fp32_value(0xB8));
}

TEST(Float8E4M3FNToFloat32Test, MaxPositiveValue) {
    EXPECT_FLOAT_EQ(448.0f, fp8e4m3fn_to_fp32_value(0x7E));
}

TEST(Float8E4M3FNToFloat32Test, MinPositiveNormalValue) {
    EXPECT_FLOAT_EQ(0.0625f, fp8e4m3fn_to_fp32_value(0x18));
}

TEST(Float8E4M3FNToFloat32Test, MinPositiveSubnormalValue) {
    EXPECT_FLOAT_EQ(0.001953125f, fp8e4m3fn_to_fp32_value(0x01));
}

TEST(Float8E4M3FNToFloat32Test, MaxNegativeValue) {
    EXPECT_FLOAT_EQ(-448.0f, fp8e4m3fn_to_fp32_value(0xFE));
}

TEST(Float8E4M3FNToFloat32Test, NaNValue) {
    EXPECT_TRUE(std::isnan(fp8e4m3fn_to_fp32_value(0x7F)));
}

TEST(Float8E4M3FNToFloat32Test, RandomPositiveValue) {
    EXPECT_FLOAT_EQ(1.5f, fp8e4m3fn_to_fp32_value(0x3C));
}

TEST(Float8E4M3FNToFloat32Test, RandomNegativeValue) {
    EXPECT_FLOAT_EQ(-1.5f, fp8e4m3fn_to_fp32_value(0xBC));
}

TEST(Float8E4M3FNToFloat32Test, DenormalizedValues) {
    // Smallest denormalized value
    EXPECT_FLOAT_EQ(0.001953125f, fp8e4m3fn_to_fp32_value(0x01));

    // Largest denormalized value
    EXPECT_NEAR(0.013671875f, fp8e4m3fn_to_fp32_value(0x07), 0.0001f);
}

TEST(Float8E4M3FNToFloat32Test, NormalizedValues) {
    // Smallest normalized value
    EXPECT_FLOAT_EQ(0.25f, fp8e4m3fn_to_fp32_value(0x28));

    // Value with exponent bias 0
    EXPECT_NEAR(1.0f, fp8e4m3fn_to_fp32_value(0x38), 0.0001f);

    // Largest normalized value
    EXPECT_NEAR(448.0f, fp8e4m3fn_to_fp32_value(0x7E), 0.1f);
}

TEST(Float8E4M3FNToFloat32Test, RoundingCases) {
    // Test values that might trigger rounding
    EXPECT_NEAR(0.5f, fp8e4m3fn_to_fp32_value(0x30), 0.0001f);
    EXPECT_NEAR(2.0f, fp8e4m3fn_to_fp32_value(0x40), 0.0001f);
    EXPECT_NEAR(3.5f, fp8e4m3fn_to_fp32_value(0x46), 0.0001f);
}

TEST(FP8E4M3FNFromFP32Test, ZeroValues) {
    // 正零
    EXPECT_EQ(fp8e4m3fn_from_fp32_value(0.0f), 0x00);

    // 负零
    EXPECT_EQ(fp8e4m3fn_from_fp32_value(-0.0f), 0x80);
}

TEST(FP8E4M3FNFromFP32Test, InfinityAndNaN) {
    // 正无穷 -> 转换为最大规格化数
    EXPECT_EQ(fp8e4m3fn_from_fp32_value(std::numeric_limits<float>::infinity()), 0x7E);

    // 负无穷 -> 转换为负的最大规格化数
    EXPECT_EQ(fp8e4m3fn_from_fp32_value(-std::numeric_limits<float>::infinity()), 0xFE);

    // NaN -> 转换为NaN表示
    float quiet_nan = std::numeric_limits<float>::quiet_NaN();
    uint8_t result = fp8e4m3fn_from_fp32_value(quiet_nan);
    EXPECT_EQ(result, 0x7F);// 正NaN

    float signaling_nan = std::numeric_limits<float>::signaling_NaN();
    result = fp8e4m3fn_from_fp32_value(signaling_nan);
    EXPECT_EQ(result, 0x7F);// 正NaN

    // 负NaN
    result = fp8e4m3fn_from_fp32_value(-quiet_nan);
    EXPECT_EQ(result, 0xFF);// 负NaN
}

TEST(FP8E4M3FNFromFP32Test, NormalizedNumbers) {
    // 1.0
    EXPECT_EQ(fp8e4m3fn_from_fp32_value(1.0f), 0x38);// 符号0, 指数7+0=7, 尾数0

    // 2.0
    EXPECT_EQ(fp8e4m3fn_from_fp32_value(2.0f), 0x40);// 指数7+1=8

    // 0.5
    EXPECT_EQ(fp8e4m3fn_from_fp32_value(0.5f), 0x30);// 指数7-1=6

    // -1.0
    EXPECT_EQ(fp8e4m3fn_from_fp32_value(-1.0f), 0xB8);// 符号1, 指数7+0=7, 尾数0

    // 最小正规格化数 (2^-6 = 0.015625)
    EXPECT_EQ(fp8e4m3fn_from_fp32_value(0.015625f), 0x08);

    // 最大规格化数 (240.0)
    EXPECT_EQ(fp8e4m3fn_from_fp32_value(240.0f), 0x77);
}

TEST(FP8E4M3FNFromFP32Test, DenormalizedNumbers) {
    // 小于最小规格化数的值会下溢到零
    float tiny_positive = 1e-10f;
    EXPECT_EQ(fp8e4m3fn_from_fp32_value(tiny_positive), 0x00);

    float tiny_negative = -1e-10f;
    EXPECT_EQ(fp8e4m3fn_from_fp32_value(tiny_negative), 0x80);
}

TEST(FP8E4M3FNFromFP32Test, Overflow) {
    // 超过最大规格化数的值会溢出到最大规格化数
    float overflow = 300.0f;
    EXPECT_EQ(fp8e4m3fn_from_fp32_value(overflow), 0x7E);

    float negative_overflow = -300.0f;
    EXPECT_EQ(fp8e4m3fn_from_fp32_value(negative_overflow), 0xFE);
}

TEST(FP8E4M3FNFromFP32Test, Rounding) {
    // 测试舍入到最近偶数
    // 值刚好在中间时，向偶数舍入
    float value1 = 1.125f;// 二进制: 1.001000...
    EXPECT_EQ(fp8e4m3fn_from_fp32_value(value1), 0x39);

    // 超过中间值，向上舍入
    float value2 = 1.126f;
    EXPECT_EQ(fp8e4m3fn_from_fp32_value(value2), 0x39);
}

TEST(FP8E4M3FNFromFP32Test, SpecialValues) {
    // PI
    EXPECT_EQ(fp8e4m3fn_from_fp32_value(3.141592653589793f), 0x45);

    // E
    EXPECT_EQ(fp8e4m3fn_from_fp32_value(2.718281828459045f), 0x43);

    // 常见机器学习值
    EXPECT_EQ(fp8e4m3fn_from_fp32_value(0.1f), 0x1D);
    EXPECT_EQ(fp8e4m3fn_from_fp32_value(0.01f), 0x00);
    EXPECT_EQ(fp8e4m3fn_from_fp32_value(10.0f), 0x4C);
    EXPECT_EQ(fp8e4m3fn_from_fp32_value(100.0f), 0x5E);
}

}// namespace