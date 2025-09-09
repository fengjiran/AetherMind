//
// Created by 赵丹 on 2025/9/9.
//
#include "utils/half.h"

#include <gtest/gtest.h>

using namespace aethermind::details;

namespace {

TEST(HalfToFP32Test, Zero) {
    // 测试正零
    uint16_t pos_zero = 0x0000;
    EXPECT_EQ(0.0f, half_to_fp32_value(pos_zero));

    // 测试负零
    uint16_t neg_zero = 0x8000;
    EXPECT_EQ(-0.0f, half_to_fp32_value(neg_zero));
}

TEST(HalfToFP32Test, Denormalized) {
    // 测试最小非规约数
    uint16_t min_denorm = 0x0001;
    EXPECT_FLOAT_EQ(5.96046448e-8f, half_to_fp32_value(min_denorm));

    // 测试最大非规约数
    uint16_t max_denorm = 0x03FF;
    EXPECT_FLOAT_EQ(6.09755516e-5f, half_to_fp32_value(max_denorm));
}

TEST(HalfToFP32Test, Normalized) {
    // 测试最小规约数
    uint16_t min_norm = 0x0400;
    EXPECT_FLOAT_EQ(6.10351562e-5f, half_to_fp32_value(min_norm));

    // 测试1.0
    uint16_t one = 0x3C00;
    EXPECT_FLOAT_EQ(1.0f, half_to_fp32_value(one));

    // 测试-1.0
    uint16_t neg_one = 0xBC00;
    EXPECT_FLOAT_EQ(-1.0f, half_to_fp32_value(neg_one));

    // 测试最大规约数
    uint16_t max_norm = 0x7BFF;
    EXPECT_FLOAT_EQ(65504.0f, half_to_fp32_value(max_norm));
}

TEST(HalfToFP32Test, Infinity) {
    // 测试正无穷
    uint16_t pos_inf = 0x7C00;
    EXPECT_TRUE(std::isinf(half_to_fp32_value(pos_inf)));
    EXPECT_GT(half_to_fp32_value(pos_inf), 0);

    // 测试负无穷
    uint16_t neg_inf = 0xFC00;
    EXPECT_TRUE(std::isinf(half_to_fp32_value(neg_inf)));
    EXPECT_LT(half_to_fp32_value(neg_inf), 0);
}

TEST(HalfToFP32Test, NaN) {
    // 测试NaN
    uint16_t nan = 0x7FFF;
    EXPECT_TRUE(std::isnan(half_to_fp32_value(nan)));

    // 测试带符号的NaN
    uint16_t neg_nan = 0xFFFF;
    EXPECT_TRUE(std::isnan(half_to_fp32_value(neg_nan)));
}

TEST(HalfToFP32Test, RandomValues) {
    // 测试随机值
    uint16_t random1 = 0x3555; // ~0.33325
    EXPECT_FLOAT_EQ(0.333251953125f, half_to_fp32_value(random1));

    uint16_t random2 = 0x4D12; // ~1234.5
    EXPECT_FLOAT_EQ(1234.5f, half_to_fp32_value(random2));
}

}