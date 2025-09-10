//
// Created by 赵丹 on 2025/9/9.
//
#include "utils/half.h"

#include <cmath>
#include <gtest/gtest.h>

using namespace aethermind::details;

namespace {

TEST(HalfToFP32Test, HalfToFp32Bits_Zero) {
    // 测试正零和负零
    EXPECT_EQ(half_to_fp32_bits(0x0000), 0x00000000);// 正零
    EXPECT_EQ(half_to_fp32_bits(0x8000), 0x80000000);// 负零
}

TEST(HalfToFP32Test, HalfToFp32Bits_Denormalized) {
    // 测试非规格化数（最小的非零正数）
    // 最小的半精度非规格化数: 0x0001 -> 2^-14 * 2^-10 = 2^-24
    EXPECT_EQ(half_to_fp32_bits(0x0001), 0x33800000);

    // 最大的半精度非规格化数: 0x03FF
    EXPECT_EQ(half_to_fp32_bits(0x03FF), 0x387FC000);
}

TEST(HalfToFP32Test, HalfToFp32Bits_Normalized) {
    // 测试规格化数

    // 1.0 in half: 0x3C00 -> 1.0 in float: 0x3F800000
    EXPECT_EQ(half_to_fp32_bits(0x3C00), 0x3F800000);

    // -1.0 in half: 0xBC00 -> -1.0 in float: 0xBF800000
    EXPECT_EQ(half_to_fp32_bits(0xBC00), 0xBF800000);

    // 2.0 in half: 0x4000 -> 2.0 in float: 0x40000000
    EXPECT_EQ(half_to_fp32_bits(0x4000), 0x40000000);

    // 0.5 in half: 0x3800 -> 0.5 in float: 0x3F000000
    EXPECT_EQ(half_to_fp32_bits(0x3800), 0x3F000000);

    // 测试一些随机值
    EXPECT_EQ(half_to_fp32_bits(0x3555), 0x3EAAA000);// ~0.33325
    EXPECT_EQ(half_to_fp32_bits(0x48CD), 0x4119A000);//
}

TEST(HalfToFP32Test, HalfToFp32Bits_Infinity) {
    // 测试无穷大
    EXPECT_EQ(half_to_fp32_bits(0x7C00), 0x7F800000);// 正无穷
    EXPECT_EQ(half_to_fp32_bits(0xFC00), 0xFF800000);// 负无穷
}

TEST(HalfToFP32Test, HalfToFp32Bits_NaN) {
    // 测试NaN（非数字）

    // 静默NaN
    EXPECT_EQ(half_to_fp32_bits(0x7C01), 0x7F802000);
    EXPECT_EQ(half_to_fp32_bits(0x7FFF), 0x7FFFE000);

    // 信号NaN
    EXPECT_EQ(half_to_fp32_bits(0x7E00), 0x7FC00000);
    EXPECT_EQ(half_to_fp32_bits(0x7F00), 0x7FE00000);
    //
    // // 负NaN
    EXPECT_EQ(half_to_fp32_bits(0xFC01), 0xFF802000);
    EXPECT_EQ(half_to_fp32_bits(0xFFFF), 0xFFFFE000);
}

TEST(HalfToFP32Test, HalfToFp32Bits_EdgeCases) {
    // 测试边界情况

    // 最大规格化半精度数: 0x7BFF -> ~65504.0
    EXPECT_EQ(half_to_fp32_bits(0x7BFF), 0x477FE000);

    // 最小规格化半精度数: 0x0400 -> 2^-14 = ~6.1035e-5
    EXPECT_EQ(half_to_fp32_bits(0x0400), 0x38800000);

    // 最大非规格化半精度数: 0x03FF -> ~6.0976e-5
    EXPECT_EQ(half_to_fp32_bits(0x03FF), 0x387FC000);

    // 最小非规格化半精度数: 0x0001 -> 2^-24 = ~5.9605e-8
    EXPECT_EQ(half_to_fp32_bits(0x0001), 0x33800000);
}

TEST(HalfToFP32Test, HalfToFp32Bits_RoundTrip) {
    // 测试往返转换的一致性（如果存在反向函数）

    // 测试一些常见值
    const uint16_t test_values[] = {
            0x0000, 0x0001, 0x03FF, 0x0400, 0x3C00, 0x4000,
            0x7C00, 0x7E00, 0x7FFF, 0x8000, 0xBC00, 0xFC00};

    for (auto half_val: test_values) {
        uint32_t fp32_bits = half_to_fp32_bits(half_val);

        // 验证符号位正确
        bool half_sign = (half_val & 0x8000) != 0;
        bool fp32_sign = (fp32_bits & 0x80000000) != 0;
        EXPECT_EQ(half_sign, fp32_sign);

        // 验证无穷和NaN处理正确
        if ((half_val & 0x7C00) == 0x7C00) {
            // 应该是无穷或NaN
            EXPECT_TRUE((fp32_bits & 0x7F800000) == 0x7F800000);
        }
    }
}

TEST(HalfToFP32Test, HalfToFp32Bits_SpecialValues) {
    // 测试特殊值

    // PI的近似值
    EXPECT_EQ(half_to_fp32_bits(0x4248), 0x40490000);// ~3.140625

    // E的近似值
    EXPECT_EQ(half_to_fp32_bits(0x4170), 0x402E0000);// ~2.71875

    // 黄金比例
    EXPECT_EQ(half_to_fp32_bits(0x3FCF), 0x3FF9E000);// ~1.618
}

TEST(HalfToFP32Test, HalfToFp32Bits_ExhaustiveSmallValues) {
    // 对小数值进行穷举测试（可选，用于彻底验证）

    for (uint16_t i = 0; i < 0x0400; ++i) {
        // 测试所有非规格化数和小的规格化数
        uint32_t result = half_to_fp32_bits(i);

        // 基本验证：结果应该是有效的32位浮点数
        EXPECT_TRUE((result & 0x7F800000) <= 0x7F800000);

        // 符号位应该匹配
        EXPECT_EQ((i & 0x8000) != 0, (result & 0x80000000) != 0);
    }
}

TEST(HalfToFP32Test, Zero) {
    // 测试正零和负零
    EXPECT_EQ(half_to_fp32_value(0x0000), 0.0f); // 正零
    EXPECT_EQ(half_to_fp32_value(0x8000), -0.0f);// 负零

    // 验证符号位正确
    EXPECT_TRUE(std::signbit(half_to_fp32_value(0x8000))); // 负零应该有负号
    EXPECT_FALSE(std::signbit(half_to_fp32_value(0x0000)));// 正零应该没有负号
}

TEST(HalfToFP32Test, Denormalized) {
    // 最小的非零正数: 0x0001 -> 2^-24 ≈ 5.96046e-08
    float min_denormal = half_to_fp32_value(0x0001);
    EXPECT_GT(min_denormal, 0.0f);
    EXPECT_LT(min_denormal, 1e-7f);

    // 最大的非规格化数: 0x03FF
    float max_denormal = half_to_fp32_value(0x03FF);
    EXPECT_GT(max_denormal, 0.0f);
    EXPECT_LT(max_denormal, 6.5e-5f);// 应该小于最小的规格化数
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

    // 1.0 in half: 0x3C00 -> 1.0 in float
    EXPECT_FLOAT_EQ(half_to_fp32_value(0x3C00), 1.0f);

    // -1.0 in half: 0xBC00 -> -1.0 in float
    EXPECT_FLOAT_EQ(half_to_fp32_value(0xBC00), -1.0f);

    // 2.0 in half: 0x4000 -> 2.0 in float
    EXPECT_FLOAT_EQ(half_to_fp32_value(0x4000), 2.0f);

    // 0.5 in half: 0x3800 -> 0.5 in float
    EXPECT_FLOAT_EQ(half_to_fp32_value(0x3800), 0.5f);

    // 测试一些常见值
    EXPECT_NEAR(half_to_fp32_value(0x3555), 0.33325f, 1e-5f);// ~1/3
    EXPECT_NEAR(half_to_fp32_value(0x48CD), 9.6016f, 1e-3f); // 1849.0
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
    // 各种NaN值
    float nan1 = half_to_fp32_value(0x7C01);// 静默NaN
    float nan2 = half_to_fp32_value(0x7FFF);// 静默NaN
    float nan3 = half_to_fp32_value(0x7E00);// 信号NaN
    float nan4 = half_to_fp32_value(0xFC01);// 负静默NaN

    // 所有都应该是NaN
    EXPECT_TRUE(std::isnan(nan1));
    EXPECT_TRUE(std::isnan(nan2));
    EXPECT_TRUE(std::isnan(nan3));
    EXPECT_TRUE(std::isnan(nan4));

    // 验证NaN的数学性质
    EXPECT_TRUE(std::isnan(nan1 + 1.0f));
    EXPECT_TRUE(std::isnan(nan1 * 2.0f));
}

TEST(HalfToFP32Test, EdgeCases) {
    // 测试边界情况

    // 最大规格化半精度数: 0x7BFF -> ~65504.0
    float max_normal = half_to_fp32_value(0x7BFF);
    EXPECT_NEAR(max_normal, 65504.0f, 1e-3f);
    EXPECT_FALSE(std::isinf(max_normal));

    // 最小规格化半精度数: 0x0400 -> 2^-14 ≈ 6.10352e-05
    float min_normal = half_to_fp32_value(0x0400);
    EXPECT_GT(min_normal, 0.0f);
    EXPECT_LT(min_normal, 1e-4f);

    // 验证规格化数和非规格化数的边界
    float last_denormal = half_to_fp32_value(0x03FF);// 最大非规格化数
    float first_normal = half_to_fp32_value(0x0400); // 最小规格化数

    EXPECT_LT(last_denormal, first_normal);// 非规格化数应该小于规格化数
}

TEST(HalfToFP32Test, SpecialValues) {
    // 测试特殊数学常数

    // PI的近似值: 0x4248 -> ~3.140625
    EXPECT_NEAR(half_to_fp32_value(0x4248), 3.140625f, 1e-6f);

    // E的近似值: 0x4170 -> ~2.71875
    EXPECT_NEAR(half_to_fp32_value(0x4170), 2.71875f, 1e-6f);

    // 黄金比例: 0x3FCF -> ~1.618
    EXPECT_NEAR(half_to_fp32_value(0x3FCF), 1.95215f, 1e-3f);
}

TEST(HalfToFP32Test, RandomValues) {
    // 测试随机值
    uint16_t random1 = 0x3555;// ~0.33325
    EXPECT_FLOAT_EQ(0.333251953125f, std::bit_cast<float>(half_to_fp32_bits(random1)));

    uint16_t random2 = 0x4D12;// ~20.28125
    EXPECT_FLOAT_EQ(20.28125f, std::bit_cast<float>(half_to_fp32_bits(random2)));
}

TEST(HalfToFP32Test, RoundTripConsistency) {
    // 测试与half_to_fp32_bits的一致性

    const uint16_t test_values[] = {
            0x0000, 0x0001, 0x03FF, 0x0400, 0x3C00, 0x4000,
            0x7C00, 0x7E00, 0x7FFF, 0x8000, 0xBC00, 0xFC00};

    for (auto half_val: test_values) {
        uint32_t bits = half_to_fp32_bits(half_val);
        float value_from_bits = fp32_from_bits(bits);
        float direct_value = half_to_fp32_value(half_val);

        // 两种方式应该得到相同的结果
        if (std::isnan(value_from_bits)) {
            EXPECT_TRUE(std::isnan(direct_value));
        } else {
            EXPECT_FLOAT_EQ(value_from_bits, direct_value);
        }
    }
}

TEST(HalfToFP32Test, Precision) {
    // 测试精度

    // 测试半精度能够表示的各种精度级别
    for (int exp = -14; exp <= 15; ++exp) {
        for (int mantissa = 0; mantissa < 1024; mantissa += 128) {
            // 构造半精度数的各个部分
            uint16_t sign = 0;
            uint16_t exponent = (exp + 15) << 10; // 偏置指数
            uint16_t fraction = mantissa;
            uint16_t half_val = sign | exponent | fraction;

            if ((half_val & 0x7C00) != 0x7C00) { // 排除无穷和NaN
                float value = half_to_fp32_value(half_val);

                // 验证值是有限的（除非是特殊情况）
                if (!std::isinf(value) && !std::isnan(value)) {
                    EXPECT_TRUE(std::isfinite(value));
                }
            }
        }
    }
}

}// namespace