//
// Created by richard on 9/23/25.
//
#include "utils/bfloat16.h"

#include <bitset>
#include <cmath>
#include <gtest/gtest.h>

using namespace aethermind;
using namespace aethermind::details;

namespace {

TEST(BFloat16Test, DefaultConstructor) {
    // 测试默认构造函数
    BFloat16 val;
    EXPECT_EQ(val.x, 0);
    EXPECT_FLOAT_EQ(val, 0.0f);
}

TEST(BFloat16Test, FromBitsConstructor) {
    // 测试通过位构造函数
    BFloat16 val1(0x3F80, BFloat16::from_bits());// 1.0
    EXPECT_EQ(val1.x, 0x3F80);
    EXPECT_FLOAT_EQ(val1, 1.0f);

    BFloat16 val2(0xBF80, BFloat16::from_bits());// -1.0
    EXPECT_EQ(val2.x, 0xBF80);
    EXPECT_FLOAT_EQ(val2, -1.0f);
}

TEST(BFloat16Test, FloatConversion) {
    // 测试从float转换和转换为float
    constexpr float test_values[] = {
            0.0f, 1.0f, -1.0f, 2.0f, 0.5f,
            0.3333333333f, 100.0f, -100.0f,
            std::numeric_limits<float>::max() / 2,// 大值但不会溢出BFloat16
            std::numeric_limits<float>::min() * 2 // 小值但不会下溢BFloat16
    };

    for (float f: test_values) {
        BFloat16 bf16 = f;
        float converted_back = bf16;

        // 由于精度损失，我们不能期望完全相等，但应该在合理的误差范围内
        // 对于BFloat16，相对误差通常应小于0.5%
        if (std::isnan(f)) {
            EXPECT_TRUE(std::isnan(converted_back));
        } else if (std::isinf(f)) {
            EXPECT_TRUE(std::isinf(converted_back));
            EXPECT_EQ(std::signbit(f), std::signbit(converted_back));
        } else if (f == 0.0f) {
            EXPECT_EQ(converted_back, 0.0f);
        } else {
            float relative_error = std::abs((converted_back - f) / f);
            EXPECT_LE(relative_error, 0.01f) << "Original: " << f << ", Converted: " << converted_back;
        }
    }
}

TEST(BFloat16Test, EdgeValues) {
    // 测试边界值

    // 零值
    BFloat16 zero_pos(0.0f);
    BFloat16 zero_neg(-0.0f);
    EXPECT_EQ(zero_pos.x, 0x0000);
    EXPECT_EQ(zero_neg.x, 0x8000);
    EXPECT_FLOAT_EQ(zero_pos, 0.0f);
    EXPECT_FLOAT_EQ(zero_neg, -0.0f);

    // 无穷大
    BFloat16 inf_pos(std::numeric_limits<float>::infinity());
    BFloat16 inf_neg(-std::numeric_limits<float>::infinity());
    EXPECT_EQ(inf_pos.x, 0x7F80);
    EXPECT_EQ(inf_neg.x, 0xFF80);
    EXPECT_TRUE(std::isinf(inf_pos));
    EXPECT_TRUE(std::isinf(inf_neg));

    // NaN
    BFloat16 nan_pos(std::numeric_limits<float>::quiet_NaN());
    BFloat16 nan_neg(-std::numeric_limits<float>::quiet_NaN());
    EXPECT_TRUE(std::isnan(nan_pos));
    EXPECT_TRUE(std::isnan(nan_neg));

    // 最小和最大规格化值
    BFloat16 min_norm = std::numeric_limits<BFloat16>::min();
    BFloat16 max_norm = std::numeric_limits<BFloat16>::max();
    BFloat16 lowest = std::numeric_limits<BFloat16>::lowest();

    EXPECT_EQ(min_norm.x, 0x0080);
    EXPECT_EQ(max_norm.x, 0x7F7F);
    EXPECT_EQ(lowest.x, 0xFF7F);
}

TEST(BFloat16Test, ArithmeticOperations) {
    // 测试基本算术运算
    BFloat16 a(2.0f);
    BFloat16 b(3.0f);

    // 加法
    EXPECT_FLOAT_EQ(a + b, 5.0f);

    // 减法
    EXPECT_FLOAT_EQ(a - b, -1.0f);
    EXPECT_FLOAT_EQ(b - a, 1.0f);

    // 乘法
    EXPECT_FLOAT_EQ(a * b, 6.0f);

    // 除法
    EXPECT_FLOAT_EQ(a / b, 0.66796875f);
    EXPECT_FLOAT_EQ(b / a, 1.5f);

    // 一元负号
    EXPECT_FLOAT_EQ(-a, -2.0f);

    // 复合赋值
    BFloat16 c = a;
    c += b;
    EXPECT_FLOAT_EQ(c, 5.0f);

    c = a;
    c -= b;
    EXPECT_FLOAT_EQ(c, -1.0f);

    c = a;
    c *= b;
    EXPECT_FLOAT_EQ(c, 6.0f);

    c = a;
    c /= b;
    EXPECT_FLOAT_EQ(c, 0.66796875f);
}

TEST(BFloat16Test, MixedTypeOperations) {
    // 测试与其他类型的混合运算
    BFloat16 a(2.0f);

    // 与float运算
    EXPECT_FLOAT_EQ(a + 3.0f, 5.0f);
    EXPECT_FLOAT_EQ(3.0f + a, 5.0f);
    EXPECT_FLOAT_EQ(a - 3.0f, -1.0f);
    EXPECT_FLOAT_EQ(3.0f - a, 1.0f);
    EXPECT_FLOAT_EQ(a * 3.0f, 6.0f);
    EXPECT_FLOAT_EQ(3.0f * a, 6.0f);
    EXPECT_FLOAT_EQ(a / 3.0f, 2.0f / 3.0f);
    EXPECT_FLOAT_EQ(3.0f / a, 1.5f);

    // 与int运算
    BFloat16 result = a + 3;
    EXPECT_FLOAT_EQ(result, 5.0f);

    result = 3 + a;
    EXPECT_FLOAT_EQ(result, 5.0f);

    result = a - 3;
    EXPECT_FLOAT_EQ(result, -1.0f);

    result = 3 - a;
    EXPECT_FLOAT_EQ(result, 1.0f);

    // 与double运算
    double d = 3.14159;
    EXPECT_DOUBLE_EQ(a + d, 5.14159);
    EXPECT_DOUBLE_EQ(d + a, 5.14159);
}

TEST(BFloat16Test, ComparisonOperators) {
    // 测试比较运算符
    BFloat16 a(2.0f);
    BFloat16 b(3.0f);
    BFloat16 c(2.0f);

    EXPECT_FALSE(a > b);
    EXPECT_TRUE(b > a);
    EXPECT_FALSE(a > c);

    EXPECT_TRUE(a < b);
    EXPECT_FALSE(b < a);
    EXPECT_FALSE(a < c);

    // 使用隐式转换测试相等性
    EXPECT_TRUE(a == c);
    EXPECT_FALSE(a == b);
}

TEST(BFloat16Test, BitwiseOperations) {
    // 测试位运算
    BFloat16 a(2.0f);// 0x4000 in bfloat16
    BFloat16 b(3.0f);// 0x4040 in bfloat16

    BFloat16 result = a;
    result = result | b;
    EXPECT_EQ(result.x, 0x4040);// 0x4000 | 0x4040 = 0x4040

    result = a;
    result = result & b;
    EXPECT_EQ(result.x, 0x4000);// 0x4000 & 0x4040 = 0x4000

    result = a;
    result = result ^ b;
    EXPECT_EQ(result.x, 0x0040);// 0x4000 ^ 0x4040 = 0x0040
}

TEST(BFloat16Test, NumericLimits) {
    // 测试numeric_limits特性
    EXPECT_TRUE(std::numeric_limits<BFloat16>::is_signed);
    EXPECT_TRUE(std::numeric_limits<BFloat16>::is_specialized);
    EXPECT_FALSE(std::numeric_limits<BFloat16>::is_integer);
    EXPECT_FALSE(std::numeric_limits<BFloat16>::is_exact);
    EXPECT_TRUE(std::numeric_limits<BFloat16>::has_infinity);
    EXPECT_TRUE(std::numeric_limits<BFloat16>::has_quiet_NaN);
    EXPECT_TRUE(std::numeric_limits<BFloat16>::has_signaling_NaN);

    EXPECT_EQ(std::numeric_limits<BFloat16>::digits, 8);
    EXPECT_EQ(std::numeric_limits<BFloat16>::digits10, 2);
    EXPECT_EQ(std::numeric_limits<BFloat16>::max_digits10, 4);
    EXPECT_EQ(std::numeric_limits<BFloat16>::radix, 2);

    EXPECT_EQ(std::numeric_limits<BFloat16>::min_exponent, -125);
    EXPECT_EQ(std::numeric_limits<BFloat16>::min_exponent10, -37);
    EXPECT_EQ(std::numeric_limits<BFloat16>::max_exponent, 128);
    EXPECT_EQ(std::numeric_limits<BFloat16>::max_exponent10, 38);

    // 测试极值
    EXPECT_EQ(std::numeric_limits<BFloat16>::min().x, 0x0080);
    EXPECT_EQ(std::numeric_limits<BFloat16>::max().x, 0x7F7F);
    EXPECT_EQ(std::numeric_limits<BFloat16>::lowest().x, 0xFF7F);
    EXPECT_EQ(std::numeric_limits<BFloat16>::epsilon().x, 0x3C00);
    EXPECT_EQ(std::numeric_limits<BFloat16>::round_error().x, 0x3F00);
    EXPECT_EQ(std::numeric_limits<BFloat16>::infinity().x, 0x7F80);
    EXPECT_EQ(std::numeric_limits<BFloat16>::quiet_NaN().x, 0x7FC0);
    EXPECT_EQ(std::numeric_limits<BFloat16>::denorm_min().x, 0x0001);
}

TEST(BFloat16Test, RoundTripAccuracy) {
    // 测试往返转换精度
    constexpr float test_values[] = {
            1.0f, 2.0f, 0.5f, 3.14159f, 1e38f, 1e-38f,
            -1.0f, -2.0f, -0.5f, -3.14159f, -1e38f, -1e-38f};

    for (float f: test_values) {
        if (std::isnan(f) || std::isinf(f)) {
            continue;// 跳过NaN和无穷大，因为它们无法精确比较
        }

        BFloat16 bf16 = f;
        float f_roundtrip = static_cast<float>(bf16);
        BFloat16 bf16_roundtrip = f_roundtrip;

        // 验证从float到bfloat16再到float，再到bfloat16，两次的bfloat16位值应该相同
        EXPECT_EQ(bf16.x, bf16_roundtrip.x) << "Original: " << f << ", Roundtrip: " << f_roundtrip;
    }
}

}// namespace