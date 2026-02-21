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

// 通用测试函数，用于测试单参数数学函数
void TestUnaryMathFunction(BFloat16 (*func)(BFloat16), float (*std_func)(float),
                           const std::vector<float>& test_values, float max_relative_error = 0.02f) {
    for (float f: test_values) {
        BFloat16 bf16 = f;
        BFloat16 result_bf16 = func(bf16);
        float expected = std_func(f);
        float actual = result_bf16;

        if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(actual)) << "Function test failed for input: " << f;
        } else if (std::isinf(expected)) {
            EXPECT_TRUE(std::isinf(actual)) << "Function test failed for input: " << f;
            EXPECT_EQ(std::signbit(expected), std::signbit(actual)) << "Function test failed for input: " << f;
        } else if (expected == 0.0f) {
            EXPECT_FLOAT_EQ(actual, 0.0f);
        } else if (f == 0.0f) {
            // 对于零输入，直接比较结果
            EXPECT_NEAR(actual, expected, max_relative_error) << "Function test failed for input: " << f;
        } else {
            // 计算相对误差
            float relative_error = std::abs((actual - expected) / expected);
            EXPECT_LE(relative_error, max_relative_error)
                    << "Function test failed for input: " << f
                    << ", Expected: " << expected
                    << ", Actual: " << actual
                    << ", Relative error: " << relative_error;
        }
    }
}

// 测试acos函数
TEST(BFloat16MathTest, Acos) {
    std::vector<float> test_values = {
            0.0f, 1.0f, -1.0f, 0.5f, -0.5f,
            0.7071f, -0.7071f, 0.8660f, -0.8660f};
    TestUnaryMathFunction(std::acos<BFloat16>, std::acos, test_values);
}

// 测试asin函数
TEST(BFloat16MathTest, Asin) {
    std::vector<float> test_values = {
            0.0f, 1.0f, -1.0f, 0.5f, -0.5f,
            0.7071f, -0.7071f, 0.8660f, -0.8660f};
    TestUnaryMathFunction(std::asin<BFloat16>, std::asin, test_values);
}

// 测试atan函数
TEST(BFloat16MathTest, Atan) {
    std::vector<float> test_values = {
            0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f,
            100.0f, -100.0f, 0.333333f, -0.333333f};
    TestUnaryMathFunction(std::atan<BFloat16>, std::atan, test_values);
}

// 测试erf函数
TEST(BFloat16MathTest, Erf) {
    std::vector<float> test_values = {
            0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f,
            3.0f, -3.0f, 0.1f, -0.1f};
    TestUnaryMathFunction(std::erf<BFloat16>, std::erf, test_values);
}

// 测试erfc函数
TEST(BFloat16MathTest, Erfc) {
    std::vector<float> test_values = {
            0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f,
            3.0f, -3.0f, 0.1f, -0.1f};
    TestUnaryMathFunction(std::erfc<BFloat16>, std::erfc, test_values);
}

// 测试exp函数
TEST(BFloat16MathTest, Exp) {
    std::vector<float> test_values = {
            0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f,
            0.693147f, -0.693147f, 1.098612f, -1.098612f};
    // 对于指数函数，放宽误差要求
    TestUnaryMathFunction(std::exp<BFloat16>, std::exp, test_values, 0.05f);
}

// 测试expm1函数
TEST(BFloat16MathTest, Expm1) {
    std::vector<float> test_values = {
            0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 0.1f, -0.1f,
            0.01f, -0.01f, 0.001f, -0.001f};
    TestUnaryMathFunction(std::expm1<BFloat16>, std::expm1, test_values);
}


// 测试isfinite函数
TEST(BFloat16MathTest, IsFinite) {
    BFloat16 finite_value(1.0f);
    BFloat16 inf_pos(std::numeric_limits<float>::infinity());
    BFloat16 inf_neg(-std::numeric_limits<float>::infinity());
    BFloat16 nan(std::numeric_limits<float>::quiet_NaN());

    EXPECT_TRUE(std::isfinite(finite_value));
    EXPECT_FALSE(std::isfinite(inf_pos));
    EXPECT_FALSE(std::isfinite(inf_neg));
    EXPECT_FALSE(std::isfinite(nan));
}

// 测试log函数
TEST(BFloat16MathTest, Log) {
    std::vector<float> test_values = {
            1.0f, 2.0f, 0.5f, M_E, M_PI, 10.0f, 0.1f,
            3.14159f, 0.318309f, 100.0f, 0.01f};
    TestUnaryMathFunction(std::log<BFloat16>, std::log, test_values);
}

// 测试log10函数
TEST(BFloat16MathTest, Log10) {
    std::vector<float> test_values = {
            1.0f, 10.0f, 0.1f, 100.0f, 0.01f, 1000.0f, 0.001f,
            5.0f, 2.0f, 50.0f, 20.0f};
    TestUnaryMathFunction(std::log10<BFloat16>, std::log10, test_values);
}

// 测试log2函数
TEST(BFloat16MathTest, Log2) {
    std::vector<float> test_values = {
            1.0f, 2.0f, 0.5f, 4.0f, 0.25f, 8.0f, 0.125f,
            3.0f, 5.0f, 10.0f, 0.3f};
    TestUnaryMathFunction(std::log2<BFloat16>, std::log2, test_values);
}

// 测试log1p函数
TEST(BFloat16MathTest, Log1p) {
    std::vector<float> test_values = {
            0.0f, 1.0f, -0.5f, 0.5f, 2.0f, -0.9f, 0.1f,
            -0.1f, 0.01f, -0.01f, 0.001f};
    TestUnaryMathFunction(std::log1p<BFloat16>, std::log1p, test_values);
}

// 测试ceil函数
TEST(BFloat16MathTest, Ceil) {
    std::vector<float> test_values = {
            0.0f, 1.0f, -1.0f, 1.1f, -1.1f, 1.5f, -1.5f,
            2.999f, -2.99f, 0.0001f, -0.0001f};
    for (float f: test_values) {
        BFloat16 bf16 = f;
        BFloat16 result_bf16 = std::ceil(bf16);
        float expected = std::ceil(f);
        float actual = result_bf16;
        EXPECT_FLOAT_EQ(actual, expected)
                << "Ceil test failed for input: " << f
                << ", Expected: " << expected
                << ", Actual: " << actual;
    }
}

// 测试cos函数
TEST(BFloat16MathTest, Cos) {
    // GTEST_SKIP();
    std::vector<float> test_values = {
            0.0f, M_PI, -M_PI, M_PI_4, -M_PI_4,
            M_PI * 0.25f, M_PI * 0.75f, M_PI * 1.25f, M_PI * 1.75f};
    TestUnaryMathFunction(std::cos<BFloat16>, std::cos, test_values);
}

// 测试sin函数
TEST(BFloat16MathTest, Sin) {
    // GTEST_SKIP();
    std::vector<float> test_values = {
            0.0f, M_PI_2, -M_PI_2, M_PI_4, -M_PI_4,
            M_PI * 0.25f, M_PI * 0.75f, M_PI * 1.25f, M_PI * 1.75f};
    TestUnaryMathFunction(std::sin<BFloat16>, std::sin, test_values);
}

// 测试sinh函数
TEST(BFloat16MathTest, Sinh) {
    std::vector<float> test_values = {
            0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f,
            3.0f, -3.0f, 0.1f, -0.1f};
    TestUnaryMathFunction(std::sinh<BFloat16>, std::sinh, test_values, 0.03f);
}

// 测试cosh函数
TEST(BFloat16MathTest, Cosh) {
    std::vector<float> test_values = {
            0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f,
            3.0f, -3.0f, 0.1f, -0.1f};
    TestUnaryMathFunction(std::cosh<BFloat16>, std::cosh, test_values, 0.03f);
}

// 测试tan函数
TEST(BFloat16MathTest, Tan) {
    std::vector<float> test_values = {
            0.0f, M_PI_4, -M_PI_4,
            0.1f, -0.1f, 1.0f, -1.0f};
    // 对于tan函数，放宽误差要求
    TestUnaryMathFunction(std::tan<BFloat16>, std::tan, test_values, 0.03f);
}

// 测试tanh函数
TEST(BFloat16MathTest, Tanh) {
    std::vector<float> test_values = {
            0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f,
            3.0f, -3.0f, 0.1f, -0.1f};
    TestUnaryMathFunction(std::tanh<BFloat16>, std::tanh, test_values);
}

// 测试floor函数
TEST(BFloat16MathTest, Floor) {
    std::vector<float> test_values = {
            0.0f, 1.0f, -1.0f, 1.9f, -1.9f, 1.5f, -1.5f,
            2.001f, -2.1f, 0.99f, -0.999f};
    for (float f: test_values) {
        BFloat16 bf16 = f;
        BFloat16 result_bf16 = std::floor(bf16);
        float expected = std::floor(f);
        float actual = result_bf16;
        EXPECT_FLOAT_EQ(actual, expected)
                << "Floor test failed for input: " << f
                << ", Expected: " << expected
                << ", Actual: " << actual;
    }
}

// 测试nearbyint函数
TEST(BFloat16MathTest, Nearbyint) {
    std::vector<float> test_values = {
            0.0f, 1.0f, -1.0f, 1.1f, -1.1f, 1.5f, -1.5f,
            2.9f, -2.9f, 0.0001f, -0.0001f};
    TestUnaryMathFunction(std::nearbyint<BFloat16>, std::nearbyint, test_values, 0.01f);
}

// 测试trunc函数
TEST(BFloat16MathTest, Trunc) {
    std::vector<float> test_values = {
            0.0f, 1.0f, -1.0f, 1.1f, -1.1f, 1.9f, -1.9f,
            1.5f, -1.5f, 0.99f, -0.99f};
    for (float f: test_values) {
        BFloat16 bf16 = f;
        BFloat16 result_bf16 = std::trunc(bf16);
        float expected = std::trunc(f);
        float actual = result_bf16;
        EXPECT_FLOAT_EQ(actual, expected)
                << "Trunc test failed for input: " << f
                << ", Expected: " << expected
                << ", Actual: " << actual;
    }
}

// 测试lgamma函数
TEST(BFloat16MathTest, Lgamma) {
    std::vector<float> test_values = {
            1.0f, 2.0f, 3.0f, 0.5f, 1.5f, 5.0f, 10.0f,
            0.1f, 0.2f, 0.3f, 0.4f};
    // 对于lgamma函数，放宽误差要求
    TestUnaryMathFunction(std::lgamma<BFloat16>, std::lgamma, test_values, 0.05f);
}

// 测试sqrt函数
TEST(BFloat16MathTest, Sqrt) {
    std::vector<float> test_values = {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 0.25f, 0.5f,
            10.0f, 25.0f, 100.0f, 0.01f};
    TestUnaryMathFunction(std::sqrt<BFloat16>, std::sqrt, test_values);
}

// 测试rsqrt函数
TEST(BFloat16MathTest, Rsqrt) {
    std::vector<float> test_values = {
            1.0f, 2.0f, 3.0f, 4.0f, 0.25f, 0.5f,
            10.0f, 25.0f, 100.0f, 0.01f};
    auto rsqrt_func = [](float x) { return 1.0f / std::sqrt(x); };
    TestUnaryMathFunction(std::rsqrt<BFloat16>, rsqrt_func, test_values);
}

// 测试abs函数
TEST(BFloat16MathTest, Abs) {
    std::vector<float> test_values = {
            0.0f, 1.0f, -1.0f, 2.5f, -2.5f,
            100.0f, -100.0f};
    for (float f: test_values) {
        BFloat16 bf16 = f;
        BFloat16 result_bf16 = std::abs(bf16);
        float expected = std::abs(f);
        float actual = result_bf16;
        EXPECT_FLOAT_EQ(actual, expected)
                << "Abs test failed for input: " << f
                << ", Expected: " << expected
                << ", Actual: " << actual;
    }
}

// 测试pow函数
TEST(BFloat16MathTest, Pow) {
    std::vector<std::pair<float, double>> test_pairs = {
            {1.0f, 2.0},
            {2.0f, 3.0},
            {0.5f, 2.0},
            {10.0f, 0.5},
            {2.0f, 0.0},
            {2.0f, -1.0},
            {0.5f, -2.0},
            {3.0f, 1.5},
            {4.0f, 0.25},
            {0.1f, 2.0}};

    for (const auto& [base, exp]: test_pairs) {
        BFloat16 base_bf16 = base;
        BFloat16 result_bf16 = std::pow(base_bf16, exp);
        float expected = std::pow(base, exp);
        float actual = result_bf16;

        if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(actual))
                    << "Pow test failed for base: " << base << ", exponent: " << exp;
        } else if (std::isinf(expected)) {
            EXPECT_TRUE(std::isinf(actual))
                    << "Pow test failed for base: " << base << ", exponent: " << exp;
            EXPECT_EQ(std::signbit(expected), std::signbit(actual))
                    << "Pow test failed for base: " << base << ", exponent: " << exp;
        } else if (base == 0.0f) {
            EXPECT_NEAR(actual, expected, 0.02f)
                    << "Pow test failed for base: " << base << ", exponent: " << exp;
        } else {
            float relative_error = std::abs((actual - expected) / expected);
            EXPECT_LE(relative_error, 0.03f)
                    << "Pow test failed for base: " << base
                    << ", exponent: " << exp
                    << ", Expected: " << expected
                    << ", Actual: " << actual
                    << ", Relative error: " << relative_error;
        }
    }
}

// 测试fmod函数
TEST(BFloat16MathTest, Fmod) {
    std::vector<std::pair<float, float>> test_pairs = {
            {5.0f, 2.0f},
            {5.5f, 2.0f},
            {-5.0f, 2.0f},
            {5.0f, -2.0f},
            {0.0f, 2.0f},
            {1.0f, 0.3f},
            {2.0f, 0.5f},
            {3.14159f, 1.0f},
            {10.0f, 3.0f},
            {7.0f, 3.0f}};

    for (const auto& [x, y]: test_pairs) {
        if (y == 0.0f) continue;// 跳过除以零的情况

        BFloat16 x_bf16 = x;
        BFloat16 y_bf16 = y;
        BFloat16 result_bf16 = std::fmod(x_bf16, y_bf16);
        float expected = std::fmod(x, y);
        float actual = static_cast<float>(result_bf16);

        // fmod结果可能会有不同的符号，取决于实现，所以我们只比较绝对值
        if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(actual))
                    << "Fmod test failed for x: " << x << ", y: " << y;
        } else {
            // 检查结果的绝对值和符号
            EXPECT_NEAR(std::abs(actual), std::abs(expected), 0.02f)
                    << "Fmod test failed for x: " << x
                    << ", y: " << y
                    << ", Expected: " << expected
                    << ", Actual: " << actual;

            // 确保结果与被除数同号
            EXPECT_EQ(std::signbit(actual), std::signbit(x))
                    << "Fmod test failed for x: " << x
                    << ", y: " << y
                    << ", Result sign incorrect";
        }
    }
}

}// namespace