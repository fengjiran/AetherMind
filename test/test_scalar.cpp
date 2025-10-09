//
// Created by richard on 10/8/25.
//
#include "scalar.h"

#include <gtest/gtest.h>

using namespace aethermind;

namespace {

// 测试bool类型特化版本的overflows函数
TEST(CastOverflowsTest, BoolType) {
    // bool可以转换到任何类型，不应该溢出
    EXPECT_FALSE((is_overflow<bool, int>(true)));
    EXPECT_FALSE((is_overflow<bool, int>(false)));
    EXPECT_FALSE((is_overflow<bool, uint32_t>(true)));
    EXPECT_FALSE((is_overflow<bool, uint32_t>(false)));
    EXPECT_FALSE((is_overflow<bool, float>(true)));
    EXPECT_FALSE((is_overflow<bool, float>(false)));

    // 测试strict_unsigned参数不影响bool类型的结果
    EXPECT_FALSE((is_overflow<bool, uint32_t>(true, true)));
    EXPECT_FALSE((is_overflow<bool, uint32_t>(false, true)));
}

// 测试整数类型特化版本的overflows函数
TEST(CastOverflowsTest, IntegerType) {
    // 不会溢出的情况
    EXPECT_FALSE((is_overflow<int32_t, int32_t>(100)));
    EXPECT_FALSE((is_overflow<int32_t, int64_t>(100)));
    EXPECT_FALSE((is_overflow<uint32_t, uint64_t>(100)));

    // 会溢出的情况 - 大于目标类型最大值
    EXPECT_TRUE((is_overflow<int32_t, int8_t>(1000))); // 1000 > INT8_MAX (127)
    EXPECT_TRUE((is_overflow<uint32_t, uint8_t>(300)));// 300 > UINT8_MAX (255)

    // 会溢出的情况 - 小于目标类型最小值
    EXPECT_TRUE((is_overflow<int32_t, int8_t>(-1000)));// -1000 < INT8_MIN (-128)

    // 测试有符号到无符号的转换（strict_unsigned = false）
    // 负数转换为无符号类型不会溢出，但值会被解释为大的正数
    EXPECT_FALSE((is_overflow<int32_t, uint32_t>(-1, false)));
    EXPECT_FALSE((is_overflow<int32_t, uint64_t>(-1, false)));

    // 测试有符号到无符号的转换（strict_unsigned = true）
    // 当strict_unsigned为true时，负数应该溢出
    EXPECT_TRUE((is_overflow<int32_t, uint32_t>(-1, true)));

    // 边界情况测试
    EXPECT_FALSE((is_overflow<int32_t, int32_t>(std::numeric_limits<int32_t>::max())));
    EXPECT_FALSE((is_overflow<int32_t, int32_t>(std::numeric_limits<int32_t>::min())));
    EXPECT_TRUE((is_overflow<int32_t, int16_t>(std::numeric_limits<int32_t>::max())));
    EXPECT_TRUE((is_overflow<int32_t, int16_t>(std::numeric_limits<int32_t>::min())));

    // 测试有符号整数到无符号整数的溢出检查
    // 当值的绝对值超过目标无符号类型的最大值时应该溢出
    int64_t large_negative = -static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) - 1;
    EXPECT_TRUE((is_overflow<int64_t, uint32_t>(large_negative, false)));
}

// 测试浮点类型特化版本的overflows函数
TEST(CastOverflowsTest, FloatingPointType) {
    // 不会溢出的情况
    EXPECT_FALSE((is_overflow<float, float>(1.0f)));
    EXPECT_FALSE((is_overflow<float, double>(1.0f)));

    // 会溢出的情况 - 大于目标类型最大值
    EXPECT_TRUE((is_overflow<double, float>(static_cast<double>(std::numeric_limits<float>::max()) * 2.0)));

    // 会溢出的情况 - 小于目标类型最小值
    EXPECT_TRUE((is_overflow<double, float>(static_cast<double>(-std::numeric_limits<float>::max()) * 2.0)));

    // 测试无穷大
    // 当目标类型支持无穷大时，无穷大不应被视为溢出
    EXPECT_FALSE((is_overflow<float, float>(std::numeric_limits<float>::infinity())));
    EXPECT_FALSE((is_overflow<double, double>(std::numeric_limits<double>::infinity())));

    // 测试NaN
    // 当目标类型不支持NaN时，NaN应被视为溢出
    // 注意：标准浮点数类型都支持NaN，这里只是展示逻辑
    float nan_value = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE((is_overflow<float, float>(nan_value)));// float支持NaN

    // 边界情况测试
    EXPECT_FALSE((is_overflow<float, float>(std::numeric_limits<float>::max())));
    EXPECT_FALSE((is_overflow<float, float>(std::numeric_limits<float>::lowest())));
    EXPECT_TRUE((is_overflow<float, Half>(static_cast<float>(std::numeric_limits<Half>::max()) * 2.0f)));
}

// 测试复数类型特化版本的overflows函数
TEST(CastOverflowsTest, ComplexType) {
    // 测试复数到复数的转换
    complex<float> c1(1.0f, 2.0f);
    EXPECT_FALSE((is_overflow<complex<float>, complex<float>>(c1)));
    EXPECT_FALSE((is_overflow<complex<float>, complex<double>>(c1)));

    // 测试复数到标量的转换 - 虚部为0时不应溢出
    complex<float> c2(1.0f, 0.0f);
    EXPECT_FALSE((is_overflow<complex<float>, float>(c2)));

    // 测试复数到标量的转换 - 虚部不为0时应溢出
    complex<float> c3(1.0f, 1.0f);
    EXPECT_TRUE((is_overflow<complex<float>, float>(c3)));

    // 测试实部溢出的情况
    complex<float> c4(std::numeric_limits<float>::max(), 0.0f);
    EXPECT_TRUE((is_overflow<complex<float>, complex<Half>>(c4)));

    // 测试虚部溢出的情况
    complex<float> c5(0.0f, std::numeric_limits<float>::max());
    EXPECT_TRUE((is_overflow<complex<float>, complex<Half>>(c5)));

    // 测试实部和虚部都溢出的情况
    complex<float> c6(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    EXPECT_TRUE((is_overflow<complex<float>, complex<Half>>(c6)));

    // 测试std::complex支持
    std::complex<double> std_c1(1.0, 2.0);
    EXPECT_FALSE((is_overflow<std::complex<double>, std::complex<double>>(std_c1)));
    EXPECT_FALSE((is_overflow<std::complex<double>, std::complex<float>>(std_c1)));

    // 测试std::complex到标量的转换
    std::complex<double> std_c2(1.0, 0.0);
    EXPECT_FALSE((is_overflow<std::complex<double>, double>(std_c2)));

    std::complex<double> std_c3(1.0, 1.0);
    EXPECT_TRUE((is_overflow<std::complex<double>, double>(std_c3)));
}

// 测试混合类型和边界情况
TEST(CastOverflowsTest, MixedTypesAndEdgeCases) {
    // 测试不同整数类型之间的转换
    EXPECT_FALSE((is_overflow<int8_t, int32_t>(127)));
    EXPECT_FALSE((is_overflow<uint8_t, int32_t>(255)));
    EXPECT_FALSE((is_overflow<int16_t, int64_t>(32767)));

    // 测试整数到浮点的转换
    EXPECT_FALSE((is_overflow<int32_t, float>(1000)));
    // 大整数可能会在转换为浮点数时失去精度，但不应被视为溢出
    EXPECT_FALSE((is_overflow<int64_t, float>(1LL << 50)));

    // 测试浮点到整数的转换
    EXPECT_FALSE((is_overflow<float, int32_t>(1000.0f)));
    EXPECT_TRUE((is_overflow<float, int32_t>(static_cast<float>(std::numeric_limits<int32_t>::max()) * 2.0f)));

    // 测试unsigned和signed的边界情况
    EXPECT_FALSE((is_overflow<uint32_t, int64_t>(std::numeric_limits<uint32_t>::max())));
    EXPECT_TRUE((is_overflow<uint64_t, int64_t>(std::numeric_limits<uint64_t>::max())));

    // 测试特殊浮点值
    EXPECT_FALSE((is_overflow<float, double>(-std::numeric_limits<float>::infinity())));
    EXPECT_FALSE((is_overflow<double, float>(0.0 / 0.0)));// NaN
}

TEST(Scalar, init) {
    GTEST_SKIP();
    Scalar s1 = false;
    EXPECT_EQ(s1.toBool(), false);
    EXPECT_TRUE(s1.type() == DataType::Bool());
    EXPECT_TRUE(s1.is_bool());
    s1 = true;
    std::cout << s1 << std::endl;
    std::cout << toString(s1) << std::endl;

    Scalar s2 = 10;
    EXPECT_EQ(s2.toInt(), 10);
    EXPECT_TRUE(s2.is_integral());
    std::cout << s2 << std::endl;
    std::cout << toString(s2) << std::endl;

    Scalar s3 = 1.5;
    EXPECT_EQ(s3.toFloat(), 1.5);
    EXPECT_TRUE(s3.is_floating_point());
    EXPECT_TRUE(std::isfinite(s3.toFloat()));
    std::cout << s3 << std::endl;
    std::cout << toString(s3) << std::endl;
}

}// namespace