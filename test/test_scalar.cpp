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

// 测试 maybe_real 结构体
TEST(CastTest, MaybeReal) {
    // 测试非复数类型
    int value = 42;
    EXPECT_EQ((maybe_real<int, false>::apply(value)), 42);

    // 测试复数类型
    complex<float> c(1.0f, 2.0f);
    EXPECT_FLOAT_EQ((maybe_real<complex<float>, true>::apply(c)), 1.0f);
}

// 测试 maybe_bool 结构体
TEST(CastTest, MaybeBool) {
    // 测试非复数类型
    int value = 42;
    EXPECT_EQ((maybe_bool<int, false>::apply(value)), 42);

    // 测试复数类型
    complex<float> c1(0.0f, 0.0f);
    EXPECT_FALSE((maybe_bool<complex<float>, true>::apply(c1)));

    complex<float> c2(1.0f, 0.0f);
    EXPECT_TRUE((maybe_bool<complex<float>, true>::apply(c2)));

    complex<float> c3(0.0f, 1.0f);
    EXPECT_TRUE((maybe_bool<complex<float>, true>::apply(c3)));

    complex<float> c4(1.0f, 1.0f);
    EXPECT_TRUE((maybe_bool<complex<float>, true>::apply(c4)));
}

// 测试通用cast模板
TEST(CastTest, CastBasicTypes) {
    // 基本类型转换
    EXPECT_EQ((cast<int, double>::apply(42)), 42.0);
    EXPECT_EQ((cast<double, int>::apply(42.5)), 42);

    // 复数到实数的转换
    complex<float> c(1.5f, 2.5f);
    EXPECT_FLOAT_EQ((cast<complex<float>, float>::apply(c)), 1.5f);

    // 实数到复数的转换
    complex<double> c2 = cast<double, complex<double>>::apply(3.14);
    EXPECT_DOUBLE_EQ(c2.real(), 3.14);
    EXPECT_DOUBLE_EQ(c2.imag(), 0.0);
}

// 测试cast<bool>特化
TEST(CastTest, CastToBool) {
    // 基本类型到bool的转换
    EXPECT_TRUE((cast<int, bool>::apply(1)));
    EXPECT_FALSE((cast<int, bool>::apply(0)));
    EXPECT_TRUE((cast<double, bool>::apply(1.5)));
    EXPECT_FALSE((cast<double, bool>::apply(0.0)));

    // 复数到bool的转换
    complex<float> c1(0.0f, 0.0f);
    EXPECT_FALSE((cast<complex<float>, bool>::apply(c1)));

    complex<float> c2(1.0f, 0.0f);
    EXPECT_TRUE((cast<complex<float>, bool>::apply(c2)));

    complex<float> c3(0.0f, 1.0f);
    EXPECT_TRUE((cast<complex<float>, bool>::apply(c3)));
}

// 测试cast<uint8_t>特化
TEST(CastTest, CastToUint8) {
    // 基本类型到uint8_t的转换
    EXPECT_EQ((cast<int, uint8_t>::apply(42)), 42);
    EXPECT_EQ((cast<int64_t, uint8_t>::apply(255)), 255);

    // 复数到uint8_t的转换
    complex<int> c(100, 50);
    EXPECT_EQ((cast<complex<int>, uint8_t>::apply(c)), 100);
}

// 测试特殊浮点类型到complex<Half>的转换
TEST(CastTest, CastToComplexHalf) {
    // BFloat16到complex<Half>的转换
    BFloat16 bfloat(1.5f);
    complex<Half> c1 = cast<BFloat16, complex<Half>>::apply(bfloat);
    EXPECT_FLOAT_EQ(c1.real(), 1.5f);
    EXPECT_FLOAT_EQ(c1.imag(), 0.0f);

    // Float8_e5m2到complex<Half>的转换
    Float8_e5m2 f8e5m2(2.5f);
    complex<Half> c2 = cast<Float8_e5m2, complex<Half>>::apply(f8e5m2);
    EXPECT_FLOAT_EQ(c2.real(), 2.5f);
    EXPECT_FLOAT_EQ(c2.imag(), 0.0f);

    // Float8_e4m3fn到complex<Half>的转换
    Float8_e4m3fn f8e4m3fn(3.5f);
    complex<Half> c3 = cast<Float8_e4m3fn, complex<Half>>::apply(f8e4m3fn);
    EXPECT_FLOAT_EQ(c3.real(), 3.5f);
    EXPECT_FLOAT_EQ(c3.imag(), 0.0f);

    // Half到complex<Half>的转换
    Half half(4.5f);
    complex<Half> c4 = cast<Half, complex<Half>>::apply(half);
    EXPECT_FLOAT_EQ(c4.real(), 4.5f);
    EXPECT_FLOAT_EQ(c4.imag(), 0.0f);

    // complex<double>到complex<Half>的转换
    complex<double> cd(5.5, 6.5);
    complex<Half> c5 = cast<complex<double>, complex<Half>>::apply(cd);
    EXPECT_FLOAT_EQ(c5.real(), 5.5f);
    EXPECT_FLOAT_EQ(c5.imag(), 6.5f);
}

// 测试check_and_cast函数
TEST(CastTest, CheckAndCastNoOverflow) {
    // 不会溢出的转换
    EXPECT_EQ((check_and_cast<int, short>(32767, "short")), 32767);
    EXPECT_EQ((check_and_cast<int, unsigned>(100, "unsigned")), 100U);
    EXPECT_EQ((check_and_cast<int, uint8_t>(-1, "unsigned char")), 255);

    // 复数到实数的安全转换
    complex<double> c(1.0, 0.0);
    EXPECT_DOUBLE_EQ((check_and_cast<complex<double>, double>(c, "double")), 1.0);
}

// 测试check_and_cast函数的溢出检测
TEST(CastTest, CheckAndCastOverflow) {
    // 整数溢出情况
    EXPECT_THROW((check_and_cast<int, char>(128, "char")), Error);

    // 浮点数溢出情况
    EXPECT_THROW((check_and_cast<double, float>(1e39, "float")), Error);

    // 复数虚部不为零的情况
    complex<double> c(1.0, 2.0);
    EXPECT_THROW((check_and_cast<complex<double>, double>(c, "double")), Error);
}

// 测试check_and_cast对bool类型的特殊处理
TEST(CastTest, CheckAndCastBool) {
    // bool类型不进行溢出检查
    EXPECT_TRUE((check_and_cast<int, bool>(100, "bool")));
    EXPECT_FALSE((check_and_cast<int, bool>(0, "bool")));

    // 复数到bool的转换，即使虚部不为零也不会抛出异常
    complex<double> c(0.0, 1.0);
    EXPECT_TRUE((check_and_cast<complex<double>, bool>(c, "bool")));
}

// 测试边界值转换
TEST(CastTest, CastBoundaryValues) {
    // 测试边界值
    EXPECT_EQ((cast<int, char>::apply(127)), 127);
    EXPECT_EQ((cast<int, unsigned char>::apply(255)), 255);

    // 测试零值
    EXPECT_EQ((cast<int, float>::apply(0)), 0.0f);
    EXPECT_FALSE((cast<int, bool>::apply(0)));

    // 测试负值
    EXPECT_FLOAT_EQ((cast<int, float>::apply(-42)), -42.0f);
    EXPECT_TRUE((cast<int, bool>::apply(-1)));
}

// 测试混合类型转换
TEST(CastTest, MixedTypeCasts) {
    // 测试不同整数类型之间的转换
    int64_t big_int = 10000000000;
    EXPECT_EQ((cast<int64_t, int32_t>::apply(big_int)), static_cast<int32_t>(big_int));

    // 测试整数到浮点数的转换
    int64_t large_int = 1000000000;
    EXPECT_DOUBLE_EQ((cast<int64_t, double>::apply(large_int)), 1000000000.0);

    // 测试浮点数到整数的转换
    double pi = 3.14159;
    EXPECT_EQ((cast<double, int>::apply(pi)), 3);
}

// 测试特殊浮点值
TEST(CastTest, SpecialFloatingPointValues) {
    // 测试无穷大
    double inf = std::numeric_limits<double>::infinity();
    float inf_float = cast<double, float>::apply(inf);
    EXPECT_TRUE(std::isinf(inf_float));

    // 测试NaN
    double nan = std::numeric_limits<double>::quiet_NaN();
    float nan_float = cast<double, float>::apply(nan);
    EXPECT_TRUE(std::isnan(nan_float));

    // 注意：check_and_cast对于NaN的处理取决于目标类型是否支持NaN
    EXPECT_NO_THROW((cast<double, float>::apply(nan)));
}

TEST(Scalar, init) {
    // GTEST_SKIP();
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

// 测试Scalar类的默认构造函数和基本整数构造
TEST(ScalarTest, DefaultAndIntegralConstructors) {
    // 默认构造函数（应该是int64_t(0)）
    Scalar default_scalar;
    EXPECT_TRUE(default_scalar.is_signed_integral());
    EXPECT_EQ(default_scalar.toLong(), 0);
    EXPECT_EQ(default_scalar.type(), DataType::Int(64));

    // 有符号整数构造函数
    Scalar i8(static_cast<int8_t>(42));
    EXPECT_TRUE(i8.is_signed_integral());
    EXPECT_EQ(i8.toChar(), 42);
    EXPECT_EQ(i8.type(), DataType::Int(8));

    Scalar i16(static_cast<int16_t>(-1234));
    EXPECT_TRUE(i16.is_signed_integral());
    EXPECT_EQ(i16.toShort(), -1234);
    EXPECT_EQ(i16.type(), DataType::Int(16));

    Scalar i32(static_cast<int32_t>(123456));
    EXPECT_TRUE(i32.is_signed_integral());
    EXPECT_EQ(i32.toInt(), 123456);
    EXPECT_EQ(i32.type(), DataType::Int(32));

    Scalar i64(static_cast<int64_t>(-9876543210));
    EXPECT_TRUE(i64.is_signed_integral());
    EXPECT_EQ(i64.toLong(), -9876543210);
    EXPECT_EQ(i64.type(), DataType::Int(64));

    // 无符号整数构造函数
    Scalar u8(static_cast<uint8_t>(200));
    EXPECT_TRUE(u8.is_signed_integral());
    EXPECT_EQ(u8.toByte(), 200);
    EXPECT_EQ(u8.type(), DataType::Int(8));

    Scalar u16(static_cast<uint16_t>(40000));
    EXPECT_TRUE(u16.is_signed_integral());
    EXPECT_EQ(u16.toUInt16(), 40000);
    EXPECT_EQ(u16.type(), DataType::Int(16));

    Scalar u32(static_cast<uint32_t>(123456789));
    EXPECT_TRUE(u32.is_signed_integral());
    EXPECT_EQ(u32.toUInt32(), 123456789);
    EXPECT_EQ(u32.type(), DataType::Int(32));

    Scalar u64(static_cast<uint64_t>(18446744073709551615ULL));
    EXPECT_TRUE(u64.is_unsigned_integral());
    EXPECT_EQ(u64.toUInt64(), 18446744073709551615ULL);
    EXPECT_EQ(u64.type(), DataType::UInt(64));
}

}// namespace