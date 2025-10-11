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

    Scalar i32(123456);
    EXPECT_TRUE(i32.is_signed_integral());
    EXPECT_EQ(i32.toInt(), 123456);
    EXPECT_EQ(i32.type(), DataType::Int(32));

    Scalar i64(-9876543210);
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

// 测试Scalar类的布尔构造函数
TEST(ScalarTest, BoolConstructor) {
    Scalar true_bool(true);
    EXPECT_TRUE(true_bool.is_bool());
    EXPECT_TRUE(true_bool.equal(true));
    EXPECT_FALSE(true_bool.equal(false));
    EXPECT_EQ(true_bool.type(), DataType::Bool());

    Scalar false_bool(false);
    EXPECT_TRUE(false_bool.is_bool());
    EXPECT_FALSE(false_bool.equal(true));
    EXPECT_TRUE(false_bool.equal(false));
}

// 测试Scalar类的浮点构造函数
TEST(ScalarTest, FloatingPointConstructors) {
    // 标准浮点类型
    Scalar f32(1.234f);
    EXPECT_TRUE(f32.is_floating_point());
    EXPECT_FLOAT_EQ(f32.toFloat(), 1.234f);
    EXPECT_DOUBLE_EQ(f32.toDouble(), 1.2339999675750732);
    EXPECT_EQ(f32.type(), DataType::Float(32));

    Scalar f64(5.6789);
    EXPECT_TRUE(f64.is_floating_point());
    EXPECT_DOUBLE_EQ(f64.toDouble(), 5.6789);
    EXPECT_EQ(f64.type(), DataType::Float(64));

    // 半精度浮点类型
    Half half_val(0.123f);
    Scalar half_scalar(half_val);
    EXPECT_TRUE(half_scalar.is_floating_point());
    EXPECT_EQ(half_scalar.type(), DataType::Float(16));

    // BFloat16类型
    BFloat16 bf16_val(0.456f);
    Scalar bf16_scalar(bf16_val);
    EXPECT_TRUE(bf16_scalar.is_floating_point());
    EXPECT_EQ(bf16_scalar.type(), DataType::BFloat(16));

    // Float8类型变体
    Float8_e4m3fn f8e4m3fn_val(0.789f);
    Scalar f8e4m3fn_scalar(f8e4m3fn_val);
    EXPECT_TRUE(f8e4m3fn_scalar.is_floating_point());
    EXPECT_EQ(f8e4m3fn_scalar.type(), DataType::Float8E4M3FN());

    Float8_e5m2 f8e5m2_val(0.321f);
    Scalar f8e5m2_scalar(f8e5m2_val);
    EXPECT_TRUE(f8e5m2_scalar.is_floating_point());
    EXPECT_EQ(f8e5m2_scalar.type(), DataType::Float8E5M2());
}

// 测试Scalar类的复数构造函数
TEST(ScalarTest, ComplexConstructors) {
    // 复数float
    complex<float> cfloat(1.0f, 2.0f);
    Scalar cfloat_scalar(cfloat);
    EXPECT_TRUE(cfloat_scalar.is_complex());
    EXPECT_EQ(cfloat_scalar.type().code(), DLDataTypeCode::kComplex);

    // 复数double
    complex<double> cdouble(3.0, 4.0);
    Scalar cdouble_scalar(cdouble);
    EXPECT_TRUE(cdouble_scalar.is_complex());
    EXPECT_EQ(cdouble_scalar.type().code(), DLDataTypeCode::kComplex);

    // 复数Half
    complex<Half> chalf(Half(5.0f), Half(6.0f));
    Scalar chalf_scalar(chalf);
    EXPECT_TRUE(chalf_scalar.is_complex());
    EXPECT_EQ(chalf_scalar.type().code(), DLDataTypeCode::kComplex);
}

// 测试Scalar类的复制和移动语义
TEST(ScalarTest, CopyAndMoveSemantics) {
    // 测试复制构造函数
    Scalar original(42);
    Scalar copy(original);
    EXPECT_EQ(original.toLong(), copy.toLong());
    EXPECT_EQ(original.type(), copy.type());

    // 测试复制赋值运算符
    Scalar assigned;
    assigned = original;
    EXPECT_EQ(original.toLong(), assigned.toLong());
    EXPECT_EQ(original.type(), assigned.type());

    // 测试移动构造函数
    Scalar moved(std::move(Scalar(123)));
    EXPECT_EQ(moved.toLong(), 123);

    // 测试移动赋值运算符
    Scalar move_assigned;
    move_assigned = std::move(Scalar(456));
    EXPECT_EQ(move_assigned.toLong(), 456);

    // 测试自赋值
    Scalar self_assign(789);
    self_assign = self_assign;
    EXPECT_EQ(self_assign.toLong(), 789);
}

// 测试Scalar类的类型转换方法
TEST(ScalarTest, TypeConversionMethods) {
    // 整数类型转换
    Scalar i64(123456789L);
    EXPECT_THROW(i64.toChar(), Error);
    EXPECT_THROW(i64.toShort(), Error);
    EXPECT_EQ(i64.toInt(), static_cast<int32_t>(123456789L));
    EXPECT_EQ(i64.toLong(), 123456789L);
    EXPECT_THROW(i64.toByte(), Error);
    EXPECT_THROW(i64.toUInt16(), Error);
    EXPECT_EQ(i64.toUInt32(), static_cast<uint32_t>(123456789L));
    EXPECT_EQ(i64.toUInt64(), static_cast<uint64_t>(123456789L));
    EXPECT_FLOAT_EQ(i64.toFloat(), 123456789.0f);
    EXPECT_DOUBLE_EQ(i64.toDouble(), 123456789.0);

    // 浮点类型转换
    Scalar f64(1.23456789);
    EXPECT_EQ(f64.toChar(), static_cast<int8_t>(1.23456789));
    EXPECT_EQ(f64.toShort(), static_cast<int16_t>(1.23456789));
    EXPECT_EQ(f64.toInt(), static_cast<int32_t>(1.23456789));
    EXPECT_EQ(f64.toLong(), static_cast<int64_t>(1.23456789));
    EXPECT_EQ(f64.toByte(), static_cast<uint8_t>(1.23456789));
    EXPECT_EQ(f64.toUInt16(), static_cast<uint16_t>(1.23456789));
    EXPECT_EQ(f64.toUInt32(), static_cast<uint32_t>(1.23456789));
    EXPECT_EQ(f64.toUInt64(), static_cast<uint64_t>(1.23456789));
    EXPECT_FLOAT_EQ(f64.toFloat(), 1.23456789f);
    EXPECT_DOUBLE_EQ(f64.toDouble(), 1.23456789);

    // 布尔类型转换
    Scalar bool_true(true);
    EXPECT_TRUE(bool_true.toBool());
    EXPECT_EQ(bool_true.toChar(), 1);
    EXPECT_EQ(bool_true.toLong(), 1L);
    EXPECT_FLOAT_EQ(bool_true.toFloat(), 1.0f);

    Scalar bool_false(false);
    EXPECT_FALSE(bool_false.toBool());
    EXPECT_EQ(bool_false.toChar(), 0);
    EXPECT_EQ(bool_false.toLong(), 0L);
    EXPECT_FLOAT_EQ(bool_false.toFloat(), 0.0f);
}

// 测试Scalar类的equal方法
TEST(ScalarTest, EqualMethod) {
    // 整数相等性测试
    Scalar i64(42);
    EXPECT_TRUE(i64.equal(42));
    EXPECT_FALSE(i64.equal(43));
    EXPECT_TRUE(i64.equal(42L));
    EXPECT_TRUE(i64 == 42);
    EXPECT_TRUE(42 == i64);

    // 浮点数相等性测试
    Scalar f64(1.234);
    EXPECT_TRUE(f64.equal(1.234));
    EXPECT_FALSE(f64.equal(1.235));
    EXPECT_FALSE(f64.equal(1.234f));
    EXPECT_TRUE(f64 == 1.234);
    EXPECT_TRUE(1.234 == f64);

    // 布尔相等性测试
    Scalar bool_true(true);
    EXPECT_TRUE(bool_true.equal(true));
    EXPECT_FALSE(bool_true.equal(false));
    EXPECT_TRUE(bool_true == true);
    EXPECT_TRUE(true == bool_true);

    Scalar bool_false(false);
    EXPECT_TRUE(bool_false.equal(false));
    EXPECT_FALSE(bool_false.equal(true));
    EXPECT_TRUE(bool_false == false);
    EXPECT_TRUE(false == bool_false);

    // 复数相等性测试
    complex<double> cval(1.0, 2.0);
    Scalar cscalar(cval);
    EXPECT_TRUE(cscalar.equal(cval));
    EXPECT_FALSE(cscalar.equal(complex<double>(1.0, 3.0)));
    EXPECT_TRUE(cscalar == cval);
    EXPECT_TRUE(cval == cscalar);

    // 跨类型相等性测试
    Scalar int_42(42);
    EXPECT_TRUE(int_42.equal(42.0)); // 整数和浮点数相等
    EXPECT_FALSE(int_42.equal(true));// 整数和布尔值不等
    EXPECT_TRUE(int_42 == 42.0);
    EXPECT_TRUE(42.0 == int_42);
}

// 测试Scalar类的一元减运算符
TEST(ScalarTest, UnaryMinusOperator) {
    // 整数测试
    Scalar i64(42);
    Scalar neg_i64 = -i64;
    EXPECT_EQ(neg_i64.toLong(), -42L);

    Scalar neg_i64_2(-123);
    Scalar pos_i64 = -neg_i64_2;
    EXPECT_EQ(pos_i64.toLong(), 123L);

    // 浮点数测试
    Scalar f64(1.234);
    Scalar neg_f64 = -f64;
    EXPECT_DOUBLE_EQ(neg_f64.toDouble(), -1.234);
    EXPECT_EQ(neg_f64.type(), f64.type());

    // 边界情况：最小整数
    Scalar min_int(std::numeric_limits<int64_t>::min());
    Scalar neg_min_int = -min_int;
    // 注意：这可能会溢出，取决于实现
}

// 测试Scalar类的log方法
TEST(ScalarTest, LogMethod) {
    // 正数测试
    Scalar positive(1.0);
    Scalar log_positive = positive.log();
    EXPECT_DOUBLE_EQ(log_positive.toDouble(), 0.0);

    Scalar e_val(M_E);
    Scalar log_e = e_val.log();
    EXPECT_NEAR(log_e.toDouble(), 1.0, 1e-10);

    // 整数转换为浮点数后计算log
    Scalar int_val(2);
    Scalar log_int = int_val.log();
    EXPECT_NEAR(log_int.toDouble(), std::log(2.0), 1e-10);
    EXPECT_TRUE(log_int.is_floating_point());
}

// 测试Scalar类的conj方法（复数共轭）
TEST(ScalarTest, ConjMethod) {
    // 复数测试
    complex<double> cval(1.0, 2.0);
    Scalar cscalar(cval);
    Scalar conj_c = cscalar.conj();
    complex<double> conj_val = cval;
    conj_val = std::conj(conj_val);
    EXPECT_TRUE(conj_c.equal(conj_val));

    // 实数的共轭是其本身
    Scalar real_val(3.14);
    Scalar conj_real = real_val.conj();
    EXPECT_TRUE(conj_real.equal(3.14));

    // 整数的共轭是其本身
    Scalar int_val(42);
    Scalar conj_int = int_val.conj();
    EXPECT_TRUE(conj_int.equal(42));
}

// 测试Scalar类的swap方法
TEST(ScalarTest, SwapMethod) {
    Scalar a(10);
    Scalar b(20.5);
    DataType type_a = a.type();
    DataType type_b = b.type();

    a.swap(b);
    EXPECT_EQ(a.toDouble(), 20.5);
    EXPECT_EQ(a.type(), type_b);
    EXPECT_EQ(b.toLong(), 10L);
    EXPECT_EQ(b.type(), type_a);
}

// 测试Scalar类的边界情况
TEST(ScalarTest, EdgeCases) {
    // 最大/最小整数值
    Scalar max_int(std::numeric_limits<int64_t>::max());
    EXPECT_EQ(max_int.toLong(), std::numeric_limits<int64_t>::max());

    Scalar min_int(std::numeric_limits<int64_t>::min());
    EXPECT_EQ(min_int.toLong(), std::numeric_limits<int64_t>::min());

    Scalar max_uint(std::numeric_limits<uint64_t>::max());
    EXPECT_EQ(max_uint.toUInt64(), std::numeric_limits<uint64_t>::max());

    // 浮点边界值
    Scalar zero(0.0);
    Scalar neg_zero(-0.0);
    EXPECT_TRUE(zero.equal(neg_zero.toDouble()));// -0.0 和 0.0 在浮点数中被视为相等

    Scalar inf(std::numeric_limits<double>::infinity());
    Scalar neg_inf(-std::numeric_limits<double>::infinity());
    Scalar nan(std::numeric_limits<double>::quiet_NaN());

    // NaN不等于任何值，包括它自己
    EXPECT_FALSE(nan.equal(nan.toDouble()));
}
}// namespace