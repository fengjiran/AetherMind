//
// Created by richard on 10/7/25.
//
#include "utils/complex.h"
#include "utils/float8_e4m3fn.h"

#include <cmath>
#include <gtest/gtest.h>

using namespace aethermind;

namespace {

// 基本测试用例
TEST(ComplexTest, DefaultConstructor) {
    complex<float> c1;
    EXPECT_FLOAT_EQ(c1.real(), 0.0f);
    EXPECT_FLOAT_EQ(c1.imag(), 0.0f);

    complex<double> c2;
    EXPECT_DOUBLE_EQ(c2.real(), 0.0);
    EXPECT_DOUBLE_EQ(c2.imag(), 0.0);

    // Float8_e4m3fn x = -1;
    // complex<float> c3{x};
}

TEST(ComplexTest, ValueConstructor) {
    complex<float> c1(1.0f, 2.0f);
    EXPECT_FLOAT_EQ(c1.real(), 1.0f);
    EXPECT_FLOAT_EQ(c1.imag(), 2.0f);
    std::cout << c1;

    complex<double> c2(3.0, 4.0);
    EXPECT_DOUBLE_EQ(c2.real(), 3.0);
    EXPECT_DOUBLE_EQ(c2.imag(), 4.0);

    // 测试单参数构造函数
    complex<float> c3(5.0f);
    EXPECT_FLOAT_EQ(c3.real(), 5.0f);
    EXPECT_FLOAT_EQ(c3.imag(), 0.0f);
}

TEST(ComplexTest, TypeConversionConstructor) {
    complex<float> c1(1.0f, 2.0f);
    complex<double> c2(c1);
    EXPECT_DOUBLE_EQ(c2.real(), 1.0);
    EXPECT_DOUBLE_EQ(c2.imag(), 2.0);

    complex<double> c3(3.0, 4.0);
    complex<float> c4(c3);
    EXPECT_FLOAT_EQ(c4.real(), 3.0f);
    EXPECT_FLOAT_EQ(c4.imag(), 4.0f);
}

TEST(ComplexTest, StdComplexConversion) {
    complex<float> c1(1.0f, 2.0f);
    auto std_c1 = static_cast<std::complex<float>>(c1);
    EXPECT_FLOAT_EQ(std_c1.real(), 1.0f);
    EXPECT_FLOAT_EQ(std_c1.imag(), 2.0f);

    std::complex<double> std_c2(3.0, 4.0);
    complex<double> c2(std_c2);
    EXPECT_DOUBLE_EQ(c2.real(), 3.0);
    EXPECT_DOUBLE_EQ(c2.imag(), 4.0);
}

TEST(ComplexTest, RealImagAccessors) {
    complex<float> c(1.0f, 2.0f);
    EXPECT_FLOAT_EQ(c.real(), 1.0f);
    EXPECT_FLOAT_EQ(c.imag(), 2.0f);

    c.real(3.0f);
    c.imag(4.0f);
    EXPECT_FLOAT_EQ(c.real(), 3.0f);
    EXPECT_FLOAT_EQ(c.imag(), 4.0f);
}

TEST(ComplexTest, ScalarAssignmentOperators) {
    complex<float> c;

    // 测试赋值运算符
    c = 5.0f;
    EXPECT_FLOAT_EQ(c.real(), 5.0f);
    EXPECT_FLOAT_EQ(c.imag(), 0.0f);

    // 测试复合赋值运算符
    c += 2.0f;
    EXPECT_FLOAT_EQ(c.real(), 7.0f);
    EXPECT_FLOAT_EQ(c.imag(), 0.0f);

    c -= 3.0f;
    EXPECT_FLOAT_EQ(c.real(), 4.0f);
    EXPECT_FLOAT_EQ(c.imag(), 0.0f);

    c = complex<float>(1.0f, 2.0f);
    c *= 2.0f;
    EXPECT_FLOAT_EQ(c.real(), 2.0f);
    EXPECT_FLOAT_EQ(c.imag(), 4.0f);

    c /= 2.0f;
    EXPECT_FLOAT_EQ(c.real(), 1.0f);
    EXPECT_FLOAT_EQ(c.imag(), 2.0f);
}

TEST(ComplexTest, ComplexAssignmentOperators) {
    complex<float> c1(1.0f, 2.0f);
    complex<float> c2(3.0f, 4.0f);

    // 测试赋值运算符
    c1 = c2;
    EXPECT_FLOAT_EQ(c1.real(), 3.0f);
    EXPECT_FLOAT_EQ(c1.imag(), 4.0f);

    // 测试复合赋值运算符
    c1 = complex<float>(1.0f, 2.0f);
    c1 += c2;
    EXPECT_FLOAT_EQ(c1.real(), 4.0f);
    EXPECT_FLOAT_EQ(c1.imag(), 6.0f);

    c1 -= c2;
    EXPECT_FLOAT_EQ(c1.real(), 1.0f);
    EXPECT_FLOAT_EQ(c1.imag(), 2.0f);

    c1 *= c2;
    // (1+2i)*(3+4i) = (3-8) + (4+6)i = -5 + 10i
    EXPECT_FLOAT_EQ(c1.real(), -5.0f);
    EXPECT_FLOAT_EQ(c1.imag(), 10.0f);

    // 测试除法
    complex<float> numerator(1.0f, 0.0f);
    complex<float> denominator(2.0f, 0.0f);
    numerator /= denominator;
    EXPECT_FLOAT_EQ(numerator.real(), 0.5f);
    EXPECT_FLOAT_EQ(numerator.imag(), 0.0f);
}


TEST(ComplexTest, UnaryOperators) {
    complex<float> c(1.0f, 2.0f);

    // 测试正号运算符
    complex<float> c_plus = +c;
    EXPECT_FLOAT_EQ(c_plus.real(), 1.0f);
    EXPECT_FLOAT_EQ(c_plus.imag(), 2.0f);

    // 测试负号运算符
    complex<float> c_minus = -c;
    EXPECT_FLOAT_EQ(c_minus.real(), -1.0f);
    EXPECT_FLOAT_EQ(c_minus.imag(), -2.0f);
}

TEST(ComplexTest, BinaryOperators) {
    complex<float> c1(1.0f, 2.0f);
    complex<float> c2(3.0f, 4.0f);

    // 测试加法
    complex<float> sum = c1 + c2;
    EXPECT_FLOAT_EQ(sum.real(), 4.0f);
    EXPECT_FLOAT_EQ(sum.imag(), 6.0f);

    // 测试与标量的加法
    sum = c1 + 2.0f;
    EXPECT_FLOAT_EQ(sum.real(), 3.0f);
    EXPECT_FLOAT_EQ(sum.imag(), 2.0f);

    sum = 2.0f + c1;
    EXPECT_FLOAT_EQ(sum.real(), 3.0f);
    EXPECT_FLOAT_EQ(sum.imag(), 2.0f);

    // 测试减法
    complex<float> diff = c1 - c2;
    EXPECT_FLOAT_EQ(diff.real(), -2.0f);
    EXPECT_FLOAT_EQ(diff.imag(), -2.0f);

    // 测试与标量的减法
    diff = c1 - 2.0f;
    EXPECT_FLOAT_EQ(diff.real(), -1.0f);
    EXPECT_FLOAT_EQ(diff.imag(), 2.0f);

    diff = 2.0f - c1;
    EXPECT_FLOAT_EQ(diff.real(), 1.0f);
    EXPECT_FLOAT_EQ(diff.imag(), -2.0f);

    // 测试乘法
    complex<float> product = c1 * c2;
    // (1+2i)*(3+4i) = (3-8) + (4+6)i = -5 + 10i
    EXPECT_FLOAT_EQ(product.real(), -5.0f);
    EXPECT_FLOAT_EQ(product.imag(), 10.0f);

    // 测试与标量的乘法
    product = c1 * 2.0f;
    EXPECT_FLOAT_EQ(product.real(), 2.0f);
    EXPECT_FLOAT_EQ(product.imag(), 4.0f);

    product = 2.0f * c1;
    EXPECT_FLOAT_EQ(product.real(), 2.0f);
    EXPECT_FLOAT_EQ(product.imag(), 4.0f);

    // 测试除法
    complex<float> quotient = c1 / c2;
    // (1+2i)/(3+4i) = (11+2i)/25 = 0.44 + 0.08i
    EXPECT_NEAR(quotient.real(), 0.44f, 1e-6f);
    EXPECT_NEAR(quotient.imag(), 0.08f, 1e-6f);

    // 测试与标量的除法
    quotient = c1 / 2.0f;
    EXPECT_FLOAT_EQ(quotient.real(), 0.5f);
    EXPECT_FLOAT_EQ(quotient.imag(), 1.0f);

    quotient = 2.0f / c1;
    // 2/(1+2i) = (2-4i)/5 = 0.4 - 0.8i
    EXPECT_FLOAT_EQ(quotient.real(), 0.4f);
    EXPECT_FLOAT_EQ(quotient.imag(), -0.8f);
}

TEST(ComplexTest, ComparisonOperators) {
    complex<float> c1(1.0f, 2.0f);
    complex<float> c2(1.0f, 2.0f);
    complex<float> c3(3.0f, 4.0f);

    // 测试相等运算符
    EXPECT_TRUE(c1 == c2);
    EXPECT_FALSE(c1 == c3);

    // 测试与标量的相等
    EXPECT_TRUE(c1 == complex<float>(1.0f, 2.0f));
    EXPECT_FALSE(c1 == 1.0f);
    complex<float> c4(5.0f);
    EXPECT_TRUE(c4 == 5.0f);

    // 测试不等运算符
    EXPECT_FALSE(c1 != c2);
    EXPECT_TRUE(c1 != c3);

    // 测试与标量的不等
    EXPECT_FALSE(c4 != 5.0f);
    EXPECT_TRUE(c1 != 5.0f);
}

TEST(ComplexTest, BooleanOperator) {
    complex<float> c1;
    complex<float> c2(1.0f, 0.0f);
    complex<float> c3(0.0f, 1.0f);
    complex<float> c4(1.0f, 1.0f);

    EXPECT_FALSE(static_cast<bool>(c1));
    EXPECT_TRUE(static_cast<bool>(c2));
    EXPECT_TRUE(static_cast<bool>(c3));
    EXPECT_TRUE(static_cast<bool>(c4));
}

TEST(ComplexTest, StdFunctions) {
    complex<float> c(3.0f, 4.0f);

    // 测试real函数
    EXPECT_FLOAT_EQ(std::real(c), 3.0f);

    // 测试imag函数
    EXPECT_FLOAT_EQ(std::imag(c), 4.0f);

    // 测试abs函数 (模长)
    EXPECT_FLOAT_EQ(std::abs(c), 5.0f);

    // 测试arg函数 (幅角)
    EXPECT_NEAR(std::arg(c), std::atan2(4.0f, 3.0f), 1e-6f);

    // 测试norm函数 (模长的平方)
    EXPECT_FLOAT_EQ(std::norm(c), 25.0f);

    // 测试conj函数 (共轭复数)
    complex<float> conj_c = std::conj(c);
    EXPECT_FLOAT_EQ(conj_c.real(), 3.0f);
    EXPECT_FLOAT_EQ(conj_c.imag(), -4.0f);
}

TEST(ComplexTest, PolarFunction) {
    // 测试极坐标转换函数
    complex<float> c = polar(5.0f, std::atan2(4.0f, 3.0f));
    EXPECT_NEAR(c.real(), 3.0f, 1e-6f);
    EXPECT_NEAR(c.imag(), 4.0f, 1e-6f);

    // 测试默认角度为0
    complex<float> c2 = polar(2.0f);
    EXPECT_FLOAT_EQ(c2.real(), 2.0f);
    EXPECT_FLOAT_EQ(c2.imag(), 0.0f);
}

// 针对Half类型的特化测试
TEST(ComplexHalfTest, Constructor) {
    Half real(1.0f);
    Half imag(2.0f);
    complex<Half> c1(real, imag);
    EXPECT_EQ(c1.real(), real);
    EXPECT_EQ(c1.imag(), imag);

    // 测试从complex<float>转换
    complex<float> c_float(3.0f, 4.0f);
    complex<Half> c2(c_float);
    EXPECT_EQ(c2.real(), Half(3.0f));
    EXPECT_EQ(c2.imag(), Half(4.0f));
}

TEST(ComplexHalfTest, ConversionToFloat) {
    complex<Half> c(Half(1.0f), Half(2.0f));
    complex<float> c_float = c;
    EXPECT_FLOAT_EQ(c_float.real(), 1.0f);
    EXPECT_FLOAT_EQ(c_float.imag(), 2.0f);
}

TEST(ComplexHalfTest, AssignmentOperators) {
    complex<Half> c1(Half(1.0f), Half(2.0f));
    complex<Half> c2(Half(3.0f), Half(4.0f));

    // 测试加法赋值
    c1 += c2;
    EXPECT_EQ(c1.real(), Half(4.0f));
    EXPECT_EQ(c1.imag(), Half(6.0f));

    // 测试减法赋值
    c1 -= c2;
    EXPECT_EQ(c1.real(), Half(1.0f));
    EXPECT_EQ(c1.imag(), Half(2.0f));

    // 测试乘法赋值
    c1 *= c2;
    // (1+2i)*(3+4i) = (3-8) + (4+6)i = -5 + 10i
    EXPECT_EQ(c1.real(), Half(-5.0f));
    EXPECT_EQ(c1.imag(), Half(10.0f));
}

TEST(ComplexTest, IntegralFloatingPointOperations) {
    complex<float> c(1.0f, 2.0f);
    int i = 3;

    // 测试整数和浮点数复数的混合运算
    auto result_add = c + i;
    EXPECT_FLOAT_EQ(result_add.real(), 4.0f);
    EXPECT_FLOAT_EQ(result_add.imag(), 2.0f);

    auto result_sub = c - i;
    EXPECT_FLOAT_EQ(result_sub.real(), -2.0f);
    EXPECT_FLOAT_EQ(result_sub.imag(), 2.0f);

    auto result_mul = c * i;
    EXPECT_FLOAT_EQ(result_mul.real(), 3.0f);
    EXPECT_FLOAT_EQ(result_mul.imag(), 6.0f);

    auto result_div = c / i;
    EXPECT_FLOAT_EQ(result_div.real(), 1.0f / 3.0f);
    EXPECT_FLOAT_EQ(result_div.imag(), 2.0f / 3.0f);

    // 测试反向操作数
    auto result_add_rev = i + c;
    EXPECT_FLOAT_EQ(result_add_rev.real(), 4.0f);
    EXPECT_FLOAT_EQ(result_add_rev.imag(), 2.0f);

    auto result_sub_rev = i - c;
    EXPECT_FLOAT_EQ(result_sub_rev.real(), 2.0f);
    EXPECT_FLOAT_EQ(result_sub_rev.imag(), -2.0f);

    auto result_mul_rev = i * c;
    EXPECT_FLOAT_EQ(result_mul_rev.real(), 3.0f);
    EXPECT_FLOAT_EQ(result_mul_rev.imag(), 6.0f);

    auto result_div_rev = i / c;
    // 3/(1+2i) = (3-6i)/5 = 0.6 - 1.2i
    EXPECT_FLOAT_EQ(result_div_rev.real(), 0.6f);
    EXPECT_FLOAT_EQ(result_div_rev.imag(), -1.2f);
}

TEST(ComplexTest, EdgeCases) {
    // 测试除以零的情况
    complex<float> c(1.0f, 0.0f);
    complex<float> zero(0.0f, 0.0f);
    complex<float> result = c / zero;

    // 除以零应该产生无穷大或NaN
    EXPECT_TRUE(std::isinf(result.real()) || std::isnan(result.real()));
    EXPECT_TRUE(std::isinf(result.imag()) || std::isnan(result.imag()));

    // 测试复数的零相乘
    complex<float> c1(0.0f, 1.0f);
    complex<float> c2(2.0f, 0.0f);
    complex<float> product = c1 * c2;
    EXPECT_FLOAT_EQ(product.real(), 0.0f);
    EXPECT_FLOAT_EQ(product.imag(), 2.0f);
}

// 测试复数指数函数
template<typename T>
void TestExp() {
    using Complex = complex<T>;
    using Real = Complex::value_type;

    // 测试实数输入
    Complex z1(Real(1.0), Real(0.0));
    Complex result1 = std::exp(z1);
    EXPECT_NEAR(result1.real(), std::exp(Real(1.0)), 1e-6);
    EXPECT_NEAR(result1.imag(), 0.0, 1e-6);

    // 测试纯虚数输入
    Complex z2(Real(0.0), static_cast<Real>(M_PI));
    Complex result2 = std::exp(z2);
    EXPECT_NEAR(result2.real(), -1.0, 1e-6);
    EXPECT_NEAR(result2.imag(), 0.0, 1e-6);

    // 测试一般复数输入
    Complex z3(Real(1.0), static_cast<Real>(M_PI_2));
    Complex result3 = std::exp(z3);
    EXPECT_NEAR(result3.real(), 0.0, 1e-6);
    EXPECT_NEAR(result3.imag(), std::exp(Real(1.0)), 1e-6);
}

TEST(ComplexTest, Exp) {
    TestExp<float>();
    TestExp<double>();
}

// 测试复数对数函数
template<typename T>
void TestLog() {
    using Complex = complex<T>;
    using Real = Complex::value_type;

    // 测试实数输入
    Complex z1(Real(2.71828), Real(0.0));
    Complex result1 = std::log(z1);
    EXPECT_NEAR(result1.real(), 1.0, 1e-5);
    EXPECT_NEAR(result1.imag(), 0.0, 1e-6);

    // 测试负数输入
    Complex z2(Real(-1.0), Real(0.0));
    Complex result2 = std::log(z2);
    EXPECT_NEAR(result2.real(), 0.0, 1e-6);
    EXPECT_NEAR(result2.imag(), static_cast<Real>(M_PI), 1e-6);

    // 测试一般复数输入
    Complex z3(Real(1.0), Real(1.0));
    Complex result3 = std::log(z3);
    EXPECT_NEAR(result3.real(), std::log(std::sqrt(2.0)), 1e-6);
    EXPECT_NEAR(result3.imag(), static_cast<Real>(M_PI_4), 1e-6);
}

TEST(ComplexTest, Log) {
    TestLog<float>();
    TestLog<double>();
}

// 测试复数常用对数函数
template<typename T>
void TestLog10() {
    using Complex = complex<T>;
    using Real = Complex::value_type;

    // 测试实数输入
    Complex z1(Real(10.0), Real(0.0));
    Complex result1 = std::log10(z1);
    EXPECT_NEAR(result1.real(), 1.0, 1e-6);
    EXPECT_NEAR(result1.imag(), 0.0, 1e-6);

    // 测试一般复数输入
    Complex z2(Real(10.0), Real(10.0));
    Complex result2 = std::log10(z2);
    Complex expected = std::log(z2) / static_cast<Real>(std::log(10.0));
    EXPECT_NEAR(result2.real(), expected.real(), 1e-6);
    EXPECT_NEAR(result2.imag(), expected.imag(), 1e-6);
}

TEST(ComplexTest, Log10) {
    TestLog10<float>();
    TestLog10<double>();
}

// 测试复数以2为底的对数函数
template<typename T>
void TestLog2() {
    using Complex = complex<T>;
    using Real = Complex::value_type;

    // 测试实数输入
    Complex z1(Real(2.0), Real(0.0));
    Complex result1 = std::log2(z1);
    EXPECT_NEAR(result1.real(), 1.0, 1e-6);
    EXPECT_NEAR(result1.imag(), 0.0, 1e-6);

    // 测试一般复数输入
    Complex z2(Real(2.0), Real(2.0));
    Complex result2 = std::log2(z2);
    Complex expected = std::log(z2) / static_cast<Real>(std::log(2.0));
    EXPECT_NEAR(result2.real(), expected.real(), 1e-6);
    EXPECT_NEAR(result2.imag(), expected.imag(), 1e-6);
}

TEST(ComplexTest, Log2) {
    TestLog2<float>();
    TestLog2<double>();
}

// 测试复数平方根函数
template<typename T>
void TestSqrt() {
    using Complex = complex<T>;
    using Real = Complex::value_type;

    // 测试实数输入
    Complex z1(Real(4.0), Real(0.0));
    Complex result1 = std::sqrt(z1);
    EXPECT_NEAR(result1.real(), 2.0, 1e-6);
    EXPECT_NEAR(result1.imag(), 0.0, 1e-6);

    // 测试负数输入
    Complex z2(Real(-4.0), Real(0.0));
    Complex result2 = std::sqrt(z2);
    EXPECT_NEAR(result2.real(), 0.0, 1e-6);
    EXPECT_NEAR(result2.imag(), 2.0, 1e-6);

    // 测试一般复数输入
    Complex z3(Real(3.0), Real(4.0));
    Complex result3 = complex_math::sqrt(z3);
    EXPECT_NEAR(result3.real(), 2.0, 1e-6);
    EXPECT_NEAR(result3.imag(), 1.0, 1e-6);
}

TEST(ComplexTest, Sqrt) {
    TestSqrt<float>();
    TestSqrt<double>();
}

// 测试复数幂函数
template<typename T>
void TestPow() {
    using Complex = complex<T>;
    using Real = Complex::value_type;

    // 测试复数的复数次幂
    Complex z1(Real(1.0), Real(0.0));
    Complex z2(Real(2.0), Real(0.0));
    Complex result1 = std::pow(z1, z2);
    EXPECT_NEAR(result1.real(), 1.0, 1e-6);
    EXPECT_NEAR(result1.imag(), 0.0, 1e-6);

    // 测试复数的实数次幂
    Complex z3(Real(0.0), Real(1.0));
    T exponent = 2.0;
    Complex result2 = std::pow(z3, exponent);
    EXPECT_NEAR(result2.real(), -1.0, 1e-6);
    EXPECT_NEAR(result2.imag(), 0.0, 1e-6);

    // 测试实数的复数次幂
    T base = 2.0;
    Complex z4(Real(1.0), Real(0.0));
    Complex result3 = std::pow(base, z4);
    EXPECT_NEAR(result3.real(), 2.0, 1e-6);
    EXPECT_NEAR(result3.imag(), 0.0, 1e-6);
}

TEST(ComplexTest, Pow) {
    TestPow<float>();
    TestPow<double>();
}

// 测试复数三角函数
template<typename T>
void TestTrigonometricFunctions() {
    using Complex = complex<T>;
    using Real = Complex::value_type;

    // 测试正弦函数
    Complex z1(Real(0.0), Real(0.0));
    Complex sin_result = std::sin(z1);
    EXPECT_NEAR(sin_result.real(), 0.0, 1e-6);
    EXPECT_NEAR(sin_result.imag(), 0.0, 1e-6);

    // 测试余弦函数
    Complex z2(Real(0.0), Real(0.0));
    Complex cos_result = std::cos(z2);
    EXPECT_NEAR(cos_result.real(), 1.0, 1e-6);
    EXPECT_NEAR(cos_result.imag(), 0.0, 1e-6);

    // 测试正切函数
    Complex z3(Real(0.0), Real(0.0));
    Complex tan_result = std::tan(z3);
    EXPECT_NEAR(tan_result.real(), 0.0, 1e-6);
    EXPECT_NEAR(tan_result.imag(), 0.0, 1e-6);

    // 测试反正弦函数
    Complex z4(Real(0.0), Real(0.0));
    Complex asin_result = std::asin(z4);
    EXPECT_NEAR(asin_result.real(), 0.0, 1e-6);
    EXPECT_NEAR(asin_result.imag(), 0.0, 1e-6);

    // 测试反余弦函数
    Complex z5(Real(1.0), Real(0.0));
    Complex acos_result = std::acos(z5);
    EXPECT_NEAR(acos_result.real(), 0.0, 1e-6);
    EXPECT_NEAR(acos_result.imag(), 0.0, 1e-6);

    // 测试反正切函数
    Complex z6(Real(0.0), Real(0.0));
    Complex atan_result = std::atan(z6);
    EXPECT_NEAR(atan_result.real(), 0.0, 1e-6);
    EXPECT_NEAR(atan_result.imag(), 0.0, 1e-6);
}

TEST(ComplexTest, TrigonometricFunctions) {
    TestTrigonometricFunctions<float>();
    TestTrigonometricFunctions<double>();
}

// 测试复数双曲函数
template<typename T>
void TestHyperbolicFunctions() {
    using Complex = complex<T>;
    using Real = Complex::value_type;

    // 测试双曲正弦函数
    Complex z1(Real(0.0), Real(0.0));
    Complex sinh_result = std::sinh(z1);
    EXPECT_NEAR(sinh_result.real(), 0.0, 1e-6);
    EXPECT_NEAR(sinh_result.imag(), 0.0, 1e-6);

    // 测试双曲余弦函数
    Complex z2(Real(0.0), Real(0.0));
    Complex cosh_result = std::cosh(z2);
    EXPECT_NEAR(cosh_result.real(), 1.0, 1e-6);
    EXPECT_NEAR(cosh_result.imag(), 0.0, 1e-6);

    // 测试双曲正切函数
    Complex z3(Real(0.0), Real(0.0));
    Complex tanh_result = std::tanh(z3);
    EXPECT_NEAR(tanh_result.real(), 0.0, 1e-6);
    EXPECT_NEAR(tanh_result.imag(), 0.0, 1e-6);

    // 测试反双曲正弦函数
    Complex z4(Real(0.0), Real(0.0));
    Complex asinh_result = std::asinh(z4);
    EXPECT_NEAR(asinh_result.real(), 0.0, 1e-6);
    EXPECT_NEAR(asinh_result.imag(), 0.0, 1e-6);

    // 测试反双曲余弦函数
    Complex z5(Real(1.0), Real(0.0));
    Complex acosh_result = std::acosh(z5);
    EXPECT_NEAR(acosh_result.real(), 0.0, 1e-6);
    EXPECT_NEAR(acosh_result.imag(), 0.0, 1e-6);

    // 测试反双曲正切函数
    Complex z6(Real(0.0), Real(0.0));
    Complex atanh_result = std::atanh(z6);
    EXPECT_NEAR(atanh_result.real(), 0.0, 1e-6);
    EXPECT_NEAR(atanh_result.imag(), 0.0, 1e-6);
}

TEST(ComplexTest, HyperbolicFunctions) {
    TestHyperbolicFunctions<float>();
    TestHyperbolicFunctions<double>();
}

// 测试复数函数的边界情况
template<typename T>
void TestComplexFunctionEdgeCases() {
    using Complex = complex<T>;
    using Real = Complex::value_type;

    // 测试无穷大输入
    Complex inf(std::numeric_limits<Real>::infinity(), 0.0);
    Complex exp_inf = std::exp(inf);
    EXPECT_TRUE(std::isinf(exp_inf.real()) || std::isnan(exp_inf.real()));

    // 测试NaN输入
    Complex nan(std::numeric_limits<Real>::quiet_NaN(), 0.0);
    Complex log_nan = std::log(nan);
    EXPECT_TRUE(std::isnan(log_nan.real()));
    EXPECT_TRUE(std::isnan(log_nan.imag()));

    // 测试零输入的特殊情况
    Complex zero(0.0, 0.0);
    Complex log_zero = std::log(zero);
    EXPECT_TRUE(std::isinf(log_zero.real()));
    EXPECT_TRUE(log_zero.real() < 0);// log(0) 应该是负无穷
}

TEST(ComplexTest, FunctionEdgeCases) {
    TestComplexFunctionEdgeCases<float>();
    TestComplexFunctionEdgeCases<double>();
}

// 测试跨类型的复数函数
template<typename T1, typename T2>
void TestCrossTypeComplexFunctions() {
    using Complex1 = complex<T1>;
    using Complex2 = complex<T2>;

    // 测试跨类型的pow函数
    Complex1 z1(1.0f, 1.0f);
    Complex2 z2(2.0, 0.0);
    auto result = std::pow(z1, z2);

    // 验证结果类型
    static_assert(std::is_same_v<decltype(result), complex<decltype(T1() * T2())>>,
                  "pow return type is incorrect");

    // 验证结果值
    Complex1 expected = std::pow(z1, static_cast<T1>(2.0f));
    EXPECT_NEAR(static_cast<float>(result.real()), expected.real(), 1e-6);
    EXPECT_NEAR(static_cast<float>(result.imag()), expected.imag(), 1e-6);
}

TEST(ComplexTest, CrossTypeFunctions) {
    TestCrossTypeComplexFunctions<float, double>();
    TestCrossTypeComplexFunctions<double, float>();
}

}// namespace