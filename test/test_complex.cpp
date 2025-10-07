//
// Created by richard on 10/7/25.
//
#include "utils/complex.h"

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
}


TEST(ComplexTest, ValueConstructor) {
    complex<float> c1(1.0f, 2.0f);
    EXPECT_FLOAT_EQ(c1.real(), 1.0f);
    EXPECT_FLOAT_EQ(c1.imag(), 2.0f);

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

}// namespace