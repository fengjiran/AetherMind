/// @file test_half.cpp
/// @brief Half value API, arithmetic, comparisons, and numeric limits.
#include "aethermind/dtypes/half.h"

#include <gtest/gtest.h>

#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <type_traits>

namespace {
using namespace aethermind;

TEST(HalfTest, ConstructorAndConversion) {
    Half h1;
    EXPECT_EQ(h1.bits(), 0);

    // from_bits stores the raw pattern without conversion.
    Half h2(0x3C00, Half::from_bits()); // 1.0 in half precision
    EXPECT_EQ(h2.bits(), 0x3C00);

    Half h3(1.0f);
    EXPECT_EQ(static_cast<float>(h3), 1.0f);

    Half h4(0.0f);
    EXPECT_EQ(static_cast<float>(h4), 0.0f);

    // Negative zero must preserve sign.
    Half h5(-0.0f);
    EXPECT_EQ(static_cast<float>(h5), -0.0f);

    Half h6(std::numeric_limits<float>::infinity());
    EXPECT_TRUE(std::isinf(static_cast<float>(h6)));
}

TEST(HalfTest, ArithmeticOperations) {
    Half a(1.5f);
    Half b(2.5f);

    EXPECT_FLOAT_EQ(a + b, 4.0f);
    EXPECT_FLOAT_EQ(a - b, -1.0f);
    EXPECT_FLOAT_EQ(a * b, 3.75f);
    EXPECT_FLOAT_EQ(a / b, 0.60009766f);

    Half c(1.0f);
    c += a;
    EXPECT_FLOAT_EQ(c, 2.5f);

    EXPECT_FLOAT_EQ(-a, -1.5f);
}

TEST(HalfTest, MixedTypeArithmetic) {
    Half h(2.0f);

    static_assert(std::is_same_v<decltype(h + 3.0f), float>);
    static_assert(std::is_same_v<decltype(3.0f + h), float>);
    EXPECT_FLOAT_EQ(h + 3.0f, 5.0f);
    EXPECT_FLOAT_EQ(3.0f + h, 5.0f);

    static_assert(std::is_same_v<decltype(h + 3.0), double>);
    static_assert(std::is_same_v<decltype(3.0 + h), double>);
    EXPECT_DOUBLE_EQ(h + 3.0, 5.0);
    EXPECT_DOUBLE_EQ(3.0 + h, 5.0);

    // With int (promoted to float, matching Half-with-float semantics).
    static_assert(std::is_same_v<decltype(h + 3), float>);
    static_assert(std::is_same_v<decltype(3 + h), float>);
    EXPECT_FLOAT_EQ(h + 3, 5.0f);
    EXPECT_FLOAT_EQ(3 + h, 5.0f);

    // With int64_t (promoted to float).
    static_assert(std::is_same_v<decltype(h + int64_t(3)), float>);
    static_assert(std::is_same_v<decltype(int64_t(3) + h), float>);
    EXPECT_FLOAT_EQ(h + int64_t(3), 5.0f);
    EXPECT_FLOAT_EQ(int64_t(3) + h, 5.0f);
}

TEST(HalfTest, NumericLimits) {
    EXPECT_EQ(std::numeric_limits<Half>::round_style, std::round_to_nearest);
    EXPECT_EQ(std::numeric_limits<Half>::has_denorm, std::denorm_present);
    EXPECT_FALSE(std::numeric_limits<Half>::is_iec559);
    EXPECT_EQ(std::numeric_limits<Half>::min().bits(), 0x0400);
    EXPECT_EQ(std::numeric_limits<Half>::max().bits(), 0x7BFF);
    EXPECT_EQ(std::numeric_limits<Half>::lowest().bits(), 0xFBFF);
    EXPECT_EQ(std::numeric_limits<Half>::epsilon().bits(), 0x1400);
    EXPECT_EQ(std::numeric_limits<Half>::infinity().bits(), 0x7C00);
    EXPECT_EQ(std::numeric_limits<Half>::quiet_NaN().bits(), 0x7E00);
    EXPECT_EQ(std::numeric_limits<Half>::signaling_NaN().bits(), 0x7D00);
    EXPECT_EQ(std::numeric_limits<Half>::denorm_min().bits(), 0x0001);
}

TEST(HalfTest, EdgeCases) {
    // Denormalized values must be positive and non-zero.
    Half denorm(std::numeric_limits<Half>::denorm_min());
    EXPECT_GT(static_cast<float>(denorm), 0.0f);

    // Quiet NaN must test as NaN.
    Half nan = std::numeric_limits<Half>::quiet_NaN();
    EXPECT_TRUE(std::isnan(static_cast<float>(nan)));

    // Overflow saturates to infinity.
    Half large(1.0e8f);
    EXPECT_TRUE(std::isinf(static_cast<float>(large)));
}

TEST(HalfTest, OutputOperator) {
    Half h(3.14f);
    std::ostringstream oss;
    oss << h;
    EXPECT_FALSE(oss.str().empty());
}

TEST(HalfTest, MemoryLayout) {
    // Half wraps a single uint16_t with no padding; size and alignment must match.
    EXPECT_EQ(sizeof(Half), 2u);
    EXPECT_EQ(alignof(Half), 2u);
}

TEST(HalfTest, FromBitsConstructor) {
    // from_bits stores the raw pattern verbatim — no conversion, no canonicalization.
    Half pos_inf(0x7C00, Half::from_bits());
    EXPECT_TRUE(std::isinf(static_cast<float>(pos_inf)));
    EXPECT_GT(static_cast<float>(pos_inf), 0.0f);

    Half neg_inf(0xFC00, Half::from_bits());
    EXPECT_TRUE(std::isinf(static_cast<float>(neg_inf)));
    EXPECT_LT(static_cast<float>(neg_inf), 0.0f);

    Half qnan(0x7E00, Half::from_bits());
    EXPECT_TRUE(std::isnan(static_cast<float>(qnan)));

    // Smallest positive denormal must round-trip through float as a positive subnormal value.
    Half min_denormal(0x0001, Half::from_bits());
    float as_float = static_cast<float>(min_denormal);
    EXPECT_GT(as_float, 0.0f);
    EXPECT_LT(as_float, 1e-7f);
}

TEST(HalfTest, AllCompoundAssignments) {
    Half a(4.0f);
    a -= Half(1.5f);
    EXPECT_FLOAT_EQ(a, 2.5f);

    Half b(2.0f);
    b *= Half(3.0f);
    EXPECT_FLOAT_EQ(b, 6.0f);

    Half c(8.0f);
    c /= Half(2.0f);
    EXPECT_FLOAT_EQ(c, 4.0f);
}

TEST(HalfTest, FloatCompoundAssignments) {
    // float& OP= Half: lhs is a float reference; result accumulates in float precision.
    float a = 4.0f;
    a += Half(1.5f);
    EXPECT_FLOAT_EQ(a, 5.5f);

    float b = 4.0f;
    b -= Half(1.5f);
    EXPECT_FLOAT_EQ(b, 2.5f);

    float c = 2.0f;
    c *= Half(3.0f);
    EXPECT_FLOAT_EQ(c, 6.0f);

    float d = 8.0f;
    d /= Half(2.0f);
    EXPECT_FLOAT_EQ(d, 4.0f);
}

TEST(HalfTest, MixedTypeCompoundAssignmentsMatchBinaryOperators) {
    // 65536.0f overflows Half; the compound form must evaluate with the wider
    // operand instead of pre-narrowing the rhs (which would yield +inf).
    Half a(-65504.0f);
    a += 65536.0f;
    EXPECT_EQ(a.bits(), 0x5000); // 32.0

    Half b(-65504.0f);
    b = b + 65536.0f;
    EXPECT_EQ(b.bits(), a.bits());

    // Same contract for int, int64_t, and double right operands.
    Half c(-65504.0f);
    c += 65536;
    EXPECT_EQ(c.bits(), 0x5000);

    Half d(-65504.0f);
    d += int64_t{65536};
    EXPECT_EQ(d.bits(), 0x5000);

    Half e(-65504.0f);
    e += 65536.0;
    EXPECT_EQ(e.bits(), 0x5000);

    // Narrowing happens once, on write-back, matching the binary forms.
    Half f(1.0f);
    f /= 3.0f;
    EXPECT_EQ(f.bits(), Half(1.0f / 3.0f).bits());

    Half g(10.0f);
    g -= 0.5f;
    EXPECT_EQ(g.bits(), Half(9.5f).bits());

    Half h(10.0f);
    h *= 2.0;
    EXPECT_EQ(h.bits(), Half(20.0f).bits());

    // Return type is Half& for each mixed-type form.
    static_assert(std::is_same_v<decltype(a += 1.5f), Half&>);
    static_assert(std::is_same_v<decltype(a -= 1.5), Half&>);
    static_assert(std::is_same_v<decltype(a *= int64_t{2}), Half&>);
    static_assert(std::is_same_v<decltype(a /= 2), Half&>);
}

TEST(HalfTest, MixedTypeSubtraction) {
    Half h(5.0f);

    EXPECT_FLOAT_EQ(h - 2.0f, 3.0f);
    EXPECT_FLOAT_EQ(7.0f - h, 2.0f);
    EXPECT_DOUBLE_EQ(h - 2.0, 3.0);
    EXPECT_DOUBLE_EQ(7.0 - h, 2.0);

    static_assert(std::is_same_v<decltype(h - 2), float>);
    static_assert(std::is_same_v<decltype(7 - h), float>);
    EXPECT_FLOAT_EQ(h - 2, 3.0f);
    EXPECT_FLOAT_EQ(7 - h, 2.0f);

    static_assert(std::is_same_v<decltype(h - int64_t(2)), float>);
    static_assert(std::is_same_v<decltype(int64_t(7) - h), float>);
    EXPECT_FLOAT_EQ(h - int64_t(2), 3.0f);
    EXPECT_FLOAT_EQ(int64_t(7) - h, 2.0f);
}

TEST(HalfTest, MixedTypeMultiplication) {
    Half h(3.0f);

    EXPECT_FLOAT_EQ(h * 2.0f, 6.0f);
    EXPECT_FLOAT_EQ(2.0f * h, 6.0f);
    EXPECT_DOUBLE_EQ(h * 2.0, 6.0);
    EXPECT_DOUBLE_EQ(2.0 * h, 6.0);

    static_assert(std::is_same_v<decltype(h * 2), float>);
    static_assert(std::is_same_v<decltype(2 * h), float>);
    EXPECT_FLOAT_EQ(h * 2, 6.0f);
    EXPECT_FLOAT_EQ(2 * h, 6.0f);

    static_assert(std::is_same_v<decltype(h * int64_t(2)), float>);
    static_assert(std::is_same_v<decltype(int64_t(2) * h), float>);
    EXPECT_FLOAT_EQ(h * int64_t(2), 6.0f);
    EXPECT_FLOAT_EQ(int64_t(2) * h, 6.0f);
}

TEST(HalfTest, MixedTypeDivision) {
    Half h(6.0f);

    EXPECT_FLOAT_EQ(h / 2.0f, 3.0f);
    EXPECT_FLOAT_EQ(12.0f / h, 2.0f);
    EXPECT_DOUBLE_EQ(h / 2.0, 3.0);
    EXPECT_DOUBLE_EQ(12.0 / h, 2.0);

    static_assert(std::is_same_v<decltype(h / 2), float>);
    static_assert(std::is_same_v<decltype(12 / h), float>);
    EXPECT_FLOAT_EQ(h / 2, 3.0f);
    EXPECT_FLOAT_EQ(12 / h, 2.0f);

    static_assert(std::is_same_v<decltype(h / int64_t(2)), float>);
    static_assert(std::is_same_v<decltype(int64_t(12) / h), float>);
    EXPECT_FLOAT_EQ(h / int64_t(2), 3.0f);
    EXPECT_FLOAT_EQ(int64_t(12) / h, 2.0f);
}

TEST(HalfTest, NaNPropagation) {
    // NaN must propagate through every arithmetic operation; result is NaN regardless of operand order.
    Half nan = std::numeric_limits<Half>::quiet_NaN();
    Half one(1.0f);

    EXPECT_TRUE(std::isnan(static_cast<float>(nan + one)));
    EXPECT_TRUE(std::isnan(static_cast<float>(one + nan)));
    EXPECT_TRUE(std::isnan(static_cast<float>(nan - one)));
    EXPECT_TRUE(std::isnan(static_cast<float>(nan * one)));
    EXPECT_TRUE(std::isnan(static_cast<float>(nan / one)));

    Half acc(1.0f);
    acc += nan;
    EXPECT_TRUE(std::isnan(static_cast<float>(acc)));
}

TEST(HalfTest, InfinityArithmetic) {
    Half pos_inf = std::numeric_limits<Half>::infinity();
    Half neg_inf(-std::numeric_limits<float>::infinity());
    Half one(1.0f);

    // inf ± finite stays inf with the same sign.
    EXPECT_TRUE(std::isinf(static_cast<float>(pos_inf + one)));
    EXPECT_GT(static_cast<float>(pos_inf + one), 0.0f);

    EXPECT_TRUE(std::isinf(static_cast<float>(neg_inf - one)));
    EXPECT_LT(static_cast<float>(neg_inf - one), 0.0f);

    // inf - inf is NaN (indeterminate form).
    EXPECT_TRUE(std::isnan(static_cast<float>(pos_inf - pos_inf)));

    // 1 / 0 is +inf; -1 / 0 is -inf.
    Half zero(0.0f);
    EXPECT_TRUE(std::isinf(static_cast<float>(one / zero)));
    EXPECT_GT(static_cast<float>(one / zero), 0.0f);

    Half neg_one(-1.0f);
    EXPECT_TRUE(std::isinf(static_cast<float>(neg_one / zero)));
    EXPECT_LT(static_cast<float>(neg_one / zero), 0.0f);

    // 0 / 0 is NaN.
    EXPECT_TRUE(std::isnan(static_cast<float>(zero / zero)));
}

TEST(HalfTest, Comparisons) {
    // Comparisons go through implicit conversion to float and follow IEEE 754 ordering.
    Half a(1.5f);
    Half b(2.5f);
    Half c(1.5f);

    EXPECT_TRUE(a == c);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_FALSE(a != c);

    EXPECT_TRUE(a < b);
    EXPECT_FALSE(b < a);
    EXPECT_TRUE(b > a);
    EXPECT_FALSE(a > b);

    EXPECT_TRUE(a <= c);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(a >= c);
    EXPECT_TRUE(b >= a);

    // +0.0 == -0.0 per IEEE 754.
    Half pos_zero(0.0f);
    Half neg_zero(-0.0f);
    EXPECT_TRUE(pos_zero == neg_zero);
}

TEST(HalfTest, NaNComparisons) {
    // NaN is unordered: every comparison with NaN returns false except !=, which returns true.
    Half nan = std::numeric_limits<Half>::quiet_NaN();
    Half one(1.0f);

    EXPECT_FALSE(nan == nan);
    EXPECT_FALSE(nan == one);
    EXPECT_FALSE(one == nan);

    EXPECT_TRUE(nan != nan);
    EXPECT_TRUE(nan != one);

    EXPECT_FALSE(nan < one);
    EXPECT_FALSE(nan > one);
    EXPECT_FALSE(nan <= one);
    EXPECT_FALSE(nan >= one);
}

TEST(HalfTest, SpaceshipOperator) {
    Half a(1.5f);
    Half b(2.5f);
    Half c(1.5f);

    EXPECT_EQ(a <=> b, std::partial_ordering::less);
    EXPECT_EQ(b <=> a, std::partial_ordering::greater);
    EXPECT_EQ(a <=> c, std::partial_ordering::equivalent);

    // +0.0 and -0.0 are equivalent under IEEE 754 value comparison.
    Half pos_zero(0.0f);
    Half neg_zero(-0.0f);
    EXPECT_EQ(pos_zero <=> neg_zero, std::partial_ordering::equivalent);

    // NaN is unordered, including relative to itself.
    Half nan = std::numeric_limits<Half>::quiet_NaN();
    EXPECT_EQ(nan <=> a, std::partial_ordering::unordered);
    EXPECT_EQ(a <=> nan, std::partial_ordering::unordered);
    EXPECT_EQ(nan <=> nan, std::partial_ordering::unordered);

    // Ordering-derived operators still work as before.
    EXPECT_TRUE(a < b);
    EXPECT_TRUE(a <= c);
    EXPECT_TRUE(b > a);
    EXPECT_TRUE(b >= a);
}

template<typename T>
class HalfIntegerArithmetic : public ::testing::Test {};

using HalfIntegerTypes = ::testing::Types<bool, char, signed char, unsigned char,
                                          wchar_t, char8_t, char16_t, char32_t,
                                          short, unsigned short, int, unsigned int,
                                          long, unsigned long, long long, unsigned long long,
                                          uint32_t, uint64_t, size_t, int64_t>;
TYPED_TEST_SUITE(HalfIntegerArithmetic, HalfIntegerTypes);

TYPED_TEST(HalfIntegerArithmetic, BinaryOperatorsReturnFloatInBothOperandOrders) {
    const Half value(1.5f);
    const TypeParam integer = 1;
    static_assert(std::is_same_v<decltype(value + integer), float>);
    static_assert(std::is_same_v<decltype(integer + value), float>);
    static_assert(std::is_same_v<decltype(value - integer), float>);
    static_assert(std::is_same_v<decltype(integer - value), float>);
    static_assert(std::is_same_v<decltype(value * integer), float>);
    static_assert(std::is_same_v<decltype(integer * value), float>);
    static_assert(std::is_same_v<decltype(value / integer), float>);
    static_assert(std::is_same_v<decltype(integer / value), float>);

    EXPECT_FLOAT_EQ(value + integer, 2.5f);
    EXPECT_FLOAT_EQ(integer + value, 2.5f);
    EXPECT_FLOAT_EQ(value - integer, 0.5f);
    EXPECT_FLOAT_EQ(integer - value, -0.5f);
    EXPECT_FLOAT_EQ(value * integer, 1.5f);
    EXPECT_FLOAT_EQ(integer * value, 1.5f);
    EXPECT_FLOAT_EQ(value / integer, 1.5f);
    EXPECT_FLOAT_EQ(integer / value, 2.0f / 3.0f);
}

TYPED_TEST(HalfIntegerArithmetic, CompoundAssignmentsReturnTheLeftOperand) {
    const TypeParam integer = 1;
    Half sum(1.5f);
    Half difference(1.5f);
    Half product(1.5f);
    Half quotient(1.5f);
    static_assert(std::is_same_v<decltype(sum += integer), Half&>);
    static_assert(std::is_same_v<decltype(difference -= integer), Half&>);
    static_assert(std::is_same_v<decltype(product *= integer), Half&>);
    static_assert(std::is_same_v<decltype(quotient /= integer), Half&>);

    EXPECT_EQ(&(sum += integer), &sum);
    EXPECT_EQ(&(difference -= integer), &difference);
    EXPECT_EQ(&(product *= integer), &product);
    EXPECT_EQ(&(quotient /= integer), &quotient);
    EXPECT_EQ(sum.bits(), 0x4100U);        // 2.5
    EXPECT_EQ(difference.bits(), 0x3800U); // 0.5
    EXPECT_EQ(product.bits(), 0x3E00U);    // 1.5
    EXPECT_EQ(quotient.bits(), 0x3E00U);
}

template<typename T>
class HalfWideIntegerArithmetic : public ::testing::Test {};

using HalfWideIntegerTypes = ::testing::Types<int, unsigned int, long long, unsigned long long,
                                              int64_t, uint32_t, uint64_t, size_t>;
TYPED_TEST_SUITE(HalfWideIntegerArithmetic, HalfWideIntegerTypes);

TYPED_TEST(HalfWideIntegerArithmetic, BinaryOperatorsDoNotNarrowTheIntegerToHalf) {
    const TypeParam integer = 65536;
    const Half positive(65504.0f);
    const Half negative(-65504.0f);
    const Half half(0.5f);

    EXPECT_FLOAT_EQ(negative + integer, 32.0f);
    EXPECT_FLOAT_EQ(integer + negative, 32.0f);
    EXPECT_FLOAT_EQ(positive - integer, -32.0f);
    EXPECT_FLOAT_EQ(integer - positive, 32.0f);
    EXPECT_FLOAT_EQ(half * integer, 32768.0f);
    EXPECT_FLOAT_EQ(integer * half, 32768.0f);
    EXPECT_FLOAT_EQ(positive / integer, 0.99951171875f);
    EXPECT_FLOAT_EQ(integer / half, 131072.0f);
}

TYPED_TEST(HalfWideIntegerArithmetic, CompoundAssignmentsNarrowOnlyOnWriteBack) {
    const TypeParam integer = 65536;
    Half sum(-65504.0f);
    Half difference(65504.0f);
    Half product(0.5f);
    Half quotient(65504.0f);

    sum += integer;
    difference -= integer;
    product *= integer;
    quotient /= integer;
    EXPECT_EQ(sum.bits(), 0x5000U);        // 32
    EXPECT_EQ(difference.bits(), 0xD000U); // -32
    EXPECT_EQ(product.bits(), 0x7800U);    // 32768
    EXPECT_EQ(quotient.bits(), 0x3BFFU);   // 0.99951171875
}

TEST(HalfIntegerArithmetic, UnsignedMaximumKeepsItsMagnitudeAndSign) {
    constexpr uint64_t integer = std::numeric_limits<uint64_t>::max();
    const Half one(1.0f);

    // UINT64_MAX rounds to 2^64 in float; it must not first pass through a
    // signed integer or overflow to a Half infinity.
    EXPECT_FLOAT_EQ(one + integer, 0x1p64f);
    EXPECT_FLOAT_EQ(integer + one, 0x1p64f);
    EXPECT_FLOAT_EQ(one - integer, -0x1p64f);
    EXPECT_FLOAT_EQ(integer - one, 0x1p64f);
    EXPECT_FLOAT_EQ(one * integer, 0x1p64f);
    EXPECT_FLOAT_EQ(integer * one, 0x1p64f);
    EXPECT_FLOAT_EQ(one / integer, 0x1p-64f);
    EXPECT_FLOAT_EQ(integer / one, 0x1p64f);
}

} // namespace
