#include "aethermind/dtypes/bfloat16.h"

#include <bitset>
#include <cmath>
#include <gtest/gtest.h>
#include <type_traits>
#include <utility>

namespace {
using namespace aethermind;

TEST(BFloat16Test, DefaultConstructor) {
    BFloat16 val;
    EXPECT_EQ(val.x, 0);
    EXPECT_FLOAT_EQ(val, 0.0f);
}

TEST(BFloat16Test, FromBitsConstructor) {
    BFloat16 val1(0x3F80, BFloat16::from_bits()); // 1.0
    EXPECT_EQ(val1.x, 0x3F80);
    EXPECT_FLOAT_EQ(val1, 1.0f);

    BFloat16 val2(0xBF80, BFloat16::from_bits()); // -1.0
    EXPECT_EQ(val2.x, 0xBF80);
    EXPECT_FLOAT_EQ(val2, -1.0f);
}

TEST(BFloat16Test, FloatConversion) {
    constexpr float test_values[] = {
            0.0f, 1.0f, -1.0f, 2.0f, 0.5f,
            0.3333333333f, 100.0f, -100.0f,
            std::numeric_limits<float>::max() / 2, // Large value that still fits BF16.
            std::numeric_limits<float>::min() * 2  // Smallest normal float doubled; still normal in BF16.
    };

    for (float f: test_values) {
        BFloat16 bf16 = f;
        float converted_back = bf16;

        // BF16 keeps only 8 mantissa bits, so exact equality is not expected.
        // Round-to-nearest keeps the relative error below roughly 0.5%; the
        // assertion allows 1%.
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

TEST(BFloat16Test, ConversionOperatorsAreNoexcept) {
    static_assert(noexcept(BFloat16(1.0f)));
    static_assert(noexcept(static_cast<float>(std::declval<const BFloat16&>())));
}

TEST(BFloat16Test, EdgeValues) {
    BFloat16 zero_pos(0.0f);
    BFloat16 zero_neg(-0.0f);
    EXPECT_EQ(zero_pos.x, 0x0000);
    EXPECT_EQ(zero_neg.x, 0x8000);
    EXPECT_FLOAT_EQ(zero_pos, 0.0f);
    EXPECT_FLOAT_EQ(zero_neg, -0.0f);

    BFloat16 inf_pos(std::numeric_limits<float>::infinity());
    BFloat16 inf_neg(-std::numeric_limits<float>::infinity());
    EXPECT_EQ(inf_pos.x, 0x7F80);
    EXPECT_EQ(inf_neg.x, 0xFF80);
    EXPECT_TRUE(std::isinf(inf_pos));
    EXPECT_TRUE(std::isinf(inf_neg));

    BFloat16 nan_pos(std::numeric_limits<float>::quiet_NaN());
    BFloat16 nan_neg(-std::numeric_limits<float>::quiet_NaN());
    EXPECT_TRUE(std::isnan(nan_pos));
    EXPECT_TRUE(std::isnan(nan_neg));

    BFloat16 min_norm = std::numeric_limits<BFloat16>::min();
    BFloat16 max_norm = std::numeric_limits<BFloat16>::max();
    BFloat16 lowest = std::numeric_limits<BFloat16>::lowest();

    EXPECT_EQ(min_norm.x, 0x0080);
    EXPECT_EQ(max_norm.x, 0x7F7F);
    EXPECT_EQ(lowest.x, 0xFF7F);
}

TEST(BFloat16Test, ArithmeticOperations) {
    BFloat16 a(2.0f);
    BFloat16 b(3.0f);

    EXPECT_FLOAT_EQ(a + b, 5.0f);

    EXPECT_FLOAT_EQ(a - b, -1.0f);
    EXPECT_FLOAT_EQ(b - a, 1.0f);

    EXPECT_FLOAT_EQ(a * b, 6.0f);

    EXPECT_FLOAT_EQ(a / b, 0.66796875f);
    EXPECT_FLOAT_EQ(b / a, 1.5f);

    EXPECT_FLOAT_EQ(-a, -2.0f);

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
    BFloat16 a(2.0f);

    EXPECT_FLOAT_EQ(a + 3.0f, 5.0f);
    EXPECT_FLOAT_EQ(3.0f + a, 5.0f);
    EXPECT_FLOAT_EQ(a - 3.0f, -1.0f);
    EXPECT_FLOAT_EQ(3.0f - a, 1.0f);
    EXPECT_FLOAT_EQ(a * 3.0f, 6.0f);
    EXPECT_FLOAT_EQ(3.0f * a, 6.0f);
    EXPECT_FLOAT_EQ(a / 3.0f, 2.0f / 3.0f);
    EXPECT_FLOAT_EQ(3.0f / a, 1.5f);

    BFloat16 result = a + 3;
    EXPECT_FLOAT_EQ(result, 5.0f);

    result = 3 + a;
    EXPECT_FLOAT_EQ(result, 5.0f);

    result = a - 3;
    EXPECT_FLOAT_EQ(result, -1.0f);

    result = 3 - a;
    EXPECT_FLOAT_EQ(result, 1.0f);

    double d = 3.14159;
    EXPECT_DOUBLE_EQ(a + d, 5.14159);
    EXPECT_DOUBLE_EQ(d + a, 5.14159);
}

template<typename T>
class BFloat16IntegerArithmetic : public ::testing::Test {};

using BFloat16IntegerTypes = ::testing::Types<int, int64_t>;
TYPED_TEST_SUITE(BFloat16IntegerArithmetic, BFloat16IntegerTypes);

TYPED_TEST(BFloat16IntegerArithmetic, AdditionPreservesCancellation) {
    const TypeParam integer = 257;
    const BFloat16 positive{256.0f};
    const BFloat16 negative{-256.0f};
    static_assert(std::is_same_v<decltype(negative + integer), BFloat16>);
    static_assert(std::is_same_v<decltype(integer + negative), BFloat16>);

    EXPECT_EQ((negative + integer).x, 0x3f80U); // 1
    EXPECT_EQ((integer + negative).x, 0x3f80U);
    EXPECT_EQ((positive + -integer).x, 0xbf80U); // -1
    EXPECT_EQ((-integer + positive).x, 0xbf80U);
}

TYPED_TEST(BFloat16IntegerArithmetic, SubtractionPreservesCancellation) {
    const TypeParam integer = 257;
    const BFloat16 positive{256.0f};
    const BFloat16 negative{-256.0f};
    static_assert(std::is_same_v<decltype(positive - integer), BFloat16>);
    static_assert(std::is_same_v<decltype(integer - positive), BFloat16>);

    EXPECT_EQ((positive - integer).x, 0xbf80U); // -1
    EXPECT_EQ((integer - positive).x, 0x3f80U); // 1
    EXPECT_EQ((negative - -integer).x, 0x3f80U);
    EXPECT_EQ((-integer - negative).x, 0xbf80U);
}

TYPED_TEST(BFloat16IntegerArithmetic, MultiplicationRoundsOnlyTheProduct) {
    const TypeParam integer = 257;
    const BFloat16 value{1.5f};
    static_assert(std::is_same_v<decltype(value * integer), BFloat16>);
    static_assert(std::is_same_v<decltype(integer * value), BFloat16>);

    // 1.5 * 257 = 385.5, which rounds to 386 rather than 384.
    EXPECT_EQ((value * integer).x, 0x43c1U);
    EXPECT_EQ((integer * value).x, 0x43c1U);
    EXPECT_EQ((value * -integer).x, 0xc3c1U);
    EXPECT_EQ((-integer * value).x, 0xc3c1U);
}

TYPED_TEST(BFloat16IntegerArithmetic, DivisionRoundsOnlyTheQuotient) {
    const TypeParam integer = 257;
    const BFloat16 numerator{1.5f};
    const BFloat16 denominator{1.75f};
    static_assert(std::is_same_v<decltype(numerator / integer), BFloat16>);
    static_assert(std::is_same_v<decltype(integer / denominator), BFloat16>);

    // 1.5 / 257 rounds to 0.005828857421875; 257 / 1.75 rounds to 147.
    EXPECT_EQ((numerator / integer).x, 0x3bbfU);
    EXPECT_EQ((integer / denominator).x, 0x4313U);
    EXPECT_EQ((numerator / -integer).x, 0xbbbfU);
    EXPECT_EQ((-integer / denominator).x, 0xc313U);
}

TYPED_TEST(BFloat16IntegerArithmetic, DivisionPreservesZeroAndNonFiniteResults) {
    const TypeParam zero = 0;
    const TypeParam one = 1;
    const BFloat16 positive_zero{0x0000, BFloat16::from_bits()};
    const BFloat16 negative_zero{0x8000, BFloat16::from_bits()};

    EXPECT_EQ((BFloat16{1.0f} / zero).x, 0x7f80U);
    EXPECT_EQ((BFloat16{-1.0f} / zero).x, 0xff80U);
    EXPECT_EQ((one / positive_zero).x, 0x7f80U);
    EXPECT_EQ((one / negative_zero).x, 0xff80U);
    EXPECT_EQ((negative_zero / one).x, 0x8000U);
    EXPECT_EQ((zero / BFloat16{-1.0f}).x, 0x8000U);
    // 0/0 is an invalid operation yielding a quiet NaN; the NaN sign bit is
    // implementation-defined, so assert only the NaN/quiet structure.
    EXPECT_EQ((positive_zero / zero).x & 0x7FC0U, 0x7FC0U);
    EXPECT_EQ((zero / positive_zero).x & 0x7FC0U, 0x7FC0U);
}

TEST(BFloat16Test, Int64ArithmeticUsesFloat32Promotion) {
    constexpr int64_t integer = 16777219;
    const BFloat16 positive{16777216.0f};
    const BFloat16 negative{-16777216.0f};

    // FP32 rounds the integer to 16777220 before cancellation, giving 4, not 3.
    EXPECT_EQ((negative + integer).x, 0x4080U);
    EXPECT_EQ((integer + negative).x, 0x4080U);
    EXPECT_EQ((positive - integer).x, 0xc080U);
    EXPECT_EQ((integer - positive).x, 0x4080U);
}

TEST(BFloat16Test, Int64ArithmeticPreservesValuesBeyondInt32Range) {
    constexpr int64_t integer = (int64_t{1} << 40) + (int64_t{1} << 24);
    const BFloat16 positive{0x5380, BFloat16::from_bits()}; // 2^40
    const BFloat16 negative{0xd380, BFloat16::from_bits()}; // -2^40

    EXPECT_EQ((negative + integer).x, 0x4b80U); // 2^24
    EXPECT_EQ((integer + negative).x, 0x4b80U);
    EXPECT_EQ((positive - integer).x, 0xcb80U); // -2^24
    EXPECT_EQ((integer - positive).x, 0x4b80U);
}

TEST(BFloat16Test, ComparisonOperators) {
    BFloat16 a(2.0f);
    BFloat16 b(3.0f);
    BFloat16 c(2.0f);

    EXPECT_FALSE(a > b);
    EXPECT_TRUE(b > a);
    EXPECT_FALSE(a > c);

    EXPECT_TRUE(a < b);
    EXPECT_FALSE(b < a);
    EXPECT_FALSE(a < c);

    EXPECT_TRUE(a == c);
    EXPECT_FALSE(a == b);
}

TEST(BFloat16Test, BitwiseOperations) {
    BFloat16 a(2.0f); // 0x4000 in bfloat16
    BFloat16 b(3.0f); // 0x4040 in bfloat16

    BFloat16 result = a;
    result = result | b;
    EXPECT_EQ(result.x, 0x4040); // 0x4000 | 0x4040 = 0x4040

    result = a;
    result = result & b;
    EXPECT_EQ(result.x, 0x4000); // 0x4000 & 0x4040 = 0x4000

    result = a;
    result = result ^ b;
    EXPECT_EQ(result.x, 0x0040); // 0x4000 ^ 0x4040 = 0x0040
}

TEST(BFloat16Test, NumericLimits) {
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

    EXPECT_EQ(std::numeric_limits<BFloat16>::min().x, 0x0080);
    EXPECT_EQ(std::numeric_limits<BFloat16>::max().x, 0x7F7F);
    EXPECT_EQ(std::numeric_limits<BFloat16>::lowest().x, 0xFF7F);
    EXPECT_EQ(std::numeric_limits<BFloat16>::epsilon().x, 0x3C00);
    EXPECT_EQ(std::numeric_limits<BFloat16>::round_error().x, 0x3F00);
    EXPECT_EQ(std::numeric_limits<BFloat16>::infinity().x, 0x7F80);
    EXPECT_EQ(std::numeric_limits<BFloat16>::quiet_NaN().x, 0x7FC0);
    EXPECT_EQ(std::numeric_limits<BFloat16>::denorm_min().x, 0x0001);
}

TEST(BFloat16Test, SignalingNaN) {
    // Regression: signaling_NaN() must not collide with +infinity (0x7F80).
    // BF16 layout: 1 sign | 8 exponent | 7 mantissa.
    //   infinity       : exp=0xFF, mantissa=0
    //   signaling NaN  : exp=0xFF, mantissa!=0, mantissa MSB=0
    //   quiet NaN      : exp=0xFF, mantissa!=0, mantissa MSB=1
    constexpr uint16_t snan_bits = std::numeric_limits<BFloat16>::signaling_NaN().x;
    constexpr uint16_t inf_bits = std::numeric_limits<BFloat16>::infinity().x;
    constexpr uint16_t qnan_bits = std::numeric_limits<BFloat16>::quiet_NaN().x;

    EXPECT_NE(snan_bits, inf_bits);
    EXPECT_NE(snan_bits, qnan_bits);

    // exponent all-ones => NaN/Inf class
    EXPECT_EQ(static_cast<uint16_t>(snan_bits & 0x7F80), 0x7F80);
    // mantissa non-zero => not infinity
    EXPECT_NE(static_cast<uint16_t>(snan_bits & 0x007F), 0x0000);
    // mantissa MSB (quiet bit) clear => signaling, not quiet
    EXPECT_EQ(static_cast<uint16_t>(snan_bits & 0x0040), 0x0000);
}

// Compares a BF16 unary math result against its FP32 reference implementation.
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
            // Zero input: compare against the expected value with an absolute tolerance.
            EXPECT_NEAR(actual, expected, max_relative_error) << "Function test failed for input: " << f;
        } else {
            float relative_error = std::abs((actual - expected) / expected);
            EXPECT_LE(relative_error, max_relative_error)
                    << "Function test failed for input: " << f
                    << ", Expected: " << expected
                    << ", Actual: " << actual
                    << ", Relative error: " << relative_error;
        }
    }
}

TEST(BFloat16MathTest, StandardMathPreservesFloatResults) {
    static_assert(std::is_same_v<decltype(std::sqrt(Half{})), float>);
    static_assert(std::is_same_v<decltype(std::sqrt(BFloat16{})), float>);
    static_assert(std::is_same_v<decltype(std::exp(Half{})), float>);
    static_assert(std::is_same_v<decltype(std::exp(BFloat16{})), float>);

    EXPECT_FLOAT_EQ(std::sqrt(Half{2.0f}), std::sqrt(2.0f));
    EXPECT_FLOAT_EQ(std::sqrt(BFloat16{2.0f}), std::sqrt(2.0f));
    EXPECT_FLOAT_EQ(std::exp(Half{1.0f}), std::exp(1.0f));
    EXPECT_FLOAT_EQ(std::exp(BFloat16{1.0f}), std::exp(1.0f));
}

TEST(BFloat16MathTest, ReducedMathSupportsArgumentDependentLookup) {
    using std::sqrt;
    static_assert(std::is_same_v<decltype(sqrt(Half{})), Half>);
    static_assert(std::is_same_v<decltype(sqrt(BFloat16{})), BFloat16>);

    EXPECT_EQ(sqrt(Half{2.0f}).bits(), Half{std::sqrt(2.0f)}.bits());
    EXPECT_EQ(sqrt(BFloat16{2.0f}).x, BFloat16{std::sqrt(2.0f)}.x);
    EXPECT_EQ(aethermind::sqrt(Half{2.0f}).bits(), Half{std::sqrt(2.0f)}.bits());
}

TEST(BFloat16MathTest, Acos) {
    std::vector<float> test_values = {
            0.0f, 1.0f, -1.0f, 0.5f, -0.5f,
            0.7071f, -0.7071f, 0.8660f, -0.8660f};
    TestUnaryMathFunction(aethermind::acos<BFloat16>, std::acos, test_values);
}

TEST(BFloat16MathTest, Asin) {
    std::vector<float> test_values = {
            0.0f, 1.0f, -1.0f, 0.5f, -0.5f,
            0.7071f, -0.7071f, 0.8660f, -0.8660f};
    TestUnaryMathFunction(aethermind::asin<BFloat16>, std::asin, test_values);
}

TEST(BFloat16MathTest, Atan) {
    std::vector<float> test_values = {
            0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f,
            100.0f, -100.0f, 0.333333f, -0.333333f};
    TestUnaryMathFunction(aethermind::atan<BFloat16>, std::atan, test_values);
}

TEST(BFloat16MathTest, Erf) {
    std::vector<float> test_values = {
            0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f,
            3.0f, -3.0f, 0.1f, -0.1f};
    TestUnaryMathFunction(aethermind::erf<BFloat16>, std::erf, test_values);
}

TEST(BFloat16MathTest, Erfc) {
    std::vector<float> test_values = {
            0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f,
            3.0f, -3.0f, 0.1f, -0.1f};
    TestUnaryMathFunction(aethermind::erfc<BFloat16>, std::erfc, test_values);
}

TEST(BFloat16MathTest, Exp) {
    std::vector<float> test_values = {
            0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f,
            0.693147f, -0.693147f, 1.098612f, -1.098612f};
    // Rounding error grows with the result; relax the relative-error bound.
    TestUnaryMathFunction(aethermind::exp<BFloat16>, std::exp, test_values, 0.05f);
}

TEST(BFloat16MathTest, Expm1) {
    std::vector<float> test_values = {
            0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 0.1f, -0.1f,
            0.01f, -0.01f, 0.001f, -0.001f};
    TestUnaryMathFunction(aethermind::expm1<BFloat16>, std::expm1, test_values);
}

TEST(BFloat16MathTest, IsFinite) {
    BFloat16 finite_value(1.0f);
    BFloat16 inf_pos(std::numeric_limits<float>::infinity());
    BFloat16 inf_neg(-std::numeric_limits<float>::infinity());
    BFloat16 nan(std::numeric_limits<float>::quiet_NaN());

    EXPECT_TRUE(aethermind::isfinite(finite_value));
    EXPECT_FALSE(aethermind::isfinite(inf_pos));
    EXPECT_FALSE(aethermind::isfinite(inf_neg));
    EXPECT_FALSE(aethermind::isfinite(nan));
}

TEST(BFloat16MathTest, Log) {
    std::vector<float> test_values = {
            1.0f, 2.0f, 0.5f, M_E, M_PI, 10.0f, 0.1f,
            3.14159f, 0.318309f, 100.0f, 0.01f};
    TestUnaryMathFunction(aethermind::log<BFloat16>, std::log, test_values);
}

TEST(BFloat16MathTest, Log10) {
    std::vector<float> test_values = {
            1.0f, 10.0f, 0.1f, 100.0f, 0.01f, 1000.0f, 0.001f,
            5.0f, 2.0f, 50.0f, 20.0f};
    TestUnaryMathFunction(aethermind::log10<BFloat16>, std::log10, test_values);
}

TEST(BFloat16MathTest, Log2) {
    std::vector<float> test_values = {
            1.0f, 2.0f, 0.5f, 4.0f, 0.25f, 8.0f, 0.125f,
            3.0f, 5.0f, 10.0f, 0.3f};
    TestUnaryMathFunction(aethermind::log2<BFloat16>, std::log2, test_values);
}

TEST(BFloat16MathTest, Log1p) {
    std::vector<float> test_values = {
            0.0f, 1.0f, -0.5f, 0.5f, 2.0f, -0.9f, 0.1f,
            -0.1f, 0.01f, -0.01f, 0.001f};
    TestUnaryMathFunction(aethermind::log1p<BFloat16>, std::log1p, test_values);
}

TEST(BFloat16MathTest, Ceil) {
    std::vector<float> test_values = {
            0.0f, 1.0f, -1.0f, 1.1f, -1.1f, 1.5f, -1.5f,
            2.999f, -2.99f, 0.0001f, -0.0001f};
    for (float f: test_values) {
        BFloat16 bf16 = f;
        BFloat16 result_bf16 = aethermind::ceil(bf16);
        float expected = std::ceil(f);
        float actual = result_bf16;
        EXPECT_FLOAT_EQ(actual, expected)
                << "Ceil test failed for input: " << f
                << ", Expected: " << expected
                << ", Actual: " << actual;
    }
}

TEST(BFloat16MathTest, Cos) {
    std::vector<float> test_values = {
            0.0f, M_PI, -M_PI, M_PI_4, -M_PI_4,
            M_PI * 0.25f, M_PI * 0.75f, M_PI * 1.25f, M_PI * 1.75f};
    TestUnaryMathFunction(aethermind::cos<BFloat16>, std::cos, test_values);
}

TEST(BFloat16MathTest, Sin) {
    std::vector<float> test_values = {
            0.0f, M_PI_2, -M_PI_2, M_PI_4, -M_PI_4,
            M_PI * 0.25f, M_PI * 0.75f, M_PI * 1.25f, M_PI * 1.75f};
    TestUnaryMathFunction(aethermind::sin<BFloat16>, std::sin, test_values);
}

TEST(BFloat16MathTest, Sinh) {
    std::vector<float> test_values = {
            0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f,
            3.0f, -3.0f, 0.1f, -0.1f};
    TestUnaryMathFunction(aethermind::sinh<BFloat16>, std::sinh, test_values, 0.03f);
}

TEST(BFloat16MathTest, Cosh) {
    std::vector<float> test_values = {
            0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f,
            3.0f, -3.0f, 0.1f, -0.1f};
    TestUnaryMathFunction(aethermind::cosh<BFloat16>, std::cosh, test_values, 0.03f);
}

TEST(BFloat16MathTest, Tan) {
    std::vector<float> test_values = {
            0.0f, M_PI_4, -M_PI_4,
            0.1f, -0.1f, 1.0f, -1.0f};
    // tan() amplifies input rounding; relax the relative-error bound.
    TestUnaryMathFunction(aethermind::tan<BFloat16>, std::tan, test_values, 0.03f);
}

TEST(BFloat16MathTest, Tanh) {
    std::vector<float> test_values = {
            0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f,
            3.0f, -3.0f, 0.1f, -0.1f};
    TestUnaryMathFunction(aethermind::tanh<BFloat16>, std::tanh, test_values);
}

TEST(BFloat16MathTest, Floor) {
    std::vector<float> test_values = {
            0.0f, 1.0f, -1.0f, 1.9f, -1.9f, 1.5f, -1.5f,
            2.001f, -2.1f, 0.99f, -0.999f};
    for (float f: test_values) {
        BFloat16 bf16 = f;
        BFloat16 result_bf16 = aethermind::floor(bf16);
        float expected = std::floor(f);
        float actual = result_bf16;
        EXPECT_FLOAT_EQ(actual, expected)
                << "Floor test failed for input: " << f
                << ", Expected: " << expected
                << ", Actual: " << actual;
    }
}

TEST(BFloat16MathTest, Nearbyint) {
    std::vector<float> test_values = {
            0.0f, 1.0f, -1.0f, 1.1f, -1.1f, 1.5f, -1.5f,
            2.9f, -2.9f, 0.0001f, -0.0001f};
    TestUnaryMathFunction(aethermind::nearbyint<BFloat16>, std::nearbyint, test_values, 0.01f);
}

TEST(BFloat16MathTest, Trunc) {
    std::vector<float> test_values = {
            0.0f, 1.0f, -1.0f, 1.1f, -1.1f, 1.9f, -1.9f,
            1.5f, -1.5f, 0.99f, -0.99f};
    for (float f: test_values) {
        BFloat16 bf16 = f;
        BFloat16 result_bf16 = aethermind::trunc(bf16);
        float expected = std::trunc(f);
        float actual = result_bf16;
        EXPECT_FLOAT_EQ(actual, expected)
                << "Trunc test failed for input: " << f
                << ", Expected: " << expected
                << ", Actual: " << actual;
    }
}

TEST(BFloat16MathTest, Lgamma) {
    std::vector<float> test_values = {
            1.0f, 2.0f, 3.0f, 0.5f, 1.5f, 5.0f, 10.0f,
            0.1f, 0.2f, 0.3f, 0.4f};
    // lgamma() loses precision near the poles; relax the relative-error bound.
    TestUnaryMathFunction(aethermind::lgamma<BFloat16>, std::lgamma, test_values, 0.05f);
}

TEST(BFloat16MathTest, Sqrt) {
    std::vector<float> test_values = {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 0.25f, 0.5f,
            10.0f, 25.0f, 100.0f, 0.01f};
    TestUnaryMathFunction(aethermind::sqrt<BFloat16>, std::sqrt, test_values);
}

TEST(BFloat16MathTest, Rsqrt) {
    std::vector<float> test_values = {
            1.0f, 2.0f, 3.0f, 4.0f, 0.25f, 0.5f,
            10.0f, 25.0f, 100.0f, 0.01f};
    auto rsqrt_func = [](float x) { return 1.0f / std::sqrt(x); };
    TestUnaryMathFunction(aethermind::rsqrt<BFloat16>, rsqrt_func, test_values);
}

TEST(BFloat16MathTest, Abs) {
    std::vector<float> test_values = {
            0.0f, 1.0f, -1.0f, 2.5f, -2.5f,
            100.0f, -100.0f};
    for (float f: test_values) {
        BFloat16 bf16 = f;
        BFloat16 result_bf16 = aethermind::abs(bf16);
        float expected = std::abs(f);
        float actual = result_bf16;
        EXPECT_FLOAT_EQ(actual, expected)
                << "Abs test failed for input: " << f
                << ", Expected: " << expected
                << ", Actual: " << actual;
    }
}

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
        BFloat16 result_bf16 = aethermind::pow(base_bf16, exp);
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
        if (y == 0.0f) continue;

        BFloat16 x_bf16 = x;
        BFloat16 y_bf16 = y;
        BFloat16 result_bf16 = aethermind::fmod(x_bf16, y_bf16);
        float expected = std::fmod(x, y);
        auto actual = static_cast<float>(result_bf16);

        // Sign handling may vary across implementations, so compare magnitudes first.
        if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(actual))
                    << "Fmod test failed for x: " << x << ", y: " << y;
        } else {
            EXPECT_NEAR(std::abs(actual), std::abs(expected), 0.02f)
                    << "Fmod test failed for x: " << x
                    << ", y: " << y
                    << ", Expected: " << expected
                    << ", Actual: " << actual;

            // fmod must preserve the sign of the dividend.
            EXPECT_EQ(std::signbit(actual), std::signbit(x))
                    << "Fmod test failed for x: " << x
                    << ", y: " << y
                    << ", Result sign incorrect";
        }
    }
}

} // namespace
