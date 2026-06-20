/// \file
/// Unit tests for IEEE 754 half-precision conversion functions and the Half type.
///
/// Covers bit-exact conversion (`fp16_to_fp32_bits_for_testing`,
/// `fp16_from_fp32_value_for_testing`), value conversion (`fp16_to_fp32_value`),
/// and the `Half` class API.
#include "utils/floating_point_utils.h"
#include "utils/half.h"
#include <cmath>
#include <gtest/gtest.h>
#include <sstream>

using namespace aethermind;
using namespace aethermind::details;

namespace aethermind::details {
uint32_t fp16_to_fp32_bits_for_testing(uint16_t h);
uint16_t fp16_from_fp32_value_for_testing(float f);
}// namespace aethermind::details

namespace {

TEST(HalfToFP32Test, HalfToFp32Bits_Zero) {
    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0x0000), 0x00000000);// +0
    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0x8000), 0x80000000);// -0
}

TEST(HalfToFP32Test, HalfToFp32Bits_Denormalized) {
    // Smallest positive denormal: 0x0001 -> 2^-14 * 2^-10 = 2^-24
    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0x0001), 0x33800000);

    // Largest denormal: 0x03FF
    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0x03FF), 0x387FC000);
}

TEST(HalfToFP32Test, HalfToFp32Bits_Normalized) {
    // 1.0 in half: 0x3C00 -> 1.0 in float: 0x3F800000
    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0x3C00), 0x3F800000);

    // -1.0 in half: 0xBC00 -> -1.0 in float: 0xBF800000
    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0xBC00), 0xBF800000);

    // 2.0 in half: 0x4000 -> 2.0 in float: 0x40000000
    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0x4000), 0x40000000);

    // 0.5 in half: 0x3800 -> 0.5 in float: 0x3F000000
    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0x3800), 0x3F000000);

    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0x3555), 0x3EAAA000);// ~0.33325
    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0x48CD), 0x4119A000);// ~9.6016
    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0x4D12), 0x41A24000);// ~20.28125
}

TEST(HalfToFP32Test, HalfToFp32Bits_Infinity) {
    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0x7C00), 0x7F800000);// +inf
    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0xFC00), 0xFF800000);// -inf
}

TEST(HalfToFP32Test, HalfToFp32Bits_NaN) {
    // Quiet NaN
    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0x7C01), 0x7F802000);
    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0x7FFF), 0x7FFFE000);

    // Signaling NaN
    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0x7E00), 0x7FC00000);
    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0x7F00), 0x7FE00000);

    // Negative NaN
    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0xFC01), 0xFF802000);
    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0xFFFF), 0xFFFFE000);
}

TEST(HalfToFP32Test, HalfToFp32Bits_EdgeCases) {
    // Max normal: 0x7BFF -> ~65504.0
    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0x7BFF), 0x477FE000);

    // Min normal: 0x0400 -> 2^-14 ≈ 6.1035e-5
    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0x0400), 0x38800000);
}

TEST(HalfToFP32Test, HalfToFp32Bits_SignAndExponent) {
    const uint16_t test_values[] = {
            0x0000, 0x0001, 0x03FF, 0x0400, 0x3C00, 0x4000,
            0x7C00, 0x7E00, 0x7FFF, 0x8000, 0xBC00, 0xFC00};

    for (auto half_val: test_values) {
        uint32_t fp32_bits = fp16_to_fp32_bits_for_testing(half_val);

        // Sign bit must be preserved.
        bool half_sign = (half_val & 0x8000) != 0;
        bool fp32_sign = (fp32_bits & 0x80000000) != 0;
        EXPECT_EQ(half_sign, fp32_sign);

        // Inf and NaN: exponent field must be all-ones in fp32.
        if ((half_val & 0x7C00) == 0x7C00) {
            EXPECT_TRUE((fp32_bits & 0x7F800000) == 0x7F800000);
        }
    }
}

TEST(HalfToFP32Test, HalfToFp32Bits_ExhaustiveSmallValues) {
    for (uint16_t i = 0; i < 0x0400; ++i) {
        uint32_t result = fp16_to_fp32_bits_for_testing(i);

        // Result must be a valid fp32 bit pattern (exponent ≤ all-ones).
        EXPECT_TRUE((result & 0x7F800000) <= 0x7F800000);

        // Sign bit must match.
        EXPECT_EQ((i & 0x8000) != 0, (result & 0x80000000) != 0);
    }
}

TEST(HalfToFP32Test, HalfToFp32Value_Zero) {
    EXPECT_EQ(fp16_to_fp32_value(0x0000), 0.0f); // +0
    EXPECT_EQ(fp16_to_fp32_value(0x8000), -0.0f);// -0

    // Sign bit must be preserved for zero.
    EXPECT_TRUE(std::signbit(fp16_to_fp32_value(0x8000)));
    EXPECT_FALSE(std::signbit(fp16_to_fp32_value(0x0000)));
}

TEST(HalfToFP32Test, HalfToFp32Value_Denormalized) {
    // Smallest positive denormal: 0x0001 -> 2^-24 ≈ 5.96046e-08
    float min_denormal = fp16_to_fp32_value(0x0001);
    EXPECT_GT(min_denormal, 0.0f);
    EXPECT_LT(min_denormal, 1e-7f);

    // Largest denormal: 0x03FF, must be smaller than the smallest normal.
    float max_denormal = fp16_to_fp32_value(0x03FF);
    EXPECT_GT(max_denormal, 0.0f);
    EXPECT_LT(max_denormal, 6.5e-5f);
}

TEST(HalfToFP32Test, HalfToFp32Value_Normalized) {
    // Min normal: 0x0400 -> 2^-14
    EXPECT_FLOAT_EQ(6.10351562e-5f, fp16_to_fp32_value(0x0400));

    // 1.0
    EXPECT_FLOAT_EQ(1.0f, fp16_to_fp32_value(0x3C00));

    // -1.0
    EXPECT_FLOAT_EQ(-1.0f, fp16_to_fp32_value(0xBC00));

    // Max normal: 0x7BFF -> 65504.0
    EXPECT_FLOAT_EQ(65504.0f, fp16_to_fp32_value(0x7BFF));

    // 2.0
    EXPECT_FLOAT_EQ(fp16_to_fp32_value(0x4000), 2.0f);

    // 0.5
    EXPECT_FLOAT_EQ(fp16_to_fp32_value(0x3800), 0.5f);

    EXPECT_NEAR(fp16_to_fp32_value(0x3555), 0.33325f, 1e-5f);// ~1/3
    EXPECT_NEAR(fp16_to_fp32_value(0x48CD), 9.6016f, 1e-3f); // ~9.6
}

TEST(HalfToFP32Test, HalfToFp32Value_Infinity) {
    EXPECT_TRUE(std::isinf(fp16_to_fp32_value(0x7C00)));// +inf
    EXPECT_GT(fp16_to_fp32_value(0x7C00), 0);

    EXPECT_TRUE(std::isinf(fp16_to_fp32_value(0xFC00)));// -inf
    EXPECT_LT(fp16_to_fp32_value(0xFC00), 0);
}

TEST(HalfToFP32Test, HalfToFp32Value_NaN) {
    float nan1 = fp16_to_fp32_value(0x7C01);// quiet NaN
    float nan2 = fp16_to_fp32_value(0x7FFF);// quiet NaN
    float nan3 = fp16_to_fp32_value(0x7E00);// signaling NaN
    float nan4 = fp16_to_fp32_value(0xFC01);// negative quiet NaN

    EXPECT_TRUE(std::isnan(nan1));
    EXPECT_TRUE(std::isnan(nan2));
    EXPECT_TRUE(std::isnan(nan3));
    EXPECT_TRUE(std::isnan(nan4));

    // NaN must propagate through arithmetic.
    EXPECT_TRUE(std::isnan(nan1 + 1.0f));
    EXPECT_TRUE(std::isnan(nan1 * 2.0f));
}

TEST(HalfToFP32Test, HalfToFp32Value_EdgeCases) {
    // Max normal: 0x7BFF -> ~65504.0, must be finite.
    float max_normal = fp16_to_fp32_value(0x7BFF);
    EXPECT_NEAR(max_normal, 65504.0f, 1e-3f);
    EXPECT_FALSE(std::isinf(max_normal));

    // Min normal: 0x0400 -> 2^-14 ≈ 6.10352e-05
    float min_normal = fp16_to_fp32_value(0x0400);
    EXPECT_GT(min_normal, 0.0f);
    EXPECT_LT(min_normal, 1e-4f);

    // Denormal boundary: max denormal < min normal.
    float last_denormal = fp16_to_fp32_value(0x03FF);
    float first_normal = fp16_to_fp32_value(0x0400);
    EXPECT_LT(last_denormal, first_normal);
}

TEST(HalfToFP32Test, HalfToFp32Value_SpecialValues) {
    // PI approximation: 0x4248 -> ~3.140625
    EXPECT_NEAR(fp16_to_fp32_value(0x4248), 3.140625f, 1e-6f);

    // E approximation: 0x4170 -> ~2.71875
    EXPECT_NEAR(fp16_to_fp32_value(0x4170), 2.71875f, 1e-6f);

    // Golden ratio: 0x3FCF -> ~1.618
    EXPECT_NEAR(fp16_to_fp32_value(0x3FCF), 1.95215f, 1e-3f);
}

TEST(HalfToFP32Test, HalfToFp32Value_RoundTripConsistency) {
    // `fp16_to_fp32_value` must match `fp32_from_bits(fp16_to_fp32_bits_for_testing(...))`.
    const uint16_t test_values[] = {
            0x0000, 0x0001, 0x03FF, 0x0400, 0x3C00, 0x4000,
            0x7C00, 0x7E00, 0x7FFF, 0x8000, 0xBC00, 0xFC00};

    for (auto half_val: test_values) {
        uint32_t bits = fp16_to_fp32_bits_for_testing(half_val);
        float value_from_bits = fp32_from_bits(bits);
        float direct_value = fp16_to_fp32_value(half_val);

        if (std::isnan(value_from_bits)) {
            EXPECT_TRUE(std::isnan(direct_value));
        } else {
            EXPECT_FLOAT_EQ(value_from_bits, direct_value);
        }
    }
}

TEST(HalfToFP32Test, HalfToFp32Value_FiniteRange) {
    for (int exp = -14; exp <= 15; ++exp) {
        for (int mantissa = 0; mantissa < 1024; mantissa += 128) {
            uint16_t exponent = (exp + 15) << 10;// biased exponent
            uint16_t half_val = exponent | mantissa;

            if ((half_val & 0x7C00) != 0x7C00) {// skip inf/nan
                float value = fp16_to_fp32_value(half_val);

                if (!std::isinf(value) && !std::isnan(value)) {
                    EXPECT_TRUE(std::isfinite(value));
                }
            }
        }
    }
}


TEST(HalfFromFP32Test, ZeroValues) {
    EXPECT_EQ(fp16_from_fp32_value_for_testing(0.0f), 0x0000); // +0
    EXPECT_EQ(fp16_from_fp32_value_for_testing(-0.0f), 0x8000);// -0
}

TEST(HalfFromFP32Test, DenormalizedNumbers) {
    // Values below the half-precision minimum normal flush to zero.
    float smallest_denormal = std::numeric_limits<float>::denorm_min();
    EXPECT_EQ(fp16_from_fp32_value_for_testing(smallest_denormal), 0x0000);

    float max_denormal = 1.1754942e-38f;// ~2^-126 * (1 - 2^-23)
    EXPECT_EQ(fp16_from_fp32_value_for_testing(max_denormal), 0x0000);
}

TEST(HalfFromFP32Test, NormalizedNumbers) {
    EXPECT_EQ(fp16_from_fp32_value_for_testing(1.0f), 0x3C00);
    EXPECT_EQ(fp16_from_fp32_value_for_testing(2.0f), 0x4000);
    EXPECT_EQ(fp16_from_fp32_value_for_testing(0.5f), 0x3800);
    EXPECT_EQ(fp16_from_fp32_value_for_testing(-1.0f), 0xBC00);

    // Float denormals flush to zero in half.
    float smallest_normal = 1.17549435e-38f;// 2^-126
    EXPECT_EQ(fp16_from_fp32_value_for_testing(smallest_normal), 0);

    // Max half-precision normal.
    EXPECT_EQ(fp16_from_fp32_value_for_testing(65504.0f), 0x7BFF);
}

TEST(HalfFromFP32Test, Infinity) {
    EXPECT_EQ(fp16_from_fp32_value_for_testing(std::numeric_limits<float>::infinity()), 0x7C00);
    EXPECT_EQ(fp16_from_fp32_value_for_testing(-std::numeric_limits<float>::infinity()), 0xFC00);
}

TEST(HalfFromFP32Test, NaN) {
    // Quiet NaN: exponent all-ones, mantissa non-zero.
    float quiet_nan = std::numeric_limits<float>::quiet_NaN();
    uint16_t half_nan = fp16_from_fp32_value_for_testing(quiet_nan);
    EXPECT_TRUE((half_nan & 0x7C00) == 0x7C00);
    EXPECT_TRUE((half_nan & 0x03FF) != 0);

    // Signaling NaN: same exponent/mantissa invariants.
    float signaling_nan = std::numeric_limits<float>::signaling_NaN();
    half_nan = fp16_from_fp32_value_for_testing(signaling_nan);
    EXPECT_TRUE((half_nan & 0x7C00) == 0x7C00);
    EXPECT_TRUE((half_nan & 0x03FF) != 0);
}

TEST(HalfFromFP32Test, Overflow) {
    // Values exceeding the half-precision max saturate to ±inf.
    EXPECT_EQ(fp16_from_fp32_value_for_testing(70000.0f), 0x7C00);
    EXPECT_EQ(fp16_from_fp32_value_for_testing(-70000.0f), 0xFC00);
}

TEST(HalfFromFP32Test, Underflow) {
    // Values too small to represent flush to zero.
    EXPECT_EQ(fp16_from_fp32_value_for_testing(1e-10f), 0x0000);
    EXPECT_EQ(fp16_from_fp32_value_for_testing(-1e-10f), 0x8000);
}

TEST(HalfFromFP32Test, Rounding) {
    // Round-to-nearest-even: tie breaks to even.
    EXPECT_EQ(fp16_from_fp32_value_for_testing(1.0009765625f), 0x3C01);
    EXPECT_EQ(fp16_from_fp32_value_for_testing(1.001953125f), 0x3C02);
}

TEST(HalfFromFP32Test, SpecialValues) {
    EXPECT_EQ(fp16_from_fp32_value_for_testing(3.141592653589793f), 0x4248);// PI
    EXPECT_EQ(fp16_from_fp32_value_for_testing(2.718281828459045f), 0x4170);// E
    EXPECT_EQ(fp16_from_fp32_value_for_testing(1.618033988749895f), 0x3E79);// Golden ratio
}

TEST(HalfFromFP32Test, RoundTrip) {
    // fp32 -> fp16 -> fp32 must recover the original within half-precision error.
    float original = 1.2345f;
    uint16_t half_val = fp16_from_fp32_value_for_testing(original);
    float reconstructed = fp16_to_fp32_value(half_val);
    EXPECT_NEAR(original, reconstructed, 1e-3);
}

TEST(HalfTest, ConstructorAndConversion) {
    // Default constructor: zero.
    Half h1;
    EXPECT_EQ(h1.x, 0);

    // from_bits constructor: raw bit pattern, no conversion.
    Half h2(0x3C00, Half::from_bits());// 1.0 in half precision
    EXPECT_EQ(h2.x, 0x3C00);

    // Float constructor and conversion.
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

    // Compound assignment.
    Half c(1.0f);
    c += a;
    EXPECT_FLOAT_EQ(c, 2.5f);

    // Unary minus.
    EXPECT_FLOAT_EQ(-a, -1.5f);
}

TEST(HalfTest, MixedTypeArithmetic) {
    Half h(2.0f);

    // With float.
    EXPECT_FLOAT_EQ(h + 3.0f, 5.0f);
    EXPECT_FLOAT_EQ(3.0f + h, 5.0f);

    // With double.
    EXPECT_DOUBLE_EQ(h + 3.0, 5.0);
    EXPECT_DOUBLE_EQ(3.0 + h, 5.0);

    // With int.
    EXPECT_FLOAT_EQ(h + 3, 5.0f);
    EXPECT_FLOAT_EQ(3 + h, 5.0f);

    // With int64_t.
    EXPECT_FLOAT_EQ(h + int64_t(3), 5.0f);
    EXPECT_FLOAT_EQ(int64_t(3) + h, 5.0f);
}

TEST(HalfTest, NumericLimits) {
    EXPECT_EQ(std::numeric_limits<Half>::min().x, 0x0400);
    EXPECT_EQ(std::numeric_limits<Half>::max().x, 0x7BFF);
    EXPECT_EQ(std::numeric_limits<Half>::lowest().x, 0xFBFF);
    EXPECT_EQ(std::numeric_limits<Half>::epsilon().x, 0x1400);
    EXPECT_EQ(std::numeric_limits<Half>::infinity().x, 0x7C00);
    EXPECT_EQ(std::numeric_limits<Half>::quiet_NaN().x, 0x7E00);
    EXPECT_EQ(std::numeric_limits<Half>::signaling_NaN().x, 0x7D00);
    EXPECT_EQ(std::numeric_limits<Half>::denorm_min().x, 0x0001);
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

TEST(HalfTest, MixedTypeSubtraction) {
    Half h(5.0f);

    EXPECT_FLOAT_EQ(h - 2.0f, 3.0f);
    EXPECT_FLOAT_EQ(7.0f - h, 2.0f);
    EXPECT_DOUBLE_EQ(h - 2.0, 3.0);
    EXPECT_DOUBLE_EQ(7.0 - h, 2.0);
    EXPECT_FLOAT_EQ(h - 2, 3.0f);
    EXPECT_FLOAT_EQ(7 - h, 2.0f);
    EXPECT_FLOAT_EQ(h - int64_t(2), 3.0f);
    EXPECT_FLOAT_EQ(int64_t(7) - h, 2.0f);
}

TEST(HalfTest, MixedTypeMultiplication) {
    Half h(3.0f);

    EXPECT_FLOAT_EQ(h * 2.0f, 6.0f);
    EXPECT_FLOAT_EQ(2.0f * h, 6.0f);
    EXPECT_DOUBLE_EQ(h * 2.0, 6.0);
    EXPECT_DOUBLE_EQ(2.0 * h, 6.0);
    EXPECT_FLOAT_EQ(h * 2, 6.0f);
    EXPECT_FLOAT_EQ(2 * h, 6.0f);
    EXPECT_FLOAT_EQ(h * int64_t(2), 6.0f);
    EXPECT_FLOAT_EQ(int64_t(2) * h, 6.0f);
}

TEST(HalfTest, MixedTypeDivision) {
    Half h(6.0f);

    EXPECT_FLOAT_EQ(h / 2.0f, 3.0f);
    EXPECT_FLOAT_EQ(12.0f / h, 2.0f);
    EXPECT_DOUBLE_EQ(h / 2.0, 3.0);
    EXPECT_DOUBLE_EQ(12.0 / h, 2.0);
    EXPECT_FLOAT_EQ(h / 2, 3.0f);
    EXPECT_FLOAT_EQ(12 / h, 2.0f);
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

}// namespace
