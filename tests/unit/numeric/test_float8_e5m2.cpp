#include "aethermind/dtypes/float8_e5m2.h"

#include <bitset>
#include <cmath>
#include <gtest/gtest.h>

namespace {
using namespace aethermind;
using namespace aethermind::detail;

TEST(Float8E5M2Test, SpecialValues) {
    EXPECT_EQ(fp8e5m2_from_fp32_value(0.0f), 0x00);
    EXPECT_EQ(fp8e5m2_from_fp32_value(-0.0f), 0x80);
    EXPECT_EQ(fp8e5m2_to_fp32_value(0x00), 0.0f);
    EXPECT_EQ(fp8e5m2_to_fp32_value(0x80), -0.0f);

    EXPECT_EQ(fp8e5m2_from_fp32_value(INFINITY), 0x7C);
    EXPECT_EQ(fp8e5m2_from_fp32_value(-INFINITY), 0xFC);
    EXPECT_TRUE(std::isinf(fp8e5m2_to_fp32_value(0x7C)));
    EXPECT_TRUE(std::isinf(fp8e5m2_to_fp32_value(0xFC)));

    EXPECT_EQ(fp8e5m2_from_fp32_value(NAN), 0x7E); // Canonical NaN
    EXPECT_TRUE(std::isnan(fp8e5m2_to_fp32_value(0x7F)));
}

TEST(Float8E5M2Test, NormalRange) {
    EXPECT_EQ(fp8e5m2_from_fp32_value(1.0f), 0x3C);
    EXPECT_FLOAT_EQ(fp8e5m2_to_fp32_value(0x3C), 1.0f);

    EXPECT_EQ(fp8e5m2_from_fp32_value(2.0f), 0x40);
    EXPECT_FLOAT_EQ(fp8e5m2_to_fp32_value(0x40), 2.0f);

    EXPECT_EQ(fp8e5m2_from_fp32_value(-1.0f), 0xBC);
    EXPECT_FLOAT_EQ(fp8e5m2_to_fp32_value(0xBC), -1.0f);

    // 1.1 is nearest to 1.0; the next encoding is 1.25.
    EXPECT_EQ(fp8e5m2_from_fp32_value(1.1f), 0x3C);
}

TEST(Float8E5M2Test, OverflowAndUnderflow) {
    // Finite values at or above 2^16 saturate to infinity; 65504 lies beyond
    // the 61440 midpoint between 57344 (max normal) and 2^16, so it rounds up.
    EXPECT_EQ(fp8e5m2_from_fp32_value(65504.0f), 0x7C);
    EXPECT_EQ(fp8e5m2_from_fp32_value(100000.0f), 0x7C);

    // Values too small to round to a subnormal flush to zero.
    EXPECT_EQ(fp8e5m2_from_fp32_value(1.0e-10f), 0x00);
}

TEST(Float8E5M2Test, DenormalHandling) {
    // 2^-16 is the smallest positive subnormal (0x01).
    EXPECT_EQ(fp8e5m2_from_fp32_value(1.0f / 65536), 0x01);
    // 2^-17 is exactly half the smallest subnormal; ties-to-even flushes to zero.
    EXPECT_EQ(fp8e5m2_from_fp32_value(1.0f / 131072), 0x00);
}

TEST(Float8E5M2Test, RoundingModes) {
    // 1.5 and 1.25 are exactly representable in E5M2.
    EXPECT_EQ(fp8e5m2_from_fp32_value(1.5f), 0x3E);
    EXPECT_EQ(fp8e5m2_from_fp32_value(1.25f), 0x3D);
    // 1.375 is the midpoint between 1.25 and 1.5; ties-to-even picks 1.5.
    EXPECT_EQ(fp8e5m2_from_fp32_value(1.375f), 0x3E);
}

TEST(Float8E5M2Test, ConstructorAndBasicProperties) {
    Float8_e5m2 f1;
    EXPECT_EQ(f1.x, 0x00);

    Float8_e5m2 f2 = Float8_e5m2(0x3C, Float8_e5m2::from_bits());
    EXPECT_EQ(f2.x, 0x3C);

    Float8_e5m2 f3(1.0f);
    EXPECT_EQ(f3.x, 0x3C);

    float result = f3;
    EXPECT_FLOAT_EQ(result, 1.0f);
}

TEST(Float8E5M2Test, ConstructorSpecialValues) {
    Float8_e5m2 zero(0.0f);
    EXPECT_EQ(zero.x, 0x00);

    Float8_e5m2 neg_zero(-0.0f);
    EXPECT_EQ(neg_zero.x, 0x80);

    Float8_e5m2 inf(std::numeric_limits<float>::infinity());
    EXPECT_EQ(inf.x, 0x7C);
    EXPECT_TRUE(inf.isinf());

    Float8_e5m2 neg_inf(-std::numeric_limits<float>::infinity());
    EXPECT_EQ(neg_inf.x, 0xFC);
    EXPECT_TRUE(neg_inf.isinf());

    Float8_e5m2 nan(std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(nan.x, 0x7E);
    EXPECT_TRUE(nan.isnan());
}

TEST(Float8E5M2Test, NumericLimits) {
    auto min_val = std::numeric_limits<Float8_e5m2>::min();
    EXPECT_EQ(min_val.x, 0x04);

    auto max_val = std::numeric_limits<Float8_e5m2>::max();
    EXPECT_EQ(max_val.x, 0x7B);

    auto lowest_val = std::numeric_limits<Float8_e5m2>::lowest();
    EXPECT_EQ(lowest_val.x, 0xFB);

    auto inf_val = std::numeric_limits<Float8_e5m2>::infinity();
    EXPECT_EQ(inf_val.x, 0x7C);

    auto nan_val = std::numeric_limits<Float8_e5m2>::quiet_NaN();
    EXPECT_EQ(nan_val.x, 0x7F);

    auto denorm_min = std::numeric_limits<Float8_e5m2>::denorm_min();
    EXPECT_EQ(denorm_min.x, 0x01);
}

TEST(Float8E5M2Test, ArithmeticOperators) {
    Float8_e5m2 a(2.0f);
    Float8_e5m2 b(3.0f);

    Float8_e5m2 sum = a + b;
    EXPECT_FLOAT_EQ(sum, 5.0f);

    Float8_e5m2 diff = a - b;
    EXPECT_FLOAT_EQ(diff, -1.0f);

    Float8_e5m2 prod = a * b;
    EXPECT_FLOAT_EQ(prod, 6.0f);

    Float8_e5m2 quot = a / b;
    EXPECT_FLOAT_EQ(quot, 0.625f);

    Float8_e5m2 neg = -a;
    EXPECT_FLOAT_EQ(neg, -2.0f);
}

TEST(Float8E5M2Test, CompoundAssignmentOperators) {
    Float8_e5m2 a(2.0f);
    Float8_e5m2 b(3.0f);

    Float8_e5m2 c = a;
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
    EXPECT_FLOAT_EQ(c, 0.625f);
}

TEST(Float8E5M2Test, MixedTypeArithmetic) {
    // Operand pair (8, 2) chosen so each of {+, -, *, /} yields a distinct
    // exactly-representable E5M2 value (10, 6, 16, 4). This ensures a
    // copy-paste bug in any operator (e.g. `-` returning `+`) cannot pass
    // by coincidence — every assertion is sensitive to the operator used.
    const Float8_e5m2 a(8.0f);

    EXPECT_FLOAT_EQ(a + 2.0f, 10.0f);
    EXPECT_FLOAT_EQ(a - 2.0f, 6.0f);
    EXPECT_FLOAT_EQ(a * 2.0f, 16.0f);
    EXPECT_FLOAT_EQ(a / 2.0f, 4.0f);

    EXPECT_FLOAT_EQ(2.0f + a, 10.0f);
    EXPECT_FLOAT_EQ(2.0f - a, -6.0f);
    EXPECT_FLOAT_EQ(2.0f * a, 16.0f);
    EXPECT_FLOAT_EQ(2.0f / a, 0.25f);

    EXPECT_DOUBLE_EQ(a + 2.0, 10.0);
    EXPECT_DOUBLE_EQ(a - 2.0, 6.0);
    EXPECT_DOUBLE_EQ(a * 2.0, 16.0);
    EXPECT_DOUBLE_EQ(a / 2.0, 4.0);

    EXPECT_DOUBLE_EQ(2.0 + a, 10.0);
    EXPECT_DOUBLE_EQ(2.0 - a, -6.0);
    EXPECT_DOUBLE_EQ(2.0 * a, 16.0);
    EXPECT_DOUBLE_EQ(2.0 / a, 0.25);

    EXPECT_FLOAT_EQ(a + 2, 10.0f);
    EXPECT_FLOAT_EQ(a - 2, 6.0f);
    EXPECT_FLOAT_EQ(a * 2, 16.0f);
    EXPECT_FLOAT_EQ(a / 2, 4.0f);

    EXPECT_FLOAT_EQ(2 + a, 10.0f);
    EXPECT_FLOAT_EQ(2 - a, -6.0f);
    EXPECT_FLOAT_EQ(2 * a, 16.0f);
    EXPECT_FLOAT_EQ(2 / a, 0.25f);

    EXPECT_FLOAT_EQ(a + static_cast<int64_t>(2), 10.0f);
    EXPECT_FLOAT_EQ(a - static_cast<int64_t>(2), 6.0f);
    EXPECT_FLOAT_EQ(a * static_cast<int64_t>(2), 16.0f);
    EXPECT_FLOAT_EQ(a / static_cast<int64_t>(2), 4.0f);

    EXPECT_FLOAT_EQ(static_cast<int64_t>(2) + a, 10.0f);
    EXPECT_FLOAT_EQ(static_cast<int64_t>(2) - a, -6.0f);
    EXPECT_FLOAT_EQ(static_cast<int64_t>(2) * a, 16.0f);
    EXPECT_FLOAT_EQ(static_cast<int64_t>(2) / a, 0.25f);
}

TEST(Float8E5M2Test, EdgeCasesAndRounding) {
    Float8_e5m2 max_val(57344.0f); // Largest finite E5M2 value (0x7B).
    EXPECT_EQ(max_val.x, 0x7B);

    Float8_e5m2 min_val(0.00006103515625f); // Smallest normal E5M2 value, 2^-14 (0x04).
    EXPECT_EQ(min_val.x, 0x04);

    Float8_e5m2 tiny(1e-10f);
    EXPECT_EQ(tiny.x, 0x00); // Below half the smallest subnormal: flushes to zero.

    Float8_e5m2 huge(1e6f);
    EXPECT_EQ(huge.x, 0x7C); // Beyond the finite range: saturates to infinity.
}

TEST(Float8E5M2Test, OutputOperator) {
    Float8_e5m2 val(1.5f);
    std::ostringstream oss;
    oss << val;

    EXPECT_FALSE(oss.str().empty());
}

TEST(Float8E5M2Test, SpecialCases) {
    Float8_e5m2 nan_val = std::numeric_limits<Float8_e5m2>::quiet_NaN();
    Float8_e5m2 normal(1.0f);

    Float8_e5m2 result = nan_val + normal;
    EXPECT_TRUE(result.isnan());

    Float8_e5m2 inf_val = std::numeric_limits<Float8_e5m2>::infinity();
    result = inf_val + normal;
    EXPECT_TRUE(result.isinf());

    Float8_e5m2 zero(0.0f);
    result = normal / zero;
    EXPECT_TRUE(result.isinf());
}

TEST(Float8E5M2Test, FloatAssignmentOperators) {
    float f = 5.0f;
    Float8_e5m2 val(2.0f);

    f += val;
    EXPECT_FLOAT_EQ(f, 7.0f);

    f = 5.0f;
    f -= val;
    EXPECT_FLOAT_EQ(f, 3.0f);

    f = 5.0f;
    f *= val;
    EXPECT_FLOAT_EQ(f, 10.0f);

    f = 5.0f;
    f /= val;
    EXPECT_FLOAT_EQ(f, 2.5f);
}

} // namespace
