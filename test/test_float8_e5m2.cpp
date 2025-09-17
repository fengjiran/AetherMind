//
// Created by 赵丹 on 2025/9/17.
//
#include "utils/float8_e5m2.h"

#include <bitset>
#include <cmath>
#include <gtest/gtest.h>

using namespace aethermind;
using namespace aethermind::details;

namespace {

TEST(Float8E5M2Test, SpecialValues) {
    // Test zero values
    EXPECT_EQ(fp8e5m2_from_fp32_value(0.0f), 0x00);
    EXPECT_EQ(fp8e5m2_from_fp32_value(-0.0f), 0x80);
    EXPECT_EQ(fp8e5m2_to_fp32_value(0x00), 0.0f);
    EXPECT_EQ(fp8e5m2_to_fp32_value(0x80), -0.0f);

    // Test infinity
    EXPECT_EQ(fp8e5m2_from_fp32_value(INFINITY), 0x7C);
    EXPECT_EQ(fp8e5m2_from_fp32_value(-INFINITY), 0xFC);
    EXPECT_TRUE(std::isinf(fp8e5m2_to_fp32_value(0x7C)));
    EXPECT_TRUE(std::isinf(fp8e5m2_to_fp32_value(0xFC)));

    // Test NaN
    EXPECT_EQ(fp8e5m2_from_fp32_value(NAN), 0x7E);// Canonical NaN
    EXPECT_TRUE(std::isnan(fp8e5m2_to_fp32_value(0x7F)));
}

TEST(Float8E5M2Test, NormalRange) {
    // Test positive normal numbers
    EXPECT_EQ(fp8e5m2_from_fp32_value(1.0f), 0x3C);
    EXPECT_FLOAT_EQ(fp8e5m2_to_fp32_value(0x3C), 1.0f);

    EXPECT_EQ(fp8e5m2_from_fp32_value(2.0f), 0x40);
    EXPECT_FLOAT_EQ(fp8e5m2_to_fp32_value(0x40), 2.0f);

    // Test negative normal numbers
    EXPECT_EQ(fp8e5m2_from_fp32_value(-1.0f), 0xBC);
    EXPECT_FLOAT_EQ(fp8e5m2_to_fp32_value(0xBC), -1.0f);

    // Test subnormal numbers
    // EXPECT_EQ(fp8e5m2_from_fp32_value(1.0f / 8388608), 0x00);// smallest positive subnormal
    // EXPECT_FLOAT_EQ(fp8e5m2_to_fp32_value(0x01), 1.0f / 8388608);

    // Test rounding
    EXPECT_EQ(fp8e5m2_from_fp32_value(1.1f), 0x3C);// should round to 1.0 or 1.25
}

TEST(Float8E5M2Test, OverflowAndUnderflow) {
    // Test overflow (clamp to infinity)
    EXPECT_EQ(fp8e5m2_from_fp32_value(65504.0f), 0x7C); // max normal fp8e5m2
    EXPECT_EQ(fp8e5m2_from_fp32_value(100000.0f), 0x7C);// should clamp to infinity

    // Test underflow (flush to zero)
    EXPECT_EQ(fp8e5m2_from_fp32_value(1.0e-10f), 0x00);// should flush to zero
}

TEST(Float8E5M2Test, DenormalHandling) {
    // Test denormal inputs
    EXPECT_EQ(fp8e5m2_from_fp32_value(1.0f / 65536), 0x01); // should be represented as normal
    EXPECT_EQ(fp8e5m2_from_fp32_value(1.0f / 131072), 0x00);// should flush to zero
}

TEST(Float8E5M2Test, RoundingModes) {
    // Test rounding to nearest even
    EXPECT_EQ(fp8e5m2_from_fp32_value(1.5f), 0x3E);  // 1.5 should round to 1.5
    EXPECT_EQ(fp8e5m2_from_fp32_value(1.25f), 0x3D); // 1.25 should round to 1.25
    EXPECT_EQ(fp8e5m2_from_fp32_value(1.375f), 0x3E);// 1.375 should round to 1.25 (nearest even)
}

TEST(Float8E5M2Test, ConstructorAndBasicProperties) {
    // 默认构造函数
    Float8_e5m2 f1;
    EXPECT_EQ(f1.x, 0x00);

    // 从位模式构造
    Float8_e5m2 f2 = Float8_e5m2(0x3C, Float8_e5m2::from_bits());
    EXPECT_EQ(f2.x, 0x3C);

    // 从float构造
    Float8_e5m2 f3(1.0f);
    EXPECT_EQ(f3.x, 0x3C);

    // 转换到float
    float result = f3;
    EXPECT_FLOAT_EQ(result, 1.0f);
}

TEST(Float8E5M2Test, SpecialValues1) {
    // 零值
    Float8_e5m2 zero(0.0f);
    EXPECT_EQ(zero.x, 0x00);

    Float8_e5m2 neg_zero(-0.0f);
    EXPECT_EQ(neg_zero.x, 0x80);

    // 无穷大
    Float8_e5m2 inf(std::numeric_limits<float>::infinity());
    EXPECT_EQ(inf.x, 0x7C);
    EXPECT_TRUE(inf.isinf());

    Float8_e5m2 neg_inf(-std::numeric_limits<float>::infinity());
    EXPECT_EQ(neg_inf.x, 0xFC);
    EXPECT_TRUE(neg_inf.isinf());

    // NaN
    Float8_e5m2 nan(std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(nan.x, 0x7E);
    EXPECT_TRUE(nan.isnan());
}

TEST(Float8E5M2Test, NumericLimits) {
    // 最小值
    auto min_val = std::numeric_limits<Float8_e5m2>::min();
    EXPECT_EQ(min_val.x, 0x04);

    // 最大值
    auto max_val = std::numeric_limits<Float8_e5m2>::max();
    EXPECT_EQ(max_val.x, 0x7B);

    // 最低值（最小负值）
    auto lowest_val = std::numeric_limits<Float8_e5m2>::lowest();
    EXPECT_EQ(lowest_val.x, 0xFB);

    // 无穷大
    auto inf_val = std::numeric_limits<Float8_e5m2>::infinity();
    EXPECT_EQ(inf_val.x, 0x7C);

    // NaN
    auto nan_val = std::numeric_limits<Float8_e5m2>::quiet_NaN();
    EXPECT_EQ(nan_val.x, 0x7F);

    // 最小非规格化数
    auto denorm_min = std::numeric_limits<Float8_e5m2>::denorm_min();
    EXPECT_EQ(denorm_min.x, 0x01);
}


TEST(Float8E5M2Test, ArithmeticOperators) {
    Float8_e5m2 a(2.0f);
    Float8_e5m2 b(3.0f);

    // 加法
    Float8_e5m2 sum = a + b;
    EXPECT_FLOAT_EQ(sum, 5.0f);

    // 减法
    Float8_e5m2 diff = a - b;
    EXPECT_FLOAT_EQ(diff, -1.0f);

    // 乘法
    Float8_e5m2 prod = a * b;
    EXPECT_FLOAT_EQ(prod, 6.0f);

    // 除法
    Float8_e5m2 quot = a / b;
    EXPECT_FLOAT_EQ(quot, 2.0f / 3.0f);

    // 负号
    Float8_e5m2 neg = -a;
    EXPECT_FLOAT_EQ(neg, -2.0f);
}

TEST(Float8E5M2Test, CompoundAssignmentOperators) {
    Float8_e5m2 a(2.0f);
    Float8_e5m2 b(3.0f);

    // +=
    Float8_e5m2 c = a;
    c += b;
    EXPECT_FLOAT_EQ(c, 5.0f);

    // -=
    c = a;
    c -= b;
    EXPECT_FLOAT_EQ(c, -1.0f);

    // *=
    c = a;
    c *= b;
    EXPECT_FLOAT_EQ(c, 6.0f);

    // /=
    c = a;
    c /= b;
    EXPECT_FLOAT_EQ(c, 2.0f / 3.0f);
}

TEST(Float8E5M2Test, MixedTypeArithmetic) {
    Float8_e5m2 a(2.0f);

    // 与float运算
    float result1 = a + 3.0f;
    EXPECT_FLOAT_EQ(result1, 5.0f);

    float result2 = 3.0f + a;
    EXPECT_FLOAT_EQ(result2, 5.0f);

    // 与double运算
    double result3 = a + 3.0;
    EXPECT_DOUBLE_EQ(result3, 5.0);

    // 与int运算
    Float8_e5m2 result4 = a + 3;
    EXPECT_FLOAT_EQ(result4, 5.0f);

    // 与int64_t运算
    Float8_e5m2 result5 = a + static_cast<int64_t>(3);
    EXPECT_FLOAT_EQ(result5, 5.0f);
}

TEST(Float8E5M2Test, EdgeCasesAndRounding) {
    // 边界值测试
    Float8_e5m2 max_val(65504.0f);// E5M2最大规格化数
    EXPECT_EQ(max_val.x, 0x7B);

    Float8_e5m2 min_val(0.00006103515625f);// E5M2最小规格化数
    EXPECT_EQ(min_val.x, 0x04);

    // 下溢测试
    Float8_e5m2 tiny(1e-10f);
    EXPECT_EQ(tiny.x, 0x00);// 下溢到零

    // 溢出测试
    Float8_e5m2 huge(1e6f);
    EXPECT_EQ(huge.x, 0x7C);// 溢出到无穷大
}

TEST(Float8E5M2Test, OutputOperator) {
    Float8_e5m2 val(1.5f);
    std::ostringstream oss;
    oss << val;

    // 检查输出格式
    EXPECT_FALSE(oss.str().empty());
}

TEST(Float8E5M2Test, SpecialCases) {
    // NaN算术
    Float8_e5m2 nan_val = std::numeric_limits<Float8_e5m2>::quiet_NaN();
    Float8_e5m2 normal(1.0f);

    Float8_e5m2 result = nan_val + normal;
    EXPECT_TRUE(result.isnan());

    // 无穷大算术
    Float8_e5m2 inf_val = std::numeric_limits<Float8_e5m2>::infinity();
    result = inf_val + normal;
    EXPECT_TRUE(result.isinf());

    // 零除
    Float8_e5m2 zero(0.0f);
    result = normal / zero;
    EXPECT_TRUE(result.isinf());
}

TEST(Float8E5M2Test, FloatAssignmentOperators) {
    float f = 5.0f;
    Float8_e5m2 val(2.0f);

    // += with float
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

}// namespace