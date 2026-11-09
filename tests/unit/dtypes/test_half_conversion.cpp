/// @file test_half_conversion.cpp
/// @brief Binary16/binary32 conversion, rounding, and floating-point environment contracts.
#include "aethermind/dtypes/detail/floating_point_bits.h"
#include "aethermind/dtypes/half.h"

#include <bit>
#include <cfenv>
#include <cmath>
#include <gtest/gtest.h>
#include <initializer_list>
#include <limits>

namespace aethermind::detail {
uint32_t fp16_to_fp32_bits_for_testing(uint16_t h);
float fp16_to_fp32_value_for_testing(uint16_t h);
uint16_t fp16_from_fp32_value_for_testing(float f);
} // namespace aethermind::detail

namespace {
using namespace aethermind;
using namespace aethermind::detail;

TEST(HalfToFP32Test, HalfToFp32Bits_Zero) {
    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0x0000), 0x00000000); // +0
    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0x8000), 0x80000000); // -0
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

    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0x3555), 0x3EAAA000); // ~0.33325
    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0x48CD), 0x4119A000); // ~9.6016
    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0x4D12), 0x41A24000); // ~20.28125
}

TEST(HalfToFP32Test, HalfToFp32Bits_Infinity) {
    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0x7C00), 0x7F800000); // +inf
    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0xFC00), 0xFF800000); // -inf
}

TEST(HalfToFP32Test, HalfToFp32Bits_NaN) {
    // NaN payloads shift into the fp32 fraction, with the quiet bit forced on.
    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0x7C01), 0x7FC02000);
    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0x7FFF), 0x7FFFE000);
    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0x7E00), 0x7FC00000);
    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0x7F00), 0x7FE00000);

    // Sign is preserved.
    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0xFC01), 0xFFC02000);
    EXPECT_EQ(fp16_to_fp32_bits_for_testing(0xFFFF), 0xFFFFE000);
}

TEST(HalfToFP32Test, EveryNaNQuietsAndPreservesSignAndPayload) {
    for (uint32_t bits = 0; bits < 0x10000U; ++bits) {
        if ((bits & 0x7C00U) != 0x7C00U || (bits & 0x03FFU) == 0) {
            continue;
        }

        const Half source{static_cast<uint16_t>(bits), Half::from_bits()};
        const uint32_t results[] = {
                fp16_to_fp32_bits_for_testing(static_cast<uint16_t>(bits)),
                std::bit_cast<uint32_t>(fp16_to_fp32_value_for_testing(static_cast<uint16_t>(bits))),
                std::bit_cast<uint32_t>(static_cast<float>(source))};

        for (const uint32_t result: results) {
            EXPECT_EQ(result & 0x7FC00000U, 0x7FC00000U) << "source bits=" << bits;
            EXPECT_EQ(result >> 31, bits >> 15) << "source bits=" << bits;
            // The remaining nine payload bits are preserved; new low bits are zero.
            EXPECT_EQ(result & 0x003FFFFFU, (bits & 0x01FFU) << 13) << "source bits=" << bits;
        }
        EXPECT_EQ(source.bits(), bits) << "source bits=" << bits;
    }
}

TEST(HalfToFP32Test, SignalingNaNConversionDoesNotRaiseInvalid) {
    std::fenv_t environment;
    ASSERT_EQ(std::feholdexcept(&environment), 0);

    uint32_t quiet_bits = 0x00400000U;
    for (uint32_t bits = 0x7C01U; bits < 0x7E00U; ++bits) {
        for (const uint32_t sign: {0U, 0x8000U}) {
            const Half source{static_cast<uint16_t>(bits | sign), Half::from_bits()};
            quiet_bits &= std::bit_cast<uint32_t>(static_cast<float>(source));
        }
    }
    const int raised = std::fetestexcept(FE_INVALID);
    // Restore the caller's environment before any assertions can report failure.
    const int restored = std::fesetenv(&environment);

    EXPECT_EQ(restored, 0);
    EXPECT_EQ(quiet_bits, 0x00400000U);
    EXPECT_EQ(raised, 0);
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
    EXPECT_EQ(fp16_to_fp32_value_for_testing(0x0000), 0.0f);  // +0
    EXPECT_EQ(fp16_to_fp32_value_for_testing(0x8000), -0.0f); // -0

    // Sign bit must be preserved for zero.
    EXPECT_TRUE(std::signbit(fp16_to_fp32_value_for_testing(0x8000)));
    EXPECT_FALSE(std::signbit(fp16_to_fp32_value_for_testing(0x0000)));
}

TEST(HalfToFP32Test, HalfToFp32Value_Denormalized) {
    // Smallest positive denormal: 0x0001 -> 2^-24 ≈ 5.96046e-08
    float min_denormal = fp16_to_fp32_value_for_testing(0x0001);
    EXPECT_GT(min_denormal, 0.0f);
    EXPECT_LT(min_denormal, 1e-7f);

    // Largest denormal: 0x03FF, must be smaller than the smallest normal.
    float max_denormal = fp16_to_fp32_value_for_testing(0x03FF);
    EXPECT_GT(max_denormal, 0.0f);
    EXPECT_LT(max_denormal, 6.5e-5f);
}

TEST(HalfToFP32Test, HalfToFp32Value_Normalized) {
    // Min normal: 0x0400 -> 2^-14
    EXPECT_FLOAT_EQ(6.10351562e-5f, fp16_to_fp32_value_for_testing(0x0400));

    // 1.0
    EXPECT_FLOAT_EQ(1.0f, fp16_to_fp32_value_for_testing(0x3C00));

    // -1.0
    EXPECT_FLOAT_EQ(-1.0f, fp16_to_fp32_value_for_testing(0xBC00));

    // Max normal: 0x7BFF -> 65504.0
    EXPECT_FLOAT_EQ(65504.0f, fp16_to_fp32_value_for_testing(0x7BFF));

    // 2.0
    EXPECT_FLOAT_EQ(fp16_to_fp32_value_for_testing(0x4000), 2.0f);

    // 0.5
    EXPECT_FLOAT_EQ(fp16_to_fp32_value_for_testing(0x3800), 0.5f);

    EXPECT_NEAR(fp16_to_fp32_value_for_testing(0x3555), 0.33325f, 1e-5f); // ~1/3
    EXPECT_NEAR(fp16_to_fp32_value_for_testing(0x48CD), 9.6016f, 1e-3f);  // ~9.6
}

TEST(HalfToFP32Test, HalfToFp32Value_Infinity) {
    EXPECT_TRUE(std::isinf(fp16_to_fp32_value_for_testing(0x7C00))); // +inf
    EXPECT_GT(fp16_to_fp32_value_for_testing(0x7C00), 0);

    EXPECT_TRUE(std::isinf(fp16_to_fp32_value_for_testing(0xFC00))); // -inf
    EXPECT_LT(fp16_to_fp32_value_for_testing(0xFC00), 0);
}

TEST(HalfToFP32Test, HalfToFp32Value_NaN) {
    float nan1 = fp16_to_fp32_value_for_testing(0x7C01); // signaling input becomes quiet
    float nan2 = fp16_to_fp32_value_for_testing(0x7FFF); // quiet NaN
    float nan3 = fp16_to_fp32_value_for_testing(0x7E00); // quiet NaN
    float nan4 = fp16_to_fp32_value_for_testing(0xFC01); // negative signaling input becomes quiet

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
    float max_normal = fp16_to_fp32_value_for_testing(0x7BFF);
    EXPECT_NEAR(max_normal, 65504.0f, 1e-3f);
    EXPECT_FALSE(std::isinf(max_normal));

    // Min normal: 0x0400 -> 2^-14 ≈ 6.10352e-05
    float min_normal = fp16_to_fp32_value_for_testing(0x0400);
    EXPECT_GT(min_normal, 0.0f);
    EXPECT_LT(min_normal, 1e-4f);

    // Denormal boundary: max denormal < min normal.
    float last_denormal = fp16_to_fp32_value_for_testing(0x03FF);
    float first_normal = fp16_to_fp32_value_for_testing(0x0400);
    EXPECT_LT(last_denormal, first_normal);
}

TEST(HalfToFP32Test, HalfToFp32Value_SpecialValues) {
    // PI approximation: 0x4248 -> ~3.140625
    EXPECT_NEAR(fp16_to_fp32_value_for_testing(0x4248), 3.140625f, 1e-6f);

    // E approximation: 0x4170 -> ~2.71875
    EXPECT_NEAR(fp16_to_fp32_value_for_testing(0x4170), 2.71875f, 1e-6f);

    // 0x3FCF -> 1 + 975/1024 ~= 1.95215
    EXPECT_NEAR(fp16_to_fp32_value_for_testing(0x3FCF), 1.95215f, 1e-3f);
}

TEST(HalfToFP32Test, HalfToFp32Value_RoundTripConsistency) {
    // `fp16_to_fp32_value_for_testing` must match `fp32_from_bits(fp16_to_fp32_bits_for_testing(...))`.
    const uint16_t test_values[] = {
            0x0000, 0x0001, 0x03FF, 0x0400, 0x3C00, 0x4000,
            0x7C00, 0x7E00, 0x7FFF, 0x8000, 0xBC00, 0xFC00};

    for (auto half_val: test_values) {
        uint32_t bits = fp16_to_fp32_bits_for_testing(half_val);
        float value_from_bits = fp32_from_bits(bits);
        float direct_value = fp16_to_fp32_value_for_testing(half_val);

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
            uint16_t exponent = (exp + 15) << 10; // biased exponent
            uint16_t half_val = exponent | mantissa;

            if ((half_val & 0x7C00) != 0x7C00) { // skip inf/nan
                float value = fp16_to_fp32_value_for_testing(half_val);

                if (!std::isinf(value) && !std::isnan(value)) {
                    EXPECT_TRUE(std::isfinite(value));
                }
            }
        }
    }
}


TEST(HalfFromFP32Test, ZeroValues) {
    EXPECT_EQ(fp16_from_fp32_value_for_testing(0.0f), 0x0000);  // +0
    EXPECT_EQ(fp16_from_fp32_value_for_testing(-0.0f), 0x8000); // -0
}

TEST(HalfFromFP32Test, DenormalizedNumbers) {
    // Even the largest binary32 subnormal is below binary16's rounding range.
    float smallest_denormal = std::numeric_limits<float>::denorm_min();
    EXPECT_EQ(fp16_from_fp32_value_for_testing(smallest_denormal), 0x0000);

    float max_denormal = 1.1754942e-38f; // ~2^-126 * (1 - 2^-23)
    EXPECT_EQ(fp16_from_fp32_value_for_testing(max_denormal), 0x0000);
}

TEST(HalfFromFP32Test, NormalizedNumbers) {
    EXPECT_EQ(fp16_from_fp32_value_for_testing(1.0f), 0x3C00);
    EXPECT_EQ(fp16_from_fp32_value_for_testing(2.0f), 0x4000);
    EXPECT_EQ(fp16_from_fp32_value_for_testing(0.5f), 0x3800);
    EXPECT_EQ(fp16_from_fp32_value_for_testing(-1.0f), 0xBC00);

    // The binary32 minimum normal is far below the binary16 subnormal range.
    float smallest_normal = 1.17549435e-38f; // 2^-126
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

TEST(HalfFromFP32Test, NaNPayloadPreserved) {
    // Construct a quiet NaN with a deterministic payload in the top 10 mantissa bits.
    constexpr uint32_t qnan_payload = 0x2AAU << 13; // bits 22-13
    constexpr uint32_t qnan_bits = 0x7FC00000U | qnan_payload;
    const float qnan = std::bit_cast<float>(qnan_bits);
    const uint16_t half_nan = fp16_from_fp32_value_for_testing(qnan);

    // Sign preserved as positive, exponent is all-ones (NaN), payload is preserved.
    EXPECT_EQ(half_nan & 0x8000U, 0x0000U);
    EXPECT_EQ(half_nan & 0x7C00U, 0x7C00U);
    EXPECT_EQ(half_nan & 0x03FFU, 0x2AAU | 0x0200U);

    // A NaN whose payload is too small to survive the 13-bit right shift still
    // produces a non-zero quiet-NaN payload.
    constexpr uint32_t tiny_payload_bits = 0x7FC00001U;
    const float tiny_qnan = std::bit_cast<float>(tiny_payload_bits);
    const uint16_t tiny_half_nan = fp16_from_fp32_value_for_testing(tiny_qnan);
    EXPECT_EQ(tiny_half_nan & 0x7C00U, 0x7C00U);
    EXPECT_EQ(tiny_half_nan & 0x03FFU, 0x0200U);
}

TEST(HalfFromFP32Test, Overflow) {
    // Values exceeding the half-precision max saturate to ±inf.
    EXPECT_EQ(fp16_from_fp32_value_for_testing(70000.0f), 0x7C00);
    EXPECT_EQ(fp16_from_fp32_value_for_testing(-70000.0f), 0xFC00);
}

TEST(HalfFromFP32Test, Underflow) {
    // Magnitudes below 2^-25 round to signed zero.
    EXPECT_EQ(fp16_from_fp32_value_for_testing(1e-10f), 0x0000);
    EXPECT_EQ(fp16_from_fp32_value_for_testing(-1e-10f), 0x8000);
}

TEST(HalfFromFP32Test, ExactRepresentableValues) {
    // Both values are exactly representable in binary16; no rounding occurs.
    EXPECT_EQ(fp16_from_fp32_value_for_testing(1.0009765625f), 0x3C01);
    EXPECT_EQ(fp16_from_fp32_value_for_testing(1.001953125f), 0x3C02);
}

TEST(HalfFromFP32Test, RoundToNearestEvenTies) {
    // Midpoints between adjacent binary16 values must break to the even mantissa.
    // At exponent 0 the binary16 ULP is 2^-10.
    EXPECT_EQ(fp16_from_fp32_value_for_testing(1.00048828125f), 0x3C00); // 1 + 2^-11: even lower
    EXPECT_EQ(fp16_from_fp32_value_for_testing(1.00146484375f), 0x3C02); // 1 + 3*2^-11: even upper

    // At exponent 1 the binary16 ULP is 2^-9.
    EXPECT_EQ(fp16_from_fp32_value_for_testing(2.0009765625f), 0x4000); // 2 + 2^-10: even lower
    EXPECT_EQ(fp16_from_fp32_value_for_testing(2.0029296875f), 0x4002); // 2 + 3*2^-10: even upper

    // Sign is preserved on both tie directions.
    EXPECT_EQ(fp16_from_fp32_value_for_testing(-1.00048828125f), 0xBC00);
    EXPECT_EQ(fp16_from_fp32_value_for_testing(-1.00146484375f), 0xBC02);
}

TEST(HalfFromFP32Test, RoundingOverflowBoundary) {
    // The largest finite binary16 value is 65504; the next step is infinity and
    // the midpoint 65520 breaks to the even (infinite) encoding.
    EXPECT_EQ(fp16_from_fp32_value_for_testing(65504.0f), 0x7BFF);
    EXPECT_EQ(fp16_from_fp32_value_for_testing(65519.0f), 0x7BFF);
    EXPECT_EQ(fp16_from_fp32_value_for_testing(65520.0f), 0x7C00);
    EXPECT_EQ(fp16_from_fp32_value_for_testing(65521.0f), 0x7C00);

    // Values beyond the rounding boundary overflow directly.
    EXPECT_EQ(fp16_from_fp32_value_for_testing(65536.0f), 0x7C00);

    // Sign is preserved when rounding to infinity.
    EXPECT_EQ(fp16_from_fp32_value_for_testing(-65519.0f), 0xFBFF);
    EXPECT_EQ(fp16_from_fp32_value_for_testing(-65520.0f), 0xFC00);
}

TEST(HalfFromFP32Test, SpecialValues) {
    EXPECT_EQ(fp16_from_fp32_value_for_testing(3.141592653589793f), 0x4248); // PI
    EXPECT_EQ(fp16_from_fp32_value_for_testing(2.718281828459045f), 0x4170); // E
    EXPECT_EQ(fp16_from_fp32_value_for_testing(1.618033988749895f), 0x3E79); // Golden ratio
}

TEST(HalfFromFP32Test, RoundTrip) {
    // fp32 -> fp16 -> fp32 must recover the original within half-precision error.
    float original = 1.2345f;
    uint16_t half_val = fp16_from_fp32_value_for_testing(original);
    float reconstructed = fp16_to_fp32_value_for_testing(half_val);
    EXPECT_NEAR(original, reconstructed, 1e-3);
}

// Reconstruct finite positive binary16 values independently of production
// widening. All intermediate dyadic values are exact in double and float.
float PositiveHalfValue(uint16_t bits) {
    const uint32_t exponent = bits >> 10;
    const uint32_t fraction = bits & 0x03FFU;
    const double value = exponent == 0
                                 ? std::ldexp(static_cast<double>(fraction), -24)
                                 : std::ldexp(static_cast<double>(0x0400U + fraction),
                                              static_cast<int>(exponent) - 25);
    return static_cast<float>(value);
}

void ExpectNarrowingForBothSigns(float positive, uint16_t expected) {
    for (const uint32_t sign: {0U, 0x80000000U}) {
        const uint32_t bits = std::bit_cast<uint32_t>(positive) | sign;
        const auto value = std::bit_cast<float>(bits);
        EXPECT_EQ(Half(value).bits(), expected | (sign >> 16)) << "source bits=" << bits;
    }
}

TEST(HalfNarrowing, CreatesSubnormalResults) {
    ExpectNarrowingForBothSigns(0x1p-24f, 0x0001);
    ExpectNarrowingForBothSigns(0x1p-23f, 0x0002);
    ExpectNarrowingForBothSigns(0x1.ff8p-15f, 0x03FF);
}

TEST(HalfNarrowing, EveryFiniteHalfValueNarrowsExactly) {
    for (uint32_t bits = 0; bits < 0x7C00U; ++bits) {
        const auto expected = static_cast<uint16_t>(bits);
        ExpectNarrowingForBothSigns(PositiveHalfValue(expected), expected);
    }
}

TEST(HalfNarrowing, EveryFiniteMidpointRoundsToNearestEven) {
    for (uint32_t lower = 0; lower < 0x7BFFU; ++lower) {
        const double a = PositiveHalfValue(static_cast<uint16_t>(lower));
        const double b = PositiveHalfValue(static_cast<uint16_t>(lower + 1U));
        // Adjacent binary16 midpoints are exactly representable in binary32.
        const auto midpoint = static_cast<float>((a + b) * 0.5);
        const auto even = static_cast<uint16_t>(lower + (lower & 1U));

        ExpectNarrowingForBothSigns(std::nextafter(midpoint, 0.0f), static_cast<uint16_t>(lower));
        ExpectNarrowingForBothSigns(midpoint, even);
        ExpectNarrowingForBothSigns(std::nextafter(midpoint, std::numeric_limits<float>::infinity()),
                                    static_cast<uint16_t>(lower + 1U));
    }
}

TEST(HalfNarrowing, OverflowMidpointRoundsToInfinity) {
    constexpr float midpoint = 65520.0f;
    ExpectNarrowingForBothSigns(std::nextafter(midpoint, 0.0f), 0x7BFF);
    ExpectNarrowingForBothSigns(midpoint, 0x7C00);
    ExpectNarrowingForBothSigns(std::nextafter(midpoint, std::numeric_limits<float>::infinity()), 0x7C00);
}

TEST(HalfNarrowing, NaNsQuietAndPreserveRetainedPayload) {
    for (uint32_t retained = 0; retained < 0x0400U; ++retained) {
        for (const uint32_t tail: {0U, 1U, 0x1FFFU}) {
            const uint32_t fraction = (retained << 13) | tail;
            if (fraction == 0) {
                continue;
            }
            const auto value = std::bit_cast<float>(0x7F800000U | fraction);
            ExpectNarrowingForBothSigns(value, static_cast<uint16_t>(0x7E00U | (retained & 0x01FFU)));
        }
    }
}

TEST(HalfNarrowing, ArithmeticPreservesSubnormalResults) {
    const Half normal{0x0400, Half::from_bits()};
    const Half below{0x03FF, Half::from_bits()};
    EXPECT_EQ((normal - below).bits(), 0x0001U);
    EXPECT_EQ((normal * Half(0.5f)).bits(), 0x0200U);

    Half compound = normal;
    compound /= Half(1024.0f);
    EXPECT_EQ(compound.bits(), 0x0001U);
}

struct ConversionCase {
    uint32_t source;
    uint16_t expected;
};

constexpr ConversionCase kEnvironmentCases[] = {
        {0x00000000U, 0x0000}, // zero
        {0x00000001U, 0x0000}, // binary32 subnormal
        {0x33000000U, 0x0000}, // zero/subnormal midpoint
        {0x33000001U, 0x0001}, // immediately above that midpoint
        {0x33800000U, 0x0001}, // smallest binary16 subnormal
        {0x33C00000U, 0x0002}, // tie between first and second subnormals
        {0x387FE000U, 0x0400}, // subnormal/normal midpoint
        {0x3F801000U, 0x3C00}, // normal tie, even lower
        {0x3F803000U, 0x3C02}, // normal tie, even upper
        {0x477FF000U, 0x7C00}, // overflow midpoint
        {0x7F7FFFFFU, 0x7C00}, // largest binary32 finite value
        {0x7F800000U, 0x7C00}, // infinity
        {0x7F800001U, 0x7E00}, // signaling NaN, retained payload is zero
        {0x7FC00000U, 0x7E00}, // quiet NaN
};

class HalfNarrowingEnvironment : public ::testing::Test {
protected:
    void SetUp() override {
        saved_ = std::feholdexcept(&environment_) == 0;
        ASSERT_TRUE(saved_);
    }

    void TearDown() override {
        if (saved_) {
            EXPECT_EQ(std::fesetenv(&environment_), 0);
        }
    }

private:
    std::fenv_t environment_{};
    bool saved_ = false;
};

TEST_F(HalfNarrowingEnvironment, FixedRoundingAcrossAllModes) {
    for (const int mode: {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        ASSERT_EQ(std::fesetround(mode), 0);
        for (const auto& [bits, expected]: kEnvironmentCases) {
            ExpectNarrowingForBothSigns(std::bit_cast<float>(bits), expected);
        }
        EXPECT_EQ(std::fegetround(), mode);
    }
}

TEST_F(HalfNarrowingEnvironment, ConversionDoesNotRaiseFloatingPointExceptions) {
    for (const auto& [bits, expected]: kEnvironmentCases) {
        ExpectNarrowingForBothSigns(std::bit_cast<float>(bits), expected);
    }
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), 0);
}

} // namespace
