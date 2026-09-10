#include "aethermind/operators/rope_frequency_resolver.h"

#include <gtest/gtest.h>

#include <cmath>

namespace {

using namespace aethermind;

RoPEParams MakeParams() {
    return RoPEParams{
            .head_dim = 4,
            .rotary_dim = 4,
            .num_attention_heads = 1,
            .num_key_value_heads = 1,
            .max_pos_embeddings = 4,
            .theta = 4.0,
    };
}

TEST(RoPEFrequencyResolver, ResolvesStandardLinearAndDynamicNtkGolden) {
    auto params = MakeParams();
    auto standard = ResolveStaticRoPEFreqs(params);
    ASSERT_TRUE(standard.ok()) << standard.status().ToString();
    ASSERT_EQ(standard->inv_freqs.size(), 2U);
    EXPECT_DOUBLE_EQ(standard->inv_freqs[0], 1.0);
    EXPECT_DOUBLE_EQ(standard->inv_freqs[1], 0.5);

    params.algorithm = LinearRoPE{.factor = 2.0};
    auto linear = ResolveStaticRoPEFreqs(params);
    ASSERT_TRUE(linear.ok()) << linear.status().ToString();
    EXPECT_DOUBLE_EQ(linear->inv_freqs[0], 0.5);
    EXPECT_DOUBLE_EQ(linear->inv_freqs[1], 0.25);

    params.algorithm = DynamicNtkRoPE{.factor = 2.0, .original_context_length = 4};
    auto dynamic = ResolveDynamicRoPEFreqs(params, 8);
    ASSERT_TRUE(dynamic.ok()) << dynamic.status().ToString();
    // base = 4 * (2 * 8 / 4 - 1) ^ (4 / (4 - 2)) = 36.
    EXPECT_DOUBLE_EQ(dynamic->inv_freqs[0], 1.0);
    EXPECT_NEAR(dynamic->inv_freqs[1], 1.0 / 6.0, 1.0e-12);
}

TEST(RoPEFrequencyResolver, ResolvesYarnLlama3AndLongRope) {
    auto params = MakeParams();
    params.algorithm = YarnRoPE{.factor = 2.0,
                                .original_context_length = 8,
                                .beta_fast = 32.0,
                                .beta_slow = 1.0,
                                .rotary_output_scale = 1.5,
                                .truncate_correction_range = false};
    auto yarn = ResolveStaticRoPEFreqs(params);
    ASSERT_TRUE(yarn.ok()) << yarn.status().ToString();
    EXPECT_DOUBLE_EQ(yarn->rotary_output_scale, 1.5);
    // HF's extrapolation ramp is zero for pair 0 and one for pair 1 here.
    EXPECT_NEAR(yarn->inv_freqs[0], 1.0, 1.0e-12);
    EXPECT_NEAR(yarn->inv_freqs[1], 0.25, 1.0e-12);

    params.theta = 100.0;
    params.algorithm = YarnRoPE{.factor = 2.0,
                                .original_context_length = 128,
                                .beta_fast = 32.0,
                                .beta_slow = 1.0,
                                .rotary_output_scale = 1.0,
                                .truncate_correction_range = false};
    auto yarn_raw = ResolveStaticRoPEFreqs(params);
    ASSERT_TRUE(yarn_raw.ok()) << yarn_raw.status().ToString();
    std::get<YarnRoPE>(params.algorithm).truncate_correction_range = true;
    auto yarn_truncated = ResolveStaticRoPEFreqs(params);
    ASSERT_TRUE(yarn_truncated.ok()) << yarn_truncated.status().ToString();
    // Raw high≈1.309 makes pair 1's ramp≈0.764; truncating it to ceil(2)
    // makes the same ramp 0.5. The unscaled second frequency is 0.1.
    const double raw_high = 4.0 * std::log(128.0 / (2.0 * std::acos(-1.0))) /
                            (2.0 * std::log(100.0));
    const double raw_ramp = 1.0 / raw_high;
    EXPECT_NEAR(yarn_raw->inv_freqs[1],
                0.1 * (1.0 - raw_ramp) + 0.05 * raw_ramp, 1.0e-12);
    EXPECT_NEAR(yarn_truncated->inv_freqs[1], 0.075, 1.0e-12);

    params.theta = 4.0;
    params.algorithm = Llama3RoPE{.factor = 2.0,
                                  .low_frequency_factor = 1.0,
                                  .high_frequency_factor = 4.0,
                                  .original_context_length = 16};
    auto llama3 = ResolveStaticRoPEFreqs(params);
    ASSERT_TRUE(llama3.ok()) << llama3.status().ToString();
    // Both wavelengths fall in Llama3's smooth interval for this fixture.
    const double first_wavelength = 2.0 * std::acos(-1.0);
    const double first_smooth = (16.0 / first_wavelength - 1.0) / 3.0;
    EXPECT_NEAR(llama3->inv_freqs[0],
                (1.0 - first_smooth) * 0.5 + first_smooth, 1.0e-12);
    const double wavelength = 4.0 * std::acos(-1.0);
    const double smooth = (16.0 / wavelength - 1.0) / 3.0;
    EXPECT_NEAR(llama3->inv_freqs[1],
                (1.0 - smooth) * 0.25 + smooth * 0.5, 1.0e-12);

    params.algorithm = LongRoPE{.short_factors = {1.0, 2.0},
                                .long_factors = {2.0, 4.0},
                                .original_context_length = 4,
                                .rotary_output_scale = 1.25};
    auto short_table = ResolveDynamicRoPEFreqs(params, 4);
    auto long_table = ResolveDynamicRoPEFreqs(params, 5);
    ASSERT_TRUE(short_table.ok()) << short_table.status().ToString();
    ASSERT_TRUE(long_table.ok()) << long_table.status().ToString();
    EXPECT_DOUBLE_EQ(short_table->inv_freqs[0], 1.0);
    EXPECT_DOUBLE_EQ(short_table->inv_freqs[1], 0.25);
    EXPECT_DOUBLE_EQ(long_table->inv_freqs[0], 0.5);
    EXPECT_DOUBLE_EQ(long_table->inv_freqs[1], 0.125);
    EXPECT_DOUBLE_EQ(long_table->rotary_output_scale, 1.25);
}

TEST(RoPEFrequencyResolver, RejectsInvalidTypedVariants) {
    auto params = MakeParams();
    params.rotary_dim = 3;
    EXPECT_FALSE(ValidateRoPEFreqParams(params).ok());
    params.rotary_dim = 4;
    params.algorithm = DynamicNtkRoPE{.factor = 2.0, .original_context_length = 0};
    EXPECT_FALSE(ValidateRoPEFreqParams(params).ok());
    params.algorithm = YarnRoPE{.factor = 2.0,
                                .original_context_length = 8,
                                .beta_fast = 1.0,
                                .beta_slow = 1.0};
    EXPECT_FALSE(ValidateRoPEFreqParams(params).ok());
    params.algorithm = LongRoPE{.short_factors = {1.0},
                                .long_factors = {1.0},
                                .original_context_length = 4};
    EXPECT_FALSE(ValidateRoPEFreqParams(params).ok());
}

} // namespace
