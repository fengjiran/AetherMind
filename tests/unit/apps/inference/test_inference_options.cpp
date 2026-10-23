#include "inference_runner.h"

#include "aethermind/model/formats/hf/hf_model_config.h"
#include "model/test_llama_checkpoint_helpers.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace {

using namespace aethermind;
using namespace aethermind::apps::inference;

HfModelConfig Configuration() {
    return test::MakeTinyLlamaConfig(2, false);
}

RunOptions Options() {
    RunOptions options;
    options.model_dir = "model-directory";
    options.prompt_tokens = {1, 2};
    options.context_tokens = 8;
    options.generation.max_new_tokens = 3;
    return options;
}

TEST(InferenceRunOptions, DefaultsToO1Plain) {
    EXPECT_EQ(RunOptions{}.preset, CpuInferencePreset::kO1Plain);
}

TEST(InferenceRunOptions, AcceptsValidRequestForBothPresets) {
    for (const CpuInferencePreset preset: {CpuInferencePreset::kO1Plain, CpuInferencePreset::kO2Packed}) {
        RunOptions options = Options();
        options.preset = preset;
        const Status status = ValidateRunOptions(options, Configuration());
        EXPECT_TRUE(status.ok()) << status.ToString();
    }
}

TEST(InferenceRunOptions, RejectsEmptyModelDirectory) {
    RunOptions options = Options();
    options.model_dir.clear();
    const Status status = ValidateRunOptions(options, Configuration());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(InferenceRunOptions, RejectsEmptyPrompt) {
    RunOptions options = Options();
    options.prompt_tokens.clear();
    const Status status = ValidateRunOptions(options, Configuration());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(InferenceRunOptions, ChecksVocabularyBoundaryForEveryPromptToken) {
    const HfModelConfig config = Configuration();
    RunOptions options = Options();
    options.prompt_tokens = {0, static_cast<uint32_t>(config.vocab_size - 1)};
    EXPECT_TRUE(ValidateRunOptions(options, config).ok());

    options.prompt_tokens.back() = static_cast<uint32_t>(config.vocab_size);
    const Status status = ValidateRunOptions(options, config);
    EXPECT_EQ(status.code(), StatusCode::kOutOfRange);
}

TEST(InferenceRunOptions, ChecksExplicitEosVocabularyBoundary) {
    const HfModelConfig config = Configuration();
    RunOptions options = Options();
    options.generation.eos_token_id = static_cast<uint32_t>(config.vocab_size - 1);
    EXPECT_TRUE(ValidateRunOptions(options, config).ok());

    options.generation.eos_token_id = static_cast<uint32_t>(config.vocab_size);
    const Status status = ValidateRunOptions(options, config);
    EXPECT_EQ(status.code(), StatusCode::kOutOfRange);
}

TEST(InferenceRunOptions, RejectsPromptBeyondModelContext) {
    HfModelConfig config = Configuration();
    config.max_position_embeddings = 1;
    RunOptions options = Options();
    options.context_tokens = 1;
    options.generation.max_new_tokens = 0;
    const Status status = ValidateRunOptions(options, config);
    EXPECT_EQ(status.code(), StatusCode::kOutOfRange);
}

TEST(InferenceRunOptions, RequiresPositiveCapacityEvenForZeroGeneration) {
    RunOptions options = Options();
    options.context_tokens = 0;
    options.generation.max_new_tokens = 0;
    const Status status = ValidateRunOptions(options, Configuration());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(InferenceRunOptions, RejectsCapacityBeyondModelContext) {
    const HfModelConfig config = Configuration();
    RunOptions options = Options();
    options.context_tokens = static_cast<size_t>(config.max_position_embeddings) + 1;
    const Status status = ValidateRunOptions(options, config);
    EXPECT_EQ(status.code(), StatusCode::kOutOfRange);
}

TEST(InferenceRunOptions, CountsOnlyDecodeAppendsAgainstCapacity) {
    RunOptions options = Options();
    options.context_tokens = options.prompt_tokens.size() + options.generation.max_new_tokens - 1;
    const Status status = ValidateRunOptions(options, Configuration());
    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(InferenceRunOptions, RejectsOneTokenBeyondCapacity) {
    RunOptions options = Options();
    options.context_tokens = options.prompt_tokens.size() + options.generation.max_new_tokens - 2;
    const Status status = ValidateRunOptions(options, Configuration());
    EXPECT_EQ(status.code(), StatusCode::kOutOfRange);
}

TEST(InferenceRunOptions, DoesNotRelyOnEosToReduceRequestedCapacity) {
    RunOptions options = Options();
    options.context_tokens = options.prompt_tokens.size();
    options.generation.eos_token_id = 1;
    const Status status = ValidateRunOptions(options, Configuration());
    EXPECT_EQ(status.code(), StatusCode::kOutOfRange);
}

TEST(InferenceRunOptions, RejectsRequestedTokenCountOverflow) {
    RunOptions options = Options();
    options.generation.max_new_tokens = std::numeric_limits<size_t>::max();
    const Status status = ValidateRunOptions(options, Configuration());
    EXPECT_EQ(status.code(), StatusCode::kOverflow);
}

TEST(InferenceRunOptions, ZeroGenerationAllowsPromptBeyondPhysicalCapacity) {
    RunOptions options = Options();
    options.context_tokens = 1;
    options.generation.max_new_tokens = 0;
    const Status status = ValidateRunOptions(options, Configuration());
    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(InferenceRunOptions, ZeroGenerationStillRejectsInvalidPromptToken) {
    const HfModelConfig config = Configuration();
    RunOptions options = Options();
    options.generation.max_new_tokens = 0;
    options.prompt_tokens.back() = static_cast<uint32_t>(config.vocab_size);
    const Status status = ValidateRunOptions(options, config);
    EXPECT_EQ(status.code(), StatusCode::kOutOfRange);
}

TEST(InferenceRunOptions, ZeroGenerationStillRejectsInvalidEos) {
    const HfModelConfig config = Configuration();
    RunOptions options = Options();
    options.generation.max_new_tokens = 0;
    options.generation.eos_token_id = static_cast<uint32_t>(config.vocab_size);
    const Status status = ValidateRunOptions(options, config);
    EXPECT_EQ(status.code(), StatusCode::kOutOfRange);
}

TEST(InferenceRunOptions, RejectsUnknownPreset) {
    RunOptions options = Options();
    options.preset = static_cast<CpuInferencePreset>(255);
    const Status status = ValidateRunOptions(options, Configuration());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(InferenceRunOptions, RejectsNonpositiveModelBoundsBeforeUnsignedConversion) {
    for (const int64_t value: {int64_t{0}, int64_t{-1}}) {
        SCOPED_TRACE(value);
        HfModelConfig config = Configuration();
        config.vocab_size = value;
        const Status vocab_status = ValidateRunOptions(Options(), config);
        EXPECT_EQ(vocab_status.code(), StatusCode::kInvalidArgument);
        EXPECT_EQ(vocab_status.message(), "vocab_size must be positive");
        config = Configuration();
        config.max_position_embeddings = value;
        const Status context_status = ValidateRunOptions(Options(), config);
        EXPECT_EQ(context_status.code(), StatusCode::kInvalidArgument);
        EXPECT_EQ(context_status.message(), "max_position_embeddings must be positive");
    }
}

TEST(InferenceRuntimeOptions, MapsGeometryAndSelectsOnlyCpuFp32Resources) {
    const HfModelConfig config = Configuration();
    const auto mapped = MakeRuntimeOptions(config, 8);
    ASSERT_TRUE(mapped.ok()) << mapped.status().ToString();
    EXPECT_TRUE(mapped->allocator.enable_cpu);
    EXPECT_FALSE(mapped->allocator.enable_cuda);
    EXPECT_FALSE(mapped->allocator.enable_cann);
    EXPECT_TRUE(mapped->backend.enable_cpu);
    EXPECT_FALSE(mapped->backend.enable_cuda);
    EXPECT_FALSE(mapped->backend.enable_cann);
    EXPECT_TRUE(mapped->kv_cache.enable_manager);
    EXPECT_EQ(mapped->kv_cache.num_layers, 2U);
    EXPECT_EQ(mapped->kv_cache.num_kv_heads, 2U);
    EXPECT_EQ(mapped->kv_cache.head_dim, 2U);
    EXPECT_EQ(mapped->kv_cache.max_tokens, 8U);
    EXPECT_EQ(mapped->kv_cache.kv_dtype, DataType::Float32());
    EXPECT_EQ(mapped->kv_cache.alignment, 64U);
    EXPECT_FALSE(mapped->workspace.enable_workspace_manager);
    EXPECT_EQ(mapped->workspace.default_workspace_limit_bytes, 0U);
}

TEST(InferenceRuntimeOptions, UsesPositiveExplicitHeadDimension) {
    HfModelConfig config = Configuration();
    config.head_dim = 4;
    config.hidden_size = config.num_attention_heads * config.head_dim;
    const auto mapped = MakeRuntimeOptions(config, 8);
    ASSERT_TRUE(mapped.ok()) << mapped.status().ToString();
    EXPECT_EQ(mapped->kv_cache.head_dim, 4U);
}

TEST(InferenceRuntimeOptions, DerivesAbsentHeadDimensionAfterExactDivision) {
    HfModelConfig config = Configuration();
    config.head_dim = 0;
    config.hidden_size = 12;
    const auto mapped = MakeRuntimeOptions(config, 8);
    ASSERT_TRUE(mapped.ok()) << mapped.status().ToString();
    EXPECT_EQ(mapped->kv_cache.head_dim, 3U);
}

TEST(InferenceRuntimeOptions, RejectsNonintegralDerivedHeadDimension) {
    HfModelConfig config = Configuration();
    config.head_dim = 0;
    config.hidden_size = 9;
    const auto mapped = MakeRuntimeOptions(config, 8);
    EXPECT_FALSE(mapped.ok());
}

TEST(InferenceRuntimeOptions, RejectsNonpositiveDimensionsBeforeUnsignedConversion) {
    constexpr std::array<int64_t HfModelConfig::*, 2> fields{
            &HfModelConfig::num_hidden_layers, &HfModelConfig::num_key_value_heads};
    for (const auto field: fields) {
        for (const int64_t value: {int64_t{0}, int64_t{-1}}) {
            SCOPED_TRACE(value);
            HfModelConfig config = Configuration();
            config.*field = value;
            const auto mapped = MakeRuntimeOptions(config, 8);
            EXPECT_FALSE(mapped.ok());
        }
    }
}

TEST(InferenceRuntimeOptions, RejectsNegativeExplicitHeadDimension) {
    HfModelConfig config = Configuration();
    config.head_dim = -1;
    const auto mapped = MakeRuntimeOptions(config, 8);
    EXPECT_FALSE(mapped.ok());
}

TEST(InferenceRuntimeOptions, RejectsInvalidOperandsWhenDerivingHeadDimension) {
    constexpr std::array<int64_t HfModelConfig::*, 2> fields{
            &HfModelConfig::hidden_size, &HfModelConfig::num_attention_heads};
    for (const auto field: fields) {
        for (const int64_t value: {int64_t{0}, int64_t{-1}}) {
            SCOPED_TRACE(value);
            HfModelConfig config = Configuration();
            config.head_dim = 0;
            config.*field = value;
            EXPECT_FALSE(MakeRuntimeOptions(config, 8).ok());
        }
    }
}

TEST(InferenceRuntimeOptions, RejectsInvalidCapacity) {
    const HfModelConfig config = Configuration();
    EXPECT_FALSE(MakeRuntimeOptions(config, 0).ok());
    EXPECT_FALSE(MakeRuntimeOptions(config, static_cast<size_t>(config.max_position_embeddings) + 1).ok());
}

TEST(InferenceRuntimeOptions, ReturnsValuesIndependentOfSourceConfigLifetime) {
    const auto mapped = [] {
        HfModelConfig config = Configuration();
        auto result = MakeRuntimeOptions(config, 8);
        config.num_hidden_layers = 99;
        config.num_key_value_heads = 99;
        config.head_dim = 99;
        return result;
    }();
    ASSERT_TRUE(mapped.ok()) << mapped.status().ToString();
    EXPECT_EQ(mapped->kv_cache.num_layers, 2U);
    EXPECT_EQ(mapped->kv_cache.num_kv_heads, 2U);
    EXPECT_EQ(mapped->kv_cache.head_dim, 2U);
}

TEST(InferenceCpuPreset, ExplicitlySelectsO1PlainAndBothPhases) {
    const auto compile = MakeCpuCompileOptions(CpuInferencePreset::kO1Plain);
    ASSERT_TRUE(compile.ok()) << compile.status().ToString();
    EXPECT_EQ(compile->optimization.opt_level, 1);
    EXPECT_FALSE(compile->lowering.enable_packed_weights);
    EXPECT_EQ(compile->lowering.selector.device_type, DeviceType::kCPU);
    EXPECT_EQ(compile->lowering.selector.weight_format, WeightFormat::kPlain);
    EXPECT_EQ(compile->lowering.selector.phase, ExecPhase::kBoth);
    EXPECT_TRUE(compile->lowering.selector.act_dtype.IsUndefined());
    EXPECT_TRUE(compile->lowering.selector.weight_dtype.IsUndefined());
}

TEST(InferenceCpuPreset, ExplicitlySelectsO2PackedWithoutForcingAllStepsPacked) {
    const auto compile = MakeCpuCompileOptions(CpuInferencePreset::kO2Packed);
    ASSERT_TRUE(compile.ok()) << compile.status().ToString();
    EXPECT_EQ(compile->optimization.opt_level, 2);
    EXPECT_TRUE(compile->lowering.enable_packed_weights);
    EXPECT_EQ(compile->lowering.selector.device_type, DeviceType::kCPU);
    EXPECT_EQ(compile->lowering.selector.weight_format, WeightFormat::kPlain);
    EXPECT_EQ(compile->lowering.selector.phase, ExecPhase::kBoth);
}

TEST(InferenceCpuPreset, RejectsUnknownPresetWithoutFallback) {
    const auto compile = MakeCpuCompileOptions(static_cast<CpuInferencePreset>(255));
    EXPECT_FALSE(compile.ok());
    EXPECT_EQ(compile.status().code(), StatusCode::kInvalidArgument);
}

} // namespace
