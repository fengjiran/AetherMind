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
