#include "inference_runner.h"

#include "aethermind/inference/executable_model.h"
#include "aethermind/model/model_loader.h"
#include "aethermind/runtime/runtime_builder.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace aethermind;
using namespace aethermind::apps::inference;
namespace fs = std::filesystem;

RunOptions SuccessOptions(const char* fixture = "f32-explicit") {
    RunOptions options;
    options.model_dir = fs::path(AETHERMIND_TEST_MODELS_DIR) / "inference-admission" / fixture;
    options.prompt_tokens = {7};
    options.context_tokens = 8;
    options.generation.max_new_tokens = 3;
    return options;
}

// Keep the expected path independent of the application's configuration helpers.
// Existing scalar-oracle tests retain responsibility for numerical/phase evidence.
StatusOr<std::vector<uint32_t>> IndependentSessionGenerate(const RunOptions& options) {
    AM_ASSIGN_OR_RETURN(auto loaded, ModelLoader::Load(options.model_dir));
    const auto& config = loaded->GetConfig();

    RuntimeOptions runtime_options;
    runtime_options.allocator = {.enable_cpu = true, .enable_cuda = false, .enable_cann = false};
    runtime_options.backend.enable_cpu = true;
    runtime_options.backend.enable_cuda = false;
    runtime_options.backend.enable_cann = false;
    runtime_options.kv_cache = {
            .enable_manager = true,
            .num_layers = static_cast<size_t>(config.num_hidden_layers),
            .num_kv_heads = static_cast<size_t>(config.num_key_value_heads),
            .max_tokens = options.context_tokens,
            .head_dim = static_cast<size_t>(config.head_dim > 0
                                                    ? config.head_dim
                                                    : config.hidden_size / config.num_attention_heads),
            .kv_dtype = DataType::Float32(),
            .alignment = 64,
    };

    ModelCompileOptions compile_options;
    compile_options.optimization.opt_level = options.preset == CpuInferencePreset::kO2Packed ? 2 : 1;
    compile_options.lowering.enable_packed_weights = options.preset == CpuInferencePreset::kO2Packed;
    compile_options.lowering.selector = {
            .device_type = DeviceType::kCPU,
            .act_dtype = {},
            .weight_dtype = {},
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kBoth,
    };

    AM_ASSIGN_OR_RETURN(auto artifact, ModelCompiler::Compile(std::move(loaded), compile_options));
    RuntimeBuilder builder;
    builder.WithOptions(runtime_options);
    AM_ASSIGN_OR_RETURN(auto runtime, builder.TryBuild());
    AM_ASSIGN_OR_RETURN(auto executable, PrepareExecutableModel(runtime, std::move(artifact)));
    auto model = std::make_shared<const ExecutableModel>(std::move(executable));
    AM_ASSIGN_OR_RETURN(auto session, InferenceSession::Create(runtime, std::move(model)));
    return session.Generate(options.prompt_tokens, options.generation);
}

struct RunnerFixtureCase {
    const char* fixture;
    const char* name;
    CpuInferencePreset preset;
};

constexpr std::array<RunnerFixtureCase, 8> kRunnerFixtures{{
        {"f32-missing", "F32MissingO1Plain", CpuInferencePreset::kO1Plain},
        {"f32-missing", "F32MissingO2Packed", CpuInferencePreset::kO2Packed},
        {"f32-auto", "F32AutoO1Plain", CpuInferencePreset::kO1Plain},
        {"f32-auto", "F32AutoO2Packed", CpuInferencePreset::kO2Packed},
        {"f32-explicit", "F32ExplicitO1Plain", CpuInferencePreset::kO1Plain},
        {"f32-explicit", "F32ExplicitO2Packed", CpuInferencePreset::kO2Packed},
        {"f32-tied", "F32TiedO1Plain", CpuInferencePreset::kO1Plain},
        {"f32-tied", "F32TiedO2Packed", CpuInferencePreset::kO2Packed},
}};

class InferenceRunnerEquivalence : public ::testing::TestWithParam<RunnerFixtureCase> {};

TEST_P(InferenceRunnerEquivalence, MatchesIndependentSessionOutput) {
    RunOptions options = SuccessOptions(GetParam().fixture);
    options.preset = GetParam().preset;
    const auto expected = IndependentSessionGenerate(options);
    ASSERT_TRUE(expected.ok()) << expected.status().ToString();

    RunStage failure_stage = RunStage::kGenerate;
    const auto result = RunInference(options, &failure_stage);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(failure_stage, RunStage::kNone);
    EXPECT_EQ(*result, *expected);
    EXPECT_EQ(result->size(), options.generation.max_new_tokens);
}

INSTANTIATE_TEST_SUITE_P(
        HfPresets, InferenceRunnerEquivalence, ::testing::ValuesIn(kRunnerFixtures),
        [](const ::testing::TestParamInfo<RunnerFixtureCase>& info) { return info.param.name; });

TEST(InferenceRunnerSuccess, DefaultPresetRunsO1Plain) {
    const RunOptions options = SuccessOptions();
    const auto expected = IndependentSessionGenerate(options);
    ASSERT_TRUE(expected.ok()) << expected.status().ToString();

    const auto result = RunInference(options);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(options.preset, CpuInferencePreset::kO1Plain);
    EXPECT_EQ(*result, *expected);
}

TEST(InferenceRunnerSuccess, RepeatedRequestsReturnTheSameNewTokens) {
    for (const CpuInferencePreset preset: {CpuInferencePreset::kO1Plain, CpuInferencePreset::kO2Packed}) {
        SCOPED_TRACE(static_cast<int>(preset));
        RunOptions options = SuccessOptions();
        options.preset = preset;
        const auto first = RunInference(options);
        ASSERT_TRUE(first.ok()) << first.status().ToString();
        const auto second = RunInference(options);
        ASSERT_TRUE(second.ok()) << second.status().ToString();
        EXPECT_EQ(*second, *first);
        EXPECT_EQ(second->size(), options.generation.max_new_tokens);
    }
}

TEST(InferenceRunnerSuccess, GeneratesAtExactKvCapacity) {
    for (const CpuInferencePreset preset: {CpuInferencePreset::kO1Plain, CpuInferencePreset::kO2Packed}) {
        SCOPED_TRACE(static_cast<int>(preset));
        RunOptions options = SuccessOptions();
        options.preset = preset;
        options.prompt_tokens = {7, 11};
        options.context_tokens = options.prompt_tokens.size() + options.generation.max_new_tokens - 1;
        const auto expected = IndependentSessionGenerate(options);
        ASSERT_TRUE(expected.ok()) << expected.status().ToString();

        const auto result = RunInference(options);
        ASSERT_TRUE(result.ok()) << result.status().ToString();
        EXPECT_EQ(*result, *expected);
        EXPECT_EQ(result->size(), options.generation.max_new_tokens);
    }
}

TEST(InferenceRunnerSuccess, ZeroGenerationAllowsPromptBeyondPhysicalKvCapacity) {
    for (const CpuInferencePreset preset: {CpuInferencePreset::kO1Plain, CpuInferencePreset::kO2Packed}) {
        SCOPED_TRACE(static_cast<int>(preset));
        RunOptions options = SuccessOptions();
        options.preset = preset;
        options.prompt_tokens = {7, 11};
        options.context_tokens = 1;
        options.generation.max_new_tokens = 0;
        const auto expected = IndependentSessionGenerate(options);
        ASSERT_TRUE(expected.ok()) << expected.status().ToString();

        RunStage failure_stage = RunStage::kLoad;
        const auto result = RunInference(options, &failure_stage);
        ASSERT_TRUE(result.ok()) << result.status().ToString();
        EXPECT_EQ(failure_stage, RunStage::kNone);
        EXPECT_TRUE(result->empty());
        EXPECT_EQ(*result, *expected);
    }
}

TEST(InferenceRunnerSuccess, IncludesEosAndStopsAtTheFirstPrediction) {
    for (const CpuInferencePreset preset: {CpuInferencePreset::kO1Plain, CpuInferencePreset::kO2Packed}) {
        SCOPED_TRACE(static_cast<int>(preset));
        RunOptions options = SuccessOptions();
        options.preset = preset;
        const auto baseline = IndependentSessionGenerate(options);
        ASSERT_TRUE(baseline.ok()) << baseline.status().ToString();
        ASSERT_FALSE(baseline->empty());
        options.generation.max_new_tokens = 5;
        options.generation.eos_token_id = baseline->front();
        const auto expected = IndependentSessionGenerate(options);
        ASSERT_TRUE(expected.ok()) << expected.status().ToString();

        const auto result = RunInference(options);
        ASSERT_TRUE(result.ok()) << result.status().ToString();
        EXPECT_EQ(*result, *expected);
        ASSERT_EQ(result->size(), 1U);
        EXPECT_EQ(result->front(), *options.generation.eos_token_id);
    }
}

} // namespace
