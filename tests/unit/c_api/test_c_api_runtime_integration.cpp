#include "aethermind/compiler/model_compiler.h"
#include "aethermind/inference/executable_model.h"
#include "aethermind/inference/inference_session.h"
#include "aethermind/model/model_loader.h"
#include "aethermind/runtime/runtime_builder.h"
#include "test_c_api_runtime_consumer.h"

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
namespace fs = std::filesystem;

struct CAbiFixtureCase {
    const char* directory;
    const char* name;
    uint32_t preset;
    bool packed;
};

constexpr std::array<CAbiFixtureCase, 8> kCAbiFixtures{{
        {"f32-missing", "F32MissingO1Plain", AM_CPU_PRESET_O1_PLAIN, false},
        {"f32-missing", "F32MissingO2Packed", AM_CPU_PRESET_O2_PACKED, true},
        {"f32-auto", "F32AutoO1Plain", AM_CPU_PRESET_O1_PLAIN, false},
        {"f32-auto", "F32AutoO2Packed", AM_CPU_PRESET_O2_PACKED, true},
        {"f32-explicit", "F32ExplicitO1Plain", AM_CPU_PRESET_O1_PLAIN, false},
        {"f32-explicit", "F32ExplicitO2Packed", AM_CPU_PRESET_O2_PACKED, true},
        {"f32-tied", "F32TiedO1Plain", AM_CPU_PRESET_O1_PLAIN, false},
        {"f32-tied", "F32TiedO2Packed", AM_CPU_PRESET_O2_PACKED, true},
}};

struct CAbiRequest {
    std::vector<uint32_t> prompt{7};
    size_t capacity = 8;
    GenerationConfig generation{.max_new_tokens = 3};
};

// Expected results use only public library construction blocks and independently
// configured options, never the adapter's shared preset or geometry helpers.
StatusOr<std::vector<uint32_t>> DirectSessionGenerate(
        const fs::path& directory, bool packed, const CAbiRequest& request) {
    AM_ASSIGN_OR_RETURN(auto loaded, ModelLoader::Load(directory));
    RuntimeOptions runtime_options;
    {
        const auto& config = loaded->GetConfig();
        runtime_options.allocator = {.enable_cpu = true, .enable_cuda = false, .enable_cann = false};
        runtime_options.backend.enable_cpu = true;
        runtime_options.backend.enable_cuda = false;
        runtime_options.backend.enable_cann = false;
        runtime_options.kv_cache = {
                .enable_manager = true,
                .num_layers = static_cast<size_t>(config.num_hidden_layers),
                .num_kv_heads = static_cast<size_t>(config.num_key_value_heads),
                .max_tokens = request.capacity,
                .head_dim = static_cast<size_t>(config.head_dim > 0
                                                        ? config.head_dim
                                                        : config.hidden_size / config.num_attention_heads),
                .kv_dtype = DataType::Float32(),
                .alignment = 64,
        };
    }
    ModelCompileOptions compile_options;
    compile_options.optimization.opt_level = packed ? 2 : 1;
    compile_options.lowering.enable_packed_weights = packed;
    compile_options.lowering.selector = {.device_type = DeviceType::kCPU,
                                         .act_dtype = {},
                                         .weight_dtype = {},
                                         .weight_format = WeightFormat::kPlain,
                                         .phase = ExecPhase::kBoth};
    AM_ASSIGN_OR_RETURN(auto artifact, ModelCompiler::Compile(std::move(loaded), compile_options));
    RuntimeBuilder builder;
    builder.WithOptions(runtime_options);
    AM_ASSIGN_OR_RETURN(auto runtime, builder.TryBuild());
    AM_ASSIGN_OR_RETURN(auto executable, PrepareExecutableModel(runtime, std::move(artifact)));
    auto model = std::make_shared<const ExecutableModel>(std::move(executable));
    AM_ASSIGN_OR_RETURN(auto session, InferenceSession::Create(runtime, std::move(model)));
    return session.Generate(request.prompt, request.generation);
}

struct CAbiGeneration {
    AetherMindCInferenceResult status{};
    std::vector<uint32_t> tokens;
    std::vector<uint32_t> repeated_tokens;
};

class CAbiRuntimeIntegration : public ::testing::TestWithParam<CAbiFixtureCase> {
protected:
    void SetUp() override {
        directory_ = fs::path(AETHERMIND_TEST_MODELS_DIR) / "inference-admission" / GetParam().directory;
        ASSERT_TRUE(fs::is_regular_file(directory_ / "config.json"))
                << "Generate and check the M0 fixtures before the C inference gate: " << directory_;
        ASSERT_TRUE(fs::is_regular_file(directory_ / "model.safetensors")) << directory_;
        const auto loaded = ModelLoader::Load(directory_);
        ASSERT_TRUE(loaded.ok()) << loaded.status().ToString();
        config_ = (*loaded)->GetConfig();
    }

    CAbiGeneration GenerateThroughC(
            const CAbiRequest& request, bool repeat = false,
            unsigned int release_owners = AETHERMIND_C_RELEASE_NO_OWNERS) const {
        const am_config_t runtime_config{
                .struct_size = sizeof(am_config_t),
                .abi_version = AM_ABI_VERSION,
                .num_layers = static_cast<uint64_t>(config_.num_hidden_layers),
                .num_kv_heads = static_cast<uint64_t>(config_.num_key_value_heads),
                .head_dim = static_cast<uint64_t>(config_.head_dim > 0
                                                          ? config_.head_dim
                                                          : config_.hidden_size / config_.num_attention_heads),
                .max_tokens = request.capacity,
                .kv_dtype = AM_KV_DTYPE_FLOAT32,
                .cpu_preset = GetParam().preset,
        };
        const am_session_config_t session_config{
                .struct_size = sizeof(am_session_config_t),
                .abi_version = AM_ABI_VERSION,
                .max_new_tokens = request.generation.max_new_tokens,
                .has_eos_token_id = request.generation.eos_token_id.has_value() ? 1U : 0U,
                .eos_token_id = request.generation.eos_token_id.value_or(0),
        };
        CAbiGeneration result;
        result.tokens.resize(request.generation.max_new_tokens);
        if (repeat) {
            result.repeated_tokens.resize(request.generation.max_new_tokens);
        }
        const std::string path = directory_.string();
        AetherMindRunCInference(
                &runtime_config, path.c_str(), &session_config,
                request.prompt.data(), request.prompt.size(),
                result.tokens.empty() ? nullptr : result.tokens.data(), result.tokens.size(),
                result.repeated_tokens.empty() ? nullptr : result.repeated_tokens.data(),
                result.repeated_tokens.size(), repeat, release_owners, &result.status);
        return result;
    }

    void ExpectSuccess(const CAbiGeneration& result, const std::vector<uint32_t>& expected,
                       bool repeat = false) const {
        ASSERT_EQ(result.status.failure_stage, AETHERMIND_C_INFERENCE_SUCCESS)
                << result.status.diagnostic;
        EXPECT_EQ(result.status.code, AM_STATUS_OK);
        EXPECT_EQ(result.status.abi_version, AM_ABI_VERSION);
        EXPECT_EQ(result.status.unexpected_success_error, 0);
        ASSERT_LE(result.status.output_len, result.tokens.size());
        EXPECT_EQ(std::vector<uint32_t>(result.tokens.begin(), result.tokens.begin() + result.status.output_len), expected);
        if (repeat) {
            ASSERT_LE(result.status.repeated_output_len, result.repeated_tokens.size());
            EXPECT_EQ(std::vector<uint32_t>(result.repeated_tokens.begin(),
                                            result.repeated_tokens.begin() + result.status.repeated_output_len),
                      expected);
        }
    }

    fs::path directory_;
    HfModelConfig config_;
};

TEST_P(CAbiRuntimeIntegration, MatchesDirectSessionForThreeNewTokens) {
    const CAbiRequest request;
    const auto expected = DirectSessionGenerate(directory_, GetParam().packed, request);
    ASSERT_TRUE(expected.ok()) << expected.status().ToString();
    ASSERT_EQ(expected->size(), 3U);
    ExpectSuccess(GenerateThroughC(request), *expected);
}

TEST_P(CAbiRuntimeIntegration, RepeatedRequestsMatchTheSameDirectSession) {
    const CAbiRequest request;
    const auto expected = DirectSessionGenerate(directory_, GetParam().packed, request);
    ASSERT_TRUE(expected.ok()) << expected.status().ToString();
    ExpectSuccess(GenerateThroughC(request, true), *expected, true);
}

TEST_P(CAbiRuntimeIntegration, MatchesDirectSessionAtExactKvCapacity) {
    CAbiRequest request;
    request.prompt = {7, 8};
    request.capacity = 4;
    const auto expected = DirectSessionGenerate(directory_, GetParam().packed, request);
    ASSERT_TRUE(expected.ok()) << expected.status().ToString();
    ExpectSuccess(GenerateThroughC(request), *expected);
}

TEST_P(CAbiRuntimeIntegration, ZeroGenerationDoesNotRequirePromptFittingPhysicalCapacity) {
    CAbiRequest request;
    request.prompt = {7, 8};
    request.capacity = 1;
    request.generation.max_new_tokens = 0;
    const auto expected = DirectSessionGenerate(directory_, GetParam().packed, request);
    ASSERT_TRUE(expected.ok()) << expected.status().ToString();
    ASSERT_TRUE(expected->empty());
    ExpectSuccess(GenerateThroughC(request), *expected);
}

TEST_P(CAbiRuntimeIntegration, GeneratedEosIsIncludedAndStopsBothPaths) {
    CAbiRequest request;
    const auto baseline = DirectSessionGenerate(directory_, GetParam().packed, request);
    ASSERT_TRUE(baseline.ok()) << baseline.status().ToString();
    ASSERT_FALSE(baseline->empty());
    request.generation.eos_token_id = baseline->front();
    const auto expected = DirectSessionGenerate(directory_, GetParam().packed, request);
    ASSERT_TRUE(expected.ok()) << expected.status().ToString();
    EXPECT_EQ(*expected, (std::vector<uint32_t>{*request.generation.eos_token_id}));
    ExpectSuccess(GenerateThroughC(request), *expected);
}

TEST_P(CAbiRuntimeIntegration, RetainedOwnersSurviveExternalRuntimeAndModelRelease) {
    const CAbiRequest request;
    const auto expected = DirectSessionGenerate(directory_, GetParam().packed, request);
    ASSERT_TRUE(expected.ok()) << expected.status().ToString();
    ExpectSuccess(GenerateThroughC(request, true,
                                   AETHERMIND_C_RELEASE_RUNTIME | AETHERMIND_C_RELEASE_MODEL),
                  *expected, true);
}

INSTANTIATE_TEST_SUITE_P(
        HfPresets, CAbiRuntimeIntegration, ::testing::ValuesIn(kCAbiFixtures),
        [](const ::testing::TestParamInfo<CAbiFixtureCase>& info) { return info.param.name; });

} // namespace
