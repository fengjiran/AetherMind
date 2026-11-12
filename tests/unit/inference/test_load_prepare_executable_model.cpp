#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/base/device.h"
#include "aethermind/compiler/model_compiler.h"
#include "aethermind/dtypes/data_type.h"
#include "aethermind/inference/executable_model.h"
#include "aethermind/inference/inference_session.h"
#include "aethermind/model/model_loader.h"
#include "aethermind/runtime/runtime_builder.h"
#include "aethermind/runtime/runtime_options.h"

#include <array>
#include <filesystem>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <utility>

namespace {

using namespace aethermind;

namespace fs = std::filesystem;

fs::path TinyLlamaModelDir() {
    return fs::path(AETHERMIND_TEST_MODELS_DIR) / "tiny-random-LlamaForCausalLM";
}

ModelLoadOptions LoadOptions(bool packed = false) {
    return {.preset = packed ? CompilePreset::kO2Packed : CompilePreset::kO1Plain};
}

Runtime MakeTinyLlamaRuntime() {
    RuntimeOptions options;
    options.kv_cache.enable_manager = true;
    options.kv_cache.num_layers = 2;
    options.kv_cache.num_kv_heads = 4;
    options.kv_cache.max_tokens = 8;
    options.kv_cache.head_dim = 4;
    options.kv_cache.kv_dtype = DataType::Float32();
    options.kv_cache.alignment = 64;

    RuntimeBuilder builder;
    builder.WithOptions(options);
    builder.RegisterBackendFactory(DeviceType::kCPU,
                                   std::make_unique<CpuBackendFactory>());
    return builder.Build();
}

Runtime MakeCpuRuntimeWithoutBackend() {
    RuntimeOptions options;
    options.backend.enable_cpu = false;

    RuntimeBuilder builder;
    builder.WithOptions(options);
    return builder.Build();
}

class ModelLoadPath : public ::testing::TestWithParam<bool> {};

TEST_P(ModelLoadPath, LoadsPreparesAndGeneratesFromHfDirectory) {
    // Runtime outlives the prepared model and session because both borrow its
    // backend resources. These dimensions match the checked-in fixture config.
    Runtime runtime = MakeTinyLlamaRuntime();
    auto executable = ExecutableModel::Load(
            runtime, TinyLlamaModelDir(), LoadOptions(GetParam()));
    ASSERT_TRUE(executable.ok()) << executable.status().ToString();
    EXPECT_TRUE(executable->IsPreparedFor(runtime));
    EXPECT_EQ(executable->context_limit(), 2048U);
    EXPECT_EQ(executable->vocab_size(), 32000U);

    auto shared_model =
            std::make_shared<ExecutableModel>(std::move(*executable));
    auto session = InferenceSession::Create(runtime, shared_model);
    ASSERT_TRUE(session.ok()) << session.status().ToString();

    constexpr std::array<uint32_t, 1> prompt{7};
    const auto generated = session->Generate(
            prompt, GenerationConfig{.max_new_tokens = 3});
    ASSERT_TRUE(generated.ok()) << generated.status().ToString();
    ASSERT_EQ(generated->size(), 3U);
    for (const uint32_t token: *generated) {
        EXPECT_LT(token, 32000U);
    }
    const auto repeated = session->Generate(prompt, GenerationConfig{.max_new_tokens = 3});
    ASSERT_TRUE(repeated.ok()) << repeated.status().ToString();
    EXPECT_EQ(*repeated, *generated);

    Runtime other_runtime = MakeCpuRuntimeWithoutBackend();
    const auto mismatched_session =
            InferenceSession::Create(other_runtime, shared_model);
    ASSERT_FALSE(mismatched_session.ok());
    EXPECT_EQ(mismatched_session.status(),
              StatusCode::kFailedPrecondition);
}

INSTANTIATE_TEST_SUITE_P(
        PreparationPaths, ModelLoadPath, ::testing::Bool(),
        [](const ::testing::TestParamInfo<bool>& info) { return info.param ? "O2Packed" : "O1Plain"; });

TEST(ModelLoad, PreservesLoadErrorCodeAndAddsContext) {
    Runtime runtime = MakeTinyLlamaRuntime();
    const fs::path missing_dir = TinyLlamaModelDir() / "missing-model-directory";
    const ModelLoadOptions options = LoadOptions();

    const auto load_result = ModelLoader::Load(missing_dir);
    ASSERT_FALSE(load_result.ok());

    const auto result =
            ExecutableModel::Load(runtime, missing_dir, options);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), load_result.status().code());
    EXPECT_NE(result.status().message().find(
                      "ExecutableModel::Load loading failed"),
              std::string::npos);
}

TEST(ModelLoad, PreservesPreparationErrorCodeAndAddsContext) {
    Runtime runtime = MakeCpuRuntimeWithoutBackend();
    const ModelLoadOptions options = LoadOptions();

    const auto compile_options = MakeCompileOptions(options.preset, options.target);
    ASSERT_TRUE(compile_options.ok()) << compile_options.status().ToString();
    auto artifact = ModelCompiler::LoadAndCompile(TinyLlamaModelDir(), *compile_options);
    ASSERT_TRUE(artifact.ok()) << artifact.status().ToString();
    const auto prepare_result = ExecutableModel::Prepare(runtime, std::move(*artifact));
    ASSERT_FALSE(prepare_result.ok());

    const auto result =
            ExecutableModel::Load(runtime, TinyLlamaModelDir(), options);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), prepare_result.status().code());
    EXPECT_NE(result.status().message().find(
                      "ExecutableModel::Load preparation failed"),
              std::string::npos);
}

TEST(ModelLoad, PreservesNonCpuTargetsThroughCompilation) {
    for (const auto target: {DeviceType::kCUDA, DeviceType::kCANN}) {
        for (const auto preset: {CompilePreset::kO1Plain, CompilePreset::kO2Packed}) {
            SCOPED_TRACE(DeviceType2Str(target));
            const auto options = MakeCompileOptions(preset, target);
            ASSERT_TRUE(options.ok()) << options.status().ToString();
            const auto artifact = ModelCompiler::LoadAndCompile(TinyLlamaModelDir(), *options);
            ASSERT_TRUE(artifact.ok()) << artifact.status().ToString();
            ASSERT_FALSE(artifact->graph.steps().empty());
            for (const auto& step: artifact->graph.steps()) {
                EXPECT_EQ(step.spec.selector.device_type, target);
            }
        }
    }
}

TEST(ModelLoad, ReportsRequestedBackendWhenUnavailable) {
    Runtime runtime = MakeCpuRuntimeWithoutBackend();
    for (const auto target: {DeviceType::kCUDA, DeviceType::kCANN}) {
        for (const auto preset: {CompilePreset::kO1Plain, CompilePreset::kO2Packed}) {
            SCOPED_TRACE(DeviceType2Str(target));
            const auto result = ExecutableModel::Load(
                    runtime, TinyLlamaModelDir(), {.preset = preset, .target = target});
            ASSERT_FALSE(result.ok());
            EXPECT_EQ(result.status(), StatusCode::kNotFound);
            EXPECT_NE(result.status().message().find(DeviceType2Str(target)), std::string::npos)
                    << result.status().message();
        }
    }
}

} // namespace
