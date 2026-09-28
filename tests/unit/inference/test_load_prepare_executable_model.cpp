#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/base/device.h"
#include "aethermind/compiler/model_compiler.h"
#include "aethermind/dtypes/data_type.h"
#include "aethermind/inference/executable_model.h"
#include "aethermind/inference/inference_session.h"
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

ModelCompileOptions CpuPlainCompileOptions() {
    ModelCompileOptions options;
    // The CPU kernels for fused O2 operators currently require packed weights.
    options.optimization.opt_level = 1;
    options.lowering.enable_packed_weights = false;
    return options;
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

TEST(LoadAndPrepareExecutableModel, LoadsPreparesAndGeneratesFromHfDirectory) {
    // Runtime outlives the prepared model and session because both borrow its
    // backend resources. These dimensions match the checked-in fixture config.
    Runtime runtime = MakeTinyLlamaRuntime();
    auto executable = LoadAndPrepareExecutableModel(
            runtime, TinyLlamaModelDir(), CpuPlainCompileOptions());
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

    Runtime other_runtime = MakeCpuRuntimeWithoutBackend();
    const auto mismatched_session =
            InferenceSession::Create(other_runtime, shared_model);
    ASSERT_FALSE(mismatched_session.ok());
    EXPECT_EQ(mismatched_session.status().code(),
              StatusCode::kFailedPrecondition);
}

TEST(LoadAndPrepareExecutableModel, PreservesLoadCompileErrorCodeAndAddsContext) {
    Runtime runtime = MakeTinyLlamaRuntime();
    const fs::path missing_dir = TinyLlamaModelDir() / "missing-model-directory";
    const ModelCompileOptions options = CpuPlainCompileOptions();

    const auto compile_result = ModelCompiler::LoadAndCompile(missing_dir, options);
    ASSERT_FALSE(compile_result.ok());

    const auto result =
            LoadAndPrepareExecutableModel(runtime, missing_dir, options);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), compile_result.status().code());
    EXPECT_NE(result.status().message().find(
                      "LoadAndPrepareExecutableModel loading/compilation failed"),
              std::string::npos);
}

TEST(LoadAndPrepareExecutableModel, PreservesPreparationErrorCodeAndAddsContext) {
    Runtime runtime = MakeCpuRuntimeWithoutBackend();
    const ModelCompileOptions options = CpuPlainCompileOptions();

    auto artifact = ModelCompiler::LoadAndCompile(TinyLlamaModelDir(), options);
    ASSERT_TRUE(artifact.ok()) << artifact.status().ToString();
    const auto prepare_result = PrepareExecutableModel(runtime, std::move(*artifact));
    ASSERT_FALSE(prepare_result.ok());

    const auto result =
            LoadAndPrepareExecutableModel(runtime, TinyLlamaModelDir(), options);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), prepare_result.status().code());
    EXPECT_NE(result.status().message().find(
                      "LoadAndPrepareExecutableModel preparation failed"),
              std::string::npos);
}

} // namespace
