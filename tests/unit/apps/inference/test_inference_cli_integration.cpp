#include "aethermind/compiler/model_compiler.h"
#include "aethermind/inference/executable_model.h"
#include "aethermind/inference/inference_session.h"
#include "aethermind/model/model_loader.h"
#include "aethermind/runtime/runtime_builder.h"
#include "test_inference_cli_process_helpers.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

using namespace aethermind;
namespace fs = std::filesystem;

struct CliFixtureCase {
    const char* directory;
    const char* preset;
    const char* name;
    bool packed;
};

constexpr std::array<CliFixtureCase, 8> kCliFixtures{{
        {"f32-missing", "o1-plain", "F32MissingO1Plain", false},
        {"f32-missing", "o2-packed", "F32MissingO2Packed", true},
        {"f32-auto", "o1-plain", "F32AutoO1Plain", false},
        {"f32-auto", "o2-packed", "F32AutoO2Packed", true},
        {"f32-explicit", "o1-plain", "F32ExplicitO1Plain", false},
        {"f32-explicit", "o2-packed", "F32ExplicitO2Packed", true},
        {"f32-tied", "o1-plain", "F32TiedO1Plain", false},
        {"f32-tied", "o2-packed", "F32TiedO2Packed", true},
}};

struct CliRequest {
    std::vector<uint32_t> prompt{7};
    size_t capacity = 8;
    GenerationConfig generation{.max_new_tokens = 3};
};

// This expected path deliberately consumes no application helper or parser.
// Scalar-oracle tests retain ownership of independent numerical/phase evidence.
StatusOr<std::vector<uint32_t>> DirectSessionGenerate(
        const fs::path& model_directory, const CliFixtureCase& fixture, const CliRequest& request) {
    AM_ASSIGN_OR_RETURN(auto loaded, ModelLoader::Load(model_directory));
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
    compile_options.optimization.opt_level = fixture.packed ? 2 : 1;
    compile_options.lowering.enable_packed_weights = fixture.packed;
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

bool IsJsonWhitespace(char value) noexcept {
    return value == ' ' || value == '\t' || value == '\r' || value == '\n';
}

StatusOr<std::vector<uint32_t>> ParseTokenJson(std::string_view text) {
    if (text.empty()) {
        return Status::InvalidArgument("CLI stdout is empty");
    }
    const char* cursor = text.data();
    const char* const end = cursor + text.size();
    const auto skip_whitespace = [&] {
        while (cursor != end && IsJsonWhitespace(*cursor)) {
            ++cursor;
        }
    };
    skip_whitespace();
    if (cursor == end || *cursor++ != '[') {
        return Status::InvalidArgument("CLI stdout is not a JSON array");
    }
    skip_whitespace();
    std::vector<uint32_t> tokens;
    if (cursor != end && *cursor == ']') {
        ++cursor;
    } else {
        for (;;) {
            const char* const begin = cursor;
            while (cursor != end && *cursor >= '0' && *cursor <= '9') {
                ++cursor;
            }
            if (begin == cursor || (cursor - begin > 1 && *begin == '0')) {
                return Status::InvalidArgument("CLI JSON contains an invalid token integer");
            }
            uint32_t token = 0;
            const auto parsed = std::from_chars(begin, cursor, token);
            if (parsed.ec != std::errc{} || parsed.ptr != cursor) {
                return Status::OutOfRange("CLI JSON token exceeds uint32_t");
            }
            tokens.push_back(token);
            skip_whitespace();
            if (cursor == end) {
                return Status::InvalidArgument("CLI JSON array is unterminated");
            }
            if (*cursor == ']') {
                ++cursor;
                break;
            }
            if (*cursor++ != ',') {
                return Status::InvalidArgument("CLI JSON array has an invalid separator");
            }
            skip_whitespace();
        }
    }
    skip_whitespace();
    if (cursor != end) {
        return Status::InvalidArgument("CLI JSON has trailing content");
    }
    return tokens;
}

class InferenceCliIntegration : public ::testing::TestWithParam<CliFixtureCase> {
protected:
    void SetUp() override {
#if !defined(__linux__)
        FAIL() << "The CLI/direct Session integration gate requires Linux";
#else
        const char* const executable = std::getenv("AETHERMIND_INFER_BINARY");
        ASSERT_NE(executable, nullptr)
                << "Set AETHERMIND_INFER_BINARY to the separately built formal aethermind_infer executable";
        ASSERT_NE(*executable, '\0') << "AETHERMIND_INFER_BINARY must not be empty";
        binary_ = fs::absolute(executable);
        ASSERT_TRUE(fs::is_regular_file(binary_)) << "CLI executable is missing: " << binary_;
        ASSERT_EQ(access(binary_.c_str(), X_OK), 0) << "CLI file is not executable: " << binary_;
        model_directory_ = fs::path(AETHERMIND_TEST_MODELS_DIR) / "inference-admission" / GetParam().directory;
        ASSERT_TRUE(fs::is_regular_file(model_directory_ / "config.json"))
                << "Generate and check the M0 fixtures before this integration gate: " << model_directory_;
        ASSERT_TRUE(fs::is_regular_file(model_directory_ / "model.safetensors")) << model_directory_;
#endif
    }

    std::vector<std::string> Arguments(const CliRequest& request) const {
        std::string prompt;
        for (const uint32_t token: request.prompt) {
            if (!prompt.empty()) {
                prompt += ',';
            }
            prompt += std::to_string(token);
        }
        std::vector<std::string> arguments{
                "--model",
                model_directory_.string(),
                "--tokens",
                std::move(prompt),
                "--context-tokens",
                std::to_string(request.capacity),
                "--max-new-tokens",
                std::to_string(request.generation.max_new_tokens),
                "--preset",
                GetParam().preset,
        };
        if (request.generation.eos_token_id.has_value()) {
            arguments.insert(arguments.end(), {"--eos-token-id", std::to_string(*request.generation.eos_token_id)});
        }
        return arguments;
    }

    StatusOr<std::vector<uint32_t>> CliGenerate(const CliRequest& request) const {
        AM_ASSIGN_OR_RETURN(auto captured, test::cli_process::Run(binary_, Arguments(request)));
        if (captured.exit_code != 0) {
            return Status::Internal("CLI failed with exit " + std::to_string(captured.exit_code) + ": " + captured.standard_error);
        }
        if (captured.standard_output.empty() || captured.standard_output.back() != '\n' ||
            std::count(captured.standard_output.begin(), captured.standard_output.end(), '\n') != 1) {
            return Status::InvalidArgument("CLI success must contain one JSON line ending in LF");
        }
        return ParseTokenJson(captured.standard_output);
    }

    fs::path binary_;
    fs::path model_directory_;
};

TEST_P(InferenceCliIntegration, MatchesDirectSessionForThreeNewTokens) {
    const CliRequest request;
    const auto expected = DirectSessionGenerate(model_directory_, GetParam(), request);
    ASSERT_TRUE(expected.ok()) << expected.status().ToString();
    const auto actual = CliGenerate(request);
    ASSERT_TRUE(actual.ok()) << actual.status().ToString();
    EXPECT_EQ(*actual, *expected);
}

TEST_P(InferenceCliIntegration, RepeatedProcessesMatchTheSameDirectSession) {
    const CliRequest request;
    const auto expected = DirectSessionGenerate(model_directory_, GetParam(), request);
    ASSERT_TRUE(expected.ok()) << expected.status().ToString();
    const auto first = CliGenerate(request);
    ASSERT_TRUE(first.ok()) << first.status().ToString();
    const auto second = CliGenerate(request);
    ASSERT_TRUE(second.ok()) << second.status().ToString();
    EXPECT_EQ(*first, *expected);
    EXPECT_EQ(*second, *expected);
}

TEST_P(InferenceCliIntegration, MatchesDirectSessionAtExactKvCapacity) {
    CliRequest request;
    request.prompt = {7, 8};
    request.capacity = 4;
    const auto expected = DirectSessionGenerate(model_directory_, GetParam(), request);
    ASSERT_TRUE(expected.ok()) << expected.status().ToString();
    const auto actual = CliGenerate(request);
    ASSERT_TRUE(actual.ok()) << actual.status().ToString();
    EXPECT_EQ(*actual, *expected);
}

TEST_P(InferenceCliIntegration, ZeroGenerationMatchesWithoutPromptFittingPhysicalCapacity) {
    CliRequest request;
    request.prompt = {7, 8};
    request.capacity = 1;
    request.generation.max_new_tokens = 0;
    const auto expected = DirectSessionGenerate(model_directory_, GetParam(), request);
    ASSERT_TRUE(expected.ok()) << expected.status().ToString();
    const auto actual = CliGenerate(request);
    ASSERT_TRUE(actual.ok()) << actual.status().ToString();
    EXPECT_TRUE(expected->empty());
    EXPECT_EQ(*actual, *expected);
}

TEST_P(InferenceCliIntegration, GeneratedEosIsIncludedAndStopsBothPaths) {
    CliRequest request;
    const auto baseline = DirectSessionGenerate(model_directory_, GetParam(), request);
    ASSERT_TRUE(baseline.ok()) << baseline.status().ToString();
    ASSERT_FALSE(baseline->empty());
    request.generation.eos_token_id = baseline->front();
    const auto expected = DirectSessionGenerate(model_directory_, GetParam(), request);
    ASSERT_TRUE(expected.ok()) << expected.status().ToString();
    const auto actual = CliGenerate(request);
    ASSERT_TRUE(actual.ok()) << actual.status().ToString();
    EXPECT_EQ(*expected, (std::vector<uint32_t>{*request.generation.eos_token_id}));
    EXPECT_EQ(*actual, *expected);
}

TEST_P(InferenceCliIntegration, BothPathsRejectOneTokenBeyondPhysicalCapacity) {
    CliRequest request;
    request.prompt = {7, 8};
    request.capacity = 3;
    const auto expected = DirectSessionGenerate(model_directory_, GetParam(), request);
    ASSERT_FALSE(expected.ok());
    EXPECT_EQ(expected.status().code(), StatusCode::kOutOfRange);
    const auto captured = test::cli_process::Run(binary_, Arguments(request));
    ASSERT_TRUE(captured.ok()) << captured.status().ToString();
    EXPECT_EQ(captured->exit_code, 2);
    EXPECT_TRUE(captured->standard_output.empty());
    EXPECT_NE(captured->standard_error.find("parameters"), std::string::npos);
    EXPECT_NE(captured->standard_error.find("OUT_OF_RANGE"), std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(
        CliMatrix, InferenceCliIntegration, ::testing::ValuesIn(kCliFixtures),
        [](const ::testing::TestParamInfo<CliFixtureCase>& info) { return info.param.name; });

} // namespace
