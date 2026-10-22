#include "inference_runner.h"

#include "aethermind/graph/graph_types.h"
#include "aethermind/inference/model_execution_policy.h"
#include "aethermind/model/loaded_model.h"
#include "aethermind/model/model_loader.h"
#include "model/test_llama_checkpoint_helpers.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace {

using namespace aethermind;
using namespace aethermind::apps::inference;
namespace fs = std::filesystem;

struct HfAdmissionCase {
    const char* directory;
    const char* parameter_name;
    bool accepted;
    const char* diagnostic;
    DataType actual_dtype;
    std::optional<uint32_t> layer{};
};

const std::array<HfAdmissionCase, 20> kHfAdmissionCases{{
        {"f32-missing", "F32Missing", true, "", {}},
        {"f32-auto", "F32Auto", true, "", {}},
        {"f32-explicit", "F32Explicit", true, "", {}},
        {"f32-tied", "F32Tied", true, "", {}},
        {"f16-missing", "F16Missing", false, "TokenEmbedding", DataType::Float(16)},
        {"f16-auto", "F16Auto", false, "TokenEmbedding", DataType::Float(16)},
        {"bf16-missing", "BF16Missing", false, "TokenEmbedding", DataType::BFloat(16)},
        {"bf16-auto", "BF16Auto", false, "TokenEmbedding", DataType::BFloat(16)},
        {"f16-f32-hint", "F16F32Hint", false, "TokenEmbedding", DataType::Float(16)},
        {"bf16-f32-hint", "BF16F32Hint", false, "TokenEmbedding", DataType::BFloat(16)},
        {"f32-f16-hint", "F32F16Hint", false, "hint", DataType::Float(16)},
        {"f32-bf16-hint", "F32BF16Hint", false, "hint", DataType::BFloat(16)},
        {"mixed-norm-f16-missing", "MixedNormF16Missing", false, "FinalNorm", DataType::Float(16)},
        {"mixed-norm-f16-auto", "MixedNormF16Auto", false, "FinalNorm", DataType::Float(16)},
        {"mixed-norm-bf16-missing", "MixedNormBF16Missing", false, "FinalNorm", DataType::BFloat(16)},
        {"mixed-norm-bf16-auto", "MixedNormBF16Auto", false, "FinalNorm", DataType::BFloat(16)},
        {"single-norm-f16-missing", "SingleNormF16Missing", false, "PostAttentionNorm", DataType::Float(16), 1U},
        {"single-norm-f16-auto", "SingleNormF16Auto", false, "PostAttentionNorm", DataType::Float(16), 1U},
        {"single-norm-bf16-missing", "SingleNormBF16Missing", false, "PostAttentionNorm", DataType::BFloat(16), 1U},
        {"single-norm-bf16-auto", "SingleNormBF16Auto", false, "PostAttentionNorm", DataType::BFloat(16), 1U},
}};

fs::path HfDirectory(const char* name) {
    return fs::path(AETHERMIND_TEST_MODELS_DIR) / "inference-admission" / name;
}

void ExpectPrecisionDiagnostic(const Status& status, const char* role,
                               DataType actual_dtype, std::optional<uint32_t> layer = std::nullopt) {
    EXPECT_EQ(status.code(), StatusCode::kUnimplemented);
    EXPECT_NE(status.message().find(role), std::string::npos) << status.message();
    EXPECT_NE(status.message().find(ToString(actual_dtype)), std::string::npos) << status.message();
    EXPECT_NE(status.message().find("required Float32"), std::string::npos) << status.message();
    if (layer.has_value()) {
        EXPECT_NE(status.message().find("layer " + std::to_string(*layer)), std::string::npos)
                << status.message();
    }
}

class Fp32HfAdmission : public ::testing::TestWithParam<HfAdmissionCase> {};

TEST_P(Fp32HfAdmission, AppliesPolicyAfterRealLoaderAcceptsFixture) {
    const auto& expected = GetParam();
    auto loaded = ModelLoader::Load(HfDirectory(expected.directory));
    ASSERT_TRUE(loaded.ok())
            << "Generate and check M0 fixtures before testing admission: " << loaded.status().ToString();

    const Status status = ValidateFp32ExecutionInputs(**loaded);
    if (expected.accepted) {
        EXPECT_TRUE(status.ok()) << status.ToString();
    } else {
        ExpectPrecisionDiagnostic(status, expected.diagnostic, expected.actual_dtype, expected.layer);
    }
}

INSTANTIATE_TEST_SUITE_P(
        AdmissionFixtures, Fp32HfAdmission, ::testing::ValuesIn(kHfAdmissionCases),
        [](const ::testing::TestParamInfo<HfAdmissionCase>& info) { return info.param.parameter_name; });

struct HfCompileCase {
    const char* directory;
    const char* parameter_name;
    CpuInferencePreset preset;
};

constexpr std::array<HfCompileCase, 8> kHfCompileCases{{
        {"f32-missing", "F32MissingO1Plain", CpuInferencePreset::kO1Plain},
        {"f32-missing", "F32MissingO2Packed", CpuInferencePreset::kO2Packed},
        {"f32-auto", "F32AutoO1Plain", CpuInferencePreset::kO1Plain},
        {"f32-auto", "F32AutoO2Packed", CpuInferencePreset::kO2Packed},
        {"f32-explicit", "F32ExplicitO1Plain", CpuInferencePreset::kO1Plain},
        {"f32-explicit", "F32ExplicitO2Packed", CpuInferencePreset::kO2Packed},
        {"f32-tied", "F32TiedO1Plain", CpuInferencePreset::kO1Plain},
        {"f32-tied", "F32TiedO2Packed", CpuInferencePreset::kO2Packed},
}};

class Fp32HfCompile : public ::testing::TestWithParam<HfCompileCase> {};

TEST_P(Fp32HfCompile, CompilesAcceptedInputsWithExplicitCpuPreset) {
    auto loaded = ModelLoader::Load(HfDirectory(GetParam().directory));
    ASSERT_TRUE(loaded.ok()) << loaded.status().ToString();
    const Status admitted = ValidateFp32ExecutionInputs(**loaded);
    ASSERT_TRUE(admitted.ok()) << admitted.ToString();
    const auto options = MakeCpuCompileOptions(GetParam().preset);
    ASSERT_TRUE(options.ok()) << options.status().ToString();

    const auto compiled = ModelCompiler::Compile(std::move(*loaded), *options);
    ASSERT_TRUE(compiled.ok()) << compiled.status().ToString();
    EXPECT_FALSE(compiled->graph.steps().empty());
    EXPECT_NE(compiled->loaded_model, nullptr);
    for (const auto& step: compiled->graph.steps()) {
        EXPECT_EQ(step.spec.selector.device_type, DeviceType::kCPU);
        EXPECT_EQ(step.spec.selector.phase, ExecPhase::kBoth);
    }
}

INSTANTIATE_TEST_SUITE_P(
        AdmissionFixtures, Fp32HfCompile, ::testing::ValuesIn(kHfCompileCases),
        [](const ::testing::TestParamInfo<HfCompileCase>& info) { return info.param.parameter_name; });

struct RequiredRoleCase {
    TransformerWeightRole role;
    std::optional<uint32_t> layer;
    const char* diagnostic;
    const char* parameter_name;
};

constexpr std::array<RequiredRoleCase, 21> kRequiredRoles{{
        {TransformerWeightRole::kTokenEmbedding, std::nullopt, "TokenEmbedding", "TokenEmbedding"},
        {TransformerWeightRole::kFinalNorm, std::nullopt, "FinalNorm", "FinalNorm"},
        {TransformerWeightRole::kLmHead, std::nullopt, "LmHead", "LmHead"},
        {TransformerWeightRole::kInputNorm, 0U, "InputNorm", "Layer0InputNorm"},
        {TransformerWeightRole::kPostAttentionNorm, 0U, "PostAttentionNorm", "Layer0PostAttentionNorm"},
        {TransformerWeightRole::kAttentionQ, 0U, "AttentionQ", "Layer0AttentionQ"},
        {TransformerWeightRole::kAttentionK, 0U, "AttentionK", "Layer0AttentionK"},
        {TransformerWeightRole::kAttentionV, 0U, "AttentionV", "Layer0AttentionV"},
        {TransformerWeightRole::kAttentionO, 0U, "AttentionO", "Layer0AttentionO"},
        {TransformerWeightRole::kMlpGate, 0U, "MlpGate", "Layer0MlpGate"},
        {TransformerWeightRole::kMlpUp, 0U, "MlpUp", "Layer0MlpUp"},
        {TransformerWeightRole::kMlpDown, 0U, "MlpDown", "Layer0MlpDown"},
        {TransformerWeightRole::kInputNorm, 1U, "InputNorm", "Layer1InputNorm"},
        {TransformerWeightRole::kPostAttentionNorm, 1U, "PostAttentionNorm", "Layer1PostAttentionNorm"},
        {TransformerWeightRole::kAttentionQ, 1U, "AttentionQ", "Layer1AttentionQ"},
        {TransformerWeightRole::kAttentionK, 1U, "AttentionK", "Layer1AttentionK"},
        {TransformerWeightRole::kAttentionV, 1U, "AttentionV", "Layer1AttentionV"},
        {TransformerWeightRole::kAttentionO, 1U, "AttentionO", "Layer1AttentionO"},
        {TransformerWeightRole::kMlpGate, 1U, "MlpGate", "Layer1MlpGate"},
        {TransformerWeightRole::kMlpUp, 1U, "MlpUp", "Layer1MlpUp"},
        {TransformerWeightRole::kMlpDown, 1U, "MlpDown", "Layer1MlpDown"},
}};

RawWeightView& RoleForMutation(ResolvedModelWeights& weights, const RequiredRoleCase& selected) {
    switch (selected.role) {
        case TransformerWeightRole::kTokenEmbedding:
            return weights.embed_tokens;
        case TransformerWeightRole::kFinalNorm:
            return weights.final_norm;
        case TransformerWeightRole::kLmHead:
            return *weights.lm_head;
        case TransformerWeightRole::kInputNorm:
            return weights.layers.at(*selected.layer).norm.input_rmsnorm;
        case TransformerWeightRole::kPostAttentionNorm:
            return weights.layers.at(*selected.layer).norm.post_attn_rmsnorm;
        case TransformerWeightRole::kAttentionQ:
            return weights.layers.at(*selected.layer).attn.q_proj;
        case TransformerWeightRole::kAttentionK:
            return weights.layers.at(*selected.layer).attn.k_proj;
        case TransformerWeightRole::kAttentionV:
            return weights.layers.at(*selected.layer).attn.v_proj;
        case TransformerWeightRole::kAttentionO:
            return weights.layers.at(*selected.layer).attn.o_proj;
        case TransformerWeightRole::kMlpGate:
            return weights.layers.at(*selected.layer).mlp.gate_proj;
        case TransformerWeightRole::kMlpUp:
            return weights.layers.at(*selected.layer).mlp.up_proj;
        case TransformerWeightRole::kMlpDown:
            return weights.layers.at(*selected.layer).mlp.down_proj;
        case TransformerWeightRole::kMoERouter:
            ADD_FAILURE() << "MoE roles are outside the dense fixture";
            return weights.embed_tokens;
    }
    ADD_FAILURE() << "Unknown weight role";
    return weights.embed_tokens;
}

class Fp32RequiredRoleAdmission : public ::testing::TestWithParam<RequiredRoleCase> {
protected:
    void ExpectRejectedDType(DataType dtype) {
        const auto config = test::MakeTinyLlamaConfig(2, false);
        auto checkpoint = test::MakeTinyLlamaCheckpoint(config);
        auto& selected = RoleForMutation(checkpoint.weights, GetParam());
        selected.dtype = dtype;
        selected.bytes /= 2;
        const LoadedModel model(config, std::move(checkpoint.weights));

        const Status status = ValidateFp32ExecutionInputs(model);
        ExpectPrecisionDiagnostic(status, GetParam().diagnostic, dtype, GetParam().layer);
    }
};

TEST_P(Fp32RequiredRoleAdmission, RejectsFloat16AtEveryRequiredRole) {
    ExpectRejectedDType(DataType::Float(16));
}

TEST_P(Fp32RequiredRoleAdmission, RejectsBFloat16AtEveryRequiredRole) {
    ExpectRejectedDType(DataType::BFloat(16));
}

INSTANTIATE_TEST_SUITE_P(
        RequiredRoles, Fp32RequiredRoleAdmission, ::testing::ValuesIn(kRequiredRoles),
        [](const ::testing::TestParamInfo<RequiredRoleCase>& info) { return info.param.parameter_name; });

TEST(Fp32ExecutionInputs, AcceptsCompleteScalarFp32Metadata) {
    const auto config = test::MakeTinyLlamaConfig(2, false);
    auto checkpoint = test::MakeTinyLlamaCheckpoint(config);
    const LoadedModel model(config, std::move(checkpoint.weights));
    const Status status = ValidateFp32ExecutionInputs(model);
    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(Fp32ExecutionInputs, AcceptsUndefinedHintOnlyAfterCheckingActualWeights) {
    for (const char* hint_name: {"", "auto"}) {
        SCOPED_TRACE(hint_name);
        auto config = test::MakeTinyLlamaConfig(2, false);
        config.weight_dtype_hint = {};
        config.weight_dtype_hint_name = hint_name;
        auto checkpoint = test::MakeTinyLlamaCheckpoint(config);
        checkpoint.weights.layers.back().norm.post_attn_rmsnorm.dtype = DataType::BFloat(16);
        checkpoint.weights.layers.back().norm.post_attn_rmsnorm.bytes /= 2;
        const LoadedModel model(config, std::move(checkpoint.weights));
        const Status status = ValidateFp32ExecutionInputs(model);
        ExpectPrecisionDiagnostic(status, "PostAttentionNorm", DataType::BFloat(16), 1U);
    }
}

TEST(Fp32ExecutionInputs, RejectsNonFp32HintWithFp32Weights) {
    for (const DataType dtype: {DataType::Float(16), DataType::BFloat(16)}) {
        SCOPED_TRACE(ToString(dtype));
        auto config = test::MakeTinyLlamaConfig(2, false);
        config.weight_dtype_hint = dtype;
        config.weight_dtype_hint_name = ToString(dtype);
        auto checkpoint = test::MakeTinyLlamaCheckpoint(config);
        const LoadedModel model(config, std::move(checkpoint.weights));
        const Status status = ValidateFp32ExecutionInputs(model);
        ExpectPrecisionDiagnostic(status, "hint", dtype);
    }
}

TEST(Fp32ExecutionInputs, AcceptsTiedHeadByExistingEmbeddingFallback) {
    const auto config = test::MakeTinyLlamaConfig(2, true);
    auto checkpoint = test::MakeTinyLlamaCheckpoint(config);
    ASSERT_FALSE(checkpoint.weights.lm_head.has_value());
    const LoadedModel model(config, std::move(checkpoint.weights));
    const Status status = ValidateFp32ExecutionInputs(model);
    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(Fp32ExecutionInputs, RejectsInvalidRequiredViewInsteadOfSkippingIt) {
    const auto config = test::MakeTinyLlamaConfig(2, false);
    auto checkpoint = test::MakeTinyLlamaCheckpoint(config);
    checkpoint.weights.layers.back().mlp.down_proj.storage.reset();
    const LoadedModel model(config, std::move(checkpoint.weights));
    const Status status = ValidateFp32ExecutionInputs(model);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message().find("MlpDown"), std::string::npos);
    EXPECT_NE(status.message().find("layer 1"), std::string::npos);
}

TEST(Fp32ExecutionInputs, RejectsMissingEmbeddingUsedByTiedHead) {
    const auto config = test::MakeTinyLlamaConfig(2, true);
    auto checkpoint = test::MakeTinyLlamaCheckpoint(config);
    checkpoint.weights.embed_tokens = {};
    const LoadedModel model(config, std::move(checkpoint.weights));
    const Status status = ValidateFp32ExecutionInputs(model);
    EXPECT_FALSE(status.ok());
}

TEST(Fp32ExecutionInputs, RejectsResolvedLayerCountMismatch) {
    const auto config = test::MakeTinyLlamaConfig(2, false);
    auto checkpoint = test::MakeTinyLlamaCheckpoint(config);
    checkpoint.weights.layers.pop_back();
    const LoadedModel model(config, std::move(checkpoint.weights));
    const Status status = ValidateFp32ExecutionInputs(model);
    EXPECT_FALSE(status.ok());
}

TEST(Fp32ExecutionInputs, RejectsNonpositiveDeclaredLayerCount) {
    for (const int64_t count: {int64_t{0}, int64_t{-1}}) {
        auto config = test::MakeTinyLlamaConfig(2, false);
        auto checkpoint = test::MakeTinyLlamaCheckpoint(config);
        config.num_hidden_layers = count;
        const LoadedModel model(config, std::move(checkpoint.weights));
        const Status status = ValidateFp32ExecutionInputs(model);
        EXPECT_FALSE(status.ok());
    }
}

TEST(Fp32ExecutionInputs, RejectsFloat32VectorLanesAsUnsupportedPrecision) {
    const auto config = test::MakeTinyLlamaConfig(2, false);
    auto checkpoint = test::MakeTinyLlamaCheckpoint(config);
    auto& weight = checkpoint.weights.layers.back().norm.post_attn_rmsnorm;
    weight.dtype = DataType::Float(32, 4);
    const LoadedModel model(config, std::move(checkpoint.weights));
    const Status status = ValidateFp32ExecutionInputs(model);
    ExpectPrecisionDiagnostic(status, "PostAttentionNorm", DataType::Float(32, 4), 1U);
}

} // namespace
