#include "aethermind/inference/model_execution_policy.h"
#include "aethermind/model/loaded_model.h"
#include "model/test_llama_checkpoint_helpers.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <utility>

namespace {

using namespace aethermind;

LoadedModel MakeModel(HfModelConfig config) {
    auto checkpoint = test::MakeTinyLlamaCheckpoint(config);
    return LoadedModel(std::move(config), std::move(checkpoint.weights));
}

void ExpectPrecisionFailure(const Status& status, const char* role) {
    EXPECT_EQ(status.code(), StatusCode::kUnimplemented) << status.ToString();
    EXPECT_NE(status.message().find(role), std::string::npos) << status.ToString();
}

TEST(Fp32CompatibilityPolicy, CompatibleSourceAcceptsMixedScalarCheckpointPrecisions) {
    auto config = test::MakeTinyLlamaConfig(/*num_layers=*/2, /*tie_word_embeddings=*/false);
    config.weight_dtype_hint = DataType::Float(16);
    config.weight_dtype_hint_name = "float16";
    auto checkpoint = test::MakeTinyLlamaCheckpoint(config);
    checkpoint.weights.embed_tokens.dtype = DataType::Float(16);
    checkpoint.weights.final_norm.dtype = DataType::BFloat(16);
    checkpoint.weights.layers[1].attn.v_proj.dtype = DataType::Float(16);
    checkpoint.weights.layers[1].mlp.down_proj.dtype = DataType::BFloat(16);
    const LoadedModel model(config, std::move(checkpoint.weights));

    EXPECT_TRUE(ValidateFp32CompatibleSourceInputs(model).ok());
    ExpectPrecisionFailure(ValidateFp32ExecutionWeights(model), "TokenEmbedding");
    ExpectPrecisionFailure(ValidateFp32ExecutionInputs(model), "hint");
}

TEST(Fp32CompatibilityPolicy, CompatibleSourceTreatsAutoHintAsProvenance) {
    auto config = test::MakeTinyLlamaConfig(/*num_layers=*/1, /*tie_word_embeddings=*/false);
    config.weight_dtype_hint = {};
    config.weight_dtype_hint_name = "auto";
    auto checkpoint = test::MakeTinyLlamaCheckpoint(config);
    checkpoint.weights.final_norm.dtype = DataType::Float(16);
    checkpoint.weights.layers[0].norm.post_attn_rmsnorm.dtype = DataType::BFloat(16);
    const LoadedModel model(config, std::move(checkpoint.weights));

    EXPECT_TRUE(ValidateFp32CompatibleSourceInputs(model).ok());
    ExpectPrecisionFailure(ValidateFp32ExecutionWeights(model), "FinalNorm");
    ExpectPrecisionFailure(ValidateFp32ExecutionInputs(model), "FinalNorm");
}

TEST(Fp32CompatibilityPolicy, ExecutionWeightsIgnoreSourceHintAfterNormalization) {
    auto config = test::MakeTinyLlamaConfig(/*num_layers=*/1, /*tie_word_embeddings=*/false);
    config.weight_dtype_hint = DataType::BFloat(16);
    config.weight_dtype_hint_name = "bfloat16";
    const LoadedModel model = MakeModel(config);

    EXPECT_TRUE(ValidateFp32ExecutionWeights(model).ok());
    EXPECT_TRUE(ValidateFp32CompatibleSourceInputs(model).ok());
    ExpectPrecisionFailure(ValidateFp32ExecutionInputs(model), "hint");
}

TEST(Fp32CompatibilityPolicy, ExecutionWeightsRejectsNonFp32RequiredRoleWithoutReadingPayload) {
    const auto config = test::MakeTinyLlamaConfig(/*num_layers=*/1, /*tie_word_embeddings=*/false);
    auto checkpoint = test::MakeTinyLlamaCheckpoint(config);
    checkpoint.weights.layers[0].attn.q_proj.dtype = DataType::Float(16);
    const LoadedModel model(config, std::move(checkpoint.weights));

    ExpectPrecisionFailure(ValidateFp32ExecutionWeights(model), "AttentionQ");
}

TEST(Fp32CompatibilityPolicy, CompatibleSourceRejectsUnsupportedScalarPrecision) {
    const auto config = test::MakeTinyLlamaConfig(/*num_layers=*/1, /*tie_word_embeddings=*/false);
    auto checkpoint = test::MakeTinyLlamaCheckpoint(config);
    checkpoint.weights.layers[0].mlp.gate_proj.dtype = DataType::Float(64);
    const LoadedModel model(config, std::move(checkpoint.weights));

    ExpectPrecisionFailure(ValidateFp32CompatibleSourceInputs(model), "MlpGate");
}

TEST(Fp32CompatibilityPolicy, CompatibleSourceRejectsVectorScalableAndReservedPrecisions) {
    const auto config = test::MakeTinyLlamaConfig(/*num_layers=*/1, /*tie_word_embeddings=*/false);
    for (const DataType dtype: {DataType::Float(32, 4),
                                DataType(DLDataTypeCode::kFloat, 16, 2, true),
                                DataType{}}) {
        SCOPED_TRACE(ToString(dtype));
        auto checkpoint = test::MakeTinyLlamaCheckpoint(config);
        checkpoint.weights.layers[0].norm.input_rmsnorm.dtype = dtype;
        const LoadedModel model(config, std::move(checkpoint.weights));

        const Status status = ValidateFp32CompatibleSourceInputs(model);
        EXPECT_FALSE(status.ok()) << status.ToString();
        EXPECT_NE(status.message().find("InputNorm"), std::string::npos) << status.ToString();
    }
}

TEST(Fp32CompatibilityPolicy, ExecutionWeightsRejectsMissingRoleAndLayerMismatch) {
    {
        const auto config = test::MakeTinyLlamaConfig(/*num_layers=*/1, /*tie_word_embeddings=*/false);
        auto checkpoint = test::MakeTinyLlamaCheckpoint(config);
        checkpoint.weights.lm_head = RawWeightView{};
        const LoadedModel model(config, std::move(checkpoint.weights));
        const Status status = ValidateFp32ExecutionWeights(model);
        EXPECT_FALSE(status.ok());
        EXPECT_NE(status.message().find("LmHead"), std::string::npos);
    }
    {
        const auto config = test::MakeTinyLlamaConfig(/*num_layers=*/2, /*tie_word_embeddings=*/false);
        auto checkpoint = test::MakeTinyLlamaCheckpoint(config);
        checkpoint.weights.layers.pop_back();
        const LoadedModel model(config, std::move(checkpoint.weights));
        const Status status = ValidateFp32ExecutionWeights(model);
        EXPECT_FALSE(status.ok());
        EXPECT_NE(status.message().find("layer count"), std::string::npos);
    }
}

TEST(Fp32CompatibilityPolicy, RejectsNonScalarSourceHintsBeforeDTypeComparison) {
    for (const DataType hint: {DataType::Float(16, 2),
                               DataType(DLDataTypeCode::kFloat, 16, 2, true),
                               DataType(DLDataTypeCode::kBFloat, 16, 2, true)}) {
        auto config = test::MakeTinyLlamaConfig(1, false);
        config.weight_dtype_hint = hint;
        config.weight_dtype_hint_name = "invalid source hint";
        const LoadedModel model = MakeModel(config);
        ExpectPrecisionFailure(ValidateFp32CompatibleSourceInputs(model), "hint");
        ExpectPrecisionFailure(ValidateFp32ExecutionInputs(model), "hint");
        EXPECT_TRUE(ValidateFp32ExecutionWeights(model).ok());
    }
}

TEST(Fp32CompatibilityPolicy, StrictPolicyRemainsFp32SourceAndExecutionAdmission) {
    auto config = test::MakeTinyLlamaConfig(/*num_layers=*/1, /*tie_word_embeddings=*/false);
    config.weight_dtype_hint = DataType::Float(16);
    config.weight_dtype_hint_name = "float16";
    const LoadedModel model = MakeModel(config);

    ExpectPrecisionFailure(ValidateFp32ExecutionInputs(model), "hint");
    EXPECT_TRUE(ValidateFp32ExecutionWeights(model).ok());
}

} // namespace
