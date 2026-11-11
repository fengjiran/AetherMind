#include "aethermind/inference/model_execution_policy.h"
#include "aethermind/model/loaded_model.h"
#include "inference/inference_internal.h"
#include "model/test_llama_checkpoint_helpers.h"

#include <array>
#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace aethermind;

std::vector<RawWeightView*> MatrixViews(ResolvedModelWeights& weights) {
    std::vector<RawWeightView*> result{&weights.embed_tokens};
    if (weights.lm_head) result.push_back(&*weights.lm_head);
    for (auto& layer: weights.layers) {
        for (auto* view: {&layer.attn.q_proj, &layer.attn.k_proj, &layer.attn.v_proj,
                          &layer.attn.o_proj, &layer.mlp.gate_proj, &layer.mlp.up_proj,
                          &layer.mlp.down_proj}) {
            result.push_back(view);
        }
    }
    return result;
}

void SetDType(RawWeightView& view, DataType dtype) {
    const size_t elements = view.bytes / view.dtype.nbytes();
    view.dtype = dtype;
    view.bytes = elements * dtype.nbytes();
}

class W16WeightPolicy : public ::testing::TestWithParam<WeightPolicy> {
protected:
    static DataType MatrixDType() {
        return GetParam() == WeightPolicy::kKeepFp16WeightsFp32Activation
                       ? DataType::Float(16)
                       : DataType::BFloat(16);
    }

    static test::TinyLlamaCheckpoint Checkpoint(const HfModelConfig& config) {
        auto checkpoint = test::MakeTinyLlamaCheckpoint(config);
        for (auto* view: MatrixViews(checkpoint.weights)) {
            SetDType(*view, MatrixDType());
        }
        return checkpoint;
    }
};

TEST_P(W16WeightPolicy, RecognizedHintsAreProvenanceEvenWhenDifferentFromMatrices) {
    for (const auto hint: {DataType::Float32(),
                           DataType::Float(16),
                           DataType::BFloat(16),
                           DataType{}}) {
        auto config = test::MakeTinyLlamaConfig(1, false);
        config.weight_dtype_hint = hint;
        config.weight_dtype_hint_name = hint.IsUndefined() ? "auto" : ToString(hint);
        auto [weights] = Checkpoint(config);
        const LoadedModel model(config, std::move(weights));
        EXPECT_TRUE(ValidateSourceWeights(model, GetParam()).ok());
        EXPECT_TRUE(inference::internal::ValidatePreparedWeights(model, GetParam()).ok());
        EXPECT_EQ(model.GetConfig().weight_dtype_hint, hint);
        EXPECT_EQ(model.GetConfig().weight_dtype_hint_name, config.weight_dtype_hint_name);
    }
}

TEST_P(W16WeightPolicy, EveryMatrixRoleMustMatchPolicyRegardlessOfHint) {
    const auto config = test::MakeTinyLlamaConfig(1, false);
    auto checkpoint = Checkpoint(config);
    auto views = MatrixViews(checkpoint.weights);
    for (auto* view: views) {
        const RawWeightView original = *view;
        SetDType(*view, DataType::Float32());
        const LoadedModel model(config, checkpoint.weights);
        EXPECT_EQ(ValidateSourceWeights(model, GetParam()),
                  StatusCode::kUnimplemented);
        EXPECT_EQ(inference::internal::ValidatePreparedWeights(model, GetParam()),
                  StatusCode::kUnimplemented);
        *view = original;
    }
}

TEST_P(W16WeightPolicy, NormSourcesMayUseEitherReducedFormatButPreparedNormsAreFloat32) {
    const auto config = test::MakeTinyLlamaConfig(1, false);
    for (const auto norm_dtype: {DataType::Float32(),
                                 DataType::Float(16),
                                 DataType::BFloat(16)}) {
        auto [weights] = Checkpoint(config);
        for (auto* norm: {&weights.final_norm,
                          &weights.layers[0].norm.input_rmsnorm,
                          &weights.layers[0].norm.post_attn_rmsnorm}) {
            SetDType(*norm, norm_dtype);
        }
        const LoadedModel model(config, std::move(weights));
        EXPECT_TRUE(ValidateSourceWeights(model, GetParam()).ok());
        const auto prepared = inference::internal::ValidatePreparedWeights(
                model, GetParam());
        EXPECT_EQ(prepared.ok(), norm_dtype == DataType::Float32());
    }
}

TEST_P(W16WeightPolicy, RejectsUnsupportedHintsAndNonScalarViews) {
    for (const auto hint: {DataType::Float(64),
                           DataType::Float(16, 2),
                           DataType{}}) {
        auto config = test::MakeTinyLlamaConfig(1, false);
        config.weight_dtype_hint = hint;
        config.weight_dtype_hint_name = "unsupported";
        auto checkpoint = Checkpoint(config);
        const LoadedModel model(config, std::move(checkpoint.weights));
        EXPECT_EQ(ValidateSourceWeights(model, GetParam()),
                  StatusCode::kUnimplemented);
    }

    const auto config = test::MakeTinyLlamaConfig(1, false);
    for (const bool norm: {false, true}) {
        auto checkpoint = Checkpoint(config);
        auto& view = norm ? checkpoint.weights.final_norm : checkpoint.weights.embed_tokens;
        view.dtype = DataType(DLDataTypeCode::kFloat, 16, 2, true);
        const LoadedModel model(config, std::move(checkpoint.weights));
        // Non-scalar metadata fails IsValid() before the dtype whitelist, so it
        // is invalid metadata rather than an unsupported precision source.
        EXPECT_EQ(ValidateSourceWeights(model, GetParam()),
                  StatusCode::kInvalidArgument);
    }
}

TEST_P(W16WeightPolicy, PreservesAbsentAndExplicitIndependentLmHeadRules) {
    for (const bool tied: {false, true}) {
        auto config = test::MakeTinyLlamaConfig(1, tied);
        auto checkpoint = Checkpoint(config);
        if (!tied) {
            ASSERT_TRUE(checkpoint.weights.lm_head);
            EXPECT_NE(checkpoint.weights.lm_head->storage, checkpoint.weights.embed_tokens.storage);
            // A tied flag does not rewrite an explicitly supplied independent tensor.
            config.tie_word_embeddings = true;
        }
        const LoadedModel model(config, std::move(checkpoint.weights));
        EXPECT_TRUE(ValidateSourceWeights(model, GetParam()).ok());
    }
}

INSTANTIATE_TEST_SUITE_P(RetainedPrecision, W16WeightPolicy,
                         ::testing::Values(WeightPolicy::kKeepFp16WeightsFp32Activation,
                                           WeightPolicy::kKeepBf16WeightsFp32Activation));

TEST(W16PreparedWeightPolicy, RejectsUnknownPolicy) {
    const auto config = test::MakeTinyLlamaConfig(1, true);
    auto checkpoint = test::MakeTinyLlamaCheckpoint(config);
    const LoadedModel model(config, std::move(checkpoint.weights));
    EXPECT_EQ(inference::internal::ValidatePreparedWeights(model, static_cast<WeightPolicy>(255)),
              StatusCode::kInvalidArgument);
}

} // namespace
