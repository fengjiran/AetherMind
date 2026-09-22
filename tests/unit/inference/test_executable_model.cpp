#include "aethermind/inference/executable_model.h"

#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/compiler/model_compiler.h"
#include "aethermind/execution/execution_bindings.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/model/loaded_model.h"
#include "aethermind/runtime/runtime_builder.h"
#include "model/test_llama_checkpoint_helpers.h"

#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>

namespace {

using namespace aethermind;
using namespace aethermind::test;

Runtime MakeCpuRuntime() {
    RuntimeBuilder builder;
    builder.RegisterBackendFactory(DeviceType::kCPU, std::make_unique<CpuBackendFactory>());
    return builder.Build();
}

/// Compiled artifact plus the checkpoint data pointers assertions compare
/// against. They stay valid because the resolved weight views hold a shared_ptr
/// to the checkpoint storage, which the artifact owns.
struct CompiledLlama {
    LoweredModelArtifact artifact{};
    const void* embed_tokens_data = nullptr;
    const void* lm_head_data = nullptr;
};

/// opt_level 1 keeps the graph unfused, which is the only full-Llama shape the
/// CPU registry can resolve today: the fused O2 ops are packed-only.
StatusOr<CompiledLlama> CompileTinyLlama(bool tie_word_embeddings,
                                         uint32_t opt_level,
                                         bool enable_packed_weights) {
    const HfModelConfig config = MakeTinyLlamaConfig(/*num_layers=*/1, tie_word_embeddings);
    TinyLlamaCheckpoint checkpoint = MakeTinyLlamaCheckpoint(config);

    CompiledLlama compiled;
    compiled.embed_tokens_data = checkpoint.weights.embed_tokens.data;
    compiled.lm_head_data = checkpoint.weights.lm_head.has_value()
                                    ? checkpoint.weights.lm_head->data
                                    : nullptr;

    auto loaded = std::make_unique<LoadedModel>(config, std::move(checkpoint.weights));
    ModelCompileOptions options;
    options.optimization.opt_level = opt_level;
    options.lowering.enable_packed_weights = enable_packed_weights;
    auto artifact = ModelCompiler::Compile(std::move(loaded), options);
    if (!artifact.ok()) {
        return artifact.status();
    }
    compiled.artifact = std::move(*artifact);
    return compiled;
}

std::vector<uint32_t> BindingsSharingData(const ExternalTensorBindings& bindings,
                                          const void* data) {
    std::vector<uint32_t> indices;
    for (const ExternalReadOnlyValueBinding& entry: bindings.readable) {
        if (entry.tensor.data() == data) {
            indices.push_back(entry.value.index);
        }
    }
    return indices;
}

TEST(ExecutableModel, PreparesPlainUnfusedLlamaArtifact) {
    Runtime runtime = MakeCpuRuntime();
    auto compiled = CompileTinyLlama(/*tie_word_embeddings=*/false, /*opt_level=*/1,
                                     /*enable_packed_weights=*/false);
    ASSERT_TRUE(compiled.ok()) << compiled.status().ToString();

    const auto model = PrepareExecutableModel(runtime, std::move(compiled->artifact));
    ASSERT_TRUE(model.ok()) << model.status().ToString();

    EXPECT_NE(model->artifact_id(), 0U);
    EXPECT_EQ(model->phase(), ExecPhase::kBoth);

    const auto both = model->plan(ExecPhase::kBoth);
    ASSERT_TRUE(both.ok()) << both.status().ToString();
    ASSERT_NE(*both, nullptr);
    EXPECT_FALSE((*both)->steps().empty());

    // The baseline shares one immutable plan across phases, so every phase query
    // must return the same object rather than a copy.
    const auto prefill = model->plan(ExecPhase::kPrefill);
    const auto decode = model->plan(ExecPhase::kDecode);
    ASSERT_TRUE(prefill.ok()) << prefill.status().ToString();
    ASSERT_TRUE(decode.ok()) << decode.status().ToString();
    EXPECT_EQ(*prefill, *both);
    EXPECT_EQ(*decode, *both);

    const auto bindings = model->immutable_weight_bindings(ExecPhase::kBoth);
    ASSERT_TRUE(bindings.ok()) << bindings.status().ToString();
    EXPECT_TRUE((*bindings)->writable.empty());
    ASSERT_FALSE((*bindings)->readable.empty());
    for (const ExternalReadOnlyValueBinding& entry: (*bindings)->readable) {
        EXPECT_NE(entry.tensor.data(), nullptr) << "value " << entry.value.index;
        EXPECT_TRUE(entry.tensor.is_valid()) << "value " << entry.value.index;
    }
}

TEST(ExecutableModel, BindingsMatchExternalReadRequirementsExactly) {
    Runtime runtime = MakeCpuRuntime();
    auto compiled = CompileTinyLlama(/*tie_word_embeddings=*/false, /*opt_level=*/1,
                                     /*enable_packed_weights=*/false);
    ASSERT_TRUE(compiled.ok()) << compiled.status().ToString();
    const auto model = PrepareExecutableModel(runtime, std::move(compiled->artifact));
    ASSERT_TRUE(model.ok()) << model.status().ToString();

    const auto plan = model->plan(ExecPhase::kBoth);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    const auto required = ComputeExternalReadRequirements(**plan);
    ASSERT_TRUE(required.ok()) << required.status().ToString();
    const auto bindings = model->immutable_weight_bindings(ExecPhase::kBoth);
    ASSERT_TRUE(bindings.ok()) << bindings.status().ToString();

    // Reconcile in both directions against the public requirement query: model
    // inputs are session-supplied, everything else required must be bound once.
    std::vector<bool> expected((*plan)->values().size(), false);
    for (size_t i = 0; i < required->size(); ++i) {
        expected[i] = (*required)[i] &&
                      (*plan)->values()[i].kind != ExecutionValueKind::kModelInput;
    }
    std::vector<bool> bound(expected.size(), false);
    for (const ExternalReadOnlyValueBinding& entry: (*bindings)->readable) {
        ASSERT_LT(entry.value.index, bound.size());
        EXPECT_FALSE(bound[entry.value.index])
                << "value " << entry.value.index << " is bound twice";
        bound[entry.value.index] = true;
    }
    EXPECT_EQ(bound, expected);
}

TEST(ExecutableModel, PlainLlamaBindsEveryWeightAndNoModelInput) {
    Runtime runtime = MakeCpuRuntime();
    auto compiled = CompileTinyLlama(/*tie_word_embeddings=*/false, /*opt_level=*/1,
                                     /*enable_packed_weights=*/false);
    ASSERT_TRUE(compiled.ok()) << compiled.status().ToString();
    const auto model = PrepareExecutableModel(runtime, std::move(compiled->artifact));
    ASSERT_TRUE(model.ok()) << model.status().ToString();

    const auto plan = model->plan(ExecPhase::kBoth);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    const auto bindings = model->immutable_weight_bindings(ExecPhase::kBoth);
    ASSERT_TRUE(bindings.ok()) << bindings.status().ToString();

    size_t weight_values = 0;
    for (const ExecutionValueDesc& value: (*plan)->values()) {
        if (value.kind == ExecutionValueKind::kWeight) {
            ++weight_values;
        }
    }
    // One unfused decoder layer: embed_tokens, final_norm, lm_head, and nine
    // per-layer weights.
    EXPECT_EQ(weight_values, 12U);
    EXPECT_EQ((*bindings)->readable.size(), weight_values);
}

TEST(ExecutableModel, TiedLmHeadSharesEmbeddingBacking) {
    Runtime runtime = MakeCpuRuntime();
    auto compiled = CompileTinyLlama(/*tie_word_embeddings=*/true, /*opt_level=*/1,
                                     /*enable_packed_weights=*/false);
    ASSERT_TRUE(compiled.ok()) << compiled.status().ToString();
    ASSERT_EQ(compiled->lm_head_data, nullptr)
            << "a tied checkpoint carries no independent lm_head weight";
    const void* const embed_data = compiled->embed_tokens_data;

    const auto model = PrepareExecutableModel(runtime, std::move(compiled->artifact));
    ASSERT_TRUE(model.ok()) << model.status().ToString();
    const auto bindings = model->immutable_weight_bindings(ExecPhase::kBoth);
    ASSERT_TRUE(bindings.ok()) << bindings.status().ToString();

    // Two distinct weight values, one backing: the tie is resolved through
    // ResolveWeightBinding rather than by any session-side special case.
    const std::vector<uint32_t> sharing = BindingsSharingData(**bindings, embed_data);
    ASSERT_EQ(sharing.size(), 2U);
    EXPECT_NE(sharing[0], sharing[1]);
}

TEST(ExecutableModel, UntiedLmHeadKeepsItsOwnBacking) {
    Runtime runtime = MakeCpuRuntime();
    auto compiled = CompileTinyLlama(/*tie_word_embeddings=*/false, /*opt_level=*/1,
                                     /*enable_packed_weights=*/false);
    ASSERT_TRUE(compiled.ok()) << compiled.status().ToString();
    ASSERT_NE(compiled->lm_head_data, nullptr);
    ASSERT_NE(compiled->lm_head_data, compiled->embed_tokens_data);

    const auto model = PrepareExecutableModel(runtime, std::move(compiled->artifact));
    ASSERT_TRUE(model.ok()) << model.status().ToString();
    const auto bindings = model->immutable_weight_bindings(ExecPhase::kBoth);
    ASSERT_TRUE(bindings.ok()) << bindings.status().ToString();

    EXPECT_EQ(BindingsSharingData(**bindings, compiled->embed_tokens_data).size(), 1U);
    EXPECT_EQ(BindingsSharingData(**bindings, compiled->lm_head_data).size(), 1U);
}

TEST(ExecutableModel, SurvivesBeingMoved) {
    Runtime runtime = MakeCpuRuntime();
    auto compiled = CompileTinyLlama(/*tie_word_embeddings=*/true, /*opt_level=*/1,
                                     /*enable_packed_weights=*/false);
    ASSERT_TRUE(compiled.ok()) << compiled.status().ToString();
    const void* const embed_data = compiled->embed_tokens_data;

    auto prepared = PrepareExecutableModel(runtime, std::move(compiled->artifact));
    ASSERT_TRUE(prepared.ok()) << prepared.status().ToString();
    ExecutableModel model(std::move(*prepared));

    const auto bindings = model.immutable_weight_bindings(ExecPhase::kBoth);
    ASSERT_TRUE(bindings.ok()) << bindings.status().ToString();
    // The binding table borrows the metadata owned by the moved storage, so the
    // views must still describe the original weight bytes.
    ASSERT_EQ(BindingsSharingData(**bindings, embed_data).size(), 2U);
    for (const ExternalReadOnlyValueBinding& entry: (*bindings)->readable) {
        EXPECT_TRUE(entry.tensor.is_valid()) << "value " << entry.value.index;
    }
}

TEST(ExecutableModel, RejectsArtifactWithoutLoadedModel) {
    Runtime runtime = MakeCpuRuntime();

    const auto model = PrepareExecutableModel(runtime, LoweredModelArtifact{});

    ASSERT_FALSE(model.ok());
    EXPECT_EQ(model.status().code(), StatusCode::kInvalidArgument);
}

TEST(ExecutableModel, PackedLoweringIsUnresolvableForOpsWithoutPackedKernels) {
    // Lowering marks every weight-consuming step packed when
    // enable_packed_weights is set, but only QkvLinear, GateUpLinear and
    // AddRmsNorm register packed descriptors. Embedding is the first such step in
    // a Llama graph, so a packed full-model artifact cannot resolve; o_proj,
    // down_proj and lm_head (kLinear) would fail next. This records the gap and
    // fails loudly once packed variants land, which is when the packed
    // full-model path becomes testable.
    Runtime runtime = MakeCpuRuntime();
    auto compiled = CompileTinyLlama(/*tie_word_embeddings=*/false, /*opt_level=*/2,
                                     /*enable_packed_weights=*/true);
    ASSERT_TRUE(compiled.ok()) << compiled.status().ToString();

    const auto model = PrepareExecutableModel(runtime, std::move(compiled->artifact));

    ASSERT_FALSE(model.ok());
    // Kernel resolution, not weight resolution: the weights themselves are fine.
    EXPECT_EQ(model.status().code(), StatusCode::kNotFound);
}

} // namespace
