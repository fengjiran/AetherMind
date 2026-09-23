#include "aethermind/inference/executable_model.h"

#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/base/device.h"
#include "aethermind/compiler/graph_lowering.h"
#include "aethermind/compiler/model_compiler.h"
#include "aethermind/execution/execution_bindings.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/graph/graph.h"
#include "aethermind/memory/cpu_allocator.h"
#include "aethermind/model/loaded_model.h"
#include "aethermind/operators/op_params.h"
#include "aethermind/operators/op_type.h"
#include "aethermind/runtime/runtime_builder.h"
#include "aethermind/shape_inference/tensor_spec.h"
#include "execution/test_tensor_buffer_helpers.h"
#include "model/test_llama_checkpoint_helpers.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace aethermind;
using namespace aethermind::test;

constexpr float kEpsilon = 1.0e-5F;

Runtime MakeCpuRuntime() {
    RuntimeBuilder builder;
    builder.RegisterBackendFactory(DeviceType::kCPU, std::make_unique<CpuBackendFactory>());
    return builder.Build();
}

SymbolicShape StaticShape(std::initializer_list<int64_t> dimensions) {
    const std::vector<int64_t> copied(dimensions);
    return SymbolicShape(IntArrayView{copied});
}

TensorSpec FloatSpec(std::initializer_list<int64_t> dimensions) {
    return TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape(dimensions)};
}

std::vector<int64_t> Copy(IntArrayView view) {
    return {view.begin(), view.end()};
}

std::shared_ptr<const std::vector<std::byte>> MakeConstantBytes(size_t bytes) {
    return std::make_shared<const std::vector<std::byte>>(bytes, std::byte{0});
}

/// Every distinct weight backing in a checkpoint, in a stable order.
std::vector<const void*> CheckpointWeightData(const ResolvedModelWeights& weights) {
    std::vector<const void*> data;
    data.push_back(weights.embed_tokens.data);
    data.push_back(weights.final_norm.data);
    if (weights.lm_head.has_value()) {
        data.push_back(weights.lm_head->data);
    }
    for (const DecoderLayerRawWeights& layer: weights.layers) {
        data.push_back(layer.norm.input_rmsnorm.data);
        data.push_back(layer.norm.post_attn_rmsnorm.data);
        data.push_back(layer.attn.q_proj.data);
        data.push_back(layer.attn.k_proj.data);
        data.push_back(layer.attn.v_proj.data);
        data.push_back(layer.attn.o_proj.data);
        data.push_back(layer.mlp.gate_proj.data);
        data.push_back(layer.mlp.up_proj.data);
        data.push_back(layer.mlp.down_proj.data);
    }
    return data;
}

/// Compiled artifact plus the checkpoint data pointers assertions compare
/// against. They stay valid because the resolved weight views hold a shared_ptr
/// to their storage, which the artifact owns.
struct CompiledLlama {
    LoweredModelArtifact artifact{};
    const void* embed_tokens_data = nullptr;
    const void* lm_head_data = nullptr;
    std::vector<const void*> weight_data{};
};

/// opt_level 1 keeps the graph unfused, which is the only full-Llama shape the
/// CPU registry can resolve today: the fused O2 ops are packed-only.
StatusOr<CompiledLlama> CompileTinyLlama(int64_t num_layers,
                                         bool tie_word_embeddings,
                                         uint32_t opt_level,
                                         bool enable_packed_weights) {
    const HfModelConfig config = MakeTinyLlamaConfig(num_layers, tie_word_embeddings);
    TinyLlamaCheckpoint checkpoint = MakeTinyLlamaCheckpoint(config);

    CompiledLlama compiled;
    compiled.embed_tokens_data = checkpoint.weights.embed_tokens.data;
    compiled.lm_head_data = checkpoint.weights.lm_head.has_value()
                                    ? checkpoint.weights.lm_head->data
                                    : nullptr;
    compiled.weight_data = CheckpointWeightData(checkpoint.weights);

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

/// Wraps a finalized lowered graph into an artifact backed by `weights`.
LoweredModelArtifact MakeArtifactFrom(LoweredGraph graph, ResolvedModelWeights weights) {
    auto loaded = std::make_unique<LoadedModel>(
            MakeTinyLlamaConfig(/*num_layers=*/1, /*tie_word_embeddings=*/false),
            std::move(weights));
    return LoweredModelArtifact{.loaded_model = std::move(loaded),
                                .graph = std::move(graph)};
}

/// Builds an artifact from an arbitrary semantic graph, mirroring
/// ModelCompiler::Compile minus the Llama-specific front end. The plan is still
/// produced only by PrepareExecutableModel.
StatusOr<LoweredModelArtifact> MakeArtifact(const ModelGraph& graph,
                                            ResolvedModelWeights weights,
                                            GraphLoweringConfig config) {
    const auto lowered = LowerModelGraph(graph, config);
    if (!lowered.ok()) {
        return lowered.status();
    }
    return MakeArtifactFrom(std::move(*lowered), std::move(weights));
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
    auto compiled = CompileTinyLlama(/*num_layers=*/1, /*tie_word_embeddings=*/false,
                                     /*opt_level=*/1, /*enable_packed_weights=*/false);
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
    auto compiled = CompileTinyLlama(/*num_layers=*/1, /*tie_word_embeddings=*/false,
                                     /*opt_level=*/1, /*enable_packed_weights=*/false);
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
    auto compiled = CompileTinyLlama(/*num_layers=*/1, /*tie_word_embeddings=*/false,
                                     /*opt_level=*/1, /*enable_packed_weights=*/false);
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

TEST(ExecutableModel, MultiLayerLlamaBindsEveryWeightToItsOwnBacking) {
    Runtime runtime = MakeCpuRuntime();
    auto compiled = CompileTinyLlama(/*num_layers=*/2, /*tie_word_embeddings=*/false,
                                     /*opt_level=*/1, /*enable_packed_weights=*/false);
    ASSERT_TRUE(compiled.ok()) << compiled.status().ToString();
    const std::vector<const void*> expected = compiled->weight_data;
    ASSERT_EQ(expected.size(), 21U) << "3 model-level weights + 9 per layer";

    const auto model = PrepareExecutableModel(runtime, std::move(compiled->artifact));
    ASSERT_TRUE(model.ok()) << model.status().ToString();
    const auto bindings = model->immutable_weight_bindings(ExecPhase::kBoth);
    ASSERT_TRUE(bindings.ok()) << bindings.status().ToString();

    // Every checkpoint weight is bound exactly once, to its own backing: a layer
    // index mixed up during resolution would show up as a duplicated or missing
    // pointer rather than as a plausible-looking binding.
    std::vector<const void*> bound;
    for (const ExternalReadOnlyValueBinding& entry: (*bindings)->readable) {
        bound.push_back(entry.tensor.data());
    }
    ASSERT_EQ(bound.size(), expected.size());
    std::sort(bound.begin(), bound.end());
    std::vector<const void*> sorted_expected = expected;
    std::sort(sorted_expected.begin(), sorted_expected.end());
    EXPECT_EQ(bound, sorted_expected);
    EXPECT_EQ(std::unique(bound.begin(), bound.end()), bound.end())
            << "two weight values share one backing in an untied checkpoint";
}

TEST(ExecutableModel, TiedLmHeadSharesEmbeddingBacking) {
    Runtime runtime = MakeCpuRuntime();
    auto compiled = CompileTinyLlama(/*num_layers=*/1, /*tie_word_embeddings=*/true,
                                     /*opt_level=*/1, /*enable_packed_weights=*/false);
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
    auto compiled = CompileTinyLlama(/*num_layers=*/1, /*tie_word_embeddings=*/false,
                                     /*opt_level=*/1, /*enable_packed_weights=*/false);
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

TEST(ExecutableModel, OneBindingTableSpecializesBothPrefillAndDecode) {
    Runtime runtime = MakeCpuRuntime();
    auto compiled = CompileTinyLlama(/*num_layers=*/1, /*tie_word_embeddings=*/false,
                                     /*opt_level=*/1, /*enable_packed_weights=*/false);
    ASSERT_TRUE(compiled.ok()) << compiled.status().ToString();
    const auto model = PrepareExecutableModel(runtime, std::move(compiled->artifact));
    ASSERT_TRUE(model.ok()) << model.status().ToString();

    const auto plan_result = model->plan(ExecPhase::kBoth);
    ASSERT_TRUE(plan_result.ok()) << plan_result.status().ToString();
    const ExecutionPlan& plan = **plan_result;
    const auto table = model->immutable_weight_bindings(ExecPhase::kBoth);
    ASSERT_TRUE(table.ok()) << table.status().ToString();

    // token_ids and position_ids are the only model inputs, both Int64 {seq_len}.
    ASSERT_EQ(plan.model_inputs().size(), 2U);
    const ExecutionValueId tokens_id = plan.model_inputs()[0];
    const ExecutionValueId positions_id = plan.model_inputs()[1];
    ASSERT_EQ(plan.values()[tokens_id.index].spec.dtype, DataType::Int(64));
    ASSERT_EQ(plan.values()[positions_id.index].spec.dtype, DataType::Int(64));

    // The buffers must outlive the prepared bindings, which borrow their data.
    const TestBuffer prefill_tokens(DataType::Int(64), {4});
    const TestBuffer prefill_positions(DataType::Int(64), {4});
    const TestBuffer decode_tokens(DataType::Int(64), {1});
    const TestBuffer decode_positions(DataType::Int(64), {1});
    CPUAllocator allocator(Device::CPU());

    const auto specialize = [&](TensorView tokens, TensorView positions) {
        ExternalTensorBindings external = **table;
        external.readable.push_back({.value = tokens_id, .tensor = tokens});
        external.readable.push_back({.value = positions_id, .tensor = positions});
        return PrepareExecutionBindings(plan, external, allocator);
    };

    const auto prefill = specialize(prefill_tokens.view(), prefill_positions.view());
    ASSERT_TRUE(prefill.ok()) << prefill.status().ToString();
    const auto decode = specialize(decode_tokens.view(), decode_positions.view());
    ASSERT_TRUE(decode.ok()) << decode.status().ToString();

    // Both specializations reuse the same immutable weight bytes, so a session
    // never re-materializes weights when it switches phase.
    ASSERT_FALSE((*table)->readable.empty());
    const ExecutionValueId weight_id = (*table)->readable.front().value;
    const void* const weight_data = (*table)->readable.front().tensor.data();
    EXPECT_EQ(prefill->values()[weight_id.index].readable.data(), weight_data);
    EXPECT_EQ(decode->values()[weight_id.index].readable.data(), weight_data);
    EXPECT_EQ(prefill_tokens.view().data(), prefill_tokens.data());
    EXPECT_EQ(decode->values()[tokens_id.index].readable.data(), decode_tokens.data());
}

TEST(ExecutableModel, SurvivesBeingMoved) {
    Runtime runtime = MakeCpuRuntime();
    auto compiled = CompileTinyLlama(/*num_layers=*/1, /*tie_word_embeddings=*/true,
                                     /*opt_level=*/1, /*enable_packed_weights=*/false);
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

TEST(ExecutableModel, IsMoveConstructibleButNotAssignable) {
    // Owning identity: replacing a prepared model in place would invalidate
    // every plan/binding pointer already handed out, so only move construction
    // is allowed and StatusOr<ExecutableModel> relies on that alone.
    static_assert(std::is_nothrow_move_constructible_v<ExecutableModel>);
    static_assert(!std::is_move_assignable_v<ExecutableModel>);
    static_assert(!std::is_copy_constructible_v<ExecutableModel>);
    static_assert(!std::is_copy_assignable_v<ExecutableModel>);
}

TEST(ExecutableModel, RejectsArtifactWithoutLoadedModel) {
    Runtime runtime = MakeCpuRuntime();

    const auto model = PrepareExecutableModel(runtime, LoweredModelArtifact{});

    ASSERT_FALSE(model.ok());
    EXPECT_EQ(model.status().code(), StatusCode::kInvalidArgument);
}

TEST(ExecutableModel, MaterializesConstantFromInlineData) {
    const auto inline_data = MakeConstantBytes(4 * sizeof(float));
    ModelGraph graph;
    const GraphValueId input =
            graph.AddConstant(FloatSpec({4}), ConstantBinding{.inline_data = inline_data});
    const GraphValueId weight = graph.AddWeight(
            FloatSpec({4}), MakeTransformerWeightBinding(0U, TransformerWeightRole::kInputNorm));
    const auto norm = graph.AddNode(
            OpType::kRmsNorm, 0U, {input, weight},
            {NodeOutputDesc{.payload = ActivationValue{}}}, RmsNormParams{.eps = kEpsilon});
    ASSERT_TRUE(norm.ok()) << norm.status().ToString();
    graph.MarkOutput(norm->outputs[0]);

    RawWeightCarver carver;
    ResolvedModelWeights weights;
    weights.layers.resize(1);
    weights.layers[0].norm.input_rmsnorm = carver.Carve({4});
    const void* const weight_data = weights.layers[0].norm.input_rmsnorm.data;

    Runtime runtime = MakeCpuRuntime();
    auto artifact = MakeArtifact(graph, std::move(weights),
                                 GraphLoweringConfig{.enable_packed_weights = false});
    ASSERT_TRUE(artifact.ok()) << artifact.status().ToString();

    const auto model = PrepareExecutableModel(runtime, std::move(*artifact));
    ASSERT_TRUE(model.ok()) << model.status().ToString();
    const auto bindings = model->immutable_weight_bindings(ExecPhase::kBoth);
    ASSERT_TRUE(bindings.ok()) << bindings.status().ToString();

    // Execution never reads ConstantBinding::inline_data itself, so preparation
    // must turn those payload bytes into a bound view.
    ASSERT_EQ((*bindings)->readable.size(), 2U);
    bool found_constant = false;
    bool found_weight = false;
    for (const ExternalReadOnlyValueBinding& entry: (*bindings)->readable) {
        if (entry.tensor.data() == inline_data->data()) {
            found_constant = true;
            EXPECT_EQ(entry.tensor.dtype(), DataType::Float32());
            EXPECT_EQ(Copy(entry.tensor.shape()), (std::vector<int64_t>{4}));
            EXPECT_EQ(Copy(entry.tensor.strides()), (std::vector<int64_t>{1}));
        } else if (entry.tensor.data() == weight_data) {
            found_weight = true;
        }
    }
    EXPECT_TRUE(found_constant) << "constant value was not materialized";
    EXPECT_TRUE(found_weight);
}

TEST(ExecutableModel, RejectsConstantWithoutInlineData) {
    ModelGraph graph;
    const GraphValueId input = graph.AddConstant(FloatSpec({4}), ConstantBinding{});
    const GraphValueId weight = graph.AddWeight(
            FloatSpec({4}), MakeTransformerWeightBinding(0U, TransformerWeightRole::kInputNorm));
    const auto norm = graph.AddNode(
            OpType::kRmsNorm, 0U, {input, weight},
            {NodeOutputDesc{.payload = ActivationValue{}}}, RmsNormParams{.eps = kEpsilon});
    ASSERT_TRUE(norm.ok()) << norm.status().ToString();
    graph.MarkOutput(norm->outputs[0]);

    RawWeightCarver carver;
    ResolvedModelWeights weights;
    weights.layers.resize(1);
    weights.layers[0].norm.input_rmsnorm = carver.Carve({4});

    Runtime runtime = MakeCpuRuntime();
    auto artifact = MakeArtifact(graph, std::move(weights),
                                 GraphLoweringConfig{.enable_packed_weights = false});
    ASSERT_TRUE(artifact.ok()) << artifact.status().ToString();

    const auto model = PrepareExecutableModel(runtime, std::move(*artifact));

    ASSERT_FALSE(model.ok());
    EXPECT_EQ(model.status().code(), StatusCode::kFailedPrecondition);
}

TEST(ExecutableModel, RejectsConstantWhoseInlineSizeDisagreesWithShape) {
    // Three floats behind a four-element shape: binding it would let a kernel
    // read past the payload.
    const auto inline_data = MakeConstantBytes(3 * sizeof(float));
    ModelGraph graph;
    const GraphValueId input =
            graph.AddConstant(FloatSpec({4}), ConstantBinding{.inline_data = inline_data});
    const GraphValueId weight = graph.AddWeight(
            FloatSpec({4}), MakeTransformerWeightBinding(0U, TransformerWeightRole::kInputNorm));
    const auto norm = graph.AddNode(
            OpType::kRmsNorm, 0U, {input, weight},
            {NodeOutputDesc{.payload = ActivationValue{}}}, RmsNormParams{.eps = kEpsilon});
    ASSERT_TRUE(norm.ok()) << norm.status().ToString();
    graph.MarkOutput(norm->outputs[0]);

    RawWeightCarver carver;
    ResolvedModelWeights weights;
    weights.layers.resize(1);
    weights.layers[0].norm.input_rmsnorm = carver.Carve({4});

    Runtime runtime = MakeCpuRuntime();
    auto artifact = MakeArtifact(graph, std::move(weights),
                                 GraphLoweringConfig{.enable_packed_weights = false});
    ASSERT_TRUE(artifact.ok()) << artifact.status().ToString();

    const auto model = PrepareExecutableModel(runtime, std::move(*artifact));

    ASSERT_FALSE(model.ok());
    EXPECT_EQ(model.status().code(), StatusCode::kFailedPrecondition);
}

TEST(ExecutableModel, PackedSubgraphKeepsWeightOutOfBindingTable) {
    // A full packed Llama cannot resolve yet (see the gap test below), so the
    // packed contract is proven on the smallest artifact the registry supports:
    // one fused AddRmsNorm whose weight is served by a packed artifact.
    const auto input_bytes = MakeConstantBytes(6 * sizeof(float));
    const auto residual_bytes = MakeConstantBytes(6 * sizeof(float));
    ModelGraph graph;
    const TensorSpec act_spec = FloatSpec({2, 3});
    const GraphValueId input =
            graph.AddConstant(act_spec, ConstantBinding{.inline_data = input_bytes});
    const GraphValueId residual =
            graph.AddConstant(act_spec, ConstantBinding{.inline_data = residual_bytes});
    const GraphValueId weight = graph.AddWeight(
            FloatSpec({3}), MakeTransformerWeightBinding(0U, TransformerWeightRole::kInputNorm));
    const auto fused = graph.AddNode(
            OpType::kAddRmsNorm, 0U, {input, residual, weight},
            {NodeOutputDesc{.payload = ActivationValue{}},
             NodeOutputDesc{.payload = ActivationValue{}}},
            AddRmsNormParams{.eps = kEpsilon});
    ASSERT_TRUE(fused.ok()) << fused.status().ToString();
    graph.MarkOutput(fused->outputs[0]);
    graph.MarkOutput(fused->outputs[1]);

    RawWeightCarver carver;
    ResolvedModelWeights weights;
    weights.layers.resize(1);
    weights.layers[0].norm.input_rmsnorm = carver.Carve({3});
    const void* const weight_data = weights.layers[0].norm.input_rmsnorm.data;

    Runtime runtime = MakeCpuRuntime();
    auto artifact = MakeArtifact(graph, std::move(weights),
                                 GraphLoweringConfig{.enable_packed_weights = true});
    ASSERT_TRUE(artifact.ok()) << artifact.status().ToString();

    const auto model = PrepareExecutableModel(runtime, std::move(*artifact));
    ASSERT_TRUE(model.ok()) << model.status().ToString();

    const auto plan = model->plan(ExecPhase::kBoth);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ((*plan)->size(), 1U);
    const ExecutionStep& step = (*plan)->steps().front();
    EXPECT_NE(step.packed_weights, nullptr);
    EXPECT_EQ(step.inputs.size(), 3U);
    EXPECT_EQ(step.kernel_input_ports.size(), 2U);

    // Only the two constants are bound; the packed weight is served by the
    // artifact and must not also appear as an external binding.
    const auto bindings = model->immutable_weight_bindings(ExecPhase::kBoth);
    ASSERT_TRUE(bindings.ok()) << bindings.status().ToString();
    ASSERT_EQ((*bindings)->readable.size(), 2U);
    EXPECT_TRUE(BindingsSharingData(**bindings, weight_data).empty());
    EXPECT_EQ(BindingsSharingData(**bindings, input_bytes->data()).size(), 1U);
    EXPECT_EQ(BindingsSharingData(**bindings, residual_bytes->data()).size(), 1U);
}

TEST(ExecutableModel, PackedLoweringPreparesAllWeightConsumers) {
    // Packed enablement is graph-wide, so every current Llama weight consumer
    // must resolve and receive an artifact before model preparation succeeds.
    Runtime runtime = MakeCpuRuntime();
    auto compiled = CompileTinyLlama(/*num_layers=*/1, /*tie_word_embeddings=*/false,
                                     /*opt_level=*/2, /*enable_packed_weights=*/true);
    ASSERT_TRUE(compiled.ok()) << compiled.status().ToString();

    const auto model = PrepareExecutableModel(runtime, std::move(compiled->artifact));
    ASSERT_TRUE(model.ok()) << model.status().ToString();
    const auto plan = model->plan(ExecPhase::kBoth);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();

    size_t packed_steps = 0;
    for (const ExecutionStep& step: (*plan)->steps()) {
        if (step.selector.weight_format != WeightFormat::kPacked) {
            continue;
        }
        ++packed_steps;
        ASSERT_NE(step.packed_weights, nullptr);
        EXPECT_EQ(step.packed_weights->recipe(),
                  step.kernel.expected_packing_recipe);
    }
    EXPECT_GT(packed_steps, 0U);
}

TEST(ExecutableModel, RejectsWeightWithNoDenseStorageRole) {
    ModelGraph graph;
    const GraphValueId input = graph.AddConstant(
            FloatSpec({4}), ConstantBinding{.inline_data = MakeConstantBytes(4 * sizeof(float))});
    const GraphValueId weight = graph.AddWeight(
            FloatSpec({4, 4}),
            MakeTransformerWeightBinding(0U, TransformerWeightRole::kMoERouter));
    const auto linear = graph.AddNode(
            OpType::kLinear, 0U, {input, weight},
            {NodeOutputDesc{.payload = ActivationValue{}}}, LinearParams{});
    ASSERT_TRUE(linear.ok()) << linear.status().ToString();
    graph.MarkOutput(linear->outputs[0]);

    ResolvedModelWeights weights;
    weights.layers.resize(1);

    Runtime runtime = MakeCpuRuntime();
    auto artifact = MakeArtifact(graph, std::move(weights),
                                 GraphLoweringConfig{.enable_packed_weights = false});
    ASSERT_TRUE(artifact.ok()) << artifact.status().ToString();

    // Dense checkpoints carry no router weight, so the binding resolves to
    // nothing. Preparation must reject it with the role named instead of
    // leaving the weight silently unbound.
    const auto model = PrepareExecutableModel(runtime, std::move(*artifact));

    ASSERT_FALSE(model.ok());
    EXPECT_EQ(model.status().code(), StatusCode::kFailedPrecondition);
    const std::string message = model.status().message();
    EXPECT_NE(message.find("value " + std::to_string(weight.index)), std::string::npos)
            << message;
    EXPECT_NE(message.find("MoERouter"), std::string::npos) << message;
}

TEST(ExecutableModel, RejectsWeightWhoseLayerIndexHasNoStorage) {
    ModelGraph graph;
    const GraphValueId input = graph.AddConstant(
            FloatSpec({4}), ConstantBinding{.inline_data = MakeConstantBytes(4 * sizeof(float))});
    const GraphValueId weight = graph.AddWeight(
            FloatSpec({4, 4}),
            MakeTransformerWeightBinding(1U, TransformerWeightRole::kAttentionQ));
    const auto linear = graph.AddNode(
            OpType::kLinear, 1U, {input, weight},
            {NodeOutputDesc{.payload = ActivationValue{}}}, LinearParams{});
    ASSERT_TRUE(linear.ok()) << linear.status().ToString();
    graph.MarkOutput(linear->outputs[0]);

    // One layer in the checkpoint while the graph addresses layer 1: the
    // binding is structurally valid yet resolves to nothing.
    ResolvedModelWeights weights;
    weights.layers.resize(1);

    Runtime runtime = MakeCpuRuntime();
    auto artifact = MakeArtifact(graph, std::move(weights),
                                 GraphLoweringConfig{.enable_packed_weights = false});
    ASSERT_TRUE(artifact.ok()) << artifact.status().ToString();

    const auto model = PrepareExecutableModel(runtime, std::move(*artifact));

    ASSERT_FALSE(model.ok());
    EXPECT_EQ(model.status().code(), StatusCode::kFailedPrecondition);
    const std::string message = model.status().message();
    EXPECT_NE(message.find("value " + std::to_string(weight.index)), std::string::npos)
            << message;
    EXPECT_NE(message.find("AttentionQ"), std::string::npos) << message;
    EXPECT_NE(message.find("layer=1"), std::string::npos) << message;
}

TEST(ExecutableModel, PhaseSpecificArtifactRejectsUnmatchedPhaseQueries) {
    ModelGraph graph;
    const GraphValueId input = graph.AddConstant(
            FloatSpec({4}), ConstantBinding{.inline_data = MakeConstantBytes(4 * sizeof(float))});
    const GraphValueId weight = graph.AddWeight(
            FloatSpec({4}), MakeTransformerWeightBinding(0U, TransformerWeightRole::kInputNorm));
    const auto norm = graph.AddNode(
            OpType::kRmsNorm, 0U, {input, weight},
            {NodeOutputDesc{.payload = ActivationValue{}}}, RmsNormParams{.eps = kEpsilon});
    ASSERT_TRUE(norm.ok()) << norm.status().ToString();
    graph.MarkOutput(norm->outputs[0]);

    RawWeightCarver carver;
    ResolvedModelWeights weights;
    weights.layers.resize(1);
    weights.layers[0].norm.input_rmsnorm = carver.Carve({4});

    GraphLoweringConfig config{.enable_packed_weights = false};
    config.selector.phase = ExecPhase::kPrefill;

    Runtime runtime = MakeCpuRuntime();
    auto artifact = MakeArtifact(graph, std::move(weights), config);
    ASSERT_TRUE(artifact.ok()) << artifact.status().ToString();

    const auto model = PrepareExecutableModel(runtime, std::move(*artifact));
    ASSERT_TRUE(model.ok()) << model.status().ToString();
    EXPECT_EQ(model->phase(), ExecPhase::kPrefill);
    EXPECT_TRUE(model->plan(ExecPhase::kPrefill).ok());

    // A prefill-only artifact must not answer kDecode, and a kBoth query claims
    // coverage the artifact does not have; neither may silently reuse the plan.
    for (const ExecPhase mismatched: {ExecPhase::kDecode, ExecPhase::kBoth}) {
        const auto plan = model->plan(mismatched);
        ASSERT_FALSE(plan.ok()) << ToString(mismatched);
        EXPECT_EQ(plan.status().code(), StatusCode::kFailedPrecondition);
        const auto bindings = model->immutable_weight_bindings(mismatched);
        ASSERT_FALSE(bindings.ok()) << ToString(mismatched);
        EXPECT_EQ(bindings.status().code(), StatusCode::kFailedPrecondition);
    }
}

TEST(ExecutableModel, RejectsArtifactWhoseStepsMixPhases) {
    // Lowering stamps one selector onto every step, so no lowering can produce
    // a mixed-phase artifact today. The Builder test seam reproduces the
    // type-level shape, and preparation must reject it: silently accepting it
    // would make the shared-plan contract undecidable.
    ModelGraph graph;
    const GraphValueId input = graph.AddConstant(
            FloatSpec({4}), ConstantBinding{.inline_data = MakeConstantBytes(4 * sizeof(float))});
    const GraphValueId first_weight = graph.AddWeight(
            FloatSpec({4}), MakeTransformerWeightBinding(0U, TransformerWeightRole::kInputNorm));
    const auto first = graph.AddNode(
            OpType::kRmsNorm, 0U, {input, first_weight},
            {NodeOutputDesc{.payload = ActivationValue{}}}, RmsNormParams{.eps = kEpsilon});
    ASSERT_TRUE(first.ok()) << first.status().ToString();
    const GraphValueId second_weight = graph.AddWeight(
            FloatSpec({4}),
            MakeTransformerWeightBinding(0U, TransformerWeightRole::kPostAttentionNorm));
    const auto second = graph.AddNode(
            OpType::kRmsNorm, 0U, {first->outputs[0], second_weight},
            {NodeOutputDesc{.payload = ActivationValue{}}}, RmsNormParams{.eps = kEpsilon});
    ASSERT_TRUE(second.ok()) << second.status().ToString();
    graph.MarkOutput(second->outputs[0]);

    const auto lowered = LowerModelGraph(graph);
    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();
    ASSERT_EQ(lowered->steps().size(), 2U);

    LoweredGraph::Builder builder;
    builder.steps.assign(lowered->steps().begin(), lowered->steps().end());
    builder.values.assign(lowered->values().begin(), lowered->values().end());
    builder.model_inputs.assign(lowered->model_inputs().begin(), lowered->model_inputs().end());
    builder.model_outputs.assign(lowered->model_outputs().begin(),
                                 lowered->model_outputs().end());
    builder.state_aliases.assign(lowered->state_aliases().begin(),
                                 lowered->state_aliases().end());
    builder.steps[1].spec.selector.phase = ExecPhase::kPrefill;
    auto mixed = std::move(builder).Build();
    ASSERT_TRUE(mixed.ok()) << mixed.status().ToString();

    RawWeightCarver carver;
    ResolvedModelWeights weights;
    weights.layers.resize(1);
    weights.layers[0].norm.input_rmsnorm = carver.Carve({4});
    weights.layers[0].norm.post_attn_rmsnorm = carver.Carve({4});

    Runtime runtime = MakeCpuRuntime();
    const auto model = PrepareExecutableModel(
            runtime, MakeArtifactFrom(std::move(*mixed), std::move(weights)));

    ASSERT_FALSE(model.ok());
    EXPECT_EQ(model.status().code(), StatusCode::kFailedPrecondition);
    EXPECT_NE(model.status().message().find("mixes execution phases"), std::string::npos)
            << model.status().message();
}

TEST(ExecutableModel, PreparedBindingsAreReleasedBeforeTheModel) {
    // Teardown contract: prepared bindings borrow weight data owned by the
    // model, so they are released first and the model must outlive them. Under
    // ASAN/TSAN this pins the ownership order; the assertions after the scope
    // also prove the model holds no back-reference to session state.
    Runtime runtime = MakeCpuRuntime();
    auto compiled = CompileTinyLlama(/*num_layers=*/1, /*tie_word_embeddings=*/false,
                                     /*opt_level=*/1, /*enable_packed_weights=*/false);
    ASSERT_TRUE(compiled.ok()) << compiled.status().ToString();
    auto model = PrepareExecutableModel(runtime, std::move(compiled->artifact));
    ASSERT_TRUE(model.ok()) << model.status().ToString();

    const void* weight_data = nullptr;
    {
        const auto plan = model->plan(ExecPhase::kBoth);
        ASSERT_TRUE(plan.ok()) << plan.status().ToString();
        const auto table = model->immutable_weight_bindings(ExecPhase::kBoth);
        ASSERT_TRUE(table.ok()) << table.status().ToString();
        ASSERT_FALSE((*table)->readable.empty());
        const ExecutionValueId weight_value = (*table)->readable.front().value;
        weight_data = (*table)->readable.front().tensor.data();

        const TestBuffer tokens(DataType::Int(64), {1});
        const TestBuffer positions(DataType::Int(64), {1});
        ExternalTensorBindings external = **table;
        external.readable.push_back(
                {.value = (*plan)->model_inputs()[0], .tensor = tokens.view()});
        external.readable.push_back(
                {.value = (*plan)->model_inputs()[1], .tensor = positions.view()});

        CPUAllocator allocator(Device::CPU());
        const auto prepared = PrepareExecutionBindings(**plan, external, allocator);
        ASSERT_TRUE(prepared.ok()) << prepared.status().ToString();
        EXPECT_EQ(prepared->values()[weight_value.index].readable.data(), weight_data);
    }

    const auto table = model->immutable_weight_bindings(ExecPhase::kBoth);
    ASSERT_TRUE(table.ok()) << table.status().ToString();
    ASSERT_FALSE((*table)->readable.empty());
    EXPECT_EQ((*table)->readable.front().tensor.data(), weight_data);
}

} // namespace
