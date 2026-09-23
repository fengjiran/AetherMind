#include "aethermind/execution/execution_bindings.h"

#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/base/device.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/compiler/graph_lowering.h"
#include "aethermind/compiler/packing_request_builder.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/graph/graph.h"
#include "aethermind/memory/cpu_allocator.h"
#include "aethermind/model/packed_weight_store.h"
#include "aethermind/model/resolved_model_weights.h"
#include "aethermind/model/weight_prepack_planner.h"
#include "aethermind/operators/op_params.h"
#include "aethermind/operators/op_type.h"
#include "aethermind/operators/ops/embedding_op.h"
#include "aethermind/runtime/runtime_builder.h"
#include "aethermind/shape_inference/tensor_spec.h"
#include "execution/test_tensor_buffer_helpers.h"

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

namespace {

using namespace aethermind;
using aethermind::test::TestBuffer;

constexpr float kEpsilon = 1.0e-5F;

SymbolicShape StaticShape(std::initializer_list<int64_t> dimensions) {
    const std::vector<int64_t> copied(dimensions);
    return SymbolicShape(IntArrayView{copied});
}

TensorSpec FloatSpec(std::initializer_list<int64_t> dimensions) {
    return TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape(dimensions)};
}

class FloatRawStorage final : public RawStorage {
public:
    std::vector<float> values{};
};

RawWeightView MakeRawWeight(const std::shared_ptr<FloatRawStorage>& storage,
                            std::initializer_list<int64_t> shape) {
    return RawWeightView{
            .data = reinterpret_cast<const std::byte*>(storage->values.data()),
            .bytes = storage->values.size() * sizeof(float),
            .dtype = DataType::Float32(),
            .shape = std::vector<int64_t>(shape),
            .storage = storage,
            .is_contiguous = true,
    };
}

Runtime MakeCpuRuntime() {
    RuntimeBuilder builder;
    builder.RegisterBackendFactory(DeviceType::kCPU, std::make_unique<CpuBackendFactory>());
    return builder.Build();
}

std::vector<uint32_t> ValueIndicesOfKind(const ExecutionPlan& plan, ExecutionValueKind kind) {
    std::vector<uint32_t> indices;
    for (uint32_t i = 0; i < plan.values().size(); ++i) {
        if (plan.values()[i].kind == kind) {
            indices.push_back(i);
        }
    }
    return indices;
}

size_t CountStepsConsuming(const ExecutionPlan& plan, ExecutionValueId value) {
    size_t count = 0;
    for (const ExecutionStep& step: plan.steps()) {
        for (const uint32_t port: step.kernel_input_ports) {
            if (step.inputs[port] == value) {
                ++count;
            }
        }
    }
    return count;
}

size_t CountRequired(const std::vector<bool>& required) {
    size_t count = 0;
    for (const bool flag: required) {
        if (flag) {
            ++count;
        }
    }
    return count;
}

/// Lowered-graph value ids reused as execution value ids: lowering copies values
/// in order, so the two index spaces coincide.
ExecutionValueId AsExecutionValue(GraphValueId value) {
    return ExecutionValueId{.index = value.index};
}

/// Embedding + Add + two RMSNorm steps that share one norm weight, giving one
/// model input, one constant and two plain weights.
struct PlainFixture {
    ExecutionPlan plan;
    ExecutionValueId tokens{};
    ExecutionValueId table{};
    ExecutionValueId bias{};
    ExecutionValueId norm_weight{};
};

/// `runtime` must outlive the returned plan: resolved kernels borrow the backend
/// that `runtime` owns.
StatusOr<PlainFixture> MakeSharedWeightPlainPlan(Runtime& runtime) {
    ModelGraph graph;
    const GraphValueId tokens = graph.AddInput(
            TensorSpec{.dtype = DataType::Int(64), .shape = StaticShape({1})});
    const GraphValueId table = graph.AddWeight(
            FloatSpec({32, 8}),
            MakeTransformerWeightBinding(std::nullopt, TransformerWeightRole::kTokenEmbedding));
    const auto embedding = graph.AddNode(
            OpType::kEmbedding, std::nullopt, {tokens, table},
            {NodeOutputDesc{.payload = ActivationValue{}}}, EmbeddingParams{});
    if (!embedding.ok()) return embedding.status();
    const GraphValueId bias = graph.AddConstant(FloatSpec({8}), ConstantBinding{});
    const auto shifted = graph.AddNode(
            OpType::kAdd, 0U, {embedding->outputs[0], bias},
            {NodeOutputDesc{.payload = ActivationValue{}}}, AddParams{});
    if (!shifted.ok()) return shifted.status();
    const GraphValueId norm_weight = graph.AddWeight(
            FloatSpec({8}), MakeTransformerWeightBinding(0U, TransformerWeightRole::kInputNorm));
    const auto first = graph.AddNode(
            OpType::kRmsNorm, 0U, {shifted->outputs[0], norm_weight},
            {NodeOutputDesc{.payload = ActivationValue{}}}, RmsNormParams{.eps = kEpsilon});
    if (!first.ok()) return first.status();
    const auto second = graph.AddNode(
            OpType::kRmsNorm, 0U, {first->outputs[0], norm_weight},
            {NodeOutputDesc{.payload = ActivationValue{}}}, RmsNormParams{.eps = kEpsilon});
    if (!second.ok()) return second.status();
    graph.MarkOutput(second->outputs[0]);

    const auto lowered = LowerModelGraph(graph);
    if (!lowered.ok()) return lowered.status();
    auto plan = ExecutionPlanBuilder::Build(runtime, *lowered);
    if (!plan.ok()) return plan.status();
    return PlainFixture{.plan = std::move(*plan),
                        .tokens = AsExecutionValue(tokens),
                        .table = AsExecutionValue(table),
                        .bias = AsExecutionValue(bias),
                        .norm_weight = AsExecutionValue(norm_weight)};
}

/// One fused AddRmsNorm step lowered with packed weights, so its kWeight port is
/// projected out of the kernel-facing inputs.
struct PackedFixture {
    ExecutionPlan plan;
    ExecutionValueId input{};
    ExecutionValueId residual{};
    ExecutionValueId weight{};
};

/// `runtime` must outlive the returned plan. The packed store need not: plan
/// steps hold their own reference to each packed artifact.
StatusOr<PackedFixture> MakePackedAddRmsNormPlan(Runtime& runtime) {
    ModelGraph graph;
    const TensorSpec act_spec = FloatSpec({2, 3});
    const GraphValueId input = graph.AddConstant(act_spec, ConstantBinding{});
    const GraphValueId residual = graph.AddConstant(act_spec, ConstantBinding{});
    const GraphValueId weight = graph.AddWeight(
            FloatSpec({3}), MakeTransformerWeightBinding(0U, TransformerWeightRole::kInputNorm));
    const auto fused = graph.AddNode(
            OpType::kAddRmsNorm, 0U, {input, residual, weight},
            {NodeOutputDesc{.payload = ActivationValue{}},
             NodeOutputDesc{.payload = ActivationValue{}}},
            AddRmsNormParams{.eps = kEpsilon});
    if (!fused.ok()) return fused.status();
    graph.MarkOutput(fused->outputs[0]);
    graph.MarkOutput(fused->outputs[1]);

    const auto lowered =
            LowerModelGraph(graph, GraphLoweringConfig{.enable_packed_weights = true});
    if (!lowered.ok()) return lowered.status();

    auto storage = std::make_shared<FloatRawStorage>();
    storage->values = {1.0F, 0.5F, -1.5F};
    ResolvedModelWeights resolved;
    resolved.layers.resize(1);
    resolved.layers[0].norm.input_rmsnorm = MakeRawWeight(storage, {3});

    const auto requests = BuildWeightPackingRequests(*lowered, resolved);
    if (!requests.ok()) return requests.status();
    PackedWeightStore packed_store;
    const Status stored = WeightPrepackPlanner::PrepackAndStore(packed_store, *requests);
    if (!stored.ok()) return stored;
    auto plan = ExecutionPlanBuilder::Build(runtime, packed_store, *lowered);
    if (!plan.ok()) return plan.status();
    return PackedFixture{.plan = std::move(*plan),
                         .input = AsExecutionValue(input),
                         .residual = AsExecutionValue(residual),
                         .weight = AsExecutionValue(weight)};
}

TEST(ExternalReadRequirements, PlainGraphRequiresInputsConstantsAndWeights) {
    Runtime runtime = MakeCpuRuntime();
    const auto fixture = MakeSharedWeightPlainPlan(runtime);
    ASSERT_TRUE(fixture.ok()) << fixture.status().ToString();
    const ExecutionPlan& plan = fixture->plan;

    const auto required = ComputeExternalReadRequirements(plan);
    ASSERT_TRUE(required.ok()) << required.status().ToString();
    ASSERT_EQ(required->size(), plan.values().size());

    EXPECT_TRUE((*required)[fixture->tokens.index]);
    EXPECT_TRUE((*required)[fixture->table.index]);
    EXPECT_TRUE((*required)[fixture->bias.index]);
    EXPECT_TRUE((*required)[fixture->norm_weight.index]);

    for (const uint32_t activation: ValueIndicesOfKind(plan, ExecutionValueKind::kActivation)) {
        EXPECT_FALSE((*required)[activation]) << "activation " << activation;
    }
    EXPECT_EQ(CountRequired(*required), 4U);
}

TEST(ExternalReadRequirements, WeightSharedByTwoStepsIsRequiredOnce) {
    Runtime runtime = MakeCpuRuntime();
    const auto fixture = MakeSharedWeightPlainPlan(runtime);
    ASSERT_TRUE(fixture.ok()) << fixture.status().ToString();
    const ExecutionPlan& plan = fixture->plan;

    ASSERT_EQ(CountStepsConsuming(plan, fixture->norm_weight), 2U);
    ASSERT_EQ(CountStepsConsuming(plan, fixture->table), 1U);

    const auto required = ComputeExternalReadRequirements(plan);
    ASSERT_TRUE(required.ok()) << required.status().ToString();

    // The result is indexed by value, so a weight consumed by two steps still
    // contributes exactly one entry: two model-visible weights plus the model
    // input and the constant, and nothing else.
    EXPECT_TRUE((*required)[fixture->norm_weight.index]);
    EXPECT_EQ(ValueIndicesOfKind(plan, ExecutionValueKind::kWeight).size(), 2U);
    EXPECT_EQ(CountRequired(*required), 4U);
}

TEST(ExternalReadRequirements, PackedWeightIsNotRequired) {
    Runtime runtime = MakeCpuRuntime();
    const auto fixture = MakePackedAddRmsNormPlan(runtime);
    ASSERT_TRUE(fixture.ok()) << fixture.status().ToString();
    const ExecutionPlan& plan = fixture->plan;
    ASSERT_EQ(plan.size(), 1U);

    // The fused step keeps three semantic inputs but exposes only the two
    // activations to the kernel, which is what makes the weight packed.
    const ExecutionStep& step = plan.steps().front();
    ASSERT_EQ(step.inputs.size(), 3U);
    ASSERT_EQ(step.kernel_input_ports.size(), 2U);
    ASSERT_NE(step.packed_weights, nullptr);
    ASSERT_EQ(CountStepsConsuming(plan, fixture->weight), 0U);

    const auto required = ComputeExternalReadRequirements(plan);
    ASSERT_TRUE(required.ok()) << required.status().ToString();

    EXPECT_FALSE((*required)[fixture->weight.index])
            << "packed weight is served by the plan's packed artifact";
    EXPECT_TRUE((*required)[fixture->input.index]);
    EXPECT_TRUE((*required)[fixture->residual.index]);
    for (const uint32_t activation: ValueIndicesOfKind(plan, ExecutionValueKind::kActivation)) {
        EXPECT_FALSE((*required)[activation]) << "activation " << activation;
    }
    EXPECT_EQ(CountRequired(*required), 2U);
}

TEST(ExternalReadRequirements, QueryAgreesWithPrepareExecutionBindings) {
    Runtime runtime = MakeCpuRuntime();
    const auto fixture = MakeSharedWeightPlainPlan(runtime);
    ASSERT_TRUE(fixture.ok()) << fixture.status().ToString();
    const ExecutionPlan& plan = fixture->plan;

    const auto required = ComputeExternalReadRequirements(plan);
    ASSERT_TRUE(required.ok()) << required.status().ToString();

    const TestBuffer tokens(DataType::Int(64), {1});
    const TestBuffer table(DataType::Float32(), {32, 8});
    const TestBuffer activations(DataType::Float32(), {8});
    const std::vector<std::pair<ExecutionValueId, const TestBuffer*>> sources{
            {fixture->tokens, &tokens},
            {fixture->table, &table},
            {fixture->bias, &activations},
            {fixture->norm_weight, &activations},
    };

    // Bind exactly the reported set, keyed by value id rather than by position.
    ExternalTensorBindings external;
    for (uint32_t i = 0; i < required->size(); ++i) {
        if (!(*required)[i]) {
            continue;
        }
        const TestBuffer* source = nullptr;
        for (const auto& [value, buffer]: sources) {
            if (value.index == i) {
                source = buffer;
            }
        }
        ASSERT_NE(source, nullptr) << "required value " << i << " has no test buffer";
        external.readable.push_back({.value = {.index = i}, .tensor = source->view()});
    }
    ASSERT_EQ(external.readable.size(), 4U);

    CPUAllocator allocator(Device::CPU());
    const auto complete = PrepareExecutionBindings(plan, external, allocator);
    ASSERT_TRUE(complete.ok()) << complete.status().ToString();

    // Dropping any single reported value must be rejected, which is what makes
    // the query and the prepare-time completeness check one authority.
    for (size_t dropped = 0; dropped < external.readable.size(); ++dropped) {
        ExternalTensorBindings incomplete = external;
        incomplete.readable.erase(incomplete.readable.begin() +
                                  static_cast<ptrdiff_t>(dropped));
        const auto rejected = PrepareExecutionBindings(plan, incomplete, allocator);
        ASSERT_FALSE(rejected.ok()) << "dropping entry " << dropped << " was accepted";
        EXPECT_EQ(rejected.status().code(), StatusCode::kFailedPrecondition);
    }
}

} // namespace
