#include "aethermind/model/weight/weight_packing.h"

#include "aethermind/backend/backend.h"
#include "aethermind/backend/backend_factory.h"
#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/cpu/cpu_weight_prepacker.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/packed_weights.h"
#include "aethermind/base/device.h"
#include "aethermind/base/kernel_selector.h"
#include "aethermind/base/status.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/compiler/graph_lowering.h"
#include "aethermind/compiler/packing_request_builder.h"
#include "aethermind/execution/execution_bindings.h"
#include "aethermind/execution/execution_context.h"
#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/execution/executor.h"
#include "aethermind/graph/graph.h"
#include "aethermind/graph/graph_types.h"
#include "aethermind/memory/buffer.h"
#include "aethermind/model/resolved_model_weights.h"
#include "aethermind/operators/ops/embedding_op.h"
#include "aethermind/operators/ops/rmsnorm_op.h"
#include "aethermind/runtime/runtime_builder.h"
#include "aethermind/shape_inference/tensor_spec.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace {

using namespace aethermind;

struct TestStorage : RawStorage {
    explicit TestStorage(size_t nbytes) : data(nbytes) {}
    std::vector<std::byte> data;
};

SymbolicShape StaticShape(std::initializer_list<int64_t> dims) {
    return SymbolicShape(IntArrayView{std::vector<int64_t>(dims)});
}

RawWeightView MakeWeightView(const std::shared_ptr<TestStorage>& storage,
                             size_t offset,
                             size_t nbytes,
                             DataType dtype,
                             const std::vector<int64_t>& shape) {
    return RawWeightView{
            .data = storage->data.data() + offset,
            .bytes = nbytes,
            .dtype = dtype,
            .shape = shape,
            .storage = storage,
    };
}

KernelSelector MakeExpectedSelector() {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
            .phase = ExecPhase::kBoth,
    };
}

void SetIdentityPackingRecipes(std::vector<WeightPackingRequest>& requests) {
    for (WeightPackingRequest& request: requests) {
        request.recipe = CpuIdentityPackingRecipe();
    }
}

// Packs through the real CPU identity prepacker so model-level prepack tests
// exercise the Backend::PackWeights contract end to end.
StatusOr<std::unique_ptr<PackedWeights>> PackViaCpuIdentity(
        OpType op_type,
        std::span<const TensorView> components,
        const KernelSelector& selector) {
    CpuWeightPrepacker prepacker;
    return prepacker.Pack(op_type, components, selector);
}

// Minimal backend for prepack-only tests: resolves nothing, packs everything
// through the CPU identity prepacker.
class PackingOnlyTestBackend final : public Backend {
public:
    DeviceType device_type() const noexcept override {
        return DeviceType::kCPU;
    }

    StatusOr<ResolvedKernel> PrepareKernel(
            OpType,
            const KernelSelector&,
            const OpParams&) const override {
        return Status::NotFound("PackingOnlyTestBackend only packs weights");
    }

    StatusOr<std::unique_ptr<PackedWeights>> PackWeights(
            OpType op_type,
            std::span<const TensorView> components,
            const KernelSelector& selector) const override {
        return PackViaCpuIdentity(op_type, components, selector);
    }

    const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override {
        return nullptr;
    }
};

// Backend that relies on the default Backend::PackWeights implementation, so
// PrepackWeightRequests must fail loudly with Unimplemented.
class NoPackingTestBackend final : public Backend {
public:
    DeviceType device_type() const noexcept override {
        return DeviceType::kCPU;
    }

    StatusOr<ResolvedKernel> PrepareKernel(
            OpType,
            const KernelSelector&,
            const OpParams&) const override {
        return Status::NotFound("NoPackingTestBackend packs nothing");
    }

    const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override {
        return nullptr;
    }
};

// The default Backend::PackWeights returns Unimplemented; model-level prepack
// must surface that loudly instead of falling back to a concrete CPU packer.
TEST(WeightPacking, PrepackWeightRequestsRejectsBackendWithoutPacking) {
    auto storage = std::make_shared<TestStorage>(64);
    for (auto& b: storage->data) b = std::byte{0};

    const WeightPackingRequest request{
            .op_type = OpType::kLinear,
            .source_id = 1,
            .binding = MakeTransformerWeightBinding(0U, TransformerWeightRole::kAttentionQ),
            .raw_weight = MakeWeightView(storage, 0, 8, DataType::Float32(), {2, 1}),
            .selector = MakeExpectedSelector(),
            .recipe = CpuIdentityPackingRecipe(),
    };

    NoPackingTestBackend backend;
    PackedWeightStore store;
    const Status status = PrepackWeightRequests(backend, store, {request});
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kUnimplemented);
}

TEST(WeightPacking, PrepackWeightRequestsMakesWeightsFindable) {
    auto storage = std::make_shared<TestStorage>(256);
    // Fill with zeros so Pack can safely memcpy.
    for (auto& b: storage->data) b = std::byte{0};

    const std::vector<WeightPackingRequest> requests{
            {.op_type = OpType::kLinear,
             .binding = MakeTransformerWeightBinding(0U, TransformerWeightRole::kAttentionQ),
             .raw_weight = MakeWeightView(storage, 0, 8, DataType::Float32(), {2, 1}),
             .selector = MakeExpectedSelector(),
             .recipe = CpuIdentityPackingRecipe()},
    };

    PackingOnlyTestBackend backend;
    PackedWeightStore packed_weight_store;
    ASSERT_TRUE(PrepackWeightRequests(backend, packed_weight_store, requests).ok());

    const KernelSelector expected_selector = MakeExpectedSelector();
    const WeightArtifactKey key{.binding = requests.front().binding,
                                .selector = requests.front().selector,
                                .recipe = CpuWeightPrepacker::RecipeFor(
                                        requests.front().selector)};
    const auto found = packed_weight_store.Find(key);
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->op_type(), OpType::kLinear);
    EXPECT_EQ(found->selector(), expected_selector);
    EXPECT_TRUE(found->storage().is_initialized());
}

TEST(WeightPacking, PrepackWeightRequestsStoresAllLayerWeightsDistinctly) {
    auto storage = std::make_shared<TestStorage>(256);
    for (auto& b: storage->data) b = std::byte{0};

    // Two layers × 7 linear weights share the same (op_type, selector) but
    // distinct bindings. Every weight must be packed, not silently dropped.
    const std::vector<TransformerWeightRole> roles{
            TransformerWeightRole::kAttentionQ, TransformerWeightRole::kAttentionK,
            TransformerWeightRole::kAttentionV, TransformerWeightRole::kAttentionO,
            TransformerWeightRole::kMlpGate, TransformerWeightRole::kMlpUp,
            TransformerWeightRole::kMlpDown};
    std::vector<WeightPackingRequest> requests;
    for (uint32_t layer = 0; layer < 2; ++layer) {
        for (size_t role = 0; role < roles.size(); ++role) {
            requests.push_back(
                    {.op_type = OpType::kLinear,
                     .binding = MakeTransformerWeightBinding(layer, roles[role]),
                     .raw_weight = MakeWeightView(storage, (layer * roles.size() + role) * 8U,
                                                  8, DataType::Float32(), {2, 1}),
                     .selector = MakeExpectedSelector(),
                     .recipe = CpuIdentityPackingRecipe()});
        }
    }
    ASSERT_EQ(requests.size(), 14U);

    PackingOnlyTestBackend backend;
    PackedWeightStore packed_weight_store;
    ASSERT_TRUE(PrepackWeightRequests(backend, packed_weight_store, requests).ok());

    // All 14 distinct keys are stored; the same role across layers differs by
    // its layer index and every role is individually findable.
    EXPECT_EQ(packed_weight_store.size(), 14U);
    for (const auto& req: requests) {
        const WeightArtifactKey key{.binding = req.binding,
                                    .selector = req.selector,
                                    .recipe = CpuWeightPrepacker::RecipeFor(req.selector)};
        EXPECT_NE(packed_weight_store.Find(key), nullptr) << "missing key for layer";
    }
}

TEST(WeightPacking, RawViewsRemainAccessibleAfterPrepack) {
    auto storage = std::make_shared<TestStorage>(256);
    for (auto& b: storage->data) b = std::byte{0};

    const RawWeightView raw_weight =
            MakeWeightView(storage, 0, 8, DataType::Float32(), {2, 1});
    const std::vector<WeightPackingRequest> requests{
            {.op_type = OpType::kLinear,
             .binding = MakeTransformerWeightBinding(0U, TransformerWeightRole::kAttentionQ),
             .raw_weight = raw_weight,
             .selector = MakeExpectedSelector(),
             .recipe = CpuIdentityPackingRecipe()},
    };

    PackingOnlyTestBackend backend;
    PackedWeightStore packed_weight_store;
    ASSERT_TRUE(PrepackWeightRequests(backend, packed_weight_store, requests).ok());

    // Prepacking borrows the request's raw view; the caller's view stays valid.
    EXPECT_TRUE(raw_weight.IsValid());
    EXPECT_TRUE(storage->data.data() == raw_weight.data);
}

int g_planner_packed_kernel_calls = 0;

// Backend that resolves any kPacked selector and declares the same packing
// recipe the CPU prepacker produces, so exact-key resolution succeeds.
Status PlannerPackedKernel(const KernelContext&) noexcept {
    ++g_planner_packed_kernel_calls;
    return Status::Ok();
}

class PlannerPackedTestBackend final : public Backend {
public:
    DeviceType device_type() const noexcept override {
        return DeviceType::kCPU;
    }

    StatusOr<ResolvedKernel> PrepareKernel(
            OpType op_type,
            const KernelSelector& selector,
            const OpParams&) const override {
        if (selector.weight_format != WeightFormat::kPacked) {
            return Status::NotFound(
                    "Planner test backend only resolves packed selectors");
        }
        return ResolvedKernel{
                .op_type = op_type,
                .fn = &PlannerPackedKernel,
                .attrs = {},
                .name = "test::planner_packed_kernel",
                .expected_packing_recipe = CpuWeightPrepacker::RecipeFor(selector),
        };
    }

    const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override {
        return nullptr;
    }
};

class PlannerPackedTestBackendFactory final : public BackendFactory {
public:
    DeviceType device_type() const noexcept override {
        return DeviceType::kCPU;
    }

    StatusOr<std::unique_ptr<Backend>> Create() const override {
        return std::make_unique<PlannerPackedTestBackend>();
    }
};

// End-to-end identity loop: lowering marks kWeight steps packed, the
// lower-driven request builder derives key material from the artifact, the
// prepacker stores artifacts under {source, value, binding, selector, recipe},
// and the plan builder resolves each step to its own artifact.
TEST(WeightPacking, LoweredDrivenPrepackAndResolve) {
    auto storage = std::make_shared<TestStorage>(512);
    for (auto& b: storage->data) b = std::byte{0};

    ModelGraph graph;
    const GraphValueId tokens = graph.AddInput(
            TensorSpec{.dtype = DataType::Int(64), .shape = StaticShape({1})});
    const GraphValueId embedding_weight = graph.AddWeight(
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({32, 8})},
            MakeTransformerWeightBinding(std::nullopt,
                                         TransformerWeightRole::kTokenEmbedding));
    const auto embedding = graph.AddNode(
            OpType::kEmbedding, std::nullopt, {tokens, embedding_weight},
            {NodeOutputDesc{.payload = ActivationValue{}}}, EmbeddingParams{});
    ASSERT_TRUE(embedding.ok()) << embedding.status().ToString();
    const GraphValueId norm0_weight = graph.AddWeight(
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({8})},
            MakeTransformerWeightBinding(0, TransformerWeightRole::kInputNorm));
    const auto norm0 = graph.AddNode(
            OpType::kRmsNorm, 0U, {embedding->outputs[0], norm0_weight},
            {NodeOutputDesc{.payload = ActivationValue{}}}, RmsNormParams{.eps = 1.0e-5F});
    ASSERT_TRUE(norm0.ok()) << norm0.status().ToString();
    const GraphValueId norm1_weight = graph.AddWeight(
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({8})},
            MakeTransformerWeightBinding(1, TransformerWeightRole::kInputNorm));
    const auto norm1 = graph.AddNode(
            OpType::kRmsNorm, 1U, {norm0->outputs[0], norm1_weight},
            {NodeOutputDesc{.payload = ActivationValue{}}}, RmsNormParams{.eps = 1.0e-5F});
    ASSERT_TRUE(norm1.ok()) << norm1.status().ToString();
    graph.MarkOutput(norm1->outputs[0]);

    // Backing raw weights for the three weight values.
    ResolvedModelWeights resolved;
    resolved.embed_tokens = MakeWeightView(storage, 0, 32 * 8 * sizeof(float),
                                           DataType::Float32(), {32, 8});
    resolved.layers.resize(2);
    resolved.layers[0].norm.input_rmsnorm = MakeWeightView(
            storage, 256, 8 * sizeof(float), DataType::Float32(), {8});
    resolved.layers[1].norm.input_rmsnorm = MakeWeightView(
            storage, 264, 8 * sizeof(float), DataType::Float32(), {8});

    GraphLoweringConfig config;
    config.enable_packed_weights = true;
    const auto lowered = LowerModelGraph(graph, config);
    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();
    ASSERT_EQ(lowered->steps().size(), 3U);

    auto requests = BuildWeightPackingRequests(*lowered, resolved);
    ASSERT_TRUE(requests.ok()) << requests.status().ToString();
    SetIdentityPackingRecipes(*requests);
    ASSERT_EQ(requests->size(), 3U);
    for (const auto& req: *requests) {
        EXPECT_EQ(req.source_id, lowered->artifact_id());
        EXPECT_EQ(req.selector.weight_format, WeightFormat::kPacked);
    }
    // Requests appear in lowered order: embedding value, norm0 value, norm1.
    EXPECT_EQ((*requests)[0].value_index, embedding_weight.index);
    EXPECT_EQ((*requests)[1].value_index, norm0_weight.index);
    EXPECT_EQ((*requests)[2].value_index, norm1_weight.index);

    PackingOnlyTestBackend prepack_backend;
    PackedWeightStore packed_weight_store;
    ASSERT_TRUE(PrepackWeightRequests(
                        prepack_backend, packed_weight_store, *requests)
                        .ok());
    ASSERT_EQ(packed_weight_store.size(), 3U);
    EXPECT_EQ(packed_weight_store.source_id(), lowered->artifact_id());

    RuntimeBuilder builder;
    builder.RegisterBackendFactory(
            DeviceType::kCPU, std::make_unique<PlannerPackedTestBackendFactory>());
    Runtime runtime = builder.Build();
    const auto plan =
            ExecutionPlanBuilder::Build(runtime, packed_weight_store, *lowered);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->size(), 3U);
    ASSERT_NE(plan->steps()[1].packed_weights, nullptr);
    ASSERT_NE(plan->steps()[2].packed_weights, nullptr);
    EXPECT_NE(plan->steps()[1].packed_weights, plan->steps()[2].packed_weights);

    // Phase C: bind and execute the frozen plan end to end. The model token
    // input and the three packed weights are external read-only values; the
    // activations are allocated from the runtime allocator's arena.
    std::vector<ExternalReadOnlyValueBinding> readable;
    const ExecutionValueId token_value = plan->model_inputs().front();
    int64_t token_ids[1] = {7};
    const int64_t token_shape[1] = {1};
    const int64_t contiguous1[1] = {1};
    readable.push_back({.value = token_value,
                        .tensor = TensorView(token_ids, DataType::Int(64),
                                             token_shape, contiguous1)});
    float embedding_data[32 * 8]{};
    float norm0_data[8]{};
    float norm1_data[8]{};
    const int64_t emb_shape[2] = {32, 8};
    const int64_t emb_strides[2] = {8, 1};
    const int64_t norm_shape[1] = {8};
    int norm_index = 0;
    for (size_t i = 0; i < plan->values().size(); ++i) {
        if (plan->values()[i].kind != ExecutionValueKind::kWeight) {
            continue;
        }
        const ExecutionValueId value{.index = static_cast<uint32_t>(i)};
        if (plan->values()[i].spec.shape.rank() == std::optional<size_t>(2)) {
            readable.push_back({.value = value,
                                .tensor = TensorView(embedding_data, DataType::Float32(),
                                                     emb_shape, emb_strides)});
        } else {
            float* data = norm_index == 0 ? norm0_data : norm1_data;
            readable.push_back({.value = value,
                                .tensor = TensorView(data, DataType::Float32(),
                                                     norm_shape, contiguous1)});
            ++norm_index;
        }
    }
    ASSERT_EQ(readable.size(), 4U);

    auto table = PrepareExecutionBindings(
            *plan,
            ExternalTensorBindings{.readable = std::move(readable)},
            runtime.GetAllocator(Device::CPU()));
    ASSERT_TRUE(table.ok()) << table.status().ToString();
    auto context = ExecutionContext::Create(*plan, std::move(*table), nullptr);
    ASSERT_TRUE(context.ok()) << context.status().ToString();
    g_planner_packed_kernel_calls = 0;
    const Status status = Executor::Execute(*plan, *context);
    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_EQ(g_planner_packed_kernel_calls, 3);
}

TEST(WeightPacking, LoweredDrivenPrepackResolvesCompositeBindings) {
    auto storage = std::make_shared<TestStorage>(4096);
    for (auto& b: storage->data) b = std::byte{0};

    ModelGraph graph;
    const GraphValueId tokens = graph.AddInput(
            TensorSpec{.dtype = DataType::Int(64), .shape = StaticShape({1})});
    const GraphValueId embedding_weight = graph.AddWeight(
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({32, 8})},
            MakeTransformerWeightBinding(std::nullopt,
                                         TransformerWeightRole::kTokenEmbedding));
    const auto embedding = graph.AddNode(
            OpType::kEmbedding, std::nullopt, {tokens, embedding_weight},
            {NodeOutputDesc{.payload = ActivationValue{}}}, EmbeddingParams{});
    ASSERT_TRUE(embedding.ok()) << embedding.status().ToString();

    const GraphValueId qkv_weight = graph.AddWeight(
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({24, 8})},
            MakeQkvWeightBinding(0U));
    const auto qkv = graph.AddNode(
            OpType::kQkvLinear, 0U, {embedding->outputs[0], qkv_weight},
            {{.payload = ActivationValue{}},
             {.payload = ActivationValue{}},
             {.payload = ActivationValue{}}},
            QkvLinearParams{.q_out_features = 8,
                            .k_out_features = 8,
                            .v_out_features = 8});
    ASSERT_TRUE(qkv.ok()) << qkv.status().ToString();

    const GraphValueId gate_up_weight = graph.AddWeight(
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({16, 8})},
            MakeGateUpWeightBinding(0U));
    const auto gate_up = graph.AddNode(
            OpType::kGateUpLinear, 0U, {embedding->outputs[0], gate_up_weight},
            {{.payload = ActivationValue{}}, {.payload = ActivationValue{}}},
            GateUpLinearParams{.gate_out_features = 8, .up_out_features = 8});
    ASSERT_TRUE(gate_up.ok()) << gate_up.status().ToString();
    graph.MarkOutput(qkv->outputs[0]);
    graph.MarkOutput(gate_up->outputs[0]);

    // Backing raw weights: one direct embedding view plus both composite
    // recipes, each component sized [8, 8] = 256 bytes.
    ResolvedModelWeights resolved;
    resolved.embed_tokens = MakeWeightView(storage, 0, 32 * 8 * sizeof(float),
                                           DataType::Float32(), {32, 8});
    resolved.layers.resize(1);
    auto& attn = resolved.layers[0].attn;
    attn.q_proj = MakeWeightView(storage, 1024, 8 * 8 * sizeof(float),
                                 DataType::Float32(), {8, 8});
    attn.k_proj = MakeWeightView(storage, 1280, 8 * 8 * sizeof(float),
                                 DataType::Float32(), {8, 8});
    attn.v_proj = MakeWeightView(storage, 1536, 8 * 8 * sizeof(float),
                                 DataType::Float32(), {8, 8});
    auto& mlp = resolved.layers[0].mlp;
    mlp.gate_proj = MakeWeightView(storage, 1792, 8 * 8 * sizeof(float),
                                   DataType::Float32(), {8, 8});
    mlp.up_proj = MakeWeightView(storage, 2048, 8 * 8 * sizeof(float),
                                 DataType::Float32(), {8, 8});
    // Fill the component regions with a distinct byte pattern so the fused
    // layout is verifiable byte-by-byte after prepack.
    for (size_t i = 1024; i < 2304; ++i) {
        storage->data[i] = static_cast<std::byte>(i);
    }

    GraphLoweringConfig config;
    config.enable_packed_weights = true;
    const auto lowered = LowerModelGraph(graph, config);
    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();

    auto requests = BuildWeightPackingRequests(*lowered, resolved);
    ASSERT_TRUE(requests.ok()) << requests.status().ToString();
    SetIdentityPackingRecipes(*requests);
    // Embedding value + fused QKV weight + fused Gate-Up weight.
    ASSERT_EQ(requests->size(), 3U);

    auto qkv_request = std::find_if(
            requests->begin(), requests->end(),
            [](const WeightPackingRequest& req) {
                return req.op_type == OpType::kQkvLinear;
            });
    ASSERT_NE(qkv_request, requests->end());
    EXPECT_TRUE(std::holds_alternative<QkvWeightBinding>(qkv_request->binding.spec));
    ASSERT_EQ(qkv_request->components.size(), 3U);
    EXPECT_EQ(qkv_request->components[0].data, attn.q_proj.data);
    EXPECT_EQ(qkv_request->components[1].data, attn.k_proj.data);
    EXPECT_EQ(qkv_request->components[2].data, attn.v_proj.data);
    EXPECT_EQ(qkv_request->source_id, lowered->artifact_id());

    auto gate_up_request = std::find_if(
            requests->begin(), requests->end(),
            [](const WeightPackingRequest& req) {
                return req.op_type == OpType::kGateUpLinear;
            });
    ASSERT_NE(gate_up_request, requests->end());
    EXPECT_TRUE(std::holds_alternative<GateUpWeightBinding>(gate_up_request->binding.spec));
    ASSERT_EQ(gate_up_request->components.size(), 2U);
    EXPECT_EQ(gate_up_request->components[0].data, mlp.gate_proj.data);
    EXPECT_EQ(gate_up_request->components[1].data, mlp.up_proj.data);

    auto embedding_request = std::find_if(
            requests->begin(), requests->end(),
            [](const WeightPackingRequest& req) {
                return req.op_type == OpType::kEmbedding;
            });
    ASSERT_NE(embedding_request, requests->end());
    EXPECT_TRUE(embedding_request->components.empty());
    EXPECT_EQ(embedding_request->raw_weight.data, resolved.embed_tokens.data);

    PackingOnlyTestBackend prepack_backend;
    PackedWeightStore packed_weight_store;
    ASSERT_TRUE(PrepackWeightRequests(
                        prepack_backend, packed_weight_store, *requests)
                        .ok());
    ASSERT_EQ(packed_weight_store.size(), 3U);
    EXPECT_EQ(packed_weight_store.source_id(), lowered->artifact_id());

    // Stored fused artifacts carry the fused logical shape and exactly the
    // recipe-ordered concatenation of their components.
    const auto expect_fused = [&](const WeightPackingRequest& req) {
        const WeightArtifactKey key{.source_id = req.source_id,
                                    .value_index = req.value_index,
                                    .binding = req.binding,
                                    .selector = req.selector,
                                    .recipe = CpuWeightPrepacker::RecipeFor(
                                            req.selector)};
        const auto found = packed_weight_store.Find(key);
        ASSERT_NE(found, nullptr);
        ASSERT_EQ(found->logical_shape().size(), 2U);
        int64_t rows = 0;
        size_t cursor = 0;
        for (const auto& component: req.components) {
            rows += component.shape[0];
            ASSERT_LE(cursor + component.bytes, found->storage().nbytes());
            EXPECT_EQ(std::memcmp(
                              static_cast<const char*>(found->storage().data()) + cursor,
                              component.data,
                              component.bytes),
                      0);
            cursor += component.bytes;
        }
        EXPECT_EQ(found->logical_shape()[0], rows);
        EXPECT_EQ(found->logical_shape()[1], req.components.front().shape[1]);
    };
    expect_fused(*qkv_request);
    expect_fused(*gate_up_request);

    // The bound plan resolves the two fused steps to their own artifacts.
    RuntimeBuilder builder;
    builder.RegisterBackendFactory(
            DeviceType::kCPU, std::make_unique<PlannerPackedTestBackendFactory>());
    Runtime runtime = builder.Build();
    const auto plan =
            ExecutionPlanBuilder::Build(runtime, packed_weight_store, *lowered);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->size(), 3U);
    bool saw_qkv = false;
    bool saw_gate_up = false;
    for (const auto& step: plan->steps()) {
        if (step.kernel.op_type == OpType::kQkvLinear) {
            EXPECT_NE(step.packed_weights, nullptr);
            EXPECT_EQ(step.packed_weights->logical_shape(),
                      std::vector<int64_t>({24, 8}));
            saw_qkv = true;
        } else if (step.kernel.op_type == OpType::kGateUpLinear) {
            EXPECT_NE(step.packed_weights, nullptr);
            EXPECT_EQ(step.packed_weights->logical_shape(),
                      std::vector<int64_t>({16, 8}));
            saw_gate_up = true;
        }
    }
    EXPECT_TRUE(saw_qkv);
    EXPECT_TRUE(saw_gate_up);
}

Status PrepackSingleRequest(const WeightPackingRequest& request) {
    PackingOnlyTestBackend backend;
    PackedWeightStore store;
    return PrepackWeightRequests(backend, store, {request});
}

// RawWeightView::bytes must exactly match shape × dtype byte size: the
// prepacker copies the shape-derived logical size, so any mismatch would
// read out of bounds or corrupt fused layouts. These cases must be rejected
// eagerly at the PrepackWeightRequests boundary.
TEST(WeightPacking, PrepackWeightRequestsRejectsDirectWeightWithUndersizedBytes) {
    auto storage = std::make_shared<TestStorage>(64);
    for (auto& b: storage->data) b = std::byte{0};

    const WeightPackingRequest request{
            .op_type = OpType::kLinear,
            .source_id = 1,
            .binding = MakeTransformerWeightBinding(0U, TransformerWeightRole::kAttentionQ),
            .raw_weight = MakeWeightView(storage, 0, 4, DataType::Float32(), {2, 1}),
            .selector = MakeExpectedSelector(),
            .recipe = CpuIdentityPackingRecipe(),
    };

    const Status status = PrepackSingleRequest(request);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message().find("does not match"), std::string::npos);
}

TEST(WeightPacking, PrepackWeightRequestsRejectsDirectWeightWithOversizedBytes) {
    auto storage = std::make_shared<TestStorage>(64);
    for (auto& b: storage->data) b = std::byte{0};

    const WeightPackingRequest request{
            .op_type = OpType::kLinear,
            .source_id = 1,
            .binding = MakeTransformerWeightBinding(0U, TransformerWeightRole::kAttentionQ),
            .raw_weight = MakeWeightView(storage, 0, 12, DataType::Float32(), {2, 1}),
            .selector = MakeExpectedSelector(),
            .recipe = CpuIdentityPackingRecipe(),
    };

    const Status status = PrepackSingleRequest(request);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message().find("does not match"), std::string::npos);
}

TEST(WeightPacking, PrepackWeightRequestsRejectsCompositeComponentWithUndersizedBytes) {
    auto storage = std::make_shared<TestStorage>(64);
    for (auto& b: storage->data) b = std::byte{0};

    const WeightPackingRequest request{
            .op_type = OpType::kQkvLinear,
            .source_id = 1,
            .binding = MakeQkvWeightBinding(0U),
            .components = {
                    MakeWeightView(storage, 0, 8, DataType::Float32(), {2, 1}),
                    MakeWeightView(storage, 16, 4, DataType::Float32(), {2, 1}),
                    MakeWeightView(storage, 24, 8, DataType::Float32(), {2, 1}),
            },
            .selector = MakeExpectedSelector(),
            .recipe = CpuIdentityPackingRecipe(),
    };

    const Status status = PrepackSingleRequest(request);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message().find("does not match"), std::string::npos);
}

TEST(WeightPacking, PrepackWeightRequestsRejectsNegativeWeightDimension) {
    auto storage = std::make_shared<TestStorage>(64);
    for (auto& b: storage->data) b = std::byte{0};

    const WeightPackingRequest request{
            .op_type = OpType::kLinear,
            .source_id = 1,
            .binding = MakeTransformerWeightBinding(0U, TransformerWeightRole::kAttentionQ),
            .raw_weight = MakeWeightView(storage, 0, 0, DataType::Float32(), {-3}),
            .selector = MakeExpectedSelector(),
            .recipe = CpuIdentityPackingRecipe(),
    };

    const Status status = PrepackSingleRequest(request);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message().find("negative dimension"), std::string::npos);
}

TEST(WeightPacking, PrepackWeightRequestsRejectsOverflowingWeightByteSize) {
    auto storage = std::make_shared<TestStorage>(64);
    for (auto& b: storage->data) b = std::byte{0};

    const WeightPackingRequest request{
            .op_type = OpType::kLinear,
            .source_id = 1,
            .binding = MakeTransformerWeightBinding(0U, TransformerWeightRole::kAttentionQ),
            .raw_weight = RawWeightView{
                    .data = storage->data.data(),
                    .bytes = 64,
                    .dtype = DataType::Float32(),
                    .shape = {static_cast<int64_t>(1) << 30,
                              static_cast<int64_t>(1) << 30,
                              static_cast<int64_t>(1) << 30,
                              static_cast<int64_t>(1) << 30},
                    .storage = storage,
            },
            .selector = MakeExpectedSelector(),
            .recipe = CpuIdentityPackingRecipe(),
    };

    const Status status = PrepackSingleRequest(request);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message().find("overflows"), std::string::npos);
}

// Tied embeddings: a checkpoint without an independent lm_head reuses
// embed_tokens. The request builder must mirror the per-family graph builders
// and fall back to embed_tokens for the kLmHead binding, or graph-driven
// materialization fails for common tied models.
TEST(WeightPacking, BuildWeightPackingRequestsFallsBackToEmbedTokensForTiedLmHead) {
    auto storage = std::make_shared<TestStorage>(2048);
    for (auto& b: storage->data) b = std::byte{0};

    ModelGraph graph;
    const GraphValueId tokens = graph.AddInput(
            TensorSpec{.dtype = DataType::Int(64), .shape = StaticShape({1})});
    const GraphValueId embedding_weight = graph.AddWeight(
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({32, 8})},
            MakeTransformerWeightBinding(std::nullopt,
                                         TransformerWeightRole::kTokenEmbedding));
    const auto embedding = graph.AddNode(
            OpType::kEmbedding, std::nullopt, {tokens, embedding_weight},
            {NodeOutputDesc{.payload = ActivationValue{}}}, EmbeddingParams{});
    ASSERT_TRUE(embedding.ok()) << embedding.status().ToString();
    const GraphValueId lm_head_weight = graph.AddWeight(
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({32, 8})},
            MakeTransformerWeightBinding(std::nullopt,
                                         TransformerWeightRole::kLmHead));
    const auto lm_head = graph.AddNode(
            OpType::kLinear, std::nullopt, {embedding->outputs[0], lm_head_weight},
            {NodeOutputDesc{.payload = ActivationValue{}}}, LinearParams{});
    ASSERT_TRUE(lm_head.ok()) << lm_head.status().ToString();
    graph.MarkOutput(lm_head->outputs[0]);

    // Tied checkpoint: no independent lm_head backing view.
    ResolvedModelWeights resolved;
    resolved.embed_tokens = MakeWeightView(storage, 0, 32 * 8 * sizeof(float),
                                           DataType::Float32(), {32, 8});

    GraphLoweringConfig config;
    config.enable_packed_weights = true;
    const auto lowered = LowerModelGraph(graph, config);
    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();

    auto requests = BuildWeightPackingRequests(*lowered, resolved);
    ASSERT_TRUE(requests.ok()) << requests.status().ToString();
    ASSERT_EQ(requests->size(), 2U);

    // Both values resolve to the same embed_tokens bytes: the embedding
    // weight and the tied lm_head weight.
    for (const auto& req: *requests) {
        EXPECT_EQ(req.raw_weight.data, resolved.embed_tokens.data);
        EXPECT_EQ(req.raw_weight.bytes, resolved.embed_tokens.bytes);
    }
    EXPECT_EQ((*requests)[0].op_type, OpType::kEmbedding);
    EXPECT_EQ((*requests)[1].op_type, OpType::kLinear);
    EXPECT_EQ(TryGetTransformerWeightRole((*requests)[0].binding),
              TransformerWeightRole::kTokenEmbedding);
    EXPECT_EQ(TryGetTransformerWeightRole((*requests)[1].binding),
              TransformerWeightRole::kLmHead);
}
// Plain (non-packed) weight steps must not produce packing requests: only
// kPacked selectors are packed, and feeding a plain selector to the packer
// would fail.
TEST(WeightPacking, BuildWeightPackingRequestsSkipsPlainWeightSteps) {
    ModelGraph graph;
    const GraphValueId input = graph.AddConstant(
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({1, 8})},
            ConstantBinding{}, "input");
    const GraphValueId weight = graph.AddWeight(
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 8})},
            MakeTransformerWeightBinding(0U, TransformerWeightRole::kMlpUp));
    const auto linear = graph.AddNode(
            OpType::kLinear, 0U, {input, weight},
            {NodeOutputDesc{.payload = ActivationValue{}}}, LinearParams{});
    ASSERT_TRUE(linear.ok()) << linear.status().ToString();
    graph.MarkOutput(linear->outputs[0]);

    GraphLoweringConfig config; // enable_packed_weights defaults to false.
    const auto lowered = LowerModelGraph(graph, config);
    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();

    auto requests = BuildWeightPackingRequests(*lowered, ResolvedModelWeights{});
    ASSERT_TRUE(requests.ok()) << requests.status().ToString();
    EXPECT_TRUE(requests->empty());
}

// The compiler preserves all consumers until inference resolves each
// descriptor-owned recipe and can safely coalesce compatible requests.
TEST(WeightPacking, BuildWeightPackingRequestsPreservesSharedWeightConsumers) {
    auto storage = std::make_shared<TestStorage>(512);
    for (auto& b: storage->data) b = std::byte{0};

    ModelGraph graph;
    const GraphValueId input = graph.AddConstant(
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({1, 8})},
            ConstantBinding{}, "input");
    const GraphValueId weight = graph.AddWeight(
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 8})},
            MakeTransformerWeightBinding(0U, TransformerWeightRole::kMlpUp));
    const auto linear0 = graph.AddNode(
            OpType::kLinear, 0U, {input, weight},
            {NodeOutputDesc{.payload = ActivationValue{}}}, LinearParams{});
    ASSERT_TRUE(linear0.ok()) << linear0.status().ToString();
    const auto linear1 = graph.AddNode(
            OpType::kLinear, 0U, {input, weight},
            {NodeOutputDesc{.payload = ActivationValue{}}}, LinearParams{});
    ASSERT_TRUE(linear1.ok()) << linear1.status().ToString();
    graph.MarkOutput(linear0->outputs[0]);
    graph.MarkOutput(linear1->outputs[0]);

    ResolvedModelWeights resolved;
    resolved.layers.resize(1);
    resolved.layers[0].mlp.up_proj = MakeWeightView(storage, 0, 4 * 8 * sizeof(float),
                                                    DataType::Float32(), {4, 8});

    GraphLoweringConfig config;
    config.enable_packed_weights = true;
    const auto lowered = LowerModelGraph(graph, config);
    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();

    auto requests = BuildWeightPackingRequests(*lowered, resolved);
    ASSERT_TRUE(requests.ok()) << requests.status().ToString();
    ASSERT_EQ(requests->size(), 2U);
    for (const WeightPackingRequest& request: *requests) {
        EXPECT_EQ(request.value_index, weight.index);
        EXPECT_EQ(request.op_type, OpType::kLinear);
        EXPECT_TRUE(request.recipe.layout.empty());
    }
}
// Packs a contiguous FP32 test weight via the CPU identity prepacker.
std::shared_ptr<const PackedWeights> PackTestArtifact(OpType op_type,
                                                      const KernelSelector& selector,
                                                      std::vector<int64_t> shape) {
    size_t numel = 1;
    for (const int64_t dim: shape) numel *= static_cast<size_t>(dim);
    std::vector<float> data(numel, 1.0F);
    std::vector<int64_t> strides(shape.size());
    if (!strides.empty()) {
        strides.back() = 1;
        for (int64_t i = static_cast<int64_t>(strides.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    CpuWeightPrepacker prepacker;
    auto packed = prepacker.Pack(
            op_type,
            TensorView(data.data(), DataType::Float32(),
                       IntArrayView(shape), IntArrayView(strides), 0),
            selector);
    EXPECT_TRUE(packed.ok());
    if (!packed.ok()) return nullptr;
    return std::shared_ptr<const PackedWeights>(std::move(*packed));
}

ExecutionPlanNodeSpec MakePackedLinearNode() {
    return ExecutionPlanNodeSpec{
            .op_type = OpType::kLinear,
            .selector = MakeExpectedSelector(),
            .input_specs = {TensorSpec{.dtype = DataType::Float32(),
                                       .shape = StaticShape({1, 8})},
                            TensorSpec{.dtype = DataType::Float32(),
                                       .shape = StaticShape({4, 8})}},
            .output_specs = {TensorSpec{.dtype = DataType::Float32(),
                                        .shape = StaticShape({1, 4})}},
            .op_params = LinearParams{},
    };
}

// Untrusted packed nodes key their artifacts by the actual kWeight operand
// id: two packed nodes resolve distinct artifacts instead of colliding on one
// unbound key.
TEST(WeightPacking, UntrustedBuildBindsDistinctPackedArtifacts) {
    const std::vector<ExecutionPlanNodeSpec> nodes{MakePackedLinearNode(),
                                                   MakePackedLinearNode()};

    // Each untrusted node appends its operands in schema-port order
    // (activation id 0, weight id 1) before the next node's operands.
    PackedWeightStore store;
    const KernelSelector selector = MakeExpectedSelector();
    const PackingRecipe recipe = CpuWeightPrepacker::RecipeFor(selector);
    for (const uint32_t value_index: {1U, 4U}) {
        auto artifact = PackTestArtifact(OpType::kLinear, selector, {4, 8});
        ASSERT_NE(artifact, nullptr);
        ASSERT_TRUE(store.Store({.source_id = 0,
                                 .value_index = value_index,
                                 .binding = {},
                                 .selector = selector,
                                 .recipe = recipe},
                                std::move(artifact))
                            .ok());
    }

    RuntimeBuilder builder;
    builder.RegisterBackendFactory(
            DeviceType::kCPU, std::make_unique<PlannerPackedTestBackendFactory>());
    Runtime runtime = builder.Build();
    const auto plan = ExecutionPlanBuilder::Build(runtime, store, nodes);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->size(), 2U);
    ASSERT_NE(plan->steps()[0].packed_weights, nullptr);
    ASSERT_NE(plan->steps()[1].packed_weights, nullptr);
    EXPECT_NE(plan->steps()[0].packed_weights, plan->steps()[1].packed_weights);
}

TEST(WeightPacking, UntrustedBuildRejectsArtifactOpTypeMismatch) {
    const std::vector<ExecutionPlanNodeSpec> nodes{MakePackedLinearNode()};

    PackedWeightStore store;
    const KernelSelector selector = MakeExpectedSelector();
    auto artifact = PackTestArtifact(OpType::kEmbedding, selector, {4, 8});
    ASSERT_NE(artifact, nullptr);
    ASSERT_TRUE(store.Store({.source_id = 0,
                             .value_index = 1,
                             .binding = {},
                             .selector = selector,
                             .recipe = CpuWeightPrepacker::RecipeFor(selector)},
                            std::move(artifact))
                        .ok());

    RuntimeBuilder builder;
    builder.RegisterBackendFactory(
            DeviceType::kCPU, std::make_unique<PlannerPackedTestBackendFactory>());
    Runtime runtime = builder.Build();
    const auto plan = ExecutionPlanBuilder::Build(runtime, store, nodes);
    ASSERT_FALSE(plan.ok());
    EXPECT_NE(plan.status().message().find("op type"), std::string::npos);
}

TEST(WeightPacking, UntrustedBuildRejectsArtifactShapeMismatch) {
    const std::vector<ExecutionPlanNodeSpec> nodes{MakePackedLinearNode()};

    PackedWeightStore store;
    const KernelSelector selector = MakeExpectedSelector();
    auto artifact = PackTestArtifact(OpType::kLinear, selector, {3, 8});
    ASSERT_NE(artifact, nullptr);
    ASSERT_TRUE(store.Store({.source_id = 0,
                             .value_index = 1,
                             .binding = {},
                             .selector = selector,
                             .recipe = CpuWeightPrepacker::RecipeFor(selector)},
                            std::move(artifact))
                        .ok());

    RuntimeBuilder builder;
    builder.RegisterBackendFactory(
            DeviceType::kCPU, std::make_unique<PlannerPackedTestBackendFactory>());
    Runtime runtime = builder.Build();
    const auto plan = ExecutionPlanBuilder::Build(runtime, store, nodes);
    ASSERT_FALSE(plan.ok());
    EXPECT_NE(plan.status().message().find("logical metadata"), std::string::npos);
}
constexpr size_t kViewBytes = 8;
constexpr size_t kLayerStride = 72;

RawWeightView MakeWeightView(const std::shared_ptr<TestStorage>& storage, size_t offset) {
    return RawWeightView{
            .data = storage->data.data() + offset,
            .bytes = kViewBytes,
            .dtype = DataType::Float32(),
            .shape = {2, 1},
            .storage = storage,
    };
}

DecoderLayerRawWeights MakeLayer(const std::shared_ptr<TestStorage>& storage, size_t base_offset) {
    DecoderLayerRawWeights layer;
    layer.norm.input_rmsnorm = MakeWeightView(storage, base_offset + 0);
    layer.norm.post_attn_rmsnorm = MakeWeightView(storage, base_offset + 8);
    layer.attn.q_proj = MakeWeightView(storage, base_offset + 16);
    layer.attn.k_proj = MakeWeightView(storage, base_offset + 24);
    layer.attn.v_proj = MakeWeightView(storage, base_offset + 32);
    layer.attn.o_proj = MakeWeightView(storage, base_offset + 40);
    layer.mlp.gate_proj = MakeWeightView(storage, base_offset + 48);
    layer.mlp.up_proj = MakeWeightView(storage, base_offset + 56);
    layer.mlp.down_proj = MakeWeightView(storage, base_offset + 64);
    return layer;
}

/// Two-layer weights. `with_lm_head` selects an untied checkpoint; omitting it
/// produces the tied shape whose lm_head must reuse embed_tokens.
ResolvedModelWeights MakeResolved(const std::shared_ptr<TestStorage>& storage,
                                  bool with_lm_head) {
    ResolvedModelWeights resolved;
    resolved.embed_tokens = MakeWeightView(storage, 0);
    resolved.final_norm = MakeWeightView(storage, kViewBytes);
    if (with_lm_head) {
        resolved.lm_head = MakeWeightView(storage, 2 * kViewBytes);
    }
    resolved.layers.push_back(MakeLayer(storage, 3 * kViewBytes));
    resolved.layers.push_back(MakeLayer(storage, 3 * kViewBytes + kLayerStride));
    return resolved;
}

WeightBinding RoleBinding(std::optional<uint32_t> layer, TransformerWeightRole role) {
    return MakeTransformerWeightBinding(layer, role);
}

TEST(WeightBindingResolver, ModelLevelRolesResolveToTheirOwnWeight) {
    auto storage = std::make_shared<TestStorage>(256);
    const ResolvedModelWeights resolved = MakeResolved(storage, /*with_lm_head=*/true);

    EXPECT_EQ(
            ResolveWeightBinding(
                    RoleBinding(std::nullopt, TransformerWeightRole::kTokenEmbedding), resolved),
            &resolved.embed_tokens);
    EXPECT_EQ(
            ResolveWeightBinding(
                    RoleBinding(std::nullopt, TransformerWeightRole::kFinalNorm), resolved),
            &resolved.final_norm);
}

TEST(WeightBindingResolver, UntiedLmHeadResolvesToItsOwnWeight) {
    auto storage = std::make_shared<TestStorage>(256);
    const ResolvedModelWeights resolved = MakeResolved(storage, /*with_lm_head=*/true);
    ASSERT_TRUE(resolved.lm_head.has_value());

    const RawWeightView* lm_head =
            ResolveWeightBinding(RoleBinding(std::nullopt, TransformerWeightRole::kLmHead),
                                 resolved);
    ASSERT_NE(lm_head, nullptr);
    EXPECT_EQ(lm_head, &*resolved.lm_head);
    EXPECT_NE(lm_head->data, resolved.embed_tokens.data);
}

TEST(WeightBindingResolver, TiedLmHeadReusesEmbedTokensBacking) {
    auto storage = std::make_shared<TestStorage>(256);
    const ResolvedModelWeights resolved = MakeResolved(storage, /*with_lm_head=*/false);
    ASSERT_FALSE(resolved.lm_head.has_value());

    const RawWeightView* lm_head =
            ResolveWeightBinding(RoleBinding(std::nullopt, TransformerWeightRole::kLmHead),
                                 resolved);
    ASSERT_NE(lm_head, nullptr);
    EXPECT_EQ(lm_head, &resolved.embed_tokens);
    EXPECT_EQ(lm_head->data, resolved.embed_tokens.data);
    EXPECT_EQ(lm_head->storage, resolved.embed_tokens.storage);
}

TEST(WeightBindingResolver, LayerScopedRolesResolveWithinTheIndexedLayer) {
    auto storage = std::make_shared<TestStorage>(256);
    const ResolvedModelWeights resolved = MakeResolved(storage, /*with_lm_head=*/true);
    const DecoderLayerRawWeights& layer = resolved.layers[1];

    EXPECT_EQ(ResolveWeightBinding(RoleBinding(1, TransformerWeightRole::kInputNorm), resolved),
              &layer.norm.input_rmsnorm);
    EXPECT_EQ(
            ResolveWeightBinding(RoleBinding(1, TransformerWeightRole::kPostAttentionNorm),
                                 resolved),
            &layer.norm.post_attn_rmsnorm);
    EXPECT_EQ(ResolveWeightBinding(RoleBinding(1, TransformerWeightRole::kAttentionQ), resolved),
              &layer.attn.q_proj);
    EXPECT_EQ(ResolveWeightBinding(RoleBinding(1, TransformerWeightRole::kAttentionK), resolved),
              &layer.attn.k_proj);
    EXPECT_EQ(ResolveWeightBinding(RoleBinding(1, TransformerWeightRole::kAttentionV), resolved),
              &layer.attn.v_proj);
    EXPECT_EQ(ResolveWeightBinding(RoleBinding(1, TransformerWeightRole::kAttentionO), resolved),
              &layer.attn.o_proj);
    EXPECT_EQ(ResolveWeightBinding(RoleBinding(1, TransformerWeightRole::kMlpGate), resolved),
              &layer.mlp.gate_proj);
    EXPECT_EQ(ResolveWeightBinding(RoleBinding(1, TransformerWeightRole::kMlpUp), resolved),
              &layer.mlp.up_proj);
    EXPECT_EQ(ResolveWeightBinding(RoleBinding(1, TransformerWeightRole::kMlpDown), resolved),
              &layer.mlp.down_proj);
}

TEST(WeightBindingResolver, LayerIndexSelectsTheRequestedLayer) {
    auto storage = std::make_shared<TestStorage>(256);
    const ResolvedModelWeights resolved = MakeResolved(storage, /*with_lm_head=*/true);

    const RawWeightView* layer0 =
            ResolveWeightBinding(RoleBinding(0, TransformerWeightRole::kAttentionQ), resolved);
    const RawWeightView* layer1 =
            ResolveWeightBinding(RoleBinding(1, TransformerWeightRole::kAttentionQ), resolved);
    ASSERT_NE(layer0, nullptr);
    ASSERT_NE(layer1, nullptr);
    EXPECT_EQ(layer0, &resolved.layers[0].attn.q_proj);
    EXPECT_NE(layer0, layer1);
}

TEST(WeightBindingResolver, OutOfRangeLayerIndexReturnsNull) {
    auto storage = std::make_shared<TestStorage>(256);
    const ResolvedModelWeights resolved = MakeResolved(storage, /*with_lm_head=*/true);
    ASSERT_EQ(resolved.layers.size(), 2U);

    for (const uint32_t layer: {2U, 3U, 1000U}) {
        EXPECT_EQ(ResolveWeightBinding(RoleBinding(layer, TransformerWeightRole::kAttentionQ),
                                       resolved),
                  nullptr)
                << "layer " << layer;
        EXPECT_EQ(ResolveWeightBinding(RoleBinding(layer, TransformerWeightRole::kInputNorm),
                                       resolved),
                  nullptr)
                << "layer " << layer;
    }
}

TEST(WeightBindingResolver, MissingLayerIndexReturnsNullForLayerScopedRoles) {
    auto storage = std::make_shared<TestStorage>(256);
    const ResolvedModelWeights resolved = MakeResolved(storage, /*with_lm_head=*/true);

    // ModelGraph::Validate rejects a layer-scoped role without a layer index,
    // so the resolver must report null instead of silently binding layer 0.
    EXPECT_EQ(
            ResolveWeightBinding(RoleBinding(std::nullopt, TransformerWeightRole::kAttentionQ),
                                 resolved),
            nullptr);
    EXPECT_EQ(ResolveWeightBinding(RoleBinding(std::nullopt, TransformerWeightRole::kMlpDown),
                                   resolved),
              nullptr);
    EXPECT_EQ(ResolveWeightBinding(RoleBinding(std::nullopt, TransformerWeightRole::kInputNorm),
                                   resolved),
              nullptr);
    EXPECT_EQ(
            ResolveWeightBinding(RoleBinding(std::nullopt, TransformerWeightRole::kPostAttentionNorm),
                                 resolved),
            nullptr);
}

TEST(WeightBindingResolver, ModelLevelRolesIgnoreLayerIndex) {
    auto storage = std::make_shared<TestStorage>(256);
    const ResolvedModelWeights resolved = MakeResolved(storage, /*with_lm_head=*/true);

    // Layer-scope validation belongs to ModelGraph::Validate; resolution of a
    // model-level role never depends on the layer index.
    EXPECT_EQ(ResolveWeightBinding(RoleBinding(7, TransformerWeightRole::kTokenEmbedding), resolved),
              &resolved.embed_tokens);
    EXPECT_EQ(ResolveWeightBinding(RoleBinding(7, TransformerWeightRole::kFinalNorm), resolved),
              &resolved.final_norm);
}

TEST(WeightBindingResolver, MoeRouterHasNoDenseStorage) {
    auto storage = std::make_shared<TestStorage>(256);
    const ResolvedModelWeights resolved = MakeResolved(storage, /*with_lm_head=*/true);

    EXPECT_EQ(ResolveWeightBinding(RoleBinding(0, TransformerWeightRole::kMoERouter), resolved),
              nullptr);
}

TEST(WeightBindingResolver, CompositeBindingsHaveNoSingleResolution) {
    auto storage = std::make_shared<TestStorage>(256);
    const ResolvedModelWeights resolved = MakeResolved(storage, /*with_lm_head=*/true);

    EXPECT_EQ(ResolveWeightBinding(MakeQkvWeightBinding(0), resolved), nullptr);
    EXPECT_EQ(ResolveWeightBinding(MakeGateUpWeightBinding(0), resolved), nullptr);
}

TEST(WeightBindingResolver, RolelessDirectBindingReturnsNull) {
    auto storage = std::make_shared<TestStorage>(256);
    const ResolvedModelWeights resolved = MakeResolved(storage, /*with_lm_head=*/true);

    EXPECT_EQ(ResolveWeightBinding(MakeDirectWeightBinding(ParameterSlot::kKernel), resolved),
              nullptr);
}

TEST(WeightBindingResolver, EmptyLayerListRejectsLayerScopedRoles) {
    const ResolvedModelWeights resolved;

    EXPECT_EQ(resolved.layers.size(), 0U);
    EXPECT_EQ(ResolveWeightBinding(RoleBinding(0, TransformerWeightRole::kAttentionQ), resolved),
              nullptr);
    EXPECT_EQ(ResolveWeightBinding(RoleBinding(0, TransformerWeightRole::kInputNorm), resolved),
              nullptr);
}
void FreeTestBuffer(void*, void* ptr) noexcept {
    std::free(ptr);
}

Buffer MakeTestBuffer(size_t nbytes, size_t alignment = 64) {
    void* ptr = nullptr;
    const int rc = posix_memalign(&ptr, alignment, nbytes == 0 ? 1 : nbytes);
    if (rc != 0 || ptr == nullptr) {
        return {};
    }
    return Buffer{nbytes, MemoryHandle(ptr, nullptr, &FreeTestBuffer, Device::CPU(), alignment)};
}

KernelSelector MakePackedCpuSelector() {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
            .phase = ExecPhase::kBoth,
    };
}

class CountingPackedWeights final : public PackedWeights {
public:
    CountingPackedWeights(OpType op_type,
                          KernelSelector selector,
                          Buffer storage,
                          bool* destroyed_flag,
                          PackingRecipe recipe = {}) noexcept
        : op_type_(op_type),
          selector_(selector),
          storage_(std::move(storage)),
          destroyed_flag_(destroyed_flag),
          recipe_(std::move(recipe)) {}

    ~CountingPackedWeights() override {
        if (destroyed_flag_ != nullptr) {
            *destroyed_flag_ = true;
        }
    }

    OpType op_type() const noexcept override {
        return op_type_;
    }

    const KernelSelector& selector() const noexcept override {
        return selector_;
    }

    const Buffer& storage() const noexcept override {
        return storage_;
    }

    const PackingRecipe& recipe() const noexcept override {
        return recipe_;
    }

    DataType logical_dtype() const noexcept override {
        return {};
    }

    const std::vector<int64_t>& logical_shape() const noexcept override {
        return logical_shape_;
    }

private:
    OpType op_type_ = OpType::kUnknown;
    KernelSelector selector_{};
    Buffer storage_{};
    bool* destroyed_flag_ = nullptr;
    PackingRecipe recipe_{};
    std::vector<int64_t> logical_shape_{};
};
TEST(PackedWeightStoreOwnership, StoreOwnsPackedWeightsUntilItIsDestroyed) {
    bool destroyed = false;
    const KernelSelector selector = MakePackedCpuSelector();

    {
        PackedWeightStore packed_weight_store;
        auto packed = std::make_unique<CountingPackedWeights>(
                OpType::kLinear,
                selector,
                MakeTestBuffer(256),
                &destroyed);
        const PackedWeights* raw_ptr = packed.get();

        const WeightArtifactKey key{
                .binding = MakeTransformerWeightBinding(0, TransformerWeightRole::kAttentionQ),
                .selector = selector};
        ASSERT_TRUE(packed_weight_store.Store(
                                               key, std::shared_ptr<const PackedWeights>(std::move(packed)))
                            .ok());
        EXPECT_FALSE(destroyed);

        const auto found = packed_weight_store.Find(key);
        ASSERT_NE(found, nullptr);
        EXPECT_EQ(found.get(), raw_ptr);
        EXPECT_TRUE(found->storage().is_initialized());
    }

    EXPECT_TRUE(destroyed);
}

TEST(PackedWeightStoreOwnership, StoredPackedWeightsOutliveBackendInstance) {
    bool destroyed = false;
    const KernelSelector selector = MakePackedCpuSelector();

    PackedWeightStore packed_weight_store;
    {
        CpuBackend backend;
        (void) backend;

        auto packed = std::make_unique<CountingPackedWeights>(
                OpType::kLinear,
                selector,
                MakeTestBuffer(128),
                &destroyed);
        const WeightArtifactKey key{
                .binding = MakeTransformerWeightBinding(0, TransformerWeightRole::kAttentionQ),
                .selector = selector};
        ASSERT_TRUE(packed_weight_store.Store(
                                               key, std::shared_ptr<const PackedWeights>(std::move(packed)))
                            .ok());
    }

    const WeightArtifactKey key{
            .binding = MakeTransformerWeightBinding(0, TransformerWeightRole::kAttentionQ),
            .selector = selector};
    const auto found = packed_weight_store.Find(key);
    ASSERT_NE(found, nullptr);
    EXPECT_FALSE(destroyed);
    EXPECT_TRUE(found->storage().device().is_cpu());
}

TEST(PackedWeightStoreOwnership, StoreRejectsDuplicatePackedWeightEntries) {
    PackedWeightStore packed_weight_store;
    const KernelSelector selector = MakePackedCpuSelector();
    const WeightArtifactKey key{
            .binding = MakeTransformerWeightBinding(0, TransformerWeightRole::kAttentionQ),
            .selector = selector};

    ASSERT_TRUE(packed_weight_store
                        .Store(key, std::make_shared<CountingPackedWeights>(
                                            OpType::kLinear, selector,
                                            MakeTestBuffer(64), nullptr))
                        .ok());

    const Status duplicate_status = packed_weight_store.Store(
            key, std::make_shared<CountingPackedWeights>(
                         OpType::kLinear, selector, MakeTestBuffer(64), nullptr));

    ASSERT_FALSE(duplicate_status.ok());
    EXPECT_EQ(duplicate_status.code(), StatusCode::kAlreadyExists);
}

TEST(PackedWeightStoreOwnership, DistinctRecipesCoexistForSameBindingAndSelector) {
    PackedWeightStore packed_weight_store;
    const KernelSelector selector = MakePackedCpuSelector();
    const WeightBinding binding =
            MakeTransformerWeightBinding(0, TransformerWeightRole::kAttentionQ);
    const WeightArtifactKey base_key{.binding = binding, .selector = selector};
    const PackingRecipe recipe_a{.layout = "recipe_a", .alignment = 16};
    const PackingRecipe recipe_b{.layout = "recipe_b", .alignment = 32};

    // Two packing variants of the same logical weight coexist: the recipe
    // discriminates artifacts within one {binding, selector}.
    ASSERT_TRUE(packed_weight_store
                        .Store(WeightArtifactKey{.binding = binding,
                                                 .selector = selector,
                                                 .recipe = recipe_a},
                               std::make_shared<CountingPackedWeights>(
                                       OpType::kLinear, selector,
                                       MakeTestBuffer(16), nullptr, recipe_a))
                        .ok());
    ASSERT_TRUE(packed_weight_store
                        .Store(WeightArtifactKey{.binding = binding,
                                                 .selector = selector,
                                                 .recipe = recipe_b},
                               std::make_shared<CountingPackedWeights>(
                                       OpType::kLinear, selector,
                                       MakeTestBuffer(16), nullptr, recipe_b))
                        .ok());
    ASSERT_EQ(packed_weight_store.size(), 2U);
    ASSERT_NE(packed_weight_store.Find(
                      WeightArtifactKey{.binding = binding,
                                        .selector = selector,
                                        .recipe = recipe_a}),
              nullptr);
    ASSERT_NE(packed_weight_store.Find(
                      WeightArtifactKey{.binding = binding,
                                        .selector = selector,
                                        .recipe = recipe_b}),
              nullptr);
}

TEST(PackedWeightStoreOwnership, DistinctBindingsShareSelectorWithoutCollision) {
    PackedWeightStore packed_weight_store;
    const KernelSelector selector = MakePackedCpuSelector();
    const WeightArtifactKey q_key{
            .binding = MakeTransformerWeightBinding(0, TransformerWeightRole::kAttentionQ),
            .selector = selector};
    const WeightArtifactKey v_key{
            .binding = MakeTransformerWeightBinding(2, TransformerWeightRole::kAttentionV),
            .selector = selector};

    ASSERT_TRUE(packed_weight_store
                        .Store(q_key, std::make_shared<CountingPackedWeights>(
                                              OpType::kLinear, selector,
                                              MakeTestBuffer(64), nullptr))
                        .ok());
    ASSERT_TRUE(packed_weight_store
                        .Store(v_key, std::make_shared<CountingPackedWeights>(
                                              OpType::kLinear, selector,
                                              MakeTestBuffer(64), nullptr))
                        .ok());

    ASSERT_EQ(packed_weight_store.size(), 2U);
    ASSERT_NE(packed_weight_store.Find(q_key), nullptr);
    ASSERT_NE(packed_weight_store.Find(v_key), nullptr);
    EXPECT_NE(packed_weight_store.Find(q_key), packed_weight_store.Find(v_key));
}
} // namespace
