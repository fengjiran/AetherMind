#include "aethermind/model/weight_prepack_planner.h"

#include "aethermind/backend/backend.h"
#include "aethermind/backend/backend_factory.h"
#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/cpu/cpu_weight_prepacker.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/base/device.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/compiler/graph_lowering.h"
#include "aethermind/compiler/packing_request_builder.h"
#include "aethermind/execution/execution_bindings.h"
#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/execution/executor.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/graph/graph.h"
#include "aethermind/model/loaded_model.h"
#include "aethermind/model/packed_weight_store.h"
#include "aethermind/operators/ops/embedding_op.h"
#include "aethermind/operators/ops/rmsnorm_op.h"
#include "aethermind/runtime/runtime_builder.h"
#include "aethermind/shape_inference/tensor_spec.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace {

using namespace aethermind;

struct TestStorage : RawStorage {
    explicit TestStorage(size_t nbytes) : data(nbytes) {}
    std::vector<std::byte> data;
};

HfModelConfig MakeLlamaConfig(int64_t num_layers) {
    return HfModelConfig{
            .model_type = "llama",
            .architectures = {"LlamaForCausalLM"},
            .hidden_size = 64,
            .intermediate_size = 256,
            .num_hidden_layers = num_layers,
            .num_attention_heads = 8,
            .num_key_value_heads = 4,
            .vocab_size = 1000,
            .rms_norm_eps = 1e-6,
            .tie_word_embeddings = false,
    };
}

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

DecoderLayerRawWeights MakeTestLayer(const std::shared_ptr<TestStorage>& storage,
                                     size_t base_offset) {
    DecoderLayerRawWeights layer;
    layer.attn.q_proj = MakeWeightView(storage, base_offset + 0, 8, DataType::Float32(), {2, 1});
    layer.attn.k_proj = MakeWeightView(storage, base_offset + 8, 8, DataType::Float32(), {2, 1});
    layer.attn.v_proj = MakeWeightView(storage, base_offset + 16, 8, DataType::Float32(), {2, 1});
    layer.attn.o_proj = MakeWeightView(storage, base_offset + 24, 8, DataType::Float32(), {2, 1});
    layer.mlp.gate_proj = MakeWeightView(storage, base_offset + 32, 8, DataType::Float32(), {2, 1});
    layer.mlp.up_proj = MakeWeightView(storage, base_offset + 40, 8, DataType::Float32(), {2, 1});
    layer.mlp.down_proj = MakeWeightView(storage, base_offset + 48, 8, DataType::Float32(), {2, 1});
    layer.norm.input_rmsnorm = MakeWeightView(storage, base_offset + 56, 8, DataType::Float32(), {2, 1});
    layer.norm.post_attn_rmsnorm = MakeWeightView(storage, base_offset + 64, 8, DataType::Float32(), {2, 1});
    return layer;
}

KernelSelector MakeExpectedSelector() {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
            .isa = IsaLevel::kAVX2,
            .phase = ExecPhase::kBoth,
    };
}

TEST(WeightPrepackPlanner, BuildRequestsEnumeratesAllLinearWeightsPerLayer) {
    auto storage = std::make_shared<TestStorage>(256);

    ResolvedModelWeights index;
    index.embed_tokens = MakeWeightView(storage, 0, 8, DataType::Float32(), {2, 1});
    index.final_norm = MakeWeightView(storage, 8, 8, DataType::Float32(), {2, 1});
    index.layers.push_back(MakeTestLayer(storage, 16));
    index.layers.push_back(MakeTestLayer(storage, 100));

    CpuBackend backend;
    KernelRegistry registry;
    auto requests = WeightPrepackPlanner::BuildRequests(
            MakeLlamaConfig(2), index, backend, registry);

    ASSERT_TRUE(requests.ok());
    // 2 layers × 7 linear weights = 14 requests.
    EXPECT_EQ(requests->size(), 14);

    // All requests should be kLinear with the expected packed selector.
    for (const auto& req: *requests) {
        EXPECT_EQ(req.op_type, OpType::kLinear);
        EXPECT_EQ(req.selector, MakeExpectedSelector());
    }
}

TEST(WeightPrepackPlanner, BuildRequestsExcludesNormsAndEmbeddings) {
    auto storage = std::make_shared<TestStorage>(256);
    ResolvedModelWeights index;
    index.embed_tokens = MakeWeightView(storage, 0, 8, DataType::Float32(), {2, 1});
    index.final_norm = MakeWeightView(storage, 8, 8, DataType::Float32(), {2, 1});
    index.layers.push_back(MakeTestLayer(storage, 16));

    CpuBackend backend;
    KernelRegistry registry;
    auto requests = WeightPrepackPlanner::BuildRequests(
            MakeLlamaConfig(1), index, backend, registry);

    ASSERT_TRUE(requests.ok());
    for (const auto& req: *requests) {
        EXPECT_NE(req.raw_weight.data, index.embed_tokens.data);
        EXPECT_NE(req.raw_weight.data, index.final_norm.data);
        EXPECT_NE(req.raw_weight.data, index.layers[0].norm.input_rmsnorm.data);
        EXPECT_NE(req.raw_weight.data, index.layers[0].norm.post_attn_rmsnorm.data);
    }
}

TEST(WeightPrepackPlanner, BuildRequestsIncludesLmHeadWhenPresent) {
    auto storage = std::make_shared<TestStorage>(256);
    ResolvedModelWeights index;
    index.embed_tokens = MakeWeightView(storage, 0, 8, DataType::Float32(), {2, 1});
    index.final_norm = MakeWeightView(storage, 8, 8, DataType::Float32(), {2, 1});
    index.lm_head = MakeWeightView(storage, 16, 8, DataType::Float32(), {2, 1});
    index.layers.push_back(MakeTestLayer(storage, 24));

    CpuBackend backend;
    KernelRegistry registry;
    auto requests = WeightPrepackPlanner::BuildRequests(
            MakeLlamaConfig(1), index, backend, registry);

    ASSERT_TRUE(requests.ok());
    // 1 layer × 7 + lm_head = 8.
    EXPECT_EQ(requests->size(), 8);

    bool found_lm_head = false;
    for (const auto& req: *requests) {
        if (req.raw_weight.data == index.lm_head->data) {
            found_lm_head = true;
            break;
        }
    }
    EXPECT_TRUE(found_lm_head);
}

TEST(WeightPrepackPlanner, PrepackAndStoreMakesWeightsFindable) {
    auto storage = std::make_shared<TestStorage>(256);
    // Fill with zeros so Pack can safely memcpy.
    for (auto& b: storage->data) b = std::byte{0};

    ResolvedModelWeights index;
    index.embed_tokens = MakeWeightView(storage, 0, 8, DataType::Float32(), {2, 1});
    index.final_norm = MakeWeightView(storage, 8, 8, DataType::Float32(), {2, 1});
    index.layers.push_back(MakeTestLayer(storage, 16));

    LoadedModel loaded_model(MakeLlamaConfig(1), std::move(index));
    PackedWeightStore packed_weight_store;

    CpuBackend backend;
    KernelRegistry registry;
    auto requests = WeightPrepackPlanner::BuildRequests(
            loaded_model.GetConfig(), loaded_model.GetResolvedWeights(), backend, registry);
    ASSERT_TRUE(requests.ok());

    Status status = WeightPrepackPlanner::PrepackAndStore(packed_weight_store, *requests);
    ASSERT_TRUE(status.ok());

    const KernelSelector expected_selector = MakeExpectedSelector();
    const WeightArtifactKey key{.binding = requests->front().binding,
                                .selector = requests->front().selector,
                                .recipe = CpuWeightPrepacker::RecipeFor(
                                        requests->front().selector)};
    const auto found = packed_weight_store.Find(key);
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->op_type(), OpType::kLinear);
    EXPECT_EQ(found->selector(), expected_selector);
    EXPECT_TRUE(found->storage().is_initialized());
}

TEST(WeightPrepackPlanner, PrepackAndStoreStoresAllLayerWeightsDistinctly) {
    auto storage = std::make_shared<TestStorage>(256);
    for (auto& b: storage->data) b = std::byte{0};

    ResolvedModelWeights index;
    index.embed_tokens = MakeWeightView(storage, 0, 8, DataType::Float32(), {2, 1});
    index.final_norm = MakeWeightView(storage, 8, 8, DataType::Float32(), {2, 1});
    // Two layers — all linear weights share the same (op_type, selector) but
    // distinct bindings. Every weight must be packed, not silently dropped.
    index.layers.push_back(MakeTestLayer(storage, 16));
    index.layers.push_back(MakeTestLayer(storage, 100));

    LoadedModel loaded_model(MakeLlamaConfig(2), std::move(index));
    PackedWeightStore packed_weight_store;

    CpuBackend backend;
    KernelRegistry registry;
    auto requests = WeightPrepackPlanner::BuildRequests(
            loaded_model.GetConfig(), loaded_model.GetResolvedWeights(), backend, registry);
    ASSERT_TRUE(requests.ok());
    EXPECT_EQ(requests->size(), 14);

    Status status = WeightPrepackPlanner::PrepackAndStore(packed_weight_store, *requests);
    ASSERT_TRUE(status.ok());

    // All 14 distinct keys are stored; the same role across layers differs by
    // its layer index and every role is individually findable.
    EXPECT_EQ(packed_weight_store.size(), 14U);
    for (const auto& req: *requests) {
        const WeightArtifactKey key{.binding = req.binding,
                                    .selector = req.selector,
                                    .recipe = CpuWeightPrepacker::RecipeFor(
                                            req.selector)};
        EXPECT_NE(packed_weight_store.Find(key), nullptr) << "missing key for layer";
    }
}

TEST(WeightPrepackPlanner, RawViewsRemainAccessibleAfterPrepack) {
    auto storage = std::make_shared<TestStorage>(256);
    for (auto& b: storage->data) b = std::byte{0};

    ResolvedModelWeights index;
    index.embed_tokens = MakeWeightView(storage, 0, 8, DataType::Float32(), {2, 1});
    index.final_norm = MakeWeightView(storage, 8, 8, DataType::Float32(), {2, 1});
    index.layers.push_back(MakeTestLayer(storage, 16));

    LoadedModel loaded_model(MakeLlamaConfig(1), std::move(index));
    PackedWeightStore packed_weight_store;

    CpuBackend backend;
    KernelRegistry registry;
    auto requests = WeightPrepackPlanner::BuildRequests(
            loaded_model.GetConfig(), loaded_model.GetResolvedWeights(), backend, registry);
    ASSERT_TRUE(requests.ok());

    ASSERT_TRUE(WeightPrepackPlanner::PrepackAndStore(packed_weight_store, *requests).ok());

    const auto& resolved_weights = loaded_model.GetResolvedWeights();
    EXPECT_TRUE(resolved_weights.embed_tokens.IsValid());
    EXPECT_TRUE(resolved_weights.final_norm.IsValid());
    EXPECT_TRUE(resolved_weights.layers[0].attn.q_proj.IsValid());
    EXPECT_TRUE(resolved_weights.layers[0].mlp.down_proj.IsValid());
    EXPECT_TRUE(resolved_weights.layers[0].norm.input_rmsnorm.IsValid());
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

    const BackendCapabilities& capabilities() const noexcept override {
        return capabilities_;
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
                .debug_name = "test::planner_packed_kernel",
                .expected_packing_recipe = CpuWeightPrepacker::RecipeFor(selector),
        };
    }

    const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override {
        return nullptr;
    }

private:
    BackendCapabilities capabilities_{};
};

class PlannerPackedTestBackendFactory final : public BackendFactory {
public:
    DeviceType device_type() const noexcept override {
        return DeviceType::kCPU;
    }

    std::unique_ptr<Backend> Create() const override {
        return std::make_unique<PlannerPackedTestBackend>();
    }
};

// End-to-end identity loop: lowering marks kWeight steps packed, the
// lower-driven request builder derives key material from the artifact, the
// prepacker stores artifacts under {source, value, binding, selector, recipe},
// and the plan builder resolves each step to its own artifact.
TEST(WeightPrepackPlanner, LoweredDrivenPrepackAndResolve) {
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
    ASSERT_EQ(requests->size(), 3U);
    for (const auto& req: *requests) {
        EXPECT_EQ(req.source_id, lowered->artifact_id());
        EXPECT_EQ(req.selector.weight_format, WeightFormat::kPacked);
    }
    // Requests appear in lowered order: embedding value, norm0 value, norm1.
    EXPECT_EQ((*requests)[0].value_index, embedding_weight.index);
    EXPECT_EQ((*requests)[1].value_index, norm0_weight.index);
    EXPECT_EQ((*requests)[2].value_index, norm1_weight.index);

    PackedWeightStore packed_weight_store;
    ASSERT_TRUE(WeightPrepackPlanner::PrepackAndStore(
                        packed_weight_store, *requests)
                        .ok());
    ASSERT_EQ(packed_weight_store.size(), 3U);
    EXPECT_EQ(packed_weight_store.source_id(), lowered->artifact_id());

    RuntimeBuilder builder;
    builder.RegisterBackendFactory(
            DeviceType::kCPU, std::make_unique<PlannerPackedTestBackendFactory>());
    RuntimeContext runtime = builder.Build();
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

    auto table = BuildExecutionBindings(
            *plan,
            ExternalValueBindings{.readable = std::move(readable)},
            runtime.GetAllocator(Device::CPU()));
    ASSERT_TRUE(table.ok()) << table.status().ToString();
    RuntimeBindingContext context;
    context.SetBindingTable(std::move(*table));
    g_planner_packed_kernel_calls = 0;
    const Status status = Executor::Execute(*plan, context);
    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_EQ(g_planner_packed_kernel_calls, 3);
}

TEST(WeightPrepackPlanner, LoweredDrivenPrepackResolvesCompositeBindings) {
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
    // Embedding value + fused QKV weight + fused Gate-Up weight.
    ASSERT_EQ(requests->size(), 3U);

    auto qkv_request = std::find_if(
            requests->begin(), requests->end(),
            [](const WeightPrepackPlanner::Request& req) {
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
            [](const WeightPrepackPlanner::Request& req) {
                return req.op_type == OpType::kGateUpLinear;
            });
    ASSERT_NE(gate_up_request, requests->end());
    EXPECT_TRUE(std::holds_alternative<GateUpWeightBinding>(gate_up_request->binding.spec));
    ASSERT_EQ(gate_up_request->components.size(), 2U);
    EXPECT_EQ(gate_up_request->components[0].data, mlp.gate_proj.data);
    EXPECT_EQ(gate_up_request->components[1].data, mlp.up_proj.data);

    auto embedding_request = std::find_if(
            requests->begin(), requests->end(),
            [](const WeightPrepackPlanner::Request& req) {
                return req.op_type == OpType::kEmbedding;
            });
    ASSERT_NE(embedding_request, requests->end());
    EXPECT_TRUE(embedding_request->components.empty());
    EXPECT_EQ(embedding_request->raw_weight.data, resolved.embed_tokens.data);

    PackedWeightStore packed_weight_store;
    ASSERT_TRUE(WeightPrepackPlanner::PrepackAndStore(
                        packed_weight_store, *requests)
                        .ok());
    ASSERT_EQ(packed_weight_store.size(), 3U);
    EXPECT_EQ(packed_weight_store.source_id(), lowered->artifact_id());

    // Stored fused artifacts carry the fused logical shape and exactly the
    // recipe-ordered concatenation of their components.
    const auto expect_fused = [&](const WeightPrepackPlanner::Request& req) {
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
    RuntimeContext runtime = builder.Build();
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

Status PrepackSingleRequest(const WeightPrepackPlanner::Request& request) {
    PackedWeightStore store;
    return WeightPrepackPlanner::PrepackAndStore(store, {request});
}

// RawWeightView::bytes must exactly match shape × dtype byte size: the
// prepacker copies the shape-derived logical size, so any mismatch would
// read out of bounds or corrupt fused layouts. These cases must be rejected
// eagerly at the PrepackAndStore boundary.
TEST(WeightPrepackPlanner, PrepackAndStoreRejectsDirectWeightWithUndersizedBytes) {
    auto storage = std::make_shared<TestStorage>(64);
    for (auto& b: storage->data) b = std::byte{0};

    const WeightPrepackPlanner::Request request{
            .op_type = OpType::kLinear,
            .source_id = 1,
            .binding = MakeTransformerWeightBinding(0U, TransformerWeightRole::kAttentionQ),
            .raw_weight = MakeWeightView(storage, 0, 4, DataType::Float32(), {2, 1}),
            .selector = MakeExpectedSelector(),
    };

    const Status status = PrepackSingleRequest(request);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message().find("does not match"), std::string::npos);
}

TEST(WeightPrepackPlanner, PrepackAndStoreRejectsDirectWeightWithOversizedBytes) {
    auto storage = std::make_shared<TestStorage>(64);
    for (auto& b: storage->data) b = std::byte{0};

    const WeightPrepackPlanner::Request request{
            .op_type = OpType::kLinear,
            .source_id = 1,
            .binding = MakeTransformerWeightBinding(0U, TransformerWeightRole::kAttentionQ),
            .raw_weight = MakeWeightView(storage, 0, 12, DataType::Float32(), {2, 1}),
            .selector = MakeExpectedSelector(),
    };

    const Status status = PrepackSingleRequest(request);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message().find("does not match"), std::string::npos);
}

TEST(WeightPrepackPlanner, PrepackAndStoreRejectsCompositeComponentWithUndersizedBytes) {
    auto storage = std::make_shared<TestStorage>(64);
    for (auto& b: storage->data) b = std::byte{0};

    const WeightPrepackPlanner::Request request{
            .op_type = OpType::kQkvLinear,
            .source_id = 1,
            .binding = MakeQkvWeightBinding(0U),
            .components = {
                    MakeWeightView(storage, 0, 8, DataType::Float32(), {2, 1}),
                    MakeWeightView(storage, 16, 4, DataType::Float32(), {2, 1}),
                    MakeWeightView(storage, 24, 8, DataType::Float32(), {2, 1}),
            },
            .selector = MakeExpectedSelector(),
    };

    const Status status = PrepackSingleRequest(request);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message().find("does not match"), std::string::npos);
}

TEST(WeightPrepackPlanner, PrepackAndStoreRejectsNegativeWeightDimension) {
    auto storage = std::make_shared<TestStorage>(64);
    for (auto& b: storage->data) b = std::byte{0};

    const WeightPrepackPlanner::Request request{
            .op_type = OpType::kLinear,
            .source_id = 1,
            .binding = MakeTransformerWeightBinding(0U, TransformerWeightRole::kAttentionQ),
            .raw_weight = MakeWeightView(storage, 0, 0, DataType::Float32(), {-3}),
            .selector = MakeExpectedSelector(),
    };

    const Status status = PrepackSingleRequest(request);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message().find("negative dimension"), std::string::npos);
}

TEST(WeightPrepackPlanner, PrepackAndStoreRejectsOverflowingWeightByteSize) {
    auto storage = std::make_shared<TestStorage>(64);
    for (auto& b: storage->data) b = std::byte{0};

    const WeightPrepackPlanner::Request request{
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
    };

    const Status status = PrepackSingleRequest(request);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message().find("overflows"), std::string::npos);
}

// Tied embeddings: a checkpoint without an independent lm_head reuses
// embed_tokens. The request builder must mirror ModelGraphBuilder and fall
// back to embed_tokens for the kLmHead binding, or graph-driven
// materialization fails for common tied models.
TEST(WeightPrepackPlanner, BuildRequestsFallBackToEmbedTokensForTiedLmHead) {
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
// Plain (non-packed) weight steps must not produce packing requests: the
// planner packs only kPacked selectors, and feeding it plain steps would
// fail inside CpuWeightPrepacker.
TEST(WeightPrepackPlanner, BuildRequestsSkipsPlainWeightSteps) {
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

    GraphLoweringConfig config;// enable_packed_weights defaults to false.
    const auto lowered = LowerModelGraph(graph, config);
    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();

    auto requests = BuildWeightPackingRequests(*lowered, ResolvedModelWeights{});
    ASSERT_TRUE(requests.ok()) << requests.status().ToString();
    EXPECT_TRUE(requests->empty());
}

// One weight value consumed by several steps with the same selector packs
// exactly once; duplicate requests would collide on the exact artifact key
// and fail with AlreadyExists during Store.
TEST(WeightPrepackPlanner, BuildRequestsDeduplicatesSharedWeightValue) {
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
    ASSERT_EQ(requests->size(), 1U);
    EXPECT_EQ((*requests)[0].value_index, weight.index);
    EXPECT_EQ((*requests)[0].op_type, OpType::kLinear);

    PackedWeightStore store;
    ASSERT_TRUE(WeightPrepackPlanner::PrepackAndStore(store, *requests).ok());
    EXPECT_EQ(store.size(), 1U);
    EXPECT_EQ(store.source_id(), lowered->artifact_id());
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
TEST(WeightPrepackPlanner, UntrustedBuildBindsDistinctPackedArtifacts) {
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
    RuntimeContext runtime = builder.Build();
    const auto plan = ExecutionPlanBuilder::Build(runtime, store, nodes);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->size(), 2U);
    ASSERT_NE(plan->steps()[0].packed_weights, nullptr);
    ASSERT_NE(plan->steps()[1].packed_weights, nullptr);
    EXPECT_NE(plan->steps()[0].packed_weights, plan->steps()[1].packed_weights);
}

TEST(WeightPrepackPlanner, UntrustedBuildRejectsArtifactOpTypeMismatch) {
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
    RuntimeContext runtime = builder.Build();
    const auto plan = ExecutionPlanBuilder::Build(runtime, store, nodes);
    ASSERT_FALSE(plan.ok());
    EXPECT_NE(plan.status().message().find("op type"), std::string::npos);
}

TEST(WeightPrepackPlanner, UntrustedBuildRejectsArtifactShapeMismatch) {
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
    RuntimeContext runtime = builder.Build();
    const auto plan = ExecutionPlanBuilder::Build(runtime, store, nodes);
    ASSERT_FALSE(plan.ok());
    EXPECT_NE(plan.status().message().find("logical metadata"), std::string::npos);
}
}// namespace