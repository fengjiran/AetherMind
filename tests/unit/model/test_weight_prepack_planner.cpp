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

#include <cstddef>
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

}// namespace
