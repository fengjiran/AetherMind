#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/compiler/graph_lowering.h"
#include "aethermind/inference/executable_model.h"
#include "aethermind/inference/inference_session.h"
#include "aethermind/model/model_loader.h"
#include "aethermind/runtime/runtime_builder.h"
#include "test_utils/test_resource_allocator.h"

#include "aethermind/graph/graph.h"
#include <algorithm>
#include <array>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>

namespace {
using namespace aethermind;
std::filesystem::path ResourceModelDir() { return std::filesystem::path(AETHERMIND_TEST_MODELS_DIR) / "tiny-random-LlamaForCausalLM"; }
ModelCompileOptions ResourceCompile(bool packed = false) {
    ModelCompileOptions options;
    options.optimization.opt_level = packed ? 2 : 1;
    options.lowering.enable_packed_weights = packed;
    return options;
}
class ScratchBackend final : public Backend {
public:
    DeviceType device_type() const noexcept override { return DeviceType::kCPU; }
    StatusOr<ResolvedKernel> PrepareKernel(OpType op, const KernelSelector& selector, const OpParams& params) const override {
        auto result = cpu.PrepareKernel(op, selector, params);
        if (result.ok()) result->workspace_requirement = {.bytes = 64, .alignment = 64, .lifetime = WorkspaceLifetime::kPerSequence};
        return result;
    }
    StatusOr<PackingRecipe> GetPackingRecipe(OpType op, const KernelSelector& selector) const override { return cpu.GetPackingRecipe(op, selector); }
    StatusOr<AllocationRequest> GetWeightPackingRequirements(OpType op, std::span<const TensorView> values, const KernelSelector& selector, const PackingRecipe& recipe) const override {
        return cpu.GetWeightPackingRequirements(op, values, selector, recipe);
    }
    StatusOr<std::unique_ptr<PackedWeight>> PackWeights(OpType op, std::span<const TensorView> values, const KernelSelector& selector, const PackingRecipe& recipe, Allocator& allocator) const override {
        return cpu.PackWeights(op, values, selector, recipe, allocator);
    }
    const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override { return cpu.TryGetKernelRegistryForDebug(); }

private:
    CpuBackend cpu;
};
class ScratchFactory final : public BackendFactory {
public:
    DeviceType device_type() const noexcept override { return DeviceType::kCPU; }
    StatusOr<std::unique_ptr<Backend>> Create() const override { return std::unique_ptr<Backend>(std::make_unique<ScratchBackend>()); }
};
Runtime ResourceRuntime(const std::shared_ptr<test::ResourceAllocationStats>& stats,
                        size_t workspace_limit = 0, bool scratch = false) {
    RuntimeOptions options;
    options.kv_cache = {.enable_manager = true, .num_layers = 2, .num_kv_heads = 4, .max_tokens = 8, .head_dim = 4, .kv_dtype = DataType::Float32(), .alignment = 64};
    options.workspace.default_workspace_limit_bytes = workspace_limit;
    RuntimeBuilder builder;
    builder.WithOptions(options);
    builder.RegisterCustomAllocatorProvider(DeviceType::kCPU, std::make_unique<test::ResourceRecordingProvider>(stats));
    if (scratch) builder.RegisterBackendFactory(DeviceType::kCPU, std::make_unique<ScratchFactory>());
    return builder.Build();
}
std::shared_ptr<const ExecutableModel> ResourceModel(Runtime& runtime, bool packed = false) {
    auto result = LoadAndPrepareExecutableModel(runtime, ResourceModelDir(), ResourceCompile(packed));
    if (!result.ok()) {
        ADD_FAILURE() << result.status().ToString();
        return {};
    }
    return std::make_shared<ExecutableModel>(std::move(*result));
}
TEST(InferenceResourceLimits, PlanningDoesNotAllocatePayloadAndMatchesPhasePeak) {
    auto stats = std::make_shared<test::ResourceAllocationStats>();
    Runtime runtime = ResourceRuntime(stats);
    auto model = ResourceModel(runtime);
    ASSERT_NE(model, nullptr);
    auto session = InferenceSession::Create(runtime, model);
    ASSERT_TRUE(session.ok());
    const size_t before = stats->requests.size();
    auto usage = session->PlanResources(3, {.max_new_tokens = 3});
    ASSERT_TRUE(usage.ok()) << usage.status().ToString();
    EXPECT_EQ(stats->requests.size(), before);
    EXPECT_EQ(usage->peak_payload_bytes, usage->workspace_bytes + std::max(usage->prefill_activation_bytes, usage->decode_activation_bytes));
    EXPECT_EQ(usage->kv_reserved_bytes, runtime.GetKVCacheManager()->total_bytes());
    EXPECT_GT(model->resource_usage().weights.mapped_bytes, 0U);
    stats->requests.clear();
    const std::array<uint32_t, 3> prompt{7, 8, 9};
    auto generated = session->Generate(prompt, {.max_new_tokens = 3});
    ASSERT_TRUE(generated.ok());
    ASSERT_EQ(stats->requests.size(), 2U);
    EXPECT_EQ(stats->requests[0].nbytes, usage->prefill_activation_bytes);
    EXPECT_EQ(stats->requests[1].nbytes, usage->decode_activation_bytes);
}
TEST(InferenceResourceLimits, RequestBoundaryRejectsBeforeReservationAndPayloadAllocation) {
    for (bool packed: {false, true}) {
        for (bool scratch: {false, true}) {
            SCOPED_TRACE(testing::Message() << "packed=" << packed << ", scratch=" << scratch);
            auto stats = std::make_shared<test::ResourceAllocationStats>();
            Runtime runtime = ResourceRuntime(stats, 0, scratch);
            auto model = ResourceModel(runtime, packed);
            ASSERT_NE(model, nullptr);
            auto unbounded = InferenceSession::Create(runtime, model);
            ASSERT_TRUE(unbounded.ok());
            auto usage = unbounded->PlanResources(2, {.max_new_tokens = 3});
            ASSERT_TRUE(usage.ok());
            ASSERT_GT(usage->peak_payload_bytes, 1U);
            if (scratch) {
                ASSERT_GT(usage->workspace_bytes, 0U);
                ASSERT_GT(usage->peak_payload_bytes, usage->workspace_bytes);
            }
            const std::array<uint32_t, 2> prompt{7, 8};
            auto expected = unbounded->Generate(prompt, {.max_new_tokens = 3});
            ASSERT_TRUE(expected.ok());
            for (size_t cap: {usage->peak_payload_bytes - 1, usage->peak_payload_bytes, usage->peak_payload_bytes + 1}) {
                auto session = InferenceSession::Create(runtime, model, {.max_request_payload_bytes = cap});
                ASSERT_TRUE(session.ok());
                stats->requests.clear();
                auto generated = session->Generate(prompt, {.max_new_tokens = 3});
                if (cap < usage->peak_payload_bytes) {
                    EXPECT_EQ(generated.status().code(), StatusCode::kResourceExhausted);
                    EXPECT_TRUE(stats->requests.empty());
                    auto free = runtime.GetKVCacheManager()->ReserveForSession(1, 0);
                    ASSERT_TRUE(free.ok());
                    EXPECT_TRUE(runtime.GetKVCacheManager()->ReleaseSession(*free).ok());
                } else {
                    ASSERT_TRUE(generated.ok()) << generated.status().ToString();
                    EXPECT_EQ(*generated, *expected);
                }
            }
        }
    }
}
TEST(InferenceResourceLimits, WorkspaceLimitAppliesWithoutWorkspaceManager) {
    auto first_stats = std::make_shared<test::ResourceAllocationStats>();
    Runtime first = ResourceRuntime(first_stats, 0, true);
    auto first_model = ResourceModel(first);
    ASSERT_NE(first_model, nullptr);
    auto first_session = InferenceSession::Create(first, first_model);
    ASSERT_TRUE(first_session.ok());
    auto usage = first_session->PlanResources(1, {.max_new_tokens = 2});
    ASSERT_TRUE(usage.ok());
    ASSERT_GT(usage->workspace_bytes, 0U);
    const std::array<uint32_t, 1> prompt{7};
    for (size_t cap: {usage->workspace_bytes - 1, usage->workspace_bytes}) {
        auto stats = std::make_shared<test::ResourceAllocationStats>();
        Runtime runtime = ResourceRuntime(stats, cap, true);
        auto model = ResourceModel(runtime);
        ASSERT_NE(model, nullptr);
        auto session = InferenceSession::Create(runtime, model);
        ASSERT_TRUE(session.ok());
        stats->requests.clear();
        auto generated = session->Generate(prompt, {.max_new_tokens = 2});
        if (cap < usage->workspace_bytes) {
            EXPECT_EQ(generated.status().code(), StatusCode::kResourceExhausted);
            EXPECT_TRUE(stats->requests.empty());
        } else {
            EXPECT_TRUE(generated.ok()) << generated.status().ToString();
            ASSERT_FALSE(stats->requests.empty());
            EXPECT_EQ(stats->requests.front().nbytes, cap);
        }
    }
}
TEST(InferenceResourceLimits, ZeroGenerationDoesNotApplyBudgetOrPhysicalPromptCapacity) {
    auto stats = std::make_shared<test::ResourceAllocationStats>();
    Runtime runtime = ResourceRuntime(stats, 1, true);
    auto model = ResourceModel(runtime);
    ASSERT_NE(model, nullptr);
    auto session = InferenceSession::Create(runtime, model, {.max_request_payload_bytes = 1});
    ASSERT_TRUE(session.ok());
    std::array<uint32_t, 9> prompt{};
    stats->requests.clear();
    auto result = session->Generate(prompt, {.max_new_tokens = 0});
    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(result->empty());
    EXPECT_TRUE(stats->requests.empty());
    auto usage = session->PlanResources(9, {.max_new_tokens = 0});
    ASSERT_TRUE(usage.ok());
    EXPECT_EQ(usage->peak_payload_bytes, 0U);
}
TEST(InferenceResourceLimits, CapacityAndBusyErrorsPrecedeBudgetRejection) {
    auto stats = std::make_shared<test::ResourceAllocationStats>();
    Runtime runtime = ResourceRuntime(stats);
    auto model = ResourceModel(runtime);
    ASSERT_NE(model, nullptr);
    auto session = InferenceSession::Create(runtime, model, {.max_request_payload_bytes = 1});
    ASSERT_TRUE(session.ok());
    std::array<uint32_t, 9> long_prompt{};
    EXPECT_EQ(session->Generate(long_prompt, {.max_new_tokens = 1}).status().code(), StatusCode::kOutOfRange);
    auto occupied = runtime.GetKVCacheManager()->ReserveForSession(1, 0);
    ASSERT_TRUE(occupied.ok());
    std::array<uint32_t, 1> prompt{7};
    EXPECT_EQ(session->Generate(prompt, {.max_new_tokens = 1}).status().code(), StatusCode::kFailedPrecondition);
    EXPECT_TRUE(occupied->valid());
    EXPECT_TRUE(runtime.GetKVCacheManager()->ReleaseSession(*occupied).ok());
}
TEST(InferenceResourceLimits, BoundedAllocationFailureCanRetryWithoutPublishingTokens) {
    auto stats = std::make_shared<test::ResourceAllocationStats>();
    Runtime runtime = ResourceRuntime(stats);
    auto model = ResourceModel(runtime);
    ASSERT_NE(model, nullptr);
    auto unbounded = InferenceSession::Create(runtime, model);
    ASSERT_TRUE(unbounded.ok());
    auto usage = unbounded->PlanResources(1, {.max_new_tokens = 3});
    ASSERT_TRUE(usage.ok());
    auto session = InferenceSession::Create(runtime, model, {.max_request_payload_bytes = usage->peak_payload_bytes});
    ASSERT_TRUE(session.ok());
    stats->requests.clear();
    stats->fail_on_call = 1;
    std::array<uint32_t, 1> prompt{7};
    EXPECT_EQ(session->Generate(prompt, {.max_new_tokens = 3}).status().code(), StatusCode::kResourceExhausted);
    stats->fail_on_call = std::numeric_limits<size_t>::max();
    EXPECT_TRUE(session->Generate(prompt, {.max_new_tokens = 3}).ok());
}
TEST(ModelResourceLimits, FinitePackedAndModelCapsRejectBeforePacking) {
    auto stats = std::make_shared<test::ResourceAllocationStats>();
    Runtime runtime = ResourceRuntime(stats);
    auto baseline = ResourceModel(runtime, true);
    ASSERT_NE(baseline, nullptr);
    const auto usage = baseline->resource_usage();
    ASSERT_GT(usage.packed_weight_bytes, 1U);
    EXPECT_TRUE(usage.packing_plan_complete);
    EXPECT_TRUE(usage.weights.backing_complete);
    stats->requests.clear();
    auto short_packed = LoadAndPrepareExecutableModel(runtime, ResourceModelDir(), ResourceCompile(true), {.max_packed_weight_bytes = usage.packed_weight_bytes - 1});
    EXPECT_EQ(short_packed.status().code(), StatusCode::kResourceExhausted);
    EXPECT_TRUE(stats->requests.empty());
    auto short_model = LoadAndPrepareExecutableModel(runtime, ResourceModelDir(), ResourceCompile(true), {.max_model_payload_bytes = usage.managed_payload_bytes - 1});
    EXPECT_EQ(short_model.status().code(), StatusCode::kResourceExhausted);
    EXPECT_TRUE(stats->requests.empty());
    auto equal = LoadAndPrepareExecutableModel(runtime, ResourceModelDir(), ResourceCompile(true), {.max_model_payload_bytes = usage.managed_payload_bytes, .max_packed_weight_bytes = usage.packed_weight_bytes});
    ASSERT_TRUE(equal.ok()) << equal.status().ToString();
    EXPECT_EQ(equal->resource_usage().managed_payload_bytes, usage.managed_payload_bytes);
}
TEST(ModelResourceLimits, UnknownSourceOnlyRejectsCorrespondingModelCap) {
    struct Unknown final : RawStorage {
        std::shared_ptr<const RawStorage> backing;
    };
    auto stats = std::make_shared<test::ResourceAllocationStats>();
    Runtime runtime = ResourceRuntime(stats);
    const auto compile = [&]() -> StatusOr<LoweredModelArtifact> {
        AM_ASSIGN_OR_RETURN(auto loaded, ModelLoader::Load(ResourceModelDir()));
        auto weights = loaded->GetResolvedWeights();
        auto unknown = std::make_shared<Unknown>();
        unknown->backing = weights.embed_tokens.storage;
        weights.embed_tokens.storage = unknown;
        return ModelCompiler::Compile(std::make_unique<LoadedModel>(loaded->GetConfig(), std::move(weights)), ResourceCompile());
    };
    auto first = compile();
    ASSERT_TRUE(first.ok());
    stats->requests.clear();
    EXPECT_EQ(PrepareExecutableModel(runtime, std::move(*first), {.max_model_payload_bytes = 10000000}).status().code(), StatusCode::kUnimplemented);
    EXPECT_TRUE(stats->requests.empty());
    auto second = compile();
    ASSERT_TRUE(second.ok());
    auto model = PrepareExecutableModel(runtime, std::move(*second), {.max_packed_weight_bytes = 1});
    ASSERT_TRUE(model.ok()) << model.status().ToString();
    EXPECT_FALSE(model->resource_usage().weights.backing_complete);
}
TEST(ModelResourceLimits, InlineConstantsAreChargedOnceBeforeAnyPacking) {
    auto stats = std::make_shared<test::ResourceAllocationStats>();
    Runtime runtime = ResourceRuntime(stats);
    auto build = []() -> StatusOr<LoweredModelArtifact> {
        ModelGraph graph;
        const std::array<int64_t, 1> dims{4};
        const TensorSpec spec{.dtype = DataType::Float32(), .shape = SymbolicShape(dims)};
        auto data = std::make_shared<const std::vector<std::byte>>(16);
        auto first = graph.AddConstant(spec, ConstantBinding{.inline_data = data});
        graph.MarkOutput(first);
        auto second = graph.AddConstant(spec, ConstantBinding{.inline_data = data});
        graph.MarkOutput(second);
        AM_ASSIGN_OR_RETURN(auto lowered, LowerModelGraph(graph, {}));
        HfModelConfig config;
        config.vocab_size = 4;
        config.max_position_embeddings = 16;
        return LoweredModelArtifact{.loaded_model = std::make_unique<LoadedModel>(config, ResolvedModelWeights{}), .graph = std::move(lowered)};
    };
    auto bad = build();
    ASSERT_TRUE(bad.ok());
    EXPECT_EQ(PrepareExecutableModel(runtime, std::move(*bad), {.max_model_payload_bytes = 15}).status().code(), StatusCode::kResourceExhausted);
    auto good = build();
    ASSERT_TRUE(good.ok());
    auto model = PrepareExecutableModel(runtime, std::move(*good), {.max_model_payload_bytes = 16});
    ASSERT_TRUE(model.ok()) << model.status().ToString();
    EXPECT_EQ(model->resource_usage().inline_constant_bytes, 16U);
}
TEST(ModelResourceLimits, UnlimitedPreparationDoesNotValidateUnusedRawRoles) {
    auto stats = std::make_shared<test::ResourceAllocationStats>();
    Runtime runtime = ResourceRuntime(stats);
    ModelGraph graph;
    const std::array<int64_t, 1> dims{4};
    const TensorSpec spec{.dtype = DataType::Float32(), .shape = SymbolicShape(dims)};
    auto value = graph.AddConstant(spec, ConstantBinding{.inline_data = std::make_shared<const std::vector<std::byte>>(16)});
    graph.MarkOutput(value);
    auto lowered = LowerModelGraph(graph, {});
    ASSERT_TRUE(lowered.ok());
    HfModelConfig config;
    config.vocab_size = 4;
    config.max_position_embeddings = 16;
    ResolvedModelWeights weights;
    weights.final_norm.bytes = 1;
    LoweredModelArtifact artifact{.loaded_model = std::make_unique<LoadedModel>(config, std::move(weights)), .graph = std::move(*lowered)};
    auto model = PrepareExecutableModel(runtime, std::move(artifact));
    ASSERT_TRUE(model.ok()) << model.status().ToString();
    EXPECT_FALSE(model->resource_usage().weights.backing_complete);
    EXPECT_EQ(model->resource_usage().inline_constant_bytes, 16U);
}
TEST(ModelResourceLimits, SharedOwnerDistinctConstantVectorsAreBothCharged) {
    struct Pair {
        std::vector<std::byte> a = std::vector<std::byte>(16);
        std::vector<std::byte> b = std::vector<std::byte>(32);
    };
    auto pair = std::make_shared<Pair>();
    auto first = std::shared_ptr<const std::vector<std::byte>>(pair, &pair->a);
    auto second = std::shared_ptr<const std::vector<std::byte>>(pair, &pair->b);
    auto build = [&]() -> StatusOr<LoweredModelArtifact> {
        ModelGraph graph;
        const std::array<int64_t, 1> small{4}, large{8};
        graph.MarkOutput(graph.AddConstant({.dtype = DataType::Float32(), .shape = SymbolicShape(small)}, ConstantBinding{.inline_data = first}));
        graph.MarkOutput(graph.AddConstant({.dtype = DataType::Float32(), .shape = SymbolicShape(large)}, ConstantBinding{.inline_data = second}));
        AM_ASSIGN_OR_RETURN(auto lowered, LowerModelGraph(graph, {}));
        HfModelConfig config;
        config.vocab_size = 4;
        config.max_position_embeddings = 16;
        return LoweredModelArtifact{.loaded_model = std::make_unique<LoadedModel>(config, ResolvedModelWeights{}), .graph = std::move(lowered)};
    };
    auto stats = std::make_shared<test::ResourceAllocationStats>();
    Runtime runtime = ResourceRuntime(stats);
    auto artifact = build();
    ASSERT_TRUE(artifact.ok());
    EXPECT_EQ(PrepareExecutableModel(runtime, std::move(*artifact), {.max_model_payload_bytes = 47}).status().code(), StatusCode::kResourceExhausted);
    auto retry = build();
    ASSERT_TRUE(retry.ok());
    auto model = PrepareExecutableModel(runtime, std::move(*retry), {.max_model_payload_bytes = 48});
    ASSERT_TRUE(model.ok()) << model.status().ToString();
    EXPECT_EQ(model->resource_usage().inline_constant_bytes, 48U);
}
} // namespace
