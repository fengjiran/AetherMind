#include "aethermind/backend/kernel_context.h"
#include "aethermind/compiler/lowered_graph.h"
#include "aethermind/execution/execution_bindings.h"
#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/execution/executor.h"
#include "aethermind/runtime/runtime_builder.h"

#include <gtest/gtest.h>

#include <array>
#include <optional>
#include <span>
#include <vector>

namespace {

using namespace aethermind;

std::optional<KVCacheReadBinding> g_last_read_binding;

Status CaptureReadBindingKernel(const KernelContext& context) noexcept {
    if (context.kv_read != nullptr) {
        g_last_read_binding = *context.kv_read;
    }
    return context.kv_read == nullptr ? Status::FailedPrecondition("missing KV read binding")
                                      : Status::Ok();
}

Status FailingReadBindingKernel(const KernelContext& context) noexcept {
    if (context.kv_read != nullptr) {
        g_last_read_binding = *context.kv_read;
    }
    return Status::InvalidArgument("intentional attention failure");
}

TensorSpec F32Spec(std::initializer_list<int64_t> dimensions) {
    const std::vector<int64_t> copied(dimensions);
    return {.dtype = DataType::Float32(), .shape = SymbolicShape(IntArrayView(copied))};
}

KernelSelector F32CpuSelector() {
    return {
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kBoth,
    };
}

StatusOr<LoweredGraph> BuildKVCacheUpdateGraph(int64_t sequence_length,
                                               uint32_t layer_count = 1) {
    const TensorSpec activation = F32Spec({sequence_length, 4});
    const TensorSpec cache = F32Spec({2, 8, 2});
    LoweredGraph::Builder builder;
    for (uint32_t layer = 0; layer < layer_count; ++layer) {
        const uint32_t base = static_cast<uint32_t>(builder.values.size());
        builder.values.push_back({.spec = activation, .payload = ModelInputValue{}});
        builder.values.push_back({.spec = activation, .payload = ModelInputValue{}});
        builder.values.push_back({.spec = cache,
                                  .payload = StateValue{.binding = KVCacheStateBinding{
                                                                .decoder_layer_index = layer,
                                                                .slot = KVCacheSlot::kKey}}});
        builder.values.push_back({.spec = cache,
                                  .payload = StateValue{.binding = KVCacheStateBinding{
                                                                .decoder_layer_index = layer,
                                                                .slot = KVCacheSlot::kValue}}});
        builder.values.push_back({.spec = cache,
                                  .payload = StateValue{.binding = KVCacheStateBinding{
                                                                .decoder_layer_index = layer,
                                                                .slot = KVCacheSlot::kKey}}});
        builder.values.push_back({.spec = cache,
                                  .payload = StateValue{.binding = KVCacheStateBinding{
                                                                .decoder_layer_index = layer,
                                                                .slot = KVCacheSlot::kValue}}});
        builder.model_inputs.push_back({.index = base});
        builder.model_inputs.push_back({.index = base + 1});
        builder.steps.push_back({
                .spec = {
                        .op_type = OpType::kKVCacheUpdate,
                        .selector = F32CpuSelector(),
                        .input_specs = {activation, activation, cache, cache},
                        .output_specs = {cache, cache},
                        .op_params = KVCacheUpdateParams{},
                },
                .binding = {
                        .node = {.index = layer},
                        .input_values = {{.index = base}, {.index = base + 1}, {.index = base + 2}, {.index = base + 3}},
                        .output_values = {{.index = base + 4}, {.index = base + 5}},
                },
        });
        builder.state_aliases.push_back({
                .step_index = layer,
                .input_port = 2,
                .output_port = 0,
                .input = {.index = base + 2},
                .output = {.index = base + 4},
        });
        builder.state_aliases.push_back({
                .step_index = layer,
                .input_port = 3,
                .output_port = 1,
                .input = {.index = base + 3},
                .output = {.index = base + 5},
        });
    }
    return std::move(builder).Build();
}

Runtime MakeRuntime(uint32_t layer_count) {
    RuntimeOptions options;
    options.kv_cache.enable_manager = true;
    options.kv_cache.num_layers = layer_count;
    options.kv_cache.num_kv_heads = 2;
    options.kv_cache.max_tokens = 8;
    options.kv_cache.head_dim = 2;
    options.kv_cache.kv_dtype = DataType::Float32();
    options.kv_cache.alignment = 64;
    RuntimeBuilder builder;
    builder.WithOptions(options);
    return builder.Build();
}

StatusOr<ExecutionContext> MakeContext(Runtime& runtime,
                                       const ExecutionPlan& plan,
                                       KVCacheView view,
                                       std::span<const TensorView> inputs) {
    ExternalTensorBindings external;
    if (inputs.size() != plan.model_inputs().size()) {
        return Status::InvalidArgument("test input count does not match model inputs");
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        external.readable.push_back({.value = plan.model_inputs()[i], .tensor = inputs[i]});
    }
    AM_ASSIGN_OR_RETURN(PreparedExecutionBindings prepared,
                        PrepareExecutionBindings(plan, external,
                                                 runtime.GetAllocator(Device::CPU())));
    return ExecutionContext::Create(plan, std::move(prepared), nullptr, view);
}

StatusOr<ExecutionPlan> BuildUpdateThenAttentionPlan(Runtime& runtime,
                                                     KernelFunc attention_kernel) {
    AM_ASSIGN_OR_RETURN(Backend * backend, runtime.GetBackend(DeviceType::kCPU));
    AM_ASSIGN_OR_RETURN(const ResolvedKernel update_kernel,
                        backend->PrepareKernel(
                                OpType::kKVCacheUpdate, F32CpuSelector(),
                                OpParams{KVCacheUpdateParams{}}));

    const TensorSpec activation = F32Spec({2, 4});
    const TensorSpec cache = F32Spec({2, 8, 2});
    return ExecutionPlan::Create(
            {{.spec = activation, .kind = ExecutionValueKind::kModelInput},
             {.spec = activation, .kind = ExecutionValueKind::kModelInput},
             {.spec = cache,
              .kind = ExecutionValueKind::kState,
              .state_binding = ExecutionKVCacheStateIdentity{.decoder_layer_index = 0,
                                                             .slot = ExecutionKVCacheSlot::kKey}},
             {.spec = cache,
              .kind = ExecutionValueKind::kState,
              .state_binding = ExecutionKVCacheStateIdentity{.decoder_layer_index = 0,
                                                             .slot = ExecutionKVCacheSlot::kValue}},
             {.spec = activation, .kind = ExecutionValueKind::kActivation},
             {.spec = cache,
              .kind = ExecutionValueKind::kState,
              .state_binding = ExecutionKVCacheStateIdentity{.decoder_layer_index = 0,
                                                             .slot = ExecutionKVCacheSlot::kKey}},
             {.spec = cache,
              .kind = ExecutionValueKind::kState,
              .state_binding = ExecutionKVCacheStateIdentity{.decoder_layer_index = 0,
                                                             .slot = ExecutionKVCacheSlot::kValue}}},
            {{.index = 0}, {.index = 1}}, {},
            {{.selector = F32CpuSelector(),
              .kernel = update_kernel,
              .inputs = {{.index = 0}, {.index = 1}, {.index = 2}, {.index = 3}},
              .outputs = {{.index = 5}, {.index = 6}},
              .kernel_input_ports = {0, 1},
              .kernel_output_ports = {}},
             {.selector = F32CpuSelector(),
              .kernel = {.op_type = OpType::kAttention, .fn = attention_kernel},
              .inputs = {{.index = 0}, {.index = 2}, {.index = 3}},
              .outputs = {{.index = 4}},
              .kernel_input_ports = {0},
              .kernel_output_ports = {0}}},
            {.aliases = {{.step_index = 0, .input_port = 2, .output_port = 0},
                         {.step_index = 0, .input_port = 3, .output_port = 1}}});
}

float ReadKey(const KVCacheView& view, size_t layer, size_t head, size_t token, size_t dim) {
    const auto ptr = view.KeyData(layer, head, token, dim);
    EXPECT_TRUE(ptr.ok()) << ptr.status().ToString();
    return *static_cast<const float*>(ptr.value());
}

float ReadValue(const KVCacheView& view, size_t layer, size_t head, size_t token, size_t dim) {
    const auto ptr = view.ValueData(layer, head, token, dim);
    EXPECT_TRUE(ptr.ok()) << ptr.status().ToString();
    return *static_cast<const float*>(ptr.value());
}

TEST(KVCacheUpdateKernel, ExecutesThroughCpuBackendPlanAndPreservesPadding) {
    Runtime runtime = MakeRuntime(1);
    KVCacheManager* const manager = runtime.GetKVCacheManager();
    ASSERT_NE(manager, nullptr);
    auto view = manager->ReserveForSession(2, 2);
    ASSERT_TRUE(view.ok()) << view.status().ToString();

    const auto lowered = BuildKVCacheUpdateGraph(2);
    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();
    const auto plan = ExecutionPlanBuilder::Build(runtime, *lowered);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_STREQ(plan->steps()[0].kernel.name, "cpu::kvcache_update_f32_reference");

    std::array<float, 16> key_storage{};
    std::array<float, 20> value_storage{};
    key_storage[0] = 1.0F;
    key_storage[2] = 2.0F;
    key_storage[4] = 3.0F;
    key_storage[6] = 4.0F;
    key_storage[8] = 5.0F;
    key_storage[10] = 6.0F;
    key_storage[12] = 7.0F;
    key_storage[14] = 8.0F;
    value_storage[0] = 11.0F;
    value_storage[2] = 12.0F;
    value_storage[4] = 13.0F;
    value_storage[6] = 14.0F;
    value_storage[10] = 15.0F;
    value_storage[12] = 16.0F;
    value_storage[14] = 17.0F;
    value_storage[16] = 18.0F;
    const int64_t shape[2] = {2, 4};
    const int64_t key_strides[2] = {8, 2};
    const int64_t value_strides[2] = {10, 2};

    auto* const key_row = static_cast<float*>(view->MutableKeyData(0, 0, 0).value());
    auto* const value_row = static_cast<float*>(view->MutableValueData(0, 0, 0).value());
    std::fill(key_row + 2, key_row + manager->layout().head_dim_stride, -123.0F);
    std::fill(value_row + 2, value_row + manager->layout().head_dim_stride, -456.0F);
    auto* const untouched_token = static_cast<float*>(view->MutableKeyData(0, 0, 3).value());
    untouched_token[0] = 999.0F;

    const std::array<TensorView, 2> inputs = {
            TensorView(key_storage.data(), DataType::Float32(), shape, key_strides),
            TensorView(value_storage.data(), DataType::Float32(), shape, value_strides),
    };
    auto context = MakeContext(runtime, *plan, *view, inputs);
    ASSERT_TRUE(context.ok()) << context.status().ToString();
    ASSERT_TRUE(Executor::Execute(*plan, *context).ok());

    EXPECT_EQ(view->current_pos(), 2U);
    EXPECT_FALSE(view->awaiting_prefill());
    EXPECT_FLOAT_EQ(ReadKey(*view, 0, 0, 0, 0), 1.0F);
    EXPECT_FLOAT_EQ(ReadKey(*view, 0, 0, 0, 1), 2.0F);
    EXPECT_FLOAT_EQ(ReadKey(*view, 0, 1, 0, 0), 3.0F);
    EXPECT_FLOAT_EQ(ReadKey(*view, 0, 1, 0, 1), 4.0F);
    EXPECT_FLOAT_EQ(ReadKey(*view, 0, 0, 1, 0), 5.0F);
    EXPECT_FLOAT_EQ(ReadKey(*view, 0, 1, 1, 1), 8.0F);
    EXPECT_FLOAT_EQ(ReadValue(*view, 0, 0, 0, 0), 11.0F);
    EXPECT_FLOAT_EQ(ReadValue(*view, 0, 1, 1, 1), 18.0F);
    EXPECT_FLOAT_EQ(key_row[2], -123.0F);
    EXPECT_FLOAT_EQ(value_row[2], -456.0F);
    EXPECT_FLOAT_EQ(untouched_token[0], 999.0F);
}

TEST(KVCacheUpdateKernel, PrefillThenDecodeUsesOneCommittedPositionPerPlan) {
    Runtime runtime = MakeRuntime(1);
    KVCacheManager* const manager = runtime.GetKVCacheManager();
    ASSERT_NE(manager, nullptr);
    auto view = manager->ReserveForSession(2, 2);
    ASSERT_TRUE(view.ok());

    const auto prefill_graph = BuildKVCacheUpdateGraph(2);
    ASSERT_TRUE(prefill_graph.ok());
    const auto prefill_plan = ExecutionPlanBuilder::Build(runtime, *prefill_graph);
    ASSERT_TRUE(prefill_plan.ok());
    float prefill_key[8] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    float prefill_value[8] = {11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F, 17.0F, 18.0F};
    const int64_t prefill_shape[2] = {2, 4};
    const int64_t contiguous[2] = {4, 1};
    auto prefill_context = MakeContext(
            runtime, *prefill_plan, *view,
            std::array<TensorView, 2>{
                    TensorView(prefill_key, DataType::Float32(), prefill_shape, contiguous),
                    TensorView(prefill_value, DataType::Float32(), prefill_shape, contiguous)});
    ASSERT_TRUE(prefill_context.ok()) << prefill_context.status().ToString();
    ASSERT_TRUE(Executor::Execute(*prefill_plan, *prefill_context).ok());
    EXPECT_EQ(view->current_pos(), 2U);

    const auto decode_graph = BuildKVCacheUpdateGraph(1);
    ASSERT_TRUE(decode_graph.ok());
    const auto decode_plan = ExecutionPlanBuilder::Build(runtime, *decode_graph);
    ASSERT_TRUE(decode_plan.ok());
    float decode_key[4] = {21.0F, 22.0F, 23.0F, 24.0F};
    float decode_value[4] = {31.0F, 32.0F, 33.0F, 34.0F};
    const int64_t decode_shape[2] = {1, 4};
    auto decode_context = MakeContext(
            runtime, *decode_plan, *view,
            std::array<TensorView, 2>{
                    TensorView(decode_key, DataType::Float32(), decode_shape, contiguous),
                    TensorView(decode_value, DataType::Float32(), decode_shape, contiguous)});
    ASSERT_TRUE(decode_context.ok()) << decode_context.status().ToString();
    ASSERT_TRUE(Executor::Execute(*decode_plan, *decode_context).ok());

    EXPECT_EQ(view->current_pos(), 3U);
    EXPECT_FLOAT_EQ(ReadKey(*view, 0, 0, 2, 0), 21.0F);
    EXPECT_FLOAT_EQ(ReadKey(*view, 0, 1, 2, 1), 24.0F);
    EXPECT_FLOAT_EQ(ReadValue(*view, 0, 0, 2, 1), 32.0F);
    EXPECT_FLOAT_EQ(ReadValue(*view, 0, 1, 2, 0), 33.0F);
}

TEST(KVCacheUpdateKernel, MultiLayerAppendWritesOneSharedRange) {
    Runtime runtime = MakeRuntime(2);
    KVCacheManager* const manager = runtime.GetKVCacheManager();
    ASSERT_NE(manager, nullptr);
    auto view = manager->ReserveForSession(2, 1);
    ASSERT_TRUE(view.ok());
    const auto lowered = BuildKVCacheUpdateGraph(2, 2);
    ASSERT_TRUE(lowered.ok());
    const auto plan = ExecutionPlanBuilder::Build(runtime, *lowered);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();

    float layer0_key[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float layer0_value[8] = {11, 12, 13, 14, 15, 16, 17, 18};
    float layer1_key[8] = {21, 22, 23, 24, 25, 26, 27, 28};
    float layer1_value[8] = {31, 32, 33, 34, 35, 36, 37, 38};
    const int64_t shape[2] = {2, 4};
    const int64_t strides[2] = {4, 1};
    const std::array<TensorView, 4> inputs = {
            TensorView(layer0_key, DataType::Float32(), shape, strides),
            TensorView(layer0_value, DataType::Float32(), shape, strides),
            TensorView(layer1_key, DataType::Float32(), shape, strides),
            TensorView(layer1_value, DataType::Float32(), shape, strides),
    };
    auto context = MakeContext(runtime, *plan, *view, inputs);
    ASSERT_TRUE(context.ok()) << context.status().ToString();
    ASSERT_TRUE(Executor::Execute(*plan, *context).ok());

    EXPECT_EQ(view->current_pos(), 2U);
    EXPECT_FLOAT_EQ(ReadKey(*view, 0, 0, 0, 0), 1.0F);
    EXPECT_FLOAT_EQ(ReadValue(*view, 0, 1, 1, 1), 18.0F);
    EXPECT_FLOAT_EQ(ReadKey(*view, 1, 0, 0, 0), 21.0F);
    EXPECT_FLOAT_EQ(ReadValue(*view, 1, 1, 1, 1), 38.0F);
}

TEST(KVCacheUpdateKernel, MissingLayerUpdateDoesNotCommitTheTransaction) {
    Runtime runtime = MakeRuntime(2);
    KVCacheManager* const manager = runtime.GetKVCacheManager();
    ASSERT_NE(manager, nullptr);
    auto view = manager->ReserveForSession(2, 1);
    ASSERT_TRUE(view.ok());
    const auto lowered = BuildKVCacheUpdateGraph(2, 1);
    ASSERT_TRUE(lowered.ok());
    const auto plan = ExecutionPlanBuilder::Build(runtime, *lowered);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();

    float key[8] = {};
    float value[8] = {};
    const int64_t shape[2] = {2, 4};
    const int64_t strides[2] = {4, 1};
    auto context = MakeContext(runtime, *plan, *view,
                               std::array<TensorView, 2>{
                                       TensorView(key, DataType::Float32(), shape, strides),
                                       TensorView(value, DataType::Float32(), shape, strides)});
    ASSERT_TRUE(context.ok()) << context.status().ToString();

    const Status status = Executor::Execute(*plan, *context);

    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(view->current_pos(), 0U);
}

TEST(KVCacheUpdateKernel, AliasFailureDoesNotCommitTheTransaction) {
    Runtime runtime = MakeRuntime(1);
    KVCacheManager* const manager = runtime.GetKVCacheManager();
    ASSERT_NE(manager, nullptr);
    auto view = manager->ReserveForSession(2, 1);
    ASSERT_TRUE(view.ok());
    const auto lowered = BuildKVCacheUpdateGraph(2);
    ASSERT_TRUE(lowered.ok());
    const auto plan = ExecutionPlanBuilder::Build(runtime, *lowered);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();

    auto* const aliased_key = static_cast<float*>(view->MutableKeyData(0, 0, 0).value());
    float value[8] = {};
    const int64_t shape[2] = {2, 4};
    const int64_t aliased_strides[2] = {
            static_cast<int64_t>(manager->layout().token_stride / sizeof(float)), 1};
    const int64_t contiguous_strides[2] = {4, 1};
    const std::array<TensorView, 2> inputs = {
            TensorView(aliased_key, DataType::Float32(), shape, aliased_strides),
            TensorView(value, DataType::Float32(), shape, contiguous_strides),
    };
    auto context = MakeContext(runtime, *plan, *view, inputs);
    ASSERT_TRUE(context.ok()) << context.status().ToString();

    const Status status = Executor::Execute(*plan, *context);

    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(view->current_pos(), 0U);
    EXPECT_TRUE(view->awaiting_prefill());
}

TEST(KVCacheUpdateKernel, PureReadPlanUsesCommittedRangeWithoutCommit) {
    Runtime runtime = MakeRuntime(1);
    KVCacheManager* const manager = runtime.GetKVCacheManager();
    ASSERT_NE(manager, nullptr);
    auto view = manager->ReserveForSession(2, 1);
    ASSERT_TRUE(view.ok());
    ASSERT_TRUE(view->CommitUntil(2).ok());

    const TensorSpec activation = F32Spec({1, 4});
    const TensorSpec cache = F32Spec({2, 8, 2});
    const auto plan = ExecutionPlan::Create(
            {{.spec = activation, .kind = ExecutionValueKind::kModelInput},
             {.spec = cache,
              .kind = ExecutionValueKind::kState,
              .state_binding = ExecutionKVCacheStateIdentity{.decoder_layer_index = 0,
                                                             .slot = ExecutionKVCacheSlot::kKey}},
             {.spec = cache,
              .kind = ExecutionValueKind::kState,
              .state_binding = ExecutionKVCacheStateIdentity{.decoder_layer_index = 0,
                                                             .slot = ExecutionKVCacheSlot::kValue}},
             {.spec = activation, .kind = ExecutionValueKind::kActivation}},
            {{.index = 0}}, {},
            {{.selector = F32CpuSelector(),
              .kernel = {.op_type = OpType::kAttention, .fn = &CaptureReadBindingKernel},
              .inputs = {{.index = 0}, {.index = 1}, {.index = 2}},
              .outputs = {{.index = 3}},
              .kernel_input_ports = {0},
              .kernel_output_ports = {0}}});
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();

    float query[4] = {};
    const int64_t shape[2] = {1, 4};
    const int64_t strides[2] = {4, 1};
    auto context = MakeContext(
            runtime, *plan, *view,
            std::array<TensorView, 1>{
                    TensorView(query, DataType::Float32(), shape, strides)});
    ASSERT_TRUE(context.ok()) << context.status().ToString();
    g_last_read_binding.reset();

    ASSERT_TRUE(Executor::Execute(*plan, *context).ok());

    ASSERT_TRUE(g_last_read_binding.has_value());
    EXPECT_EQ(g_last_read_binding->committed_end, 2U);
    EXPECT_EQ(g_last_read_binding->visible_end, 2U);
    EXPECT_EQ(view->current_pos(), 2U);
}

TEST(KVCacheUpdateKernel, UpdateThenAttentionReadsVisibleRangeAndCommitsAtPlanEnd) {
    Runtime runtime = MakeRuntime(1);
    KVCacheManager* const manager = runtime.GetKVCacheManager();
    ASSERT_NE(manager, nullptr);
    auto view = manager->ReserveForSession(2, 1);
    ASSERT_TRUE(view.ok());
    const auto plan = BuildUpdateThenAttentionPlan(runtime, &CaptureReadBindingKernel);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();

    float key[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float value[8] = {11, 12, 13, 14, 15, 16, 17, 18};
    const int64_t shape[2] = {2, 4};
    const int64_t strides[2] = {4, 1};
    auto context = MakeContext(runtime, *plan, *view,
                               std::array<TensorView, 2>{
                                       TensorView(key, DataType::Float32(), shape, strides),
                                       TensorView(value, DataType::Float32(), shape, strides)});
    ASSERT_TRUE(context.ok()) << context.status().ToString();
    const std::span<uint8_t> scratch_before = context->ResetKVLayerTransactionScratch();
    ASSERT_EQ(scratch_before.size(), 1U);
    const uint8_t* const scratch_data = scratch_before.data();
    g_last_read_binding.reset();

    ASSERT_TRUE(Executor::Execute(*plan, *context).ok());

    EXPECT_EQ(context->ResetKVLayerTransactionScratch().data(), scratch_data);
    ASSERT_TRUE(g_last_read_binding.has_value());
    EXPECT_EQ(g_last_read_binding->committed_end, 0U);
    EXPECT_EQ(g_last_read_binding->visible_end, 2U);
    EXPECT_EQ(view->current_pos(), 2U);
    EXPECT_FLOAT_EQ(ReadKey(*view, 0, 0, 0, 0), 1.0F);
}

TEST(KVCacheUpdateKernel, AttentionRejectsStateGeometryThatDiffersFromKVCacheView) {
    Runtime runtime = MakeRuntime(1);
    KVCacheManager* const manager = runtime.GetKVCacheManager();
    ASSERT_NE(manager, nullptr);
    auto view = manager->ReserveForSession(2, 1);
    ASSERT_TRUE(view.ok());
    ASSERT_TRUE(view->CommitUntil(2).ok());

    const TensorSpec activation = F32Spec({1, 4});
    const TensorSpec wrong_cache = F32Spec({2, 7, 2});
    const auto plan = ExecutionPlan::Create(
            {{.spec = activation, .kind = ExecutionValueKind::kModelInput},
             {.spec = wrong_cache,
              .kind = ExecutionValueKind::kState,
              .state_binding = ExecutionKVCacheStateIdentity{.decoder_layer_index = 0,
                                                             .slot = ExecutionKVCacheSlot::kKey}},
             {.spec = wrong_cache,
              .kind = ExecutionValueKind::kState,
              .state_binding = ExecutionKVCacheStateIdentity{.decoder_layer_index = 0,
                                                             .slot = ExecutionKVCacheSlot::kValue}},
             {.spec = activation, .kind = ExecutionValueKind::kActivation}},
            {{.index = 0}}, {},
            {{.selector = F32CpuSelector(),
              .kernel = {.op_type = OpType::kAttention, .fn = &CaptureReadBindingKernel},
              .inputs = {{.index = 0}, {.index = 1}, {.index = 2}},
              .outputs = {{.index = 3}},
              .kernel_input_ports = {0},
              .kernel_output_ports = {0}}});
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();

    float query[4] = {};
    const int64_t shape[2] = {1, 4};
    const int64_t strides[2] = {4, 1};
    auto context = MakeContext(
            runtime, *plan, *view,
            std::array<TensorView, 1>{
                    TensorView(query, DataType::Float32(), shape, strides)});
    ASSERT_TRUE(context.ok()) << context.status().ToString();
    g_last_read_binding.reset();

    const Status status = Executor::Execute(*plan, *context);

    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_FALSE(g_last_read_binding.has_value());
    EXPECT_EQ(view->current_pos(), 2U);
}

TEST(KVCacheUpdateKernel, LaterStepFailureLeavesWrittenRangeUncommitted) {
    Runtime runtime = MakeRuntime(1);
    KVCacheManager* const manager = runtime.GetKVCacheManager();
    ASSERT_NE(manager, nullptr);
    auto view = manager->ReserveForSession(2, 1);
    ASSERT_TRUE(view.ok());
    const auto plan = BuildUpdateThenAttentionPlan(runtime, &FailingReadBindingKernel);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();

    float key[8] = {41, 42, 43, 44, 45, 46, 47, 48};
    float value[8] = {51, 52, 53, 54, 55, 56, 57, 58};
    const int64_t shape[2] = {2, 4};
    const int64_t strides[2] = {4, 1};
    auto* const key_cache = static_cast<float*>(view->MutableKeyData(0, 0, 0).value());
    auto context = MakeContext(runtime, *plan, *view,
                               std::array<TensorView, 2>{
                                       TensorView(key, DataType::Float32(), shape, strides),
                                       TensorView(value, DataType::Float32(), shape, strides)});
    ASSERT_TRUE(context.ok()) << context.status().ToString();
    g_last_read_binding.reset();

    const Status status = Executor::Execute(*plan, *context);

    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    ASSERT_TRUE(g_last_read_binding.has_value());
    EXPECT_EQ(g_last_read_binding->visible_end, 2U);
    EXPECT_FLOAT_EQ(key_cache[0], 41.0F);
    EXPECT_EQ(view->current_pos(), 0U);
    EXPECT_FALSE(view->KeyData(0, 0, 0).ok());
}

TEST(KVCacheUpdateKernel, AttentionBeforeLayerUpdateCannotReadVisibleAppendRange) {
    Runtime runtime = MakeRuntime(1);
    KVCacheManager* const manager = runtime.GetKVCacheManager();
    ASSERT_NE(manager, nullptr);
    auto view = manager->ReserveForSession(2, 1);
    ASSERT_TRUE(view.ok());
    const auto backend = runtime.GetBackend(DeviceType::kCPU);
    ASSERT_TRUE(backend.ok());
    const auto update_kernel = backend.value()->PrepareKernel(
            OpType::kKVCacheUpdate, F32CpuSelector(), OpParams{KVCacheUpdateParams{}});
    ASSERT_TRUE(update_kernel.ok()) << update_kernel.status().ToString();

    const TensorSpec activation = F32Spec({2, 4});
    const TensorSpec cache = F32Spec({2, 8, 2});
    const auto plan = ExecutionPlan::Create(
            {{.spec = activation, .kind = ExecutionValueKind::kModelInput},
             {.spec = activation, .kind = ExecutionValueKind::kModelInput},
             {.spec = cache,
              .kind = ExecutionValueKind::kState,
              .state_binding = ExecutionKVCacheStateIdentity{.decoder_layer_index = 0,
                                                             .slot = ExecutionKVCacheSlot::kKey}},
             {.spec = cache,
              .kind = ExecutionValueKind::kState,
              .state_binding = ExecutionKVCacheStateIdentity{.decoder_layer_index = 0,
                                                             .slot = ExecutionKVCacheSlot::kValue}},
             {.spec = activation, .kind = ExecutionValueKind::kActivation},
             {.spec = cache,
              .kind = ExecutionValueKind::kState,
              .state_binding = ExecutionKVCacheStateIdentity{.decoder_layer_index = 0,
                                                             .slot = ExecutionKVCacheSlot::kKey}},
             {.spec = cache,
              .kind = ExecutionValueKind::kState,
              .state_binding = ExecutionKVCacheStateIdentity{.decoder_layer_index = 0,
                                                             .slot = ExecutionKVCacheSlot::kValue}}},
            {{.index = 0}, {.index = 1}}, {},
            {{.selector = F32CpuSelector(),
              .kernel = {.op_type = OpType::kAttention, .fn = &CaptureReadBindingKernel},
              .inputs = {{.index = 0}, {.index = 2}, {.index = 3}},
              .outputs = {{.index = 4}},
              .kernel_input_ports = {0},
              .kernel_output_ports = {0}},
             {.selector = F32CpuSelector(),
              .kernel = *update_kernel,
              .inputs = {{.index = 0}, {.index = 1}, {.index = 2}, {.index = 3}},
              .outputs = {{.index = 5}, {.index = 6}},
              .kernel_input_ports = {0, 1},
              .kernel_output_ports = {}}},
            {.aliases = {{.step_index = 1, .input_port = 2, .output_port = 0},
                         {.step_index = 1, .input_port = 3, .output_port = 1}}});
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();

    float key[8] = {};
    float value[8] = {};
    const int64_t shape[2] = {2, 4};
    const int64_t strides[2] = {4, 1};
    auto context = MakeContext(runtime, *plan, *view,
                               std::array<TensorView, 2>{
                                       TensorView(key, DataType::Float32(), shape, strides),
                                       TensorView(value, DataType::Float32(), shape, strides)});
    ASSERT_TRUE(context.ok()) << context.status().ToString();
    g_last_read_binding.reset();

    const Status status = Executor::Execute(*plan, *context);

    EXPECT_EQ(status.code(), StatusCode::kFailedPrecondition);
    EXPECT_EQ(view->current_pos(), 0U);
    EXPECT_FALSE(g_last_read_binding.has_value());
}

TEST(KVCacheUpdateKernel, RejectsUntrustedStatefulNodeWithoutIdentity) {
    Runtime runtime = MakeRuntime(1);
    const TensorSpec activation = F32Spec({1, 4});
    const TensorSpec cache = F32Spec({2, 8, 2});
    const auto plan = ExecutionPlanBuilder::Build(
            runtime,
            std::vector{ExecutionPlanNodeSpec{
                    .op_type = OpType::kKVCacheUpdate,
                    .selector = F32CpuSelector(),
                    .input_specs = {activation, activation, cache, cache},
                    .output_specs = {cache, cache},
                    .op_params = KVCacheUpdateParams{},
            }});

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(plan.status().message().find("state identity"), std::string::npos);
}

TEST(KVCacheUpdateKernel, RegistryDoesNotResolveUnsupportedF16Selector) {
    Runtime runtime = MakeRuntime(1);
    const auto backend = runtime.GetBackend(DeviceType::kCPU);
    ASSERT_TRUE(backend.ok());
    KernelSelector selector = F32CpuSelector();
    selector.act_dtype = DataType::Float(16);
    selector.weight_dtype = DataType::Float(16);

    const auto kernel = backend.value()->PrepareKernel(
            OpType::kKVCacheUpdate, selector, OpParams{KVCacheUpdateParams{}});

    ASSERT_FALSE(kernel.ok());
    EXPECT_EQ(kernel.status().code(), StatusCode::kNotFound);
}

} // namespace
