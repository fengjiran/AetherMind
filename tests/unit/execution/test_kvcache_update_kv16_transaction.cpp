#include "aethermind/backend/kernel_context.h"
#include "aethermind/compiler/lowered_graph.h"
#include "aethermind/dtypes/bfloat16.h"
#include "aethermind/dtypes/half.h"
#include "aethermind/execution/execution_bindings.h"
#include "aethermind/execution/execution_context.h"
#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/execution/executor.h"
#include "aethermind/runtime/runtime_builder.h"
#include "kv_cache/test_kv_cache_helpers.h"

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

namespace {

using namespace aethermind;

// KV16 transaction regressions: the LayerRunner commit/visibility machinery is
// dtype-agnostic; these verify the KV16 reference kernels integrate with it for
// both F16 and BF16 caches. Activations stay F32; only the cache dtype varies.

TensorSpec Spec(DataType dtype, std::initializer_list<int64_t> dimensions) {
    const std::vector<int64_t> copied(dimensions);
    return {.dtype = dtype, .shape = SymbolicShape(IntArrayView(copied))};
}

KernelSelector Selector(DataType kv_dtype) {
    return {.device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .state_dtype = kv_dtype,
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kBoth};
}

Runtime MakeRuntime(uint32_t layer_count, DataType kv_dtype) {
    RuntimeOptions options;
    options.kv_cache.enable_manager = true;
    options.kv_cache.num_layers = layer_count;
    options.kv_cache.num_kv_heads = 2;
    options.kv_cache.max_tokens = 8;
    options.kv_cache.head_dim = 2;
    options.kv_cache.kv_dtype = kv_dtype;
    options.kv_cache.alignment = 64;
    RuntimeBuilder builder;
    builder.WithOptions(options);
    return builder.Build();
}

StatusOr<LoweredGraph> BuildUpdateGraph(int64_t sequence_length, uint32_t layer_count,
                                        DataType kv_dtype) {
    const TensorSpec activation = Spec(DataType::Float32(), {sequence_length, 4});
    const TensorSpec cache = Spec(kv_dtype, {2, 8, 2});
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
                .spec = {.op_type = OpType::kKVCacheUpdate,
                         .selector = Selector(kv_dtype),
                         .input_specs = {activation, activation, cache, cache},
                         .output_specs = {cache, cache},
                         .op_params = KVCacheUpdateParams{}},
                .binding = {.node = {.index = layer},
                            .input_values = {{.index = base}, {.index = base + 1}, {.index = base + 2}, {.index = base + 3}},
                            .output_values = {{.index = base + 4}, {.index = base + 5}}},
        });
        builder.state_aliases.push_back({.step_index = layer,
                                         .input_port = 2,
                                         .output_port = 0,
                                         .input = {.index = base + 2},
                                         .output = {.index = base + 4}});
        builder.state_aliases.push_back({.step_index = layer,
                                         .input_port = 3,
                                         .output_port = 1,
                                         .input = {.index = base + 3},
                                         .output = {.index = base + 5}});
    }
    return std::move(builder).Build();
}

StatusOr<ExecutionContext> MakeContext(Runtime& runtime, const ExecutionPlan& plan,
                                       KVCacheSessionView view,
                                       std::span<const TensorView> inputs) {
    ExternalTensorBindings external;
    for (size_t i = 0; i < inputs.size(); ++i) {
        external.read_only.push_back({.value = plan.model_inputs()[i], .tensor = inputs[i]});
    }
    AM_ASSIGN_OR_RETURN(PreparedExecutionBindings prepared,
                        PrepareExecutionBindings(plan, external,
                                                 runtime.GetAllocator(Device::CPU())));
    return ExecutionContext::Create(plan, std::move(prepared), nullptr, view);
}

StatusOr<ExecutionPlan> BuildUpdateThenAttentionPlan(Runtime& runtime, DataType kv_dtype,
                                                     int64_t sequence_length,
                                                     ResolvedKernel attention_kernel) {
    AM_ASSIGN_OR_RETURN(Backend * backend, runtime.GetBackend(DeviceType::kCPU));
    AM_ASSIGN_OR_RETURN(const ResolvedKernel update_kernel,
                        backend->PrepareKernel(OpType::kKVCacheUpdate, Selector(kv_dtype),
                                               OpParams{KVCacheUpdateParams{}}));
    const TensorSpec activation = Spec(DataType::Float32(), {sequence_length, 4});
    const TensorSpec cache = Spec(kv_dtype, {2, 8, 2});
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
            {{.selector = Selector(kv_dtype),
              .kernel = update_kernel,
              .inputs = {{.index = 0}, {.index = 1}, {.index = 2}, {.index = 3}},
              .outputs = {{.index = 5}, {.index = 6}},
              .kernel_input_ports = {0, 1},
              .kernel_output_ports = {}},
             {.selector = Selector(kv_dtype),
              .kernel = std::move(attention_kernel),
              .inputs = {{.index = 0}, {.index = 2}, {.index = 3}},
              .outputs = {{.index = 4}},
              .kernel_input_ports = {0},
              .kernel_output_ports = {0}}},
            {.aliases = {{.step_index = 0, .input_port = 2, .output_port = 0},
                         {.step_index = 0, .input_port = 3, .output_port = 1}}});
}

float Widen(const void* ptr, DataType kv_dtype) {
    uint16_t bits = 0;
    std::memcpy(&bits, ptr, sizeof(bits));
    return kv_dtype == DataType::Float(16)
                   ? static_cast<float>(Half(bits, Half::from_bits()))
                   : aethermind::detail::bf16_to_fp32_value(bits);
}

float ReadKey(const KVCacheSessionView& view, DataType kv_dtype, size_t layer, size_t head,
              size_t token, size_t dim) {
    const auto ptr = view.KeyData(layer, head, token, dim);
    EXPECT_TRUE(ptr.ok()) << ptr.status().ToString();
    return Widen(ptr.value(), kv_dtype);
}

float ReadValue(const KVCacheSessionView& view, DataType kv_dtype, size_t layer, size_t head,
                size_t token, size_t dim) {
    const auto ptr = view.ValueData(layer, head, token, dim);
    EXPECT_TRUE(ptr.ok()) << ptr.status().ToString();
    return Widen(ptr.value(), kv_dtype);
}

ResolvedKernel FailingAttentionKernel() {
    return {.op_type = OpType::kAttention,
            .fn = [](const KernelContext&) noexcept {
                return Status::InvalidArgument("intentional attention failure");
            }};
}

const std::array<DataType, 2> kKvDTypes = {DataType::Float(16), DataType::BFloat(16)};

TEST(KVCacheUpdateKV16Transaction, PrefillThenDecodeCommitsOnePositionPerPlan) {
    for (const DataType kv_dtype: kKvDTypes) {
        SCOPED_TRACE(ToString(kv_dtype));
        Runtime runtime = MakeRuntime(1, kv_dtype);
        auto* manager = runtime.GetKVCacheManager();
        ASSERT_NE(manager, nullptr);
        auto view = manager->ReserveForSession(2, 2);
        ASSERT_TRUE(view.ok()) << view.status().ToString();

        const auto prefill = BuildUpdateGraph(2, 1, kv_dtype);
        ASSERT_TRUE(prefill.ok()) << prefill.status().ToString();
        const auto prefill_plan = ExecutionPlanBuilder::Build(runtime, *prefill);
        ASSERT_TRUE(prefill_plan.ok()) << prefill_plan.status().ToString();
        EXPECT_EQ(prefill_plan->steps()[0].selector.state_dtype, kv_dtype);

        float key[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        float value[8] = {11, 12, 13, 14, 15, 16, 17, 18};
        const int64_t shape[2] = {2, 4};
        const int64_t strides[2] = {4, 1};
        auto ctx = MakeContext(runtime, *prefill_plan, *view,
                               std::array<TensorView, 2>{
                                       TensorView(key, DataType::Float32(), shape, strides),
                                       TensorView(value, DataType::Float32(), shape, strides)});
        ASSERT_TRUE(ctx.ok()) << ctx.status().ToString();
        ASSERT_TRUE(Executor::Execute(*prefill_plan, *ctx).ok());
        EXPECT_EQ(view->current_pos(), 2U);

        const auto decode = BuildUpdateGraph(1, 1, kv_dtype);
        ASSERT_TRUE(decode.ok());
        const auto decode_plan = ExecutionPlanBuilder::Build(runtime, *decode);
        ASSERT_TRUE(decode_plan.ok());
        float dkey[4] = {21, 22, 23, 24};
        float dvalue[4] = {31, 32, 33, 34};
        const int64_t dshape[2] = {1, 4};
        auto dctx = MakeContext(runtime, *decode_plan, *view,
                                std::array<TensorView, 2>{
                                        TensorView(dkey, DataType::Float32(), dshape, strides),
                                        TensorView(dvalue, DataType::Float32(), dshape, strides)});
        ASSERT_TRUE(dctx.ok());
        ASSERT_TRUE(Executor::Execute(*decode_plan, *dctx).ok());
        EXPECT_EQ(view->current_pos(), 3U);
        EXPECT_FLOAT_EQ(ReadKey(*view, kv_dtype, 0, 0, 2, 0), 21.0F);
        EXPECT_FLOAT_EQ(ReadValue(*view, kv_dtype, 0, 1, 2, 1), 34.0F);
    }
}

TEST(KVCacheUpdateKV16Transaction, MultiLayerAppendWritesOneSharedRange) {
    for (const DataType kv_dtype: kKvDTypes) {
        SCOPED_TRACE(ToString(kv_dtype));
        Runtime runtime = MakeRuntime(2, kv_dtype);
        auto view = runtime.GetKVCacheManager()->ReserveForSession(2, 1);
        ASSERT_TRUE(view.ok());
        const auto lowered = BuildUpdateGraph(2, 2, kv_dtype);
        ASSERT_TRUE(lowered.ok());
        const auto plan = ExecutionPlanBuilder::Build(runtime, *lowered);
        ASSERT_TRUE(plan.ok()) << plan.status().ToString();

        float l0k[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        float l0v[8] = {11, 12, 13, 14, 15, 16, 17, 18};
        float l1k[8] = {21, 22, 23, 24, 25, 26, 27, 28};
        float l1v[8] = {31, 32, 33, 34, 35, 36, 37, 38};
        const int64_t shape[2] = {2, 4};
        const int64_t strides[2] = {4, 1};
        const std::array<TensorView, 4> inputs = {
                TensorView(l0k, DataType::Float32(), shape, strides),
                TensorView(l0v, DataType::Float32(), shape, strides),
                TensorView(l1k, DataType::Float32(), shape, strides),
                TensorView(l1v, DataType::Float32(), shape, strides)};
        auto ctx = MakeContext(runtime, *plan, *view, inputs);
        ASSERT_TRUE(ctx.ok());
        ASSERT_TRUE(Executor::Execute(*plan, *ctx).ok());
        EXPECT_EQ(view->current_pos(), 2U);
        EXPECT_FLOAT_EQ(ReadKey(*view, kv_dtype, 0, 0, 0, 0), 1.0F);
        EXPECT_FLOAT_EQ(ReadValue(*view, kv_dtype, 0, 1, 1, 1), 18.0F);
        EXPECT_FLOAT_EQ(ReadKey(*view, kv_dtype, 1, 0, 0, 0), 21.0F);
        EXPECT_FLOAT_EQ(ReadValue(*view, kv_dtype, 1, 1, 1, 1), 38.0F);
    }
}

TEST(KVCacheUpdateKV16Transaction, MissingLayerUpdateDoesNotCommit) {
    for (const DataType kv_dtype: kKvDTypes) {
        SCOPED_TRACE(ToString(kv_dtype));
        Runtime runtime = MakeRuntime(2, kv_dtype);
        auto view = runtime.GetKVCacheManager()->ReserveForSession(2, 1);
        ASSERT_TRUE(view.ok());
        // Runtime has two layers but the graph updates only one.
        const auto lowered = BuildUpdateGraph(2, 1, kv_dtype);
        ASSERT_TRUE(lowered.ok());
        const auto plan = ExecutionPlanBuilder::Build(runtime, *lowered);
        ASSERT_TRUE(plan.ok());
        float key[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        float value[8] = {11, 12, 13, 14, 15, 16, 17, 18};
        const int64_t shape[2] = {2, 4};
        const int64_t strides[2] = {4, 1};
        auto ctx = MakeContext(runtime, *plan, *view,
                               std::array<TensorView, 2>{
                                       TensorView(key, DataType::Float32(), shape, strides),
                                       TensorView(value, DataType::Float32(), shape, strides)});
        ASSERT_TRUE(ctx.ok());
        EXPECT_FALSE(Executor::Execute(*plan, *ctx).ok());
        EXPECT_EQ(view->current_pos(), 0U);
        EXPECT_TRUE(view->awaiting_prefill());
    }
}

TEST(KVCacheUpdateKV16Transaction, UpdateThenAttentionReadsVisibleRangeAndCommits) {
    for (const DataType kv_dtype: kKvDTypes) {
        SCOPED_TRACE(ToString(kv_dtype));
        Runtime runtime = MakeRuntime(1, kv_dtype);
        auto view = runtime.GetKVCacheManager()->ReserveForSession(2, 1);
        ASSERT_TRUE(view.ok());
        auto backend_or = runtime.GetBackend(DeviceType::kCPU);
        ASSERT_TRUE(backend_or.ok()) << backend_or.status().ToString();
        const auto attention = (*backend_or)->PrepareKernel(OpType::kAttention, Selector(kv_dtype), OpParams{AttentionParams{.num_q_heads = 2, .num_kv_heads = 2, .head_dim = 2}});
        ASSERT_TRUE(attention.ok()) << attention.status().ToString();
        const auto plan = BuildUpdateThenAttentionPlan(runtime, kv_dtype, 2, *attention);
        ASSERT_TRUE(plan.ok()) << plan.status().ToString();

        float key[8] = {1, 0, 0, 1, 1, 0, 0, 1};
        float value[8] = {10, 0, 0, 20, 10, 0, 0, 20};
        const int64_t shape[2] = {2, 4};
        const int64_t strides[2] = {4, 1};
        auto ctx = MakeContext(runtime, *plan, *view,
                               std::array<TensorView, 2>{
                                       TensorView(key, DataType::Float32(), shape, strides),
                                       TensorView(value, DataType::Float32(), shape, strides)});
        ASSERT_TRUE(ctx.ok());
        ASSERT_TRUE(Executor::Execute(*plan, *ctx).ok());
        EXPECT_EQ(view->current_pos(), 2U);
        EXPECT_FALSE(view->awaiting_prefill());
        EXPECT_FLOAT_EQ(ReadKey(*view, kv_dtype, 0, 0, 0, 0), 1.0F);
    }
}

TEST(KVCacheUpdateKV16Transaction, LaterAttentionFailureLeavesRangeUncommitted) {
    for (const DataType kv_dtype: kKvDTypes) {
        SCOPED_TRACE(ToString(kv_dtype));
        Runtime runtime = MakeRuntime(1, kv_dtype);
        auto view = runtime.GetKVCacheManager()->ReserveForSession(2, 1);
        ASSERT_TRUE(view.ok());
        const auto plan = BuildUpdateThenAttentionPlan(runtime, kv_dtype, 2,
                                                       FailingAttentionKernel());
        ASSERT_TRUE(plan.ok()) << plan.status().ToString();
        float key[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        float value[8] = {11, 12, 13, 14, 15, 16, 17, 18};
        const int64_t shape[2] = {2, 4};
        const int64_t strides[2] = {4, 1};
        auto ctx = MakeContext(runtime, *plan, *view,
                               std::array<TensorView, 2>{
                                       TensorView(key, DataType::Float32(), shape, strides),
                                       TensorView(value, DataType::Float32(), shape, strides)});
        ASSERT_TRUE(ctx.ok());
        EXPECT_FALSE(Executor::Execute(*plan, *ctx).ok());
        // The KVUpdate wrote its window, but the plan failed: nothing committed.
        EXPECT_EQ(view->current_pos(), 0U);
        EXPECT_TRUE(view->awaiting_prefill());
    }
}

} // namespace