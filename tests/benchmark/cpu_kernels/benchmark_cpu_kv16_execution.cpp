#include "aethermind/compiler/lowered_graph.h"
#include "aethermind/execution/execution_bindings.h"
#include "aethermind/execution/execution_context.h"
#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/execution/executor.h"
#include "aethermind/runtime/runtime_builder.h"

#include <benchmark/benchmark.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <span>
#include <utility>
#include <vector>

namespace {

using namespace aethermind;

constexpr uint32_t kLayers = 2;
constexpr int64_t kQueryHeads = 4;
constexpr int64_t kKvHeads = 2;
constexpr int64_t kHeadDim = 32;
constexpr int64_t kQueryWidth = kQueryHeads * kHeadDim;
constexpr int64_t kKvWidth = kKvHeads * kHeadDim;

TensorSpec Spec(DataType dtype, std::initializer_list<int64_t> shape) {
    const std::vector<int64_t> dimensions(shape);
    return {.dtype = dtype, .shape = SymbolicShape(IntArrayView(dimensions))};
}

KernelSelector Selector(DataType kv_dtype) {
    return {.device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .state_dtype = kv_dtype,
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kBoth};
}

StatusOr<Runtime> MakeRuntime(DataType kv_dtype, size_t capacity) {
    RuntimeOptions options;
    options.kv_cache = {.enable_manager = true,
                        .num_layers = kLayers,
                        .num_kv_heads = static_cast<size_t>(kKvHeads),
                        .max_tokens = capacity,
                        .head_dim = static_cast<size_t>(kHeadDim),
                        .kv_dtype = kv_dtype,
                        .alignment = 64};
    RuntimeBuilder builder;
    builder.WithOptions(options);
    return builder.TryBuild();
}

StatusOr<LoweredGraph> MakeGraph(DataType kv_dtype, int64_t rows,
                                 int64_t capacity, bool with_attention) {
    const TensorSpec query = Spec(DataType::Float32(), {rows, kQueryWidth});
    const TensorSpec activation = Spec(DataType::Float32(), {rows, kKvWidth});
    const TensorSpec cache = Spec(kv_dtype, {kKvHeads, capacity, kHeadDim});
    LoweredGraph::Builder graph;
    const auto input = [&](const TensorSpec& spec) {
        const GraphValueId id{.index = static_cast<uint32_t>(graph.values.size())};
        graph.values.push_back({.spec = spec, .payload = ModelInputValue{}});
        graph.model_inputs.push_back(id);
        return id;
    };
    const auto state = [&](uint32_t layer, KVCacheSlot slot) {
        const GraphValueId id{.index = static_cast<uint32_t>(graph.values.size())};
        graph.values.push_back({.spec = cache,
                                .payload = StateValue{.binding = KVCacheStateBinding{
                                                              .decoder_layer_index = layer,
                                                              .slot = slot}}});
        return id;
    };
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
        GraphValueId q{};
        if (with_attention) {
            q = input(query);
        }
        const GraphValueId k = input(activation);
        const GraphValueId v = input(activation);
        const GraphValueId k_in = state(layer, KVCacheSlot::kKey);
        const GraphValueId v_in = state(layer, KVCacheSlot::kValue);
        const GraphValueId k_out = state(layer, KVCacheSlot::kKey);
        const GraphValueId v_out = state(layer, KVCacheSlot::kValue);
        const size_t update_step = graph.steps.size();
        graph.steps.push_back({
                .spec = {.op_type = OpType::kKVCacheUpdate,
                         .selector = Selector(kv_dtype),
                         .input_specs = {activation, activation, cache, cache},
                         .output_specs = {cache, cache},
                         .op_params = KVCacheUpdateParams{}},
                .binding = {.node = {.index = static_cast<uint32_t>(update_step)},
                            .input_values = {k, v, k_in, v_in},
                            .output_values = {k_out, v_out}},
        });
        graph.state_aliases.push_back({.step_index = update_step,
                                       .input_port = 2,
                                       .output_port = 0,
                                       .input = k_in,
                                       .output = k_out});
        graph.state_aliases.push_back({.step_index = update_step,
                                       .input_port = 3,
                                       .output_port = 1,
                                       .input = v_in,
                                       .output = v_out});
        if (with_attention) {
            const GraphValueId output{.index = static_cast<uint32_t>(graph.values.size())};
            graph.values.push_back({.spec = query, .payload = ActivationValue{}});
            graph.model_outputs.push_back(output);
            graph.steps.push_back({
                    .spec = {.op_type = OpType::kAttention,
                             .selector = Selector(kv_dtype),
                             .input_specs = {query, cache, cache},
                             .output_specs = {query},
                             .op_params = AttentionParams{.num_q_heads = kQueryHeads,
                                                          .num_kv_heads = kKvHeads,
                                                          .head_dim = kHeadDim}},
                    .binding = {.node = {.index = static_cast<uint32_t>(graph.steps.size())},
                                .input_values = {q, k_out, v_out},
                                .output_values = {output}},
            });
        }
    }
    return std::move(graph).Build();
}

struct Operands {
    explicit Operands(int64_t rows, float value)
        : query(static_cast<size_t>(rows * kQueryWidth), 0.0F),
          key(static_cast<size_t>(rows * kKvWidth), 0.0F),
          value(static_cast<size_t>(rows * kKvWidth), value),
          output(static_cast<size_t>(kLayers) * rows * kQueryWidth, -1.0F) {}

    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
    std::vector<float> output;
};

StatusOr<ExecutionContext> MakeContext(Runtime& runtime, const ExecutionPlan& plan,
                                       KVCacheSessionView view, int64_t rows,
                                       Operands& operands, bool with_attention) {
    const std::array<int64_t, 2> query_shape{rows, kQueryWidth};
    const std::array<int64_t, 2> query_stride{kQueryWidth, 1};
    const std::array<int64_t, 2> kv_shape{rows, kKvWidth};
    const std::array<int64_t, 2> kv_stride{kKvWidth, 1};
    const TensorView query(operands.query.data(), DataType::Float32(), query_shape, query_stride);
    const TensorView key(operands.key.data(), DataType::Float32(), kv_shape, kv_stride);
    const TensorView value(operands.value.data(), DataType::Float32(), kv_shape, kv_stride);
    ExternalTensorBindings external;
    size_t port = 0;
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
        if (with_attention) {
            external.read_only.push_back({.value = plan.model_inputs()[port++], .tensor = query});
        }
        external.read_only.push_back({.value = plan.model_inputs()[port++], .tensor = key});
        external.read_only.push_back({.value = plan.model_inputs()[port++], .tensor = value});
        if (with_attention) {
            const MutableTensorView output(
                    operands.output.data() + static_cast<size_t>(layer) * rows * kQueryWidth,
                    DataType::Float32(), query_shape, query_stride);
            external.writable.push_back({.value = plan.model_outputs()[layer], .tensor = output});
        }
    }
    AM_ASSIGN_OR_RETURN(auto allocator, runtime.TryGetAllocator(Device::CPU()));
    AM_ASSIGN_OR_RETURN(auto bindings, PrepareExecutionBindings(plan, external, *allocator));
    return ExecutionContext::Create(plan, std::move(bindings), nullptr, view);
}

class ReservationRelease {
public:
    ReservationRelease(KVCacheManager& manager, KVCacheSessionView& view) noexcept
        : manager_(manager), view_(view) {}
    ReservationRelease(const ReservationRelease&) = delete;
    ReservationRelease& operator=(const ReservationRelease&) = delete;
    ~ReservationRelease() { static_cast<void>(manager_.ReleaseSession(view_)); }

private:
    KVCacheManager& manager_;
    KVCacheSessionView& view_;
};

Status CheckResult(const KVCacheSessionView& view, const Operands& operands,
                   DataType kv_dtype, size_t committed, float expected,
                   bool decode) {
    if (view.current_pos() != committed) {
        return Status::Internal("KV16 ExecutionStep baseline did not commit the complete plan");
    }
    for (const float value: operands.output) {
        if (!std::isfinite(value) || std::abs(value - expected) > 2.0e-6F) {
            return Status::Internal("KV16 ExecutionStep Attention failed the analytic output guard");
        }
    }
    // These exact patterns are independent of the production conversion helpers.
    const uint16_t three = kv_dtype == DataType::Float(16) ? 0x4200U : 0x4040U;
    const uint16_t five = kv_dtype == DataType::Float(16) ? 0x4500U : 0x40A0U;
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
        for (size_t head = 0; head < static_cast<size_t>(kKvHeads); ++head) {
            AM_ASSIGN_OR_RETURN(const void* key, view.KeyData(layer, head, committed - 1, 0));
            AM_ASSIGN_OR_RETURN(const void* value, view.ValueData(layer, head, committed - 1, 0));
            uint16_t key_bits = 0;
            uint16_t value_bits = 0;
            std::memcpy(&key_bits, key, sizeof(key_bits));
            std::memcpy(&value_bits, value, sizeof(value_bits));
            if (key_bits != 0 || value_bits != five) {
                return Status::Internal("KV16 ExecutionStep append failed the exact cache guard");
            }
            if (decode) {
                AM_ASSIGN_OR_RETURN(const void* history, view.ValueData(layer, head, committed - 2, 0));
                std::memcpy(&value_bits, history, sizeof(value_bits));
                if (value_bits != three) {
                    return Status::Internal("KV16 ExecutionStep Decode modified committed history");
                }
            }
        }
    }
    return Status::Ok();
}

Status RunExecutionBaseline(benchmark::State& state, DataType kv_dtype, bool decode) {
    const int64_t history = state.range(0);
    if (history <= 0 || history > 1024) {
        return Status::InvalidArgument("KV16 ExecutionStep baseline context must be in [1, 1024]");
    }
    const int64_t rows = decode ? 1 : history;
    AM_ASSIGN_OR_RETURN(auto runtime, MakeRuntime(kv_dtype, static_cast<size_t>(history + 1)));
    auto* manager = runtime.GetKVCacheManager();
    if (manager == nullptr) {
        return Status::Internal("KV16 ExecutionStep baseline requires a KV manager");
    }
    AM_ASSIGN_OR_RETURN(auto view, manager->ReserveForSession(static_cast<size_t>(history), 1));
    const ReservationRelease release(*manager, view);
    AM_ASSIGN_OR_RETURN(auto graph, MakeGraph(kv_dtype, rows, history + 1, true));
    AM_ASSIGN_OR_RETURN(auto plan, ExecutionPlanBuilder::Build(runtime, graph));
    Operands measured(rows, 5.0F);
    AM_ASSIGN_OR_RETURN(auto context, MakeContext(runtime, plan, view, rows, measured, true));

    // Decode restores only K/V, outside timing; running full historical Attention
    // here would make the untimed reset quadratic in context length.
    AM_ASSIGN_OR_RETURN(auto restore_graph, MakeGraph(kv_dtype, history, history + 1, false));
    AM_ASSIGN_OR_RETURN(auto restore_plan, ExecutionPlanBuilder::Build(runtime, restore_graph));
    Operands historical(history, 3.0F);
    AM_ASSIGN_OR_RETURN(auto restore_context,
                        MakeContext(runtime, restore_plan, view, history, historical, false));
    const size_t committed = static_cast<size_t>(history + (decode ? 1 : 0));
    const float expected = decode
                                   ? static_cast<float>((3.0 * static_cast<double>(history) + 5.0) /
                                                        static_cast<double>(history + 1))
                                   : 5.0F;
    for (auto _: state) {
        state.PauseTiming();
        const Status reset = manager->ResetSession(view);
        if (!reset.ok()) {
            state.ResumeTiming();
            return reset;
        }
        if (decode) {
            const Status restore = Executor::Execute(restore_plan, restore_context);
            if (!restore.ok()) {
                state.ResumeTiming();
                return restore;
            }
        }
        state.ResumeTiming();
        const Status executed = Executor::Execute(plan, context);
        state.PauseTiming();
        const Status checked = executed.ok()
                                       ? CheckResult(view, measured, kv_dtype, committed, expected, decode)
                                       : executed;
        state.ResumeTiming();
        if (!checked.ok()) {
            return checked;
        }
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(kLayers) * rows * kQueryWidth);
    state.counters["layers"] = kLayers;
    state.counters["query_heads"] = kQueryHeads;
    state.counters["kv_heads"] = kKvHeads;
    state.counters["head_dim"] = kHeadDim;
    state.counters["history_tokens"] = decode ? history : 0;
    state.counters["append_tokens"] = rows;
    state.counters["kv_physical_bytes"] = static_cast<double>(manager->total_bytes());
    return Status::Ok();
}

void BenchmarkKV16Execution(benchmark::State& state, DataType kv_dtype, bool decode) {
    const Status result = RunExecutionBaseline(state, kv_dtype, decode);
    if (!result.ok()) {
        state.SkipWithError(result.message().c_str());
    }
}

} // namespace

BENCHMARK_CAPTURE(BenchmarkKV16Execution, f16_prefill, DataType::Float(16), false)->Arg(128)->Arg(1024);
BENCHMARK_CAPTURE(BenchmarkKV16Execution, bf16_prefill, DataType::BFloat(16), false)->Arg(128)->Arg(1024);
BENCHMARK_CAPTURE(BenchmarkKV16Execution, f16_decode, DataType::Float(16), true)->Arg(128)->Arg(1024);
BENCHMARK_CAPTURE(BenchmarkKV16Execution, bf16_decode, DataType::BFloat(16), true)->Arg(128)->Arg(1024);
