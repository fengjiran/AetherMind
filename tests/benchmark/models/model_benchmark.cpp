// Developer-only measurements: internal prepared execution and public Generate
// have distinct timing boundaries and never replace production orchestration.
#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/cpu/cpu_workspace_arena.h"
#include "aethermind/compiler/model_compiler.h"
#include "aethermind/execution/execution_context.h"
#include "aethermind/execution/executor.h"
#include "aethermind/inference/cpu_inference_options.h"
#include "aethermind/inference/executable_model.h"
#include "aethermind/inference/inference_session.h"
#include "aethermind/inference/model_execution_policy.h"
#include "aethermind/memory/cpu_allocator.h"
#include "aethermind/model/model_loader.h"
#include "aethermind/runtime/runtime_builder.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <spdlog/spdlog.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/resource.h>
#include <utility>
#include <vector>

#if !defined(AETHERMIND_MODEL_BENCH_BUILD_TYPE) || !defined(AETHERMIND_MODEL_BENCH_SANITIZER) ||   \
        !defined(AETHERMIND_MODEL_BENCH_COMPILER) || !defined(AETHERMIND_MODEL_BENCH_CXX_FLAGS) || \
        !defined(AETHERMIND_MODEL_BENCH_HAS_AVX2_FMA)
#error "The model benchmark requires explicit build provenance"
#endif

namespace {
using namespace aethermind;
using Nanoseconds = std::optional<uint64_t>;
struct Options {
    std::filesystem::path model;
    std::vector<uint32_t> prompt;
    size_t max_new_tokens = 0;
    CpuInferencePreset preset = CpuInferencePreset::kO1Plain;
    std::string preset_name;
    size_t warmup = 1;
    size_t repetitions = 5;
    std::string mode = "timing";
    std::optional<uint32_t> eos;
};
struct Geometry {
    int64_t hidden_size = 0, intermediate_size = 0, vocab_size = 0, num_hidden_layers = 0;
    int64_t num_attention_heads = 0, num_key_value_heads = 0, head_dim = 0, max_position_embeddings = 0;
    friend bool operator==(const Geometry&, const Geometry&) = default;
};
struct DecodeStep {
    size_t index = 0, kv_before = 0, kv_after = 0;
    uint32_t input_token = 0, output_token = 0;
    Nanoseconds execute_ns;
};
struct InternalSample {
    std::vector<uint32_t> generated_tokens;
    std::string stop_reason;
    size_t final_watermark = 0;
    Nanoseconds request_setup_ns, first_token_request_ns, prefill_execute_ns, decode_setup_ns;
    std::vector<DecodeStep> decode_steps;
    std::optional<uint64_t> prefill_allocator_requests;
    std::vector<std::optional<uint64_t>> decode_allocator_requests;
};
struct Sample {
    size_t index = 0;
    Nanoseconds load_ns, compile_ns, runtime_ns, prepare_ns;
    size_t raw_unique_visible_bytes = 0, packed_unique_storage_bytes = 0;
    size_t kv_capacity_bytes = 0, workspace_bytes = 0, process_peak_rss_bytes = 0;
    InternalSample internal;
    std::vector<uint32_t> generated_tokens;
    std::string generate_stop_reason;
    Nanoseconds generate_ns;
};
struct Run {
    Sample sample;
    Geometry geometry;
    CpuCapabilities capabilities;
    size_t context_capacity = 0;
};
// Diagnostic modes deliberately never query a clock, including preparation.
class Timer {
public:
    explicit Timer(bool enabled) noexcept : enabled_(enabled) {
        if (enabled_) begin_ = std::chrono::steady_clock::now();
    }
    Nanoseconds Elapsed() const noexcept {
        if (!enabled_) return std::nullopt;
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                             std::chrono::steady_clock::now() - begin_)
                                             .count());
    }

private:
    bool enabled_;
    std::chrono::steady_clock::time_point begin_{};
};
struct AllocationCounter {
    uint64_t requests = 0;
};
// This diagnostic counts Runtime allocator requests, not arbitrary malloc.
// Keeping the delegate's Buffer makes the handle independent of allocator life.
class CountingAllocator final : public Allocator {
public:
    CountingAllocator(Device device, std::shared_ptr<AllocationCounter> counter)
        : delegate_(device), counter_(std::move(counter)) {}
    Device device() const noexcept override { return delegate_.device(); }

protected:
    StatusOr<MemoryHandle> AllocateStorage(const AllocationRequest& request) override {
        if (counter_->requests == std::numeric_limits<uint64_t>::max())
            return Status::Overflow("Allocator request count overflow");
        ++counter_->requests;
        AM_ASSIGN_OR_RETURN(auto storage, delegate_.TryAllocate(request));
        auto retained = std::make_unique<Buffer>(std::move(storage));
        MemoryHandle handle(retained->mutable_data(), retained.get(), [](void* context, void*) noexcept { delete static_cast<Buffer*>(context); }, retained->device(), retained->alignment());
        (void) retained.release();
        return handle;
    }

private:
    CPUAllocator delegate_;
    std::shared_ptr<AllocationCounter> counter_;
};
class CountingProvider final : public AllocatorProvider {
public:
    explicit CountingProvider(std::shared_ptr<AllocationCounter> counter) : counter_(std::move(counter)) {}
    std::unique_ptr<Allocator> CreateAllocator(Device device) override {
        return std::make_unique<CountingAllocator>(device, counter_);
    }

private:
    std::shared_ptr<AllocationCounter> counter_;
};
class Reservation {
public:
    Reservation(KVCacheManager& manager, KVCacheSessionView view) noexcept : manager_(manager), view_(view) {}
    Reservation(const Reservation&) = delete;
    Reservation& operator=(const Reservation&) = delete;
    ~Reservation() {
        if (view_.valid()) (void) manager_.ReleaseSession(view_);
    }
    KVCacheSessionView view() const noexcept { return view_; }
    Status Release() noexcept { return manager_.ReleaseSession(view_); }

private:
    KVCacheManager& manager_;
    KVCacheSessionView view_;
};
template<typename T>
T ParseUnsigned(std::string_view text, std::string_view name) {
    T value = 0;
    if (text.empty() || !std::ranges::all_of(text, [](char c) { return c >= '0' && c <= '9'; }))
        throw std::invalid_argument(std::string(name) + " requires an unsigned decimal integer");
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size())
        throw std::invalid_argument(std::string(name) + " is outside its integer range");
    return value;
}
Options ParseOptions(int argc, char** argv) {
    Options options;
    std::set<std::string_view> seen;
    for (int index = 1; index < argc; index += 2) {
        const std::string_view name(argv[index]);
        if (!seen.insert(name).second) throw std::invalid_argument("Duplicate option: " + std::string(name));
        if (index + 1 >= argc || std::string_view(argv[index + 1]).starts_with("--"))
            throw std::invalid_argument("Missing value for option: " + std::string(name));
        const std::string_view value(argv[index + 1]);
        if (name == "--model") {
            if (value.empty()) throw std::invalid_argument("--model requires a non-empty path");
            options.model = std::string(value);
        } else if (name == "--prompt") {
            size_t begin = 0;
            for (;;) {
                const size_t comma = value.find(',', begin);
                options.prompt.push_back(ParseUnsigned<uint32_t>(value.substr(begin, comma - begin), name));
                if (comma == std::string_view::npos) break;
                begin = comma + 1;
            }
        } else if (name == "--max-new-tokens")
            options.max_new_tokens = ParseUnsigned<size_t>(value, name);
        else if (name == "--preset") {
            options.preset_name = value;
            if (value == "o1-plain") options.preset = CpuInferencePreset::kO1Plain;
            else if (value == "o2-packed")
                options.preset = CpuInferencePreset::kO2Packed;
            else
                throw std::invalid_argument("--preset requires o1-plain or o2-packed");
        } else if (name == "--warmup")
            options.warmup = ParseUnsigned<size_t>(value, name);
        else if (name == "--repetitions") {
            options.repetitions = ParseUnsigned<size_t>(value, name);
            if (!options.repetitions) throw std::invalid_argument("--repetitions must be positive");
        } else if (name == "--mode") {
            options.mode = value;
            if (value != "timing" && value != "profile" && value != "allocations")
                throw std::invalid_argument("--mode requires timing, profile or allocations");
        } else if (name == "--eos-token")
            options.eos = ParseUnsigned<uint32_t>(value, name);
        else
            throw std::invalid_argument("Unknown option: " + std::string(name));
    }
    for (const auto required: {"--model", "--prompt", "--max-new-tokens", "--preset"})
        if (!seen.contains(required)) throw std::invalid_argument("Required option: " + std::string(required));
    if (options.warmup > 10000 || options.repetitions > 10000) {
        throw std::invalid_argument("--warmup and --repetitions must not exceed 10000");
    }
    if (options.prompt.size() > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument("Requested prompt count overflows int64_t");
    }
    return options;
}
Geometry MakeGeometry(const HfModelConfig& c) {
    return {.hidden_size = c.hidden_size, .intermediate_size = c.intermediate_size, .vocab_size = c.vocab_size, .num_hidden_layers = c.num_hidden_layers, .num_attention_heads = c.num_attention_heads, .num_key_value_heads = c.num_key_value_heads, .head_dim = c.head_dim != 0 ? c.head_dim : c.hidden_size / c.num_attention_heads, .max_position_embeddings = c.max_position_embeddings};
}
StatusOr<size_t> ContextCapacity(const Options& o, const HfModelConfig& c) {
    const size_t appends = o.max_new_tokens ? o.max_new_tokens - 1 : 0;
    if (appends > std::numeric_limits<size_t>::max() - o.prompt.size())
        return Status::Overflow("Prompt plus requested KV appends overflow size_t");
    const size_t capacity = o.prompt.size() + appends;
    if (capacity > static_cast<size_t>(c.max_position_embeddings))
        return Status::OutOfRange("Prompt plus requested KV appends exceed model context");
    for (uint32_t token: o.prompt)
        if (static_cast<uint64_t>(token) >= static_cast<uint64_t>(c.vocab_size))
            return Status::OutOfRange("Prompt token exceeds the model vocabulary");
    if (o.eos && static_cast<uint64_t>(*o.eos) >= static_cast<uint64_t>(c.vocab_size))
        return Status::OutOfRange("EOS token exceeds the model vocabulary");
    return capacity;
}
Status CheckedAdd(size_t bytes, size_t& total) {
    if (bytes > std::numeric_limits<size_t>::max() - total)
        return Status::Overflow("Model memory byte count overflow");
    total += bytes;
    return Status::Ok();
}
StatusOr<size_t> RawBytes(const ResolvedModelWeights& weights) {
    std::set<std::pair<uintptr_t, size_t>> seen;
    size_t total = 0;
    const auto add = [&](const RawWeightView& weight) -> Status {
        if (seen.emplace(reinterpret_cast<uintptr_t>(weight.data), weight.bytes).second)
            return CheckedAdd(weight.bytes, total);
        return Status::Ok();
    };
    AM_RETURN_IF_ERROR(add(weights.embed_tokens));
    AM_RETURN_IF_ERROR(add(weights.final_norm));
    if (weights.lm_head) { AM_RETURN_IF_ERROR(add(*weights.lm_head)); }
    for (const auto& layer: weights.layers)
        for (const auto* weight: {&layer.norm.input_rmsnorm, &layer.norm.post_attn_rmsnorm,
                                  &layer.attn.q_proj, &layer.attn.k_proj, &layer.attn.v_proj, &layer.attn.o_proj,
                                  &layer.mlp.gate_proj, &layer.mlp.up_proj, &layer.mlp.down_proj}) {
            AM_RETURN_IF_ERROR(add(*weight));
        }
    return total;
}
StatusOr<size_t> PackedBytes(const ExecutionPlan& prefill, const ExecutionPlan& decode) {
    std::set<const BufferImpl*> seen;
    size_t total = 0;
    for (const auto* plan: {&prefill, &decode})
        for (const auto& step: plan->steps())
            if (step.packed_weights) {
                const auto& storage = step.packed_weights->storage();
                if (seen.insert(storage.impl()).second) { AM_RETURN_IF_ERROR(CheckedAdd(storage.nbytes(), total)); }
            }
    return total;
}
StatusOr<size_t> PeakRssBytes() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss < 0 ||
        static_cast<uint64_t>(usage.ru_maxrss) > std::numeric_limits<size_t>::max() / 1024U)
        return Status::Internal("Cannot read the process RSS high-water mark");
    // Linux ru_maxrss is KiB and cumulative over the whole process.
    return static_cast<size_t>(usage.ru_maxrss) * 1024U;
}
StatusOr<ExecutionContext> PrepareContext(Runtime& runtime, const ExecutionPlan& plan,
                                          const ExternalTensorBindings& weights, std::span<const int64_t> tokens, std::span<const int64_t> positions,
                                          WorkspaceArena* workspace, KVCacheSessionView view) {
    if (plan.model_inputs().size() != 2 || plan.model_outputs().size() != 1 ||
        tokens.empty() || tokens.size() != positions.size())
        return Status::InvalidArgument("Benchmark requires canonical decoder inputs and token output");
    ExternalTensorBindings external = weights;
    const std::array<int64_t, 1> shape{static_cast<int64_t>(tokens.size())};
    constexpr std::array<int64_t, 1> strides{1};
    external.read_only.reserve(external.read_only.size() + 2);
    external.read_only.push_back({.value = plan.model_inputs()[0],
                                  .tensor = TensorView(tokens.data(), DataType::Int(64), shape, strides, alignof(int64_t))});
    external.read_only.push_back({.value = plan.model_inputs()[1],
                                  .tensor = TensorView(positions.data(), DataType::Int(64), shape, strides, alignof(int64_t))});
    AM_ASSIGN_OR_RETURN(auto allocator, runtime.TryGetAllocator(Device::CPU()));
    AM_ASSIGN_OR_RETURN(auto bindings, PrepareExecutionBindings(plan, external, *allocator));
    return ExecutionContext::Create(plan, std::move(bindings), workspace, view);
}
StatusOr<uint32_t> ReadToken(const ExecutionContext& context, const ExecutionPlan& plan, size_t sequence, size_t vocab) {
    const auto* prepared = context.prepared_bindings();
    if (!prepared || plan.model_outputs().size() != 1 || plan.model_outputs()[0].index >= prepared->values().size())
        return Status::Internal("Benchmark token output has no binding");
    const auto tensor = prepared->values()[plan.model_outputs()[0].index].ReadView();
    if (!tensor.is_valid() || tensor.dtype() != DataType::Int(64) || tensor.rank() != 1 ||
        tensor.dim(0) != static_cast<int64_t>(sequence) || !tensor.is_contiguous())
        return Status::Internal("Benchmark token output has an invalid shape or dtype");
    int64_t token = 0;
    std::memcpy(&token, static_cast<const std::byte*>(tensor.data()) + (sequence - 1) * sizeof(token), sizeof(token));
    if (token < 0 || !std::in_range<uint32_t>(token) || static_cast<uint64_t>(token) >= vocab)
        return Status::OutOfRange("Benchmark token output is outside the vocabulary");
    return static_cast<uint32_t>(token);
}
struct ExecutionMeasurement {
    Nanoseconds elapsed;
    std::optional<uint64_t> allocator_requests;
};
StatusOr<ExecutionMeasurement> Execute(const ExecutionPlan& plan, ExecutionContext& context,
                                       WorkspaceArena* workspace, bool timing, const AllocationCounter* counter) {
    const uint64_t before = counter ? counter->requests : 0;
    Timer timer(timing);
    if (workspace) workspace->Reset();
    const Status status = Executor::Execute(plan, context);
    const Nanoseconds elapsed = timer.Elapsed();
    const uint64_t after = counter ? counter->requests : 0;
    AM_RETURN_IF_ERROR(status);
    return ExecutionMeasurement{elapsed, counter ? std::optional(after - before) : std::nullopt};
}
std::string StopReason(const Options& options, const std::vector<uint32_t>& generated) {
    if (!options.max_new_tokens) return "zero_limit";
    return options.eos && !generated.empty() && generated.back() == *options.eos ? "eos" : "length";
}
StatusOr<InternalSample> RunInternal(Runtime& runtime, const ExecutableModel& model, const Options& options,
                                     const AllocationCounter* counter) {
    const bool timing = options.mode == "timing";
    InternalSample result;
    if (!options.max_new_tokens) {
        result.stop_reason = "zero_limit";
        result.request_setup_ns = timing ? Nanoseconds(0) : std::nullopt;
        return result;
    }
    AM_ASSIGN_OR_RETURN(const auto prefill, model.plan(ExecPhase::kPrefill));
    AM_ASSIGN_OR_RETURN(const auto decode, model.plan(ExecPhase::kDecode));
    AM_ASSIGN_OR_RETURN(const auto prefill_weights, model.immutable_weight_bindings(ExecPhase::kPrefill));
    AM_ASSIGN_OR_RETURN(const auto decode_weights, model.immutable_weight_bindings(ExecPhase::kDecode));
    auto* manager = runtime.GetKVCacheManager();
    if (!manager) return Status::Internal("Benchmark requires initialized KV storage");
    const size_t workspace_bytes = std::max(prefill->total_workspace_bytes(), decode->total_workspace_bytes());
    const size_t workspace_alignment = std::max(prefill->workspace_alignment(), decode->workspace_alignment());
    AM_ASSIGN_OR_RETURN(auto allocator, runtime.TryGetAllocator(Device::CPU()));
    // Instrumentation arrays are allocated before any request timer starts.
    result.decode_steps.reserve(options.max_new_tokens - 1);
    result.decode_allocator_requests.reserve(options.max_new_tokens - 1);
    Timer first_token_timer(timing);
    Timer setup_timer(timing);
    AM_ASSIGN_OR_RETURN(auto view, manager->ReserveForSession(options.prompt.size(), options.max_new_tokens - 1));
    // Declared first: all contexts and backing allocations die before release.
    Reservation reservation(*manager, view);
    std::vector<int64_t> tokens(options.prompt.begin(), options.prompt.end());
    std::vector<int64_t> positions(options.prompt.size());
    for (size_t index = 0; index < positions.size(); ++index) positions[index] = static_cast<int64_t>(index);
    result.generated_tokens.reserve(options.max_new_tokens);
    Buffer workspace_storage;
    if (workspace_bytes) {
        AM_ASSIGN_OR_RETURN(workspace_storage, allocator->TryAllocate({.nbytes = workspace_bytes, .alignment = workspace_alignment}));
    }
    CpuWorkspaceArena workspace(workspace_bytes ? workspace_storage.mutable_data() : nullptr, workspace_bytes);
    WorkspaceArena* workspace_ptr = workspace_bytes ? &workspace : nullptr;
    std::optional<ExecutionContext> prefill_context;
    {
        // Keep the moved-from macro temporary out of the reservation teardown.
        AM_ASSIGN_OR_RETURN(auto prepared_prefill, PrepareContext(runtime, *prefill, *prefill_weights, tokens, positions, workspace_ptr, view));
        prefill_context.emplace(std::move(prepared_prefill));
    }
    result.request_setup_ns = setup_timer.Elapsed();
    AM_ASSIGN_OR_RETURN(const auto prefill_measurement, Execute(*prefill, *prefill_context, workspace_ptr, timing, counter));
    result.prefill_execute_ns = prefill_measurement.elapsed;
    result.prefill_allocator_requests = prefill_measurement.allocator_requests;
    AM_ASSIGN_OR_RETURN(const uint32_t first_token, ReadToken(*prefill_context, *prefill, options.prompt.size(), model.vocab_size()));
    result.generated_tokens.push_back(first_token);
    result.first_token_request_ns = first_token_timer.Elapsed();
    const auto finished = [&] {
        return result.generated_tokens.size() == options.max_new_tokens ||
               (options.eos && result.generated_tokens.back() == *options.eos);
    };
    if (!finished()) {
        Timer decode_setup_timer(timing);
        prefill_context.reset();
        std::vector<int64_t> decode_token(1, static_cast<int64_t>(first_token));
        std::vector<int64_t> decode_position(1, static_cast<int64_t>(view.current_pos()));
        AM_ASSIGN_OR_RETURN(auto context, PrepareContext(runtime, *decode, *decode_weights, decode_token, decode_position, workspace_ptr, view));
        result.decode_setup_ns = decode_setup_timer.Elapsed();
        while (!finished()) {
            DecodeStep step{.index = result.decode_steps.size(), .kv_before = view.current_pos(), .input_token = static_cast<uint32_t>(decode_token[0])};
            AM_ASSIGN_OR_RETURN(const auto measured, Execute(*decode, context, workspace_ptr, timing, counter));
            step.execute_ns = measured.elapsed;
            AM_ASSIGN_OR_RETURN(step.output_token, ReadToken(context, *decode, 1, model.vocab_size()));
            step.kv_after = view.current_pos();
            if (step.kv_after != step.kv_before + 1)
                return Status::Internal("Decode failed to commit exactly one KV position");
            result.generated_tokens.push_back(step.output_token);
            result.decode_steps.push_back(step);
            result.decode_allocator_requests.push_back(measured.allocator_requests);
            decode_token[0] = static_cast<int64_t>(step.output_token);
            decode_position[0] = static_cast<int64_t>(step.kv_after);
        }
    }
    result.final_watermark = view.current_pos();
    if (result.final_watermark != options.prompt.size() + result.generated_tokens.size() - 1)
        return Status::Internal("Internal watermark disagrees with the generated trajectory");
    result.stop_reason = StopReason(options, result.generated_tokens);
    prefill_context.reset();
    AM_RETURN_IF_ERROR(reservation.Release());
    return result;
}
StatusOr<Run> RunSample(const Options& options, size_t index) {
    Run run;
    Sample& sample = run.sample;
    sample.index = index;
    const bool timing = options.mode == "timing";
    AM_ASSIGN_OR_RETURN(auto compile_options, MakeCpuCompileOptions(options.preset));
    Timer load_timer(timing);
    AM_ASSIGN_OR_RETURN(auto loaded, ModelLoader::Load(options.model));
    sample.load_ns = load_timer.Elapsed();
    AM_RETURN_IF_ERROR(ValidateFp32ExecutionInputs(*loaded));
    const HfModelConfig config = loaded->GetConfig();
    run.geometry = MakeGeometry(config);
    AM_ASSIGN_OR_RETURN(run.context_capacity, ContextCapacity(options, config));
    AM_ASSIGN_OR_RETURN(sample.raw_unique_visible_bytes, RawBytes(loaded->GetResolvedWeights()));
    Timer compile_timer(timing);
    AM_ASSIGN_OR_RETURN(auto artifact, ModelCompiler::Compile(std::move(loaded), compile_options));
    sample.compile_ns = compile_timer.Elapsed();
    RuntimeOptions runtime_options;
    AM_ASSIGN_OR_RETURN(runtime_options.kv_cache, MakeFp32KVCacheOptions(config, run.context_capacity));
    RuntimeBuilder builder;
    builder.WithOptions(runtime_options);
    std::shared_ptr<AllocationCounter> counter;
    if (options.mode == "allocations") {
        counter = std::make_shared<AllocationCounter>();
        builder.RegisterCustomAllocatorProvider(kCPU, std::make_unique<CountingProvider>(counter));
    }
    Timer runtime_timer(timing);
    // This final local Runtime is never moved after Prepare borrows backends.
    AM_ASSIGN_OR_RETURN(auto runtime, builder.TryBuild());
    sample.runtime_ns = runtime_timer.Elapsed();
    Timer prepare_timer(timing);
    AM_ASSIGN_OR_RETURN(auto executable, PrepareExecutableModel(runtime, std::move(artifact)));
    sample.prepare_ns = prepare_timer.Elapsed();
    auto model = std::make_shared<ExecutableModel>(std::move(executable));
    AM_ASSIGN_OR_RETURN(const auto prefill, model->plan(ExecPhase::kPrefill));
    AM_ASSIGN_OR_RETURN(const auto decode, model->plan(ExecPhase::kDecode));
    AM_ASSIGN_OR_RETURN(sample.packed_unique_storage_bytes, PackedBytes(*prefill, *decode));
    sample.workspace_bytes = std::max(prefill->total_workspace_bytes(), decode->total_workspace_bytes());
    const auto* manager = runtime.GetKVCacheManager();
    if (!manager) return Status::Internal("Prepared benchmark Runtime has no KV manager");
    sample.kv_capacity_bytes = manager->total_bytes();
    AM_ASSIGN_OR_RETURN(auto backend, runtime.GetBackend(kCPU));
    const auto* cpu_backend = dynamic_cast<const CpuBackend*>(backend);
    if (!cpu_backend) return Status::Internal("Model benchmark requires the CPU backend");
    run.capabilities = cpu_backend->cpu_capabilities();
    AM_ASSIGN_OR_RETURN(sample.internal, RunInternal(runtime, *model, options, counter.get()));
    AM_ASSIGN_OR_RETURN(auto session, InferenceSession::Create(runtime, model));
    Timer generate_timer(timing);
    auto generated = session.Generate(options.prompt, GenerationConfig{.max_new_tokens = options.max_new_tokens, .eos_token_id = options.eos});
    sample.generate_ns = generate_timer.Elapsed();
    if (!generated.ok()) return generated.status();
    sample.generated_tokens = std::move(generated).value();
    sample.generate_stop_reason = StopReason(options, sample.generated_tokens);
    if (sample.generated_tokens != sample.internal.generated_tokens || sample.generate_stop_reason != sample.internal.stop_reason)
        return Status::Internal("Prepared diagnostic and public Generate trajectories differ");
    AM_ASSIGN_OR_RETURN(sample.process_peak_rss_bytes, PeakRssBytes());
    return run;
}
void String(std::ostream& stream, std::string_view value) {
    constexpr char hex[] = "0123456789abcdef";
    stream.put('"');
    for (unsigned char c: value) {
        if (c == '"' || c == '\\') stream.put('\\').put(static_cast<char>(c));
        else if (c < 0x20)
            stream << "\\u00" << hex[c >> 4] << hex[c & 15];
        else
            stream.put(static_cast<char>(c));
    }
    stream.put('"');
}
template<typename T>
void Optional(std::ostream& stream, const std::optional<T>& value) {
    if (value) stream << *value;
    else
        stream << "null";
}
template<typename T>
void Array(std::ostream& stream, const std::vector<T>& values) {
    stream.put('[');
    for (size_t index = 0; index < values.size(); ++index) {
        if (index) stream.put(',');
        stream << values[index];
    }
    stream.put(']');
}
void FeatureArray(std::ostream& stream, const CpuFeatureSet& features) {
    stream.put('[');
    bool first = true;
    for (size_t index = 0; index < static_cast<size_t>(CpuFeature::kCount); ++index) {
        const auto feature = static_cast<CpuFeature>(index);
        if (features.Contains(feature)) {
            if (!first) stream.put(',');
            first = false;
            String(stream, ToString(feature));
        }
    }
    stream.put(']');
}
void WriteSample(std::ostream& s, const Sample& sample) {
    s << "{\"index\":" << sample.index << ",\"preparation\":{\"load_ns\":";
    Optional(s, sample.load_ns);
    s << ",\"compile_ns\":";
    Optional(s, sample.compile_ns);
    s << ",\"runtime_ns\":";
    Optional(s, sample.runtime_ns);
    s << ",\"prepare_ns\":";
    Optional(s, sample.prepare_ns);
    s << "},\"memory\":{\"raw_unique_visible_bytes\":" << sample.raw_unique_visible_bytes
      << ",\"packed_unique_storage_bytes\":" << sample.packed_unique_storage_bytes
      << ",\"kv_capacity_bytes\":" << sample.kv_capacity_bytes << ",\"workspace_bytes\":" << sample.workspace_bytes
      << ",\"process_peak_rss_bytes\":" << sample.process_peak_rss_bytes << "},\"internal\":{\"generated_tokens\":";
    const auto& i = sample.internal;
    Array(s, i.generated_tokens);
    s << ",\"stop_reason\":";
    String(s, i.stop_reason);
    s << ",\"final_watermark\":" << i.final_watermark << ",\"request_setup_ns\":";
    Optional(s, i.request_setup_ns);
    s << ",\"first_token_request_ns\":";
    Optional(s, i.first_token_request_ns);
    s << ",\"prefill_execute_ns\":";
    Optional(s, i.prefill_execute_ns);
    s << ",\"decode_setup_ns\":";
    Optional(s, i.decode_setup_ns);
    s << ",\"decode_steps\":[";
    for (size_t index = 0; index < i.decode_steps.size(); ++index) {
        const auto& step = i.decode_steps[index];
        if (index) s.put(',');
        s << "{\"index\":" << step.index << ",\"kv_before\":" << step.kv_before << ",\"kv_after\":" << step.kv_after
          << ",\"input_token\":" << step.input_token << ",\"output_token\":" << step.output_token << ",\"execute_ns\":";
        Optional(s, step.execute_ns);
        s.put('}');
    }
    s << "],\"allocator_requests\":{\"prefill_execute\":";
    Optional(s, i.prefill_allocator_requests);
    s << ",\"decode_execute\":[";
    for (size_t index = 0; index < i.decode_allocator_requests.size(); ++index) {
        if (index) s.put(',');
        Optional(s, i.decode_allocator_requests[index]);
    }
    s << "]}},\"generate\":{\"tokens\":";
    Array(s, sample.generated_tokens);
    s << ",\"stop_reason\":";
    String(s, sample.generate_stop_reason);
    s << ",\"duration_ns\":";
    Optional(s, sample.generate_ns);
    s << "}}";
}
void WriteDocument(std::ostream& s, const Options& options, const std::vector<Run>& runs) {
    s << "{\"schema_version\":1,\"status\":\"complete\",\"mode\":";
    String(s, options.mode);
    s << ",\"build\":{\"compiler\":";
    String(s, AETHERMIND_MODEL_BENCH_COMPILER);
    s << ",\"build_type\":";
    String(s, AETHERMIND_MODEL_BENCH_BUILD_TYPE);
    s << ",\"sanitizer\":";
    String(s, AETHERMIND_MODEL_BENCH_SANITIZER);
    s << ",\"compiled_avx2_fma\":" << (AETHERMIND_MODEL_BENCH_HAS_AVX2_FMA ? "true" : "false") << ",\"cxx_flags\":";
    String(s, AETHERMIND_MODEL_BENCH_CXX_FLAGS);
    s << "},\"cpu\":{\"usable_features\":";
    FeatureArray(s, runs.front().capabilities.usable_features);
    s << ",\"effective_features\":";
    FeatureArray(s, runs.front().capabilities.effective_features);
    const auto& g = runs.front().geometry;
    s << "},\"geometry\":{\"hidden_size\":" << g.hidden_size << ",\"intermediate_size\":" << g.intermediate_size
      << ",\"vocab_size\":" << g.vocab_size << ",\"num_hidden_layers\":" << g.num_hidden_layers
      << ",\"num_attention_heads\":" << g.num_attention_heads << ",\"num_key_value_heads\":" << g.num_key_value_heads
      << ",\"head_dim\":" << g.head_dim << ",\"max_position_embeddings\":" << g.max_position_embeddings
      << "},\"workload\":{\"preset\":";
    String(s, options.preset_name);
    s << ",\"prompt_tokens\":";
    Array(s, options.prompt);
    s << ",\"max_new_tokens\":" << options.max_new_tokens << ",\"eos_token_id\":";
    Optional(s, options.eos);
    s << ",\"context_capacity\":" << runs.front().context_capacity << ",\"warmup\":" << options.warmup
      << ",\"repetitions\":" << options.repetitions << "},\"samples\":[";
    for (size_t index = 0; index < runs.size(); ++index) {
        if (index) s.put(',');
        WriteSample(s, runs[index].sample);
    }
    s << "]}\n";
}
int RunCommandLine(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "Usage: aethermind_model_benchmark --model PATH --prompt 1,7,3 --max-new-tokens N\n"
                     "  --preset o1-plain|o2-packed [--warmup N] [--repetitions N]\n"
                     "  [--mode timing|profile|allocations] [--eos-token ID]\n";
        return std::cout ? 0 : 1;
    }
    const Options options = ParseOptions(argc, argv);
    std::vector<Run> runs;
    runs.reserve(options.repetitions);
    std::optional<Geometry> geometry;
    std::optional<CpuCapabilities> capabilities;
    for (size_t index = 0; index < options.warmup + options.repetitions; ++index) {
        auto result = RunSample(options, index < options.warmup ? 0 : index - options.warmup);
        if (!result.ok()) {
            std::cerr << "aethermind_model_benchmark: " << result.status().ToString() << '\n';
            return 1;
        }
        if (geometry && (*geometry != result->geometry || capabilities->usable_features != result->capabilities.usable_features ||
                         capabilities->effective_features != result->capabilities.effective_features)) {
            std::cerr << "aethermind_model_benchmark: geometry or CPU features changed during the run\n";
            return 1;
        }
        geometry = result->geometry;
        capabilities = result->capabilities;
        if (index >= options.warmup) runs.push_back(std::move(result).value());
    }
    // Serialize only after all measurements; never emit a partial success.
    std::ostringstream document;
    WriteDocument(document, options, runs);
    std::cout << document.str();
    return std::cout ? 0 : 1;
}
} // namespace
int main(int argc, char** argv) {
    try {
        spdlog::set_level(spdlog::level::off);
        return RunCommandLine(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "aethermind_model_benchmark: " << error.what() << '\n';
    } catch (...) {
        std::cerr << "aethermind_model_benchmark: unknown failure\n";
    }
    return 1;
}
