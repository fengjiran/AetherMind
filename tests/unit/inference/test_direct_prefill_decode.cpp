#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/cpu/cpu_workspace_arena.h"
#include "aethermind/base/device.h"
#include "aethermind/compiler/model_compiler.h"
#include "aethermind/execution/execution_context.h"
#include "aethermind/execution/executor.h"
#include "aethermind/inference/executable_model.h"
#include "aethermind/inference/inference_session.h"
#include "aethermind/memory/cpu_allocator.h"
#include "aethermind/model/loaded_model.h"
#include "aethermind/runtime/runtime_builder.h"
#include "execution/test_tensor_buffer_helpers.h"
#include "inference/inference_session_internal.h"
#include "inference/test_malloc_interposer.h"
#include "model/test_llama_checkpoint_helpers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace {

using namespace aethermind;
using namespace aethermind::test;

constexpr int64_t kHiddenSize = 8;
constexpr int64_t kIntermediateSize = 16;
constexpr int64_t kVocabSize = 32;
constexpr int64_t kQueryHeads = 4;
constexpr int64_t kKvHeads = 2;
constexpr int64_t kHeadDim = 2;
constexpr size_t kCacheCapacity = 8;
constexpr float kRmsNormEpsilon = 1.0e-5F;
constexpr float kLogitTolerance = 2.0e-5F;
constexpr float kKvTolerance = 2.0e-6F;

void FillWeight(RawWeightView& weight, int seed, bool is_norm = false, bool is_embedding = false) {
    const int64_t count = weight.shape[0] *
                          (weight.shape.size() > 1 ? weight.shape[1] : 1);
    const int64_t columns = weight.shape.size() > 1 ? weight.shape[1] : 1;
    auto* const data = const_cast<float*>(reinterpret_cast<const float*>(weight.data));
    for (int64_t i = 0; i < count; ++i) {
        if (is_norm) {
            const int value = static_cast<int>((seed + i * 5) % 19);
            data[i] = 0.82F + static_cast<float>(value) * 0.019F;
            continue;
        }

        const int64_t row = i / columns;
        const int64_t column = i % columns;
        const int value = static_cast<int>((row * 17 + column * 11 + seed * 7 +
                                            (row * column) % 13) %
                                           31) -
                          15;
        data[i] = static_cast<float>(value) * (is_embedding ? 0.075F : 0.024F);
    }
}

void FillTinyCheckpoint(ResolvedModelWeights& weights) {
    FillWeight(weights.embed_tokens, 3, false, true);
    FillWeight(weights.final_norm, 5, true);
    ASSERT_EQ(weights.layers.size(), 1U);
    DecoderLayerRawWeights& layer = weights.layers[0];
    FillWeight(layer.norm.input_rmsnorm, 7, true);
    FillWeight(layer.norm.post_attn_rmsnorm, 11, true);
    FillWeight(layer.attn.q_proj, 13);
    FillWeight(layer.attn.k_proj, 17);
    FillWeight(layer.attn.v_proj, 19);
    FillWeight(layer.attn.o_proj, 23);
    FillWeight(layer.mlp.gate_proj, 29);
    FillWeight(layer.mlp.up_proj, 31);
    FillWeight(layer.mlp.down_proj, 37);
}

Runtime MakeCpuRuntime() {
    RuntimeBuilder builder;
    builder.RegisterBackendFactory(
            DeviceType::kCPU, std::make_unique<CpuBackendFactory>());
    return builder.Build();
}

Runtime MakeCpuRuntimeWithKVCache(size_t kv_heads = kKvHeads,
                                  size_t capacity = kCacheCapacity,
                                  bool enable_cpu_allocator = true) {
    RuntimeOptions options;
    options.allocator.enable_cpu = enable_cpu_allocator;
    options.kv_cache.enable_manager = true;
    options.kv_cache.num_layers = 1;
    options.kv_cache.num_kv_heads = kv_heads;
    options.kv_cache.max_tokens = capacity;
    options.kv_cache.head_dim = kHeadDim;
    options.kv_cache.kv_dtype = DataType::Float32();
    options.kv_cache.alignment = 64;

    RuntimeBuilder builder;
    builder.WithOptions(options);
    builder.RegisterBackendFactory(DeviceType::kCPU, std::make_unique<CpuBackendFactory>());
    return builder.Build();
}

StatusOr<ExecutableModel> PrepareTinyExecutableModel(
        Runtime& runtime,
        ResolvedModelWeights* reference_weights) {
    const HfModelConfig config = MakeTinyLlamaConfig(/*num_layers=*/1,
                                                     /*tie_word_embeddings=*/true);
    TinyLlamaCheckpoint checkpoint = MakeTinyLlamaCheckpoint(config);
    FillTinyCheckpoint(checkpoint.weights);
    *reference_weights = checkpoint.weights;

    auto loaded = std::make_unique<LoadedModel>(config, std::move(checkpoint.weights));
    ModelCompileOptions options;
    options.optimization.opt_level = 1;
    options.lowering.enable_packed_weights = false;
    auto artifact = ModelCompiler::Compile(std::move(loaded), options);
    if (!artifact.ok()) {
        return artifact.status();
    }
    return PrepareExecutableModel(runtime, std::move(*artifact));
}

class TinyLlamaScalarReference {
public:
    explicit TinyLlamaScalarReference(const ResolvedModelWeights& weights)
        : weights_(weights),
          keys_(kCacheCapacity * kKvHeads * kHeadDim, 0.0F),
          values_(kCacheCapacity * kKvHeads * kHeadDim, 0.0F) {}

    struct Output {
        std::vector<float> logits;
        std::vector<int64_t> tokens;
    };

    Output Run(std::span<const int64_t> input_tokens,
               std::span<const int64_t> positions) {
        EXPECT_EQ(input_tokens.size(), positions.size());
        Output output;
        output.logits.reserve(input_tokens.size() * kVocabSize);
        output.tokens.reserve(input_tokens.size());

        const DecoderLayerRawWeights& layer = weights_.layers[0];
        for (size_t row = 0; row < input_tokens.size(); ++row) {
            const int64_t token = input_tokens[row];
            const int64_t position = positions[row];
            EXPECT_GE(token, 0);
            EXPECT_LT(token, kVocabSize);
            EXPECT_GE(position, 0);

            const auto* const embedding = reinterpret_cast<const float*>(weights_.embed_tokens.data) +
                                          token * kHiddenSize;
            std::vector<float> hidden(embedding, embedding + kHiddenSize);

            const std::vector<float> input_norm = RmsNorm(
                    hidden, layer.norm.input_rmsnorm);
            std::vector<float> query = Linear(input_norm, layer.attn.q_proj);
            std::vector<float> key = Linear(input_norm, layer.attn.k_proj);
            const std::vector<float> value = Linear(input_norm, layer.attn.v_proj);
            ApplyRope(query, kQueryHeads, position);
            ApplyRope(key, kKvHeads, position);
            StoreKv(keys_, key, position);
            StoreKv(values_, value, position);

            std::vector<float> attention(kHiddenSize, 0.0F);
            const int64_t group_size = kQueryHeads / kKvHeads;
            const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
            for (int64_t query_head = 0; query_head < kQueryHeads; ++query_head) {
                const int64_t kv_head = query_head / group_size;
                float max_logit = -std::numeric_limits<float>::infinity();
                for (int64_t key_position = 0; key_position <= position; ++key_position) {
                    max_logit = std::max(max_logit,
                                         AttentionLogit(query, query_head, kv_head,
                                                        key_position, scale));
                }

                std::array<float, kHeadDim> head_output{};
                float denominator = 0.0F;
                for (int64_t key_position = 0; key_position <= position; ++key_position) {
                    const float weight = std::exp(
                            AttentionLogit(query, query_head, kv_head,
                                           key_position, scale) -
                            max_logit);
                    denominator += weight;
                    for (int64_t dim = 0; dim < kHeadDim; ++dim) {
                        head_output[dim] += weight * KvAt(values_, key_position, kv_head, dim);
                    }
                }

                for (int64_t dim = 0; dim < kHeadDim; ++dim) {
                    attention[query_head * kHeadDim + dim] = head_output[dim] / denominator;
                }
            }

            const std::vector<float> projected_attention =
                    Linear(attention, layer.attn.o_proj);
            for (int64_t i = 0; i < kHiddenSize; ++i) {
                hidden[i] += projected_attention[i];
            }

            const std::vector<float> post_attention_norm = RmsNorm(
                    hidden, layer.norm.post_attn_rmsnorm);
            const std::vector<float> gate = Linear(post_attention_norm, layer.mlp.gate_proj);
            const std::vector<float> up = Linear(post_attention_norm, layer.mlp.up_proj);
            std::vector<float> activated(kIntermediateSize);
            for (int64_t i = 0; i < kIntermediateSize; ++i) {
                activated[i] = Silu(gate[i]) * up[i];
            }
            const std::vector<float> down = Linear(activated, layer.mlp.down_proj);
            for (int64_t i = 0; i < kHiddenSize; ++i) {
                hidden[i] += down[i];
            }

            const std::vector<float> final_hidden = RmsNorm(hidden, weights_.final_norm);
            const int64_t winner = AppendLogits(final_hidden, output.logits);
            output.tokens.push_back(winner);
        }
        return output;
    }

    const std::vector<float>& keys() const noexcept {
        return keys_;
    }

    const std::vector<float>& values() const noexcept {
        return values_;
    }

private:
    static std::vector<float> Linear(const std::vector<float>& input,
                                     const RawWeightView& weight) {
        const int64_t output_size = weight.shape[0];
        const int64_t input_size = weight.shape[1];
        const auto* const data = reinterpret_cast<const float*>(weight.data);
        std::vector<float> output(static_cast<size_t>(output_size));
        for (int64_t out = 0; out < output_size; ++out) {
            double sum = 0.0;
            for (int64_t in = 0; in < input_size; ++in) {
                sum += static_cast<double>(input[in]) * data[out * input_size + in];
            }
            output[out] = static_cast<float>(sum);
        }
        return output;
    }

    static std::vector<float> RmsNorm(const std::vector<float>& input,
                                      const RawWeightView& weight) {
        double sum_sq = 0.0;
        for (const float value: input) {
            sum_sq += static_cast<double>(value) * value;
        }
        const double inv_rms = 1.0 / std::sqrt(
                                             sum_sq / static_cast<double>(input.size()) +
                                             static_cast<double>(kRmsNormEpsilon));
        const auto* const weight_data = reinterpret_cast<const float*>(weight.data);
        std::vector<float> output(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = static_cast<float>(static_cast<double>(input[i]) * inv_rms *
                                           static_cast<double>(weight_data[i]));
        }
        return output;
    }

    static float Silu(float value) {
        if (value >= 0.0F) {
            return value / (1.0F + std::exp(-value));
        }
        const float exp_value = std::exp(value);
        return value * exp_value / (1.0F + exp_value);
    }

    static void ApplyRope(std::vector<float>& values,
                          int64_t heads,
                          int64_t position) {
        const double angle = static_cast<double>(position);
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        for (int64_t head = 0; head < heads; ++head) {
            const size_t first_index = static_cast<size_t>(head * kHeadDim);
            const size_t second_index = first_index + 1;
            const float first = values[first_index];
            const float second = values[second_index];
            values[first_index] = static_cast<float>(
                    static_cast<double>(first) * cosine -
                    static_cast<double>(second) * sine);
            values[second_index] = static_cast<float>(
                    static_cast<double>(second) * cosine +
                    static_cast<double>(first) * sine);
        }
    }

    static size_t KvOffset(int64_t position, int64_t head, int64_t dim) {
        return static_cast<size_t>(position * kKvHeads * kHeadDim +
                                   head * kHeadDim + dim);
    }

    static void StoreKv(std::vector<float>& cache,
                        const std::vector<float>& row,
                        int64_t position) {
        for (int64_t head = 0; head < kKvHeads; ++head) {
            for (int64_t dim = 0; dim < kHeadDim; ++dim) {
                cache[KvOffset(position, head, dim)] = row[head * kHeadDim + dim];
            }
        }
    }

    float KvAt(const std::vector<float>& cache,
               int64_t position,
               int64_t head,
               int64_t dim) const {
        return cache[KvOffset(position, head, dim)];
    }

    float AttentionLogit(const std::vector<float>& query,
                         int64_t query_head,
                         int64_t kv_head,
                         int64_t key_position,
                         float scale) const {
        float dot = 0.0F;
        for (int64_t dim = 0; dim < kHeadDim; ++dim) {
            dot += query[query_head * kHeadDim + dim] *
                   KvAt(keys_, key_position, kv_head, dim);
        }
        return dot * scale;
    }

    int64_t AppendLogits(const std::vector<float>& hidden,
                         std::vector<float>& logits) const {
        const auto* const embedding = reinterpret_cast<const float*>(weights_.embed_tokens.data);
        int64_t winner = 0;
        float best = -std::numeric_limits<float>::infinity();
        for (int64_t token = 0; token < kVocabSize; ++token) {
            double sum = 0.0;
            for (int64_t dim = 0; dim < kHiddenSize; ++dim) {
                sum += static_cast<double>(hidden[dim]) *
                       embedding[token * kHiddenSize + dim];
            }
            const float value = static_cast<float>(sum);
            logits.push_back(value);
            if (token == 0 || value > best) {
                best = value;
                winner = token;
            }
        }
        return winner;
    }

    const ResolvedModelWeights& weights_;
    std::vector<float> keys_;
    std::vector<float> values_;
};

StatusOr<InferenceSession> PrepareTinySession(
        Runtime& runtime,
        ResolvedModelWeights* reference_weights) {
    auto executable = PrepareTinyExecutableModel(runtime, reference_weights);
    if (!executable.ok()) {
        return executable.status();
    }
    auto shared_model =
            std::make_shared<ExecutableModel>(std::move(*executable));
    return InferenceSession::Create(runtime, std::move(shared_model));
}

std::vector<uint32_t> ScalarGenerate(
        const ResolvedModelWeights& weights,
        std::span<const uint32_t> prompt,
        const GenerationConfig& config) {
    std::vector<uint32_t> generated;
    if (config.max_new_tokens == 0) {
        return generated;
    }

    std::vector<int64_t> prompt_ids(prompt.begin(), prompt.end());
    std::vector<int64_t> prompt_positions(prompt.size());
    for (size_t i = 0; i < prompt_positions.size(); ++i) {
        prompt_positions[i] = static_cast<int64_t>(i);
    }

    TinyLlamaScalarReference reference(weights);
    auto output = reference.Run(prompt_ids, prompt_positions);
    generated.reserve(config.max_new_tokens);
    generated.push_back(static_cast<uint32_t>(output.tokens.back()));
    if (config.eos_token_id.has_value() &&
        generated.back() == *config.eos_token_id) {
        return generated;
    }

    while (generated.size() < config.max_new_tokens) {
        const std::array<int64_t, 1> input{
                static_cast<int64_t>(generated.back())};
        const std::array<int64_t, 1> position{
                static_cast<int64_t>(prompt.size() + generated.size() - 1)};
        output = reference.Run(input, position);
        generated.push_back(static_cast<uint32_t>(output.tokens.back()));
        if (config.eos_token_id.has_value() &&
            generated.back() == *config.eos_token_id) {
            break;
        }
    }
    return generated;
}

void ReleaseTestSession(KVCacheManager& manager, KVCacheView& cache_view) {
    const Status status = manager.ReleaseSession(cache_view);
    EXPECT_TRUE(status.ok()) << status.ToString();
}

StatusOr<ExecutionContext> PrepareContext(Runtime& runtime,
                                          const ExecutionPlan& plan,
                                          const ExternalTensorBindings& immutable_bindings,
                                          const TestBuffer& tokens,
                                          const TestBuffer& positions,
                                          WorkspaceArena* workspace,
                                          KVCacheView cache_view) {
    if (plan.model_inputs().size() != 2) {
        return Status::Internal("tiny Llama plan must have token and position inputs");
    }
    ExternalTensorBindings external = immutable_bindings;
    external.readable.push_back({.value = plan.model_inputs()[0], .tensor = tokens.view()});
    external.readable.push_back({.value = plan.model_inputs()[1], .tensor = positions.view()});

    auto prepared = PrepareExecutionBindings(
            plan, external, runtime.GetAllocator(Device::CPU()));
    if (!prepared.ok()) {
        return prepared.status();
    }
    return ExecutionContext::Create(plan, std::move(*prepared), workspace,
                                    std::move(cache_view));
}

MutableTensorView FindOutput(const ExecutionContext& context,
                             const ExecutionPlan& plan,
                             ExecutionValueId value) {
    const PreparedExecutionBindings* const prepared = context.prepared_bindings();
    if (prepared == nullptr) {
        return {};
    }
    for (size_t step_index = 0; step_index < plan.steps().size(); ++step_index) {
        const ExecutionStep& step = plan.steps()[step_index];
        for (size_t compact_output = 0;
             compact_output < step.kernel_output_ports.size(); ++compact_output) {
            const uint32_t port = step.kernel_output_ports[compact_output];
            if (port < step.outputs.size() && step.outputs[port] == value) {
                return prepared->step(step_index).outputs[compact_output];
            }
        }
    }
    return {};
}

struct StageSnapshot {
    std::vector<float> logits;
    std::vector<int64_t> tokens;
    std::vector<float> keys;
    std::vector<float> values;
    size_t commit_position = 0;
};

std::optional<StageSnapshot> CaptureStage(const ExecutionContext& context,
                                          const ExecutionPlan& plan,
                                          ExecutionValueId logits_id,
                                          ExecutionValueId tokens_id,
                                          size_t sequence_length,
                                          KVCacheView cache_view) {
    const MutableTensorView logits_view = FindOutput(context, plan, logits_id);
    const MutableTensorView tokens_view = FindOutput(context, plan, tokens_id);
    if (!logits_view.is_valid() || !tokens_view.is_valid()) {
        ADD_FAILURE() << "could not locate the prepared logits or token output";
        return std::nullopt;
    }

    StageSnapshot snapshot;
    snapshot.logits.assign(static_cast<const float*>(logits_view.data()),
                           static_cast<const float*>(logits_view.data()) +
                                   sequence_length * kVocabSize);
    snapshot.tokens.assign(static_cast<const int64_t*>(tokens_view.data()),
                           static_cast<const int64_t*>(tokens_view.data()) + sequence_length);
    snapshot.commit_position = cache_view.current_pos();

    for (size_t position = 0; position < snapshot.commit_position; ++position) {
        for (size_t head = 0; head < kKvHeads; ++head) {
            for (size_t dim = 0; dim < kHeadDim; ++dim) {
                const auto key = cache_view.KeyData(0, head, position, dim);
                const auto value = cache_view.ValueData(0, head, position, dim);
                if (!key.ok() || !value.ok()) {
                    ADD_FAILURE() << "KV read failed for position " << position;
                    return std::nullopt;
                }
                snapshot.keys.push_back(*static_cast<const float*>(*key));
                snapshot.values.push_back(*static_cast<const float*>(*value));
            }
        }
    }
    return snapshot;
}

void ExpectStageMatches(const StageSnapshot& actual,
                        const TinyLlamaScalarReference::Output& expected,
                        const TinyLlamaScalarReference& reference,
                        size_t expected_commit_position) {
    EXPECT_EQ(actual.tokens, expected.tokens);
    EXPECT_EQ(actual.commit_position, expected_commit_position);
    ASSERT_EQ(actual.logits.size(), expected.logits.size());
    for (size_t i = 0; i < actual.logits.size(); ++i) {
        EXPECT_TRUE(std::isfinite(actual.logits[i])) << "logit " << i;
        EXPECT_NEAR(actual.logits[i], expected.logits[i], kLogitTolerance)
                << "logit " << i;
    }

    const size_t expected_kv_count = expected_commit_position * kKvHeads * kHeadDim;
    ASSERT_EQ(actual.keys.size(), expected_kv_count);
    ASSERT_EQ(actual.values.size(), expected_kv_count);
    for (size_t i = 0; i < expected_kv_count; ++i) {
        EXPECT_NEAR(actual.keys[i], reference.keys()[i], kKvTolerance) << "key " << i;
        EXPECT_NEAR(actual.values[i], reference.values()[i], kKvTolerance) << "value " << i;
    }
}


std::vector<const void*> PreparedAddresses(const ExecutionContext& context) {
    std::vector<const void*> addresses;
    const PreparedExecutionBindings* const prepared = context.prepared_bindings();
    if (prepared == nullptr) {
        return addresses;
    }
    for (size_t i = 0; i < prepared->step_count(); ++i) {
        const StepTensorBinding& step = prepared->step(i);
        for (const TensorView& input: step.inputs) {
            addresses.push_back(input.data());
        }
        for (const MutableTensorView& output: step.outputs) {
            addresses.push_back(output.data());
        }
        addresses.push_back(prepared->kernel_params(i));
    }
    return addresses;
}

class WorkspaceStorage {
public:
    WorkspaceStorage(size_t size, size_t alignment)
        : size_(size), alignment_(std::max(alignment, alignof(std::max_align_t))) {
        if (size_ != 0) {
            data_ = ::operator new(size_, std::align_val_t{alignment_});
        }
    }

    WorkspaceStorage(const WorkspaceStorage&) = delete;
    WorkspaceStorage& operator=(const WorkspaceStorage&) = delete;
    ~WorkspaceStorage() {
        if (data_ != nullptr) {
            ::operator delete(data_, std::align_val_t{alignment_});
        }
    }

    void* data() const noexcept {
        return data_;
    }

    size_t size() const noexcept {
        return size_;
    }

private:
    void* data_ = nullptr;
    size_t size_ = 0;
    size_t alignment_ = alignof(std::max_align_t);
};

struct DirectRunResult {
    std::array<StageSnapshot, 3> stages;
};

std::optional<DirectRunResult> RunSuccessfulSession(
        Runtime& runtime,
        const ExecutionPlan& plan,
        const ExternalTensorBindings& immutable_bindings,
        const ResolvedModelWeights& reference_weights,
        ExecutionValueId logits_id,
        ExecutionValueId tokens_id,
        std::span<const int64_t> prompt) {
    KVCacheManager* const manager = runtime.GetKVCacheManager();
    if (manager == nullptr) {
        ADD_FAILURE() << "Runtime has no KVCacheManager";
        return std::nullopt;
    }
    auto reserved = manager->ReserveForSession(prompt.size(), 2);
    if (!reserved.ok()) {
        ADD_FAILURE() << reserved.status().ToString();
        return std::nullopt;
    }
    KVCacheView cache_view = std::move(*reserved);

    WorkspaceStorage workspace_storage(plan.total_workspace_bytes(),
                                       plan.workspace_alignment());
    CpuWorkspaceArena workspace(workspace_storage.data(), workspace_storage.size());
    WorkspaceArena* const workspace_ptr = plan.total_workspace_bytes() == 0
                                                  ? nullptr
                                                  : &workspace;

    TestBuffer prefill_tokens(DataType::Int(64), {static_cast<int64_t>(prompt.size())});
    TestBuffer prefill_positions(DataType::Int(64), {static_cast<int64_t>(prompt.size())});
    std::copy(prompt.begin(), prompt.end(),
              static_cast<int64_t*>(prefill_tokens.mutable_data()));
    for (size_t i = 0; i < prompt.size(); ++i) {
        static_cast<int64_t*>(prefill_positions.mutable_data())[i] =
                static_cast<int64_t>(i);
    }

    TestBuffer decode_tokens(DataType::Int(64), {1});
    TestBuffer decode_positions(DataType::Int(64), {1});
    auto prefill_context_result = PrepareContext(
            runtime, plan, immutable_bindings, prefill_tokens, prefill_positions,
            workspace_ptr, cache_view);
    if (!prefill_context_result.ok()) {
        ADD_FAILURE() << prefill_context_result.status().ToString();
        ReleaseTestSession(*manager, cache_view);
        return std::nullopt;
    }
    ExecutionContext prefill_context(std::move(*prefill_context_result));
    auto decode_context_result = PrepareContext(
            runtime, plan, immutable_bindings, decode_tokens, decode_positions,
            workspace_ptr, cache_view);
    if (!decode_context_result.ok()) {
        ADD_FAILURE() << decode_context_result.status().ToString();
        prefill_context.Clear();
        ReleaseTestSession(*manager, cache_view);
        return std::nullopt;
    }
    ExecutionContext decode_context(std::move(*decode_context_result));

    DirectRunResult result;
    TinyLlamaScalarReference reference(reference_weights);
    std::vector<int64_t> prompt_positions(prompt.size());
    for (size_t i = 0; i < prompt_positions.size(); ++i) {
        prompt_positions[i] = static_cast<int64_t>(i);
    }
    const TinyLlamaScalarReference::Output expected_prefill =
            reference.Run(prompt, prompt_positions);

    const Status prefill_status = Executor::Execute(plan, prefill_context);
    if (!prefill_status.ok()) {
        ADD_FAILURE() << "Prefill failed: " << prefill_status.ToString();
        decode_context.Clear();
        prefill_context.Clear();
        ReleaseTestSession(*manager, cache_view);
        return std::nullopt;
    }
    auto prefill_snapshot = CaptureStage(prefill_context, plan, logits_id,
                                         tokens_id, prompt.size(), cache_view);
    if (!prefill_snapshot.has_value()) {
        decode_context.Clear();
        prefill_context.Clear();
        ReleaseTestSession(*manager, cache_view);
        return std::nullopt;
    }
    result.stages[0] = std::move(*prefill_snapshot);
    ExpectStageMatches(result.stages[0], expected_prefill, reference, prompt.size());

    int64_t decode_token = expected_prefill.tokens.back();
    int64_t decode_position = static_cast<int64_t>(prompt.size());
    TinyLlamaScalarReference::Output expected_decode1 = reference.Run(
            std::span<const int64_t>(&decode_token, 1),
            std::span<const int64_t>(&decode_position, 1));
    const std::vector<const void*> decode_addresses = PreparedAddresses(decode_context);
    const MutableTensorView decode_token_output =
            FindOutput(decode_context, plan, tokens_id);
    if (!decode_token_output.is_valid()) {
        ADD_FAILURE() << "Could not locate the prepared Decode token output";
        decode_context.Clear();
        prefill_context.Clear();
        ReleaseTestSession(*manager, cache_view);
        return std::nullopt;
    }

    const auto execute_decode_iteration = [&](int64_t input_token,
                                              int64_t position,
                                              int64_t& output_token,
                                              MallocCallCounts& allocations) {
        const auto run_iteration = [&] {
            *static_cast<int64_t*>(decode_tokens.mutable_data()) = input_token;
            *static_cast<int64_t*>(decode_positions.mutable_data()) = position;
            if (workspace_ptr != nullptr) {
                workspace_ptr->Reset();
            }
            const Status status = Executor::Execute(plan, decode_context);
            output_token = *static_cast<const int64_t*>(decode_token_output.data());
            return status;
        };

        if (!MallocInterposerAvailable()) {
            return run_iteration();
        }
        BeginMallocCallCounting();
        const Status status = run_iteration();
        allocations = EndMallocCallCounting();
        return status;
    };

    int64_t decode1_output_token = 0;
    MallocCallCounts decode1_allocations;
    const Status decode1_status = execute_decode_iteration(
            result.stages[0].tokens.back(), decode_position,
            decode1_output_token, decode1_allocations);
    if (!decode1_status.ok()) {
        ADD_FAILURE() << "Decode #1 failed: " << decode1_status.ToString();
        decode_context.Clear();
        prefill_context.Clear();
        ReleaseTestSession(*manager, cache_view);
        return std::nullopt;
    }
    if (MallocInterposerAvailable()) {
        EXPECT_EQ(decode1_allocations.malloc_calls, 0U);
        EXPECT_EQ(decode1_allocations.calloc_calls, 0U);
        EXPECT_EQ(decode1_allocations.realloc_calls, 0U);
        EXPECT_EQ(decode1_allocations.free_calls, 0U);
        EXPECT_EQ(decode1_allocations.aligned_alloc_calls, 0U);
        EXPECT_EQ(decode1_allocations.posix_memalign_calls, 0U);
        EXPECT_EQ(decode1_allocations.memalign_calls, 0U);
    }
    EXPECT_EQ(PreparedAddresses(decode_context), decode_addresses);
    auto decode1_snapshot = CaptureStage(decode_context, plan, logits_id,
                                         tokens_id, 1, cache_view);
    if (!decode1_snapshot.has_value()) {
        decode_context.Clear();
        prefill_context.Clear();
        ReleaseTestSession(*manager, cache_view);
        return std::nullopt;
    }
    result.stages[1] = std::move(*decode1_snapshot);
    EXPECT_EQ(result.stages[1].tokens.back(), decode1_output_token);
    ExpectStageMatches(result.stages[1], expected_decode1, reference,
                       prompt.size() + 1);

    decode_token = expected_decode1.tokens.back();
    ++decode_position;
    TinyLlamaScalarReference::Output expected_decode2 = reference.Run(
            std::span<const int64_t>(&decode_token, 1),
            std::span<const int64_t>(&decode_position, 1));

    int64_t decode2_output_token = 0;
    MallocCallCounts decode2_allocations;
    const Status decode2_status = execute_decode_iteration(
            decode1_output_token, decode_position,
            decode2_output_token, decode2_allocations);
    if (!decode2_status.ok()) {
        ADD_FAILURE() << "Decode #2 failed: " << decode2_status.ToString();
        decode_context.Clear();
        prefill_context.Clear();
        ReleaseTestSession(*manager, cache_view);
        return std::nullopt;
    }
    if (MallocInterposerAvailable()) {
        EXPECT_EQ(decode2_allocations.malloc_calls, 0U);
        EXPECT_EQ(decode2_allocations.calloc_calls, 0U);
        EXPECT_EQ(decode2_allocations.realloc_calls, 0U);
        EXPECT_EQ(decode2_allocations.free_calls, 0U);
        EXPECT_EQ(decode2_allocations.aligned_alloc_calls, 0U);
        EXPECT_EQ(decode2_allocations.posix_memalign_calls, 0U);
        EXPECT_EQ(decode2_allocations.memalign_calls, 0U);
    }
    EXPECT_EQ(PreparedAddresses(decode_context), decode_addresses);
    auto decode2_snapshot = CaptureStage(decode_context, plan, logits_id,
                                         tokens_id, 1, cache_view);
    if (!decode2_snapshot.has_value()) {
        decode_context.Clear();
        prefill_context.Clear();
        ReleaseTestSession(*manager, cache_view);
        return std::nullopt;
    }
    result.stages[2] = std::move(*decode2_snapshot);
    EXPECT_EQ(result.stages[2].tokens.back(), decode2_output_token);
    ExpectStageMatches(result.stages[2], expected_decode2, reference,
                       prompt.size() + 2);

    decode_context.Clear();
    prefill_context.Clear();
    const Status release_status = manager->ReleaseSession(cache_view);
    if (!release_status.ok()) {
        ADD_FAILURE() << "ReleaseSession failed: " << release_status.ToString();
        return std::nullopt;
    }
    return result;
}

bool RunFailureCleanupScenario(Runtime& runtime,
                               const ExecutionPlan& plan,
                               const ExternalTensorBindings& immutable_bindings,
                               std::span<const int64_t> prompt) {
    KVCacheManager* const manager = runtime.GetKVCacheManager();
    auto reserved = manager->ReserveForSession(prompt.size(), 2);
    if (!reserved.ok()) {
        ADD_FAILURE() << reserved.status().ToString();
        return false;
    }
    KVCacheView cache_view = std::move(*reserved);

    WorkspaceStorage workspace_storage(plan.total_workspace_bytes(),
                                       plan.workspace_alignment());
    CpuWorkspaceArena workspace(workspace_storage.data(), workspace_storage.size());
    WorkspaceArena* const workspace_ptr = plan.total_workspace_bytes() == 0
                                                  ? nullptr
                                                  : &workspace;
    TestBuffer tokens(DataType::Int(64), {static_cast<int64_t>(prompt.size())});
    TestBuffer positions(DataType::Int(64), {static_cast<int64_t>(prompt.size())});
    std::copy(prompt.begin(), prompt.end(), static_cast<int64_t*>(tokens.mutable_data()));
    for (size_t i = 0; i < prompt.size(); ++i) {
        static_cast<int64_t*>(positions.mutable_data())[i] = static_cast<int64_t>(i);
    }
    TestBuffer decode_token(DataType::Int(64), {1});
    TestBuffer decode_position(DataType::Int(64), {1});
    auto prefill_result = PrepareContext(runtime, plan, immutable_bindings,
                                         tokens, positions, workspace_ptr, cache_view);
    auto decode_result = PrepareContext(runtime, plan, immutable_bindings,
                                        decode_token, decode_position, workspace_ptr, cache_view);
    if (!prefill_result.ok() || !decode_result.ok()) {
        ADD_FAILURE() << "Could not prepare failure-path contexts";
        if (prefill_result.ok()) {
            prefill_result->Clear();
        }
        if (decode_result.ok()) {
            decode_result->Clear();
        }
        ReleaseTestSession(*manager, cache_view);
        return false;
    }
    ExecutionContext prefill_context(std::move(*prefill_result));
    ExecutionContext decode_context(std::move(*decode_result));
    const Status prefill_status = Executor::Execute(plan, prefill_context);
    if (!prefill_status.ok()) {
        ADD_FAILURE() << "Failure-path Prefill failed: " << prefill_status.ToString();
        decode_context.Clear();
        prefill_context.Clear();
        ReleaseTestSession(*manager, cache_view);
        return false;
    }
    EXPECT_EQ(cache_view.current_pos(), prompt.size());

    const MutableTensorView token_output = FindOutput(
            prefill_context, plan, plan.model_outputs().front());
    if (!token_output.is_valid()) {
        ADD_FAILURE() << "Could not locate Prefill token output";
        decode_context.Clear();
        prefill_context.Clear();
        ReleaseTestSession(*manager, cache_view);
        return false;
    }
    *static_cast<int64_t*>(decode_token.mutable_data()) =
            static_cast<const int64_t*>(token_output.data())[prompt.size() - 1];
    *static_cast<int64_t*>(decode_position.mutable_data()) = -1;

    const Status decode_status = Executor::Execute(plan, decode_context);
    EXPECT_FALSE(decode_status.ok());
    EXPECT_EQ(decode_status.code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(cache_view.current_pos(), prompt.size());

    decode_context.Clear();
    prefill_context.Clear();
    const Status release_status = manager->ReleaseSession(cache_view);
    EXPECT_TRUE(release_status.ok()) << release_status.ToString();

    auto next_reservation = manager->ReserveForSession(prompt.size(), 2);
    EXPECT_TRUE(next_reservation.ok()) << next_reservation.status().ToString();
    if (next_reservation.ok()) {
        KVCacheView next_view = std::move(*next_reservation);
        const Status next_release = manager->ReleaseSession(next_view);
        EXPECT_TRUE(next_release.ok()) << next_release.ToString();
    }
    return true;
}

void ExpectSameStage(const StageSnapshot& lhs, const StageSnapshot& rhs) {
    EXPECT_EQ(lhs.tokens, rhs.tokens);
    EXPECT_EQ(lhs.commit_position, rhs.commit_position);
    ASSERT_EQ(lhs.logits.size(), rhs.logits.size());
    ASSERT_EQ(lhs.keys.size(), rhs.keys.size());
    ASSERT_EQ(lhs.values.size(), rhs.values.size());
    EXPECT_EQ(lhs.logits, rhs.logits);
    EXPECT_EQ(lhs.keys, rhs.keys);
    EXPECT_EQ(lhs.values, rhs.values);
}

TEST(DirectPrefillDecode, TinyTiedGqaLlamaMatchesScalarOracleAndReusesDecodeBindings) {
    Runtime runtime = MakeCpuRuntimeWithKVCache();
    ResolvedModelWeights reference_weights;
    auto executable_result = PrepareTinyExecutableModel(runtime, &reference_weights);
    ASSERT_TRUE(executable_result.ok()) << executable_result.status().ToString();
    ExecutableModel executable(std::move(*executable_result));

    const auto* const embedding =
            reinterpret_cast<const float*>(reference_weights.embed_tokens.data);
    const auto* const q_projection =
            reinterpret_cast<const float*>(reference_weights.layers[0].attn.q_proj.data);
    EXPECT_NE(embedding[0], 0.0F);
    EXPECT_NE(embedding[kHiddenSize], embedding[0]);
    EXPECT_NE(q_projection[0], 0.0F);
    EXPECT_NE(q_projection[kHiddenSize], q_projection[0]);

    const auto prefill_plan_result = executable.plan(ExecPhase::kPrefill);
    const auto decode_plan_result = executable.plan(ExecPhase::kDecode);
    ASSERT_TRUE(prefill_plan_result.ok()) << prefill_plan_result.status().ToString();
    ASSERT_TRUE(decode_plan_result.ok()) << decode_plan_result.status().ToString();
    ASSERT_EQ(*prefill_plan_result, *decode_plan_result);
    const ExecutionPlan& plan = **prefill_plan_result;
    ASSERT_EQ(plan.model_inputs().size(), 2U);
    ASSERT_EQ(plan.model_outputs().size(), 1U);
    ASSERT_EQ(plan.values()[plan.model_inputs()[0].index].spec.dtype, DataType::Int(64));
    ASSERT_EQ(plan.values()[plan.model_inputs()[1].index].spec.dtype, DataType::Int(64));

    ExecutionValueId logits_id{};
    bool found_argmax = false;
    for (const ExecutionStep& step: plan.steps()) {
        if (step.kernel.op_type == OpType::kArgmax) {
            ASSERT_FALSE(found_argmax);
            ASSERT_EQ(step.inputs.size(), 1U);
            ASSERT_EQ(step.outputs.size(), 1U);
            logits_id = step.inputs[0];
            found_argmax = true;
        }
    }
    ASSERT_TRUE(found_argmax);
    const ExecutionValueId tokens_id = plan.model_outputs()[0];
    ASSERT_EQ(plan.values()[logits_id.index].spec.dtype, DataType::Float32());
    ASSERT_EQ(plan.values()[tokens_id.index].spec.dtype, DataType::Int(64));

    const auto immutable_bindings_result = executable.immutable_weight_bindings(ExecPhase::kBoth);
    ASSERT_TRUE(immutable_bindings_result.ok()) << immutable_bindings_result.status().ToString();
    const ExternalTensorBindings& immutable_bindings = **immutable_bindings_result;
    size_t embedding_backing_bindings = 0;
    for (const ExternalReadOnlyValueBinding& binding: immutable_bindings.readable) {
        if (binding.tensor.data() == reference_weights.embed_tokens.data) {
            ++embedding_backing_bindings;
        }
    }
    EXPECT_EQ(embedding_backing_bindings, 2U)
            << "tied lm-head must reuse the embedding backing";

    constexpr std::array<int64_t, 3> prompt = {1, 7, 3};
    const auto first_run = RunSuccessfulSession(
            runtime, plan, immutable_bindings, reference_weights,
            logits_id, tokens_id, prompt);
    ASSERT_TRUE(first_run.has_value());
    ASSERT_GE(first_run->stages[0].keys.size(), 3 * kKvHeads * kHeadDim);
    EXPECT_NE(first_run->stages[0].keys[0], first_run->stages[0].keys[2])
            << "KV head projections must be distinguishable";
    EXPECT_NE(first_run->stages[0].keys[0], first_run->stages[0].keys[kKvHeads * kHeadDim])
            << "RoPE positions must be distinguishable";
    const auto repeated_run = RunSuccessfulSession(
            runtime, plan, immutable_bindings, reference_weights,
            logits_id, tokens_id, prompt);
    ASSERT_TRUE(repeated_run.has_value());
    for (size_t stage = 0; stage < first_run->stages.size(); ++stage) {
        ExpectSameStage(first_run->stages[stage], repeated_run->stages[stage]);
    }

    EXPECT_TRUE(RunFailureCleanupScenario(runtime, plan, immutable_bindings, prompt));
}

TEST(InferenceSession, RejectsRuntimeDifferentFromPreparationRuntime) {
    Runtime preparation_runtime = MakeCpuRuntimeWithKVCache();
    ResolvedModelWeights reference_weights;
    auto executable = PrepareTinyExecutableModel(
            preparation_runtime, &reference_weights);
    ASSERT_TRUE(executable.ok()) << executable.status().ToString();
    auto shared_model =
            std::make_shared<ExecutableModel>(std::move(*executable));

    Runtime other_runtime = MakeCpuRuntime();
    auto session = InferenceSession::Create(other_runtime, shared_model);
    ASSERT_FALSE(session.ok());
    EXPECT_EQ(session.status().code(), StatusCode::kFailedPrecondition);
}

TEST(InferenceSession, TinyGenerateMatchesScalarOracleAndStartsFreshEachCall) {
    Runtime runtime = MakeCpuRuntimeWithKVCache();
    ResolvedModelWeights reference_weights;
    auto session_result = PrepareTinySession(runtime, &reference_weights);
    ASSERT_TRUE(session_result.ok()) << session_result.status().ToString();
    InferenceSession session = std::move(*session_result);

    constexpr std::array<uint32_t, 3> prompt{1, 7, 3};
    const GenerationConfig config{.max_new_tokens = 3};
    const std::vector<uint32_t> expected =
            ScalarGenerate(reference_weights, prompt, config);

    auto first = session.Generate(prompt, config);
    ASSERT_TRUE(first.ok()) << first.status().ToString();
    EXPECT_EQ(*first, expected);

    auto repeated = session.Generate(prompt, config);
    ASSERT_TRUE(repeated.ok()) << repeated.status().ToString();
    EXPECT_EQ(*repeated, expected);
}

TEST(InferenceSession, HandlesZeroOneEosAndExactKvCapacity) {
    Runtime runtime = MakeCpuRuntimeWithKVCache();
    ResolvedModelWeights reference_weights;
    auto session_result = PrepareTinySession(runtime, &reference_weights);
    ASSERT_TRUE(session_result.ok()) << session_result.status().ToString();
    InferenceSession session = std::move(*session_result);

    std::array<uint32_t, 3> prompt{1, 7, 3};
    const GenerationConfig one_token{.max_new_tokens = 1};
    const auto expected_one = ScalarGenerate(reference_weights, prompt, one_token);
    auto one = session.Generate(prompt, one_token);
    ASSERT_TRUE(one.ok()) << one.status().ToString();
    EXPECT_EQ(*one, expected_one);

    const GenerationConfig full_capacity{.max_new_tokens = 6};
    std::vector<uint32_t> expected_full;
    size_t mid_eos_index = full_capacity.max_new_tokens;
    for (uint32_t final_prompt_token = 0;
         final_prompt_token < static_cast<uint32_t>(kVocabSize) &&
         mid_eos_index == full_capacity.max_new_tokens;
         ++final_prompt_token) {
        prompt[2] = final_prompt_token;
        expected_full = ScalarGenerate(reference_weights, prompt, full_capacity);
        for (size_t index = 1; index < expected_full.size(); ++index) {
            if (std::find(expected_full.begin(),
                          expected_full.begin() + static_cast<std::ptrdiff_t>(index),
                          expected_full[index]) ==
                expected_full.begin() + static_cast<std::ptrdiff_t>(index)) {
                mid_eos_index = index;
                break;
            }
        }
    }
    ASSERT_LT(mid_eos_index, expected_full.size())
            << "the tiny fixture must provide a distinct later token for mid-EOS";
    ASSERT_EQ(expected_full.size(), full_capacity.max_new_tokens);
    auto full = session.Generate(prompt, full_capacity);
    ASSERT_TRUE(full.ok()) << full.status().ToString();
    EXPECT_EQ(*full, expected_full);

    const GenerationConfig first_eos{
            .max_new_tokens = 6,
            .eos_token_id = expected_full.front(),
    };
    auto first_eos_result = session.Generate(prompt, first_eos);
    ASSERT_TRUE(first_eos_result.ok()) << first_eos_result.status().ToString();
    EXPECT_EQ(*first_eos_result,
              std::vector<uint32_t>{expected_full.front()});

    const GenerationConfig middle_eos{
            .max_new_tokens = 6,
            .eos_token_id = expected_full[mid_eos_index],
    };
    auto middle_eos_result = session.Generate(prompt, middle_eos);
    ASSERT_TRUE(middle_eos_result.ok()) << middle_eos_result.status().ToString();
    EXPECT_EQ(*middle_eos_result,
              std::vector<uint32_t>(
                      expected_full.begin(),
                      expected_full.begin() +
                              static_cast<std::ptrdiff_t>(mid_eos_index + 1)));

    const GenerationConfig over_capacity{.max_new_tokens = 7};
    auto over = session.Generate(prompt, over_capacity);
    ASSERT_FALSE(over.ok());
    EXPECT_EQ(over.status().code(), StatusCode::kOutOfRange);
}

TEST(InferenceSession, RejectsContextLimitAndArithmeticOverflowBeforeReservation) {
    Runtime runtime = MakeCpuRuntimeWithKVCache(kKvHeads, 200);
    ResolvedModelWeights reference_weights;
    auto session_result = PrepareTinySession(runtime, &reference_weights);
    ASSERT_TRUE(session_result.ok()) << session_result.status().ToString();
    InferenceSession session = std::move(*session_result);

    constexpr std::array<uint32_t, 3> prompt{1, 7, 3};
    auto over_context = session.Generate(
            prompt, GenerationConfig{.max_new_tokens = 127});
    ASSERT_FALSE(over_context.ok());
    EXPECT_EQ(over_context.status().code(), StatusCode::kOutOfRange);

    auto overflow = session.Generate(
            prompt, GenerationConfig{
                            .max_new_tokens = std::numeric_limits<size_t>::max(),
                    });
    ASSERT_FALSE(overflow.ok());
    EXPECT_EQ(overflow.status().code(), StatusCode::kOverflow);

    KVCacheManager* const manager = runtime.GetKVCacheManager();
    ASSERT_NE(manager, nullptr);
    auto available = manager->ReserveForSession(prompt.size(), 197);
    ASSERT_TRUE(available.ok()) << available.status().ToString();
    ReleaseTestSession(*manager, *available);
}

TEST(InferenceSession, ZeroLimitDoesNotRequireKvAndInputsAreValidated) {
    Runtime runtime = MakeCpuRuntime();
    ResolvedModelWeights reference_weights;
    auto session_result = PrepareTinySession(runtime, &reference_weights);
    ASSERT_TRUE(session_result.ok()) << session_result.status().ToString();
    InferenceSession session = std::move(*session_result);

    constexpr std::array<uint32_t, 1> prompt{1};
    auto zero = session.Generate(
            prompt, GenerationConfig{.max_new_tokens = 0});
    ASSERT_TRUE(zero.ok()) << zero.status().ToString();
    EXPECT_TRUE(zero->empty());

    auto no_manager = session.Generate(
            prompt, GenerationConfig{.max_new_tokens = 1});
    ASSERT_FALSE(no_manager.ok());
    EXPECT_EQ(no_manager.status().code(), StatusCode::kFailedPrecondition);

    auto empty_prompt = session.Generate(
            std::span<const uint32_t>{},
            GenerationConfig{.max_new_tokens = 0});
    ASSERT_FALSE(empty_prompt.ok());
    EXPECT_EQ(empty_prompt.status().code(), StatusCode::kInvalidArgument);

    constexpr std::array<uint32_t, 1> invalid_token{
            static_cast<uint32_t>(kVocabSize)};
    auto bad_prompt = session.Generate(
            invalid_token, GenerationConfig{.max_new_tokens = 0});
    ASSERT_FALSE(bad_prompt.ok());
    EXPECT_EQ(bad_prompt.status().code(), StatusCode::kOutOfRange);

    auto bad_eos = session.Generate(
            prompt, GenerationConfig{
                            .max_new_tokens = 0,
                            .eos_token_id = static_cast<uint32_t>(kVocabSize),
                    });
    ASSERT_FALSE(bad_eos.ok());
    EXPECT_EQ(bad_eos.status().code(), StatusCode::kOutOfRange);
}

TEST(InferenceSession, MissingCpuAllocatorReturnsStatusBeforeReservation) {
    Runtime runtime =
            MakeCpuRuntimeWithKVCache(kKvHeads, kCacheCapacity, false);
    ResolvedModelWeights reference_weights;
    auto session_result = PrepareTinySession(runtime, &reference_weights);
    ASSERT_TRUE(session_result.ok()) << session_result.status().ToString();
    InferenceSession session = std::move(*session_result);

    constexpr std::array<uint32_t, 3> prompt{1, 7, 3};
    auto result = session.Generate(
            prompt, GenerationConfig{.max_new_tokens = 1});
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kFailedPrecondition);

    KVCacheManager* const manager = runtime.GetKVCacheManager();
    ASSERT_NE(manager, nullptr);
    auto available = manager->ReserveForSession(prompt.size(), 0);
    ASSERT_TRUE(available.ok()) << available.status().ToString();
    ReleaseTestSession(*manager, *available);
}

TEST(InferenceSession, HandlesPromptSizedCapacityAndReservationCleanup) {
    constexpr std::array<uint32_t, 3> prompt{1, 7, 3};
    Runtime exact_runtime = MakeCpuRuntimeWithKVCache(kKvHeads, prompt.size());
    ResolvedModelWeights exact_weights;
    auto exact_session_result = PrepareTinySession(exact_runtime, &exact_weights);
    ASSERT_TRUE(exact_session_result.ok())
            << exact_session_result.status().ToString();
    InferenceSession exact_session = std::move(*exact_session_result);

    const GenerationConfig one_token{.max_new_tokens = 1};
    auto result = exact_session.Generate(prompt, one_token);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(*result, ScalarGenerate(exact_weights, prompt, one_token));

    auto needs_decode = exact_session.Generate(
            prompt, GenerationConfig{.max_new_tokens = 2});
    ASSERT_FALSE(needs_decode.ok());
    EXPECT_EQ(needs_decode.status().code(), StatusCode::kOutOfRange);

    Runtime mismatch_runtime = MakeCpuRuntimeWithKVCache(kKvHeads - 1);
    ResolvedModelWeights mismatch_weights;
    auto mismatch_session_result =
            PrepareTinySession(mismatch_runtime, &mismatch_weights);
    ASSERT_TRUE(mismatch_session_result.ok())
            << mismatch_session_result.status().ToString();
    InferenceSession mismatch_session = std::move(*mismatch_session_result);
    KVCacheManager* const manager = mismatch_runtime.GetKVCacheManager();
    ASSERT_NE(manager, nullptr);

    auto held = manager->ReserveForSession(prompt.size(), 0);
    ASSERT_TRUE(held.ok()) << held.status().ToString();
    auto contended = mismatch_session.Generate(prompt, one_token);
    ASSERT_FALSE(contended.ok());
    EXPECT_EQ(contended.status().code(), StatusCode::kFailedPrecondition);
    ReleaseTestSession(*manager, *held);

    auto execution_failure = mismatch_session.Generate(prompt, one_token);
    ASSERT_FALSE(execution_failure.ok());
    auto after_failure = manager->ReserveForSession(prompt.size(), 0);
    ASSERT_TRUE(after_failure.ok()) << after_failure.status().ToString();
    ReleaseTestSession(*manager, *after_failure);
}

TEST(InferenceSession, SharedDecodeLoopHasNoMallocFamilyCallsAfterPreparation) {
#if !defined(__GLIBC__) || !defined(__linux__)
    GTEST_SKIP() << "The malloc interposer requires glibc/Linux";
#else
    ASSERT_TRUE(MallocInterposerAvailable());
    Runtime runtime = MakeCpuRuntimeWithKVCache();
    ResolvedModelWeights reference_weights;
    auto executable_result =
            PrepareTinyExecutableModel(runtime, &reference_weights);
    ASSERT_TRUE(executable_result.ok())
            << executable_result.status().ToString();
    ExecutableModel executable(std::move(*executable_result));
    const ExecutionPlan& plan = **executable.plan(ExecPhase::kBoth);
    const ExternalTensorBindings& immutable_bindings =
            **executable.immutable_weight_bindings(ExecPhase::kBoth);
    constexpr std::array<uint32_t, 3> prompt{1, 7, 3};
    std::vector<int64_t> prompt_tokens(prompt.begin(), prompt.end());
    std::vector<int64_t> prompt_positions(prompt.size());
    for (size_t i = 0; i < prompt_positions.size(); ++i) {
        prompt_positions[i] = static_cast<int64_t>(i);
    }

    KVCacheManager* const manager = runtime.GetKVCacheManager();
    ASSERT_NE(manager, nullptr);
    auto reservation = manager->ReserveForSession(prompt.size(), 2);
    ASSERT_TRUE(reservation.ok()) << reservation.status().ToString();
    KVCacheView cache_view = std::move(*reservation);

    WorkspaceStorage workspace_storage(
            plan.total_workspace_bytes(), plan.workspace_alignment());
    CpuWorkspaceArena workspace(
            workspace_storage.data(), workspace_storage.size());
    WorkspaceArena* const workspace_ptr =
            plan.total_workspace_bytes() == 0 ? nullptr : &workspace;

    TestBuffer prefill_tokens(
            DataType::Int(64), {static_cast<int64_t>(prompt.size())});
    TestBuffer prefill_positions(
            DataType::Int(64), {static_cast<int64_t>(prompt.size())});
    std::copy(prompt_tokens.begin(), prompt_tokens.end(),
              static_cast<int64_t*>(prefill_tokens.mutable_data()));
    std::copy(prompt_positions.begin(), prompt_positions.end(),
              static_cast<int64_t*>(prefill_positions.mutable_data()));
    auto prefill_result = PrepareContext(
            runtime, plan, immutable_bindings, prefill_tokens,
            prefill_positions, workspace_ptr, cache_view);
    ASSERT_TRUE(prefill_result.ok()) << prefill_result.status().ToString();
    ExecutionContext prefill_context(std::move(*prefill_result));
    ASSERT_TRUE(Executor::Execute(plan, prefill_context).ok());
    const ExecutionValueId token_output = plan.model_outputs().front();
    const MutableTensorView prefill_output =
            FindOutput(prefill_context, plan, token_output);
    ASSERT_TRUE(prefill_output.is_valid());
    const int64_t first_token =
            prefill_output.data<int64_t>()[prompt.size() - 1];
    prefill_context.Clear();

    TestBuffer decode_tokens(DataType::Int(64), {1});
    TestBuffer decode_positions(DataType::Int(64), {1});
    static_cast<int64_t*>(decode_tokens.mutable_data())[0] = first_token;
    static_cast<int64_t*>(decode_positions.mutable_data())[0] =
            static_cast<int64_t>(prompt.size());
    auto decode_result = PrepareContext(
            runtime, plan, immutable_bindings, decode_tokens,
            decode_positions, workspace_ptr, cache_view);
    ASSERT_TRUE(decode_result.ok()) << decode_result.status().ToString();
    ExecutionContext decode_context(std::move(*decode_result));

    std::vector<uint32_t> generated;
    generated.reserve(3);
    generated.push_back(static_cast<uint32_t>(first_token));
    int64_t next_position = static_cast<int64_t>(prompt.size());
    const Status decode_status = [&] {
        BeginMallocCallCounting();
        const Status status = inference::internal::RunDecodeLoop(
                plan, decode_context, token_output,
                static_cast<int64_t*>(decode_tokens.mutable_data()),
                static_cast<int64_t*>(decode_positions.mutable_data()),
                static_cast<size_t>(kVocabSize), 3, std::nullopt, generated);
        return status;
    }();
    const MallocCallCounts counts = EndMallocCallCounting();

    decode_context.Clear();
    ReleaseTestSession(*manager, cache_view);
    ASSERT_TRUE(decode_status.ok()) << decode_status.ToString();
    EXPECT_EQ(generated, ScalarGenerate(
                                 reference_weights, prompt,
                                 GenerationConfig{.max_new_tokens = 3}));
    EXPECT_EQ(counts.malloc_calls, 0U);
    EXPECT_EQ(counts.calloc_calls, 0U);
    EXPECT_EQ(counts.realloc_calls, 0U);
    EXPECT_EQ(counts.free_calls, 0U);
    EXPECT_EQ(counts.aligned_alloc_calls, 0U);
    EXPECT_EQ(counts.posix_memalign_calls, 0U);
    EXPECT_EQ(counts.memalign_calls, 0U);
#endif
}

TEST(DirectPrefillDecode, MallocInterposerObservesCpuAllocatorCalls) {
#if !defined(__GLIBC__) || !defined(__linux__)
    GTEST_SKIP() << "The malloc interposer calibration requires glibc/Linux";
#else
    ASSERT_TRUE(MallocInterposerAvailable());
    Runtime runtime = MakeCpuRuntimeWithKVCache();
    Allocator& allocator = runtime.GetAllocator(Device::CPU());

    BeginMallocCallCounting();
    bool allocated = false;
    {
        Buffer buffer = allocator.Allocate(128);
        allocated = buffer.data() != nullptr;
    }
    const MallocCallCounts counts = EndMallocCallCounting();
    EXPECT_TRUE(allocated);
    EXPECT_EQ(counts.posix_memalign_calls, 1U)
            << "the probe runs through alloc_cpu() in the AetherMind shared library";
    EXPECT_GE(counts.free_calls, 1U);
#endif
}

} // namespace
