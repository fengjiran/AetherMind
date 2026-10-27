#ifndef AETHERMIND_TEST_HF_FIXED_TRAJECTORY_HELPERS_H
#define AETHERMIND_TEST_HF_FIXED_TRAJECTORY_HELPERS_H

#include "aethermind/backend/cpu/cpu_workspace_arena.h"
#include "aethermind/base/status.h"
#include "aethermind/compiler/model_compiler.h"
#include "aethermind/execution/execution_context.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/execution/executor.h"
#include "aethermind/inference/cpu_inference_options.h"
#include "aethermind/inference/executable_model.h"
#include "aethermind/inference/model_execution_policy.h"
#include "aethermind/model/formats/hf/hf_safetensors_file.h"
#include "aethermind/model/loaded_model.h"
#include "aethermind/model/model_loader.h"
#include "aethermind/operators/op_type.h"
#include "aethermind/runtime/runtime_builder.h"
#include "execution/test_tensor_buffer_helpers.h"
#include "inference/test_execution_observation_helpers.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace aethermind::test::numerical {

using aethermind::test::observation::FindOutput;
using aethermind::test::observation::PrepareContext;

inline constexpr float kLogitAbsoluteTolerance = 2.0e-5F;
inline constexpr float kKvAbsoluteTolerance = 2.0e-6F;
inline constexpr float kRelativeTolerance = 2.0e-4F;
inline constexpr std::array<std::string_view, 3> kStageNames{"prefill", "decode_0", "decode_1"};
inline constexpr size_t kMaxReferenceElements = 8U * 1024U * 1024U;

template<typename T>
struct TensorData {
    std::vector<int64_t> shape;
    std::vector<T> values;
};

struct StageSnapshot {
    std::vector<int64_t> logit_rows;
    TensorData<float> logits;
    std::vector<int64_t> tokens;
    TensorData<float> keys;
    TensorData<float> values;
    size_t position = 0;
};

struct ReferenceTrajectory {
    std::vector<int64_t> prompt;
    std::vector<int64_t> decode_inputs;
    std::vector<int64_t> greedy_tokens;
    std::vector<float> greedy_margins;
    std::array<StageSnapshot, 3> stages;
};

using CapturedTrajectory = std::array<StageSnapshot, 3>;

inline StatusOr<size_t> ElementCount(std::span<const int64_t> shape) {
    size_t count = 1;
    for (const int64_t dimension: shape) {
        if (dimension <= 0 || !std::in_range<size_t>(dimension) ||
            static_cast<size_t>(dimension) > kMaxReferenceElements / count) {
            return Status::InvalidArgument("Numerical reference has invalid or excessive tensor dimensions");
        }
        count *= static_cast<size_t>(dimension);
    }
    return count;
}

template<typename T>
inline StatusOr<TensorData<T>> ReadTensor(const HfSafetensorsFile& file, const std::string& name) {
    const HfSafetensorsEntry* const entry = file.Find(name);
    if (entry == nullptr) {
        return Status::InvalidArgument("Numerical reference is missing tensor '" + name + "'");
    }
    if (entry->dtype != DataType::Make<T>()) {
        return Status::InvalidArgument("Numerical reference tensor has wrong dtype: " + name);
    }
    AM_ASSIGN_OR_RETURN(const size_t count, ElementCount(entry->shape));
    if (entry->ByteSize() != count * sizeof(T) || std::endian::native != std::endian::little) {
        return Status::InvalidArgument("Numerical reference tensor has invalid byte size or byte order: " + name);
    }
    TensorData<T> result{.shape = entry->shape, .values = std::vector<T>(count)};
    // A safetensors payload offset need not satisfy alignof(int64_t). Copying
    // also prevents any reference view from outliving its mmap owner.
    std::memcpy(result.values.data(), entry->view.data, count * sizeof(T));
    if constexpr (std::is_floating_point_v<T>) {
        if (!std::ranges::all_of(result.values, [](T value) { return std::isfinite(value); })) {
            return Status::InvalidArgument("Numerical reference contains a non-finite value: " + name);
        }
    }
    return result;
}

template<typename T>
inline StatusOr<std::vector<T>> ReadVector(const HfSafetensorsFile& file, const std::string& name) {
    AM_ASSIGN_OR_RETURN(auto tensor, ReadTensor<T>(file, name));
    if (tensor.shape.size() != 1) {
        return Status::InvalidArgument("Numerical reference tensor must be rank one: " + name);
    }
    return std::move(tensor.values);
}

inline StatusOr<ReferenceTrajectory> LoadReference(const std::filesystem::path& path) {
    AM_ASSIGN_OR_RETURN(auto file, HfSafetensorsFile::Open(path));
    std::unordered_set<std::string> expected{
            "prompt", "decode_inputs", "greedy_tokens", "greedy_margins", "tolerance.logits", "tolerance.kv"};
    for (const auto stage: kStageNames) {
        for (const auto suffix: {"logit_rows", "logits", "tokens", "keys", "values", "position"}) {
            expected.insert(std::string(stage) + "." + suffix);
        }
    }
    if (file.Entries().size() != expected.size()) {
        return Status::InvalidArgument("Numerical reference tensor set is incomplete or contains extra stages");
    }
    for (const auto& entry: file.Entries()) {
        if (!expected.contains(entry.name)) {
            return Status::InvalidArgument("Unexpected numerical reference tensor: " + entry.name);
        }
    }
    AM_ASSIGN_OR_RETURN(auto logits_tolerance, ReadVector<float>(file, "tolerance.logits"));
    AM_ASSIGN_OR_RETURN(auto kv_tolerance, ReadVector<float>(file, "tolerance.kv"));
    if (logits_tolerance != std::vector<float>{kLogitAbsoluteTolerance, kRelativeTolerance} ||
        kv_tolerance != std::vector<float>{kKvAbsoluteTolerance, kRelativeTolerance}) {
        return Status::InvalidArgument("Numerical reference tolerance differs from the frozen FP32 contract");
    }
    ReferenceTrajectory reference;
    AM_ASSIGN_OR_RETURN(reference.prompt, ReadVector<int64_t>(file, "prompt"));
    AM_ASSIGN_OR_RETURN(reference.decode_inputs, ReadVector<int64_t>(file, "decode_inputs"));
    AM_ASSIGN_OR_RETURN(reference.greedy_tokens, ReadVector<int64_t>(file, "greedy_tokens"));
    AM_ASSIGN_OR_RETURN(reference.greedy_margins, ReadVector<float>(file, "greedy_margins"));
    for (size_t index = 0; index < kStageNames.size(); ++index) {
        const std::string prefix(kStageNames[index]);
        auto& stage = reference.stages[index];
        AM_ASSIGN_OR_RETURN(stage.logit_rows, ReadVector<int64_t>(file, prefix + ".logit_rows"));
        AM_ASSIGN_OR_RETURN(stage.logits, ReadTensor<float>(file, prefix + ".logits"));
        AM_ASSIGN_OR_RETURN(stage.tokens, ReadVector<int64_t>(file, prefix + ".tokens"));
        AM_ASSIGN_OR_RETURN(stage.keys, ReadTensor<float>(file, prefix + ".keys"));
        AM_ASSIGN_OR_RETURN(stage.values, ReadTensor<float>(file, prefix + ".values"));
        AM_ASSIGN_OR_RETURN(auto position, ReadVector<int64_t>(file, prefix + ".position"));
        if (position.size() != 1 || position[0] <= 0 || !std::in_range<size_t>(position[0])) {
            return Status::InvalidArgument("Invalid numerical reference watermark: " + prefix);
        }
        stage.position = static_cast<size_t>(position[0]);
    }
    return reference;
}

inline Status ValidateReference(const ReferenceTrajectory& reference, const HfModelConfig& config,
                                bool last_prefill_row_only) {
    if (reference.prompt.empty() || reference.prompt.size() > std::numeric_limits<size_t>::max() - 2 ||
        reference.prompt.size() + 2 > static_cast<size_t>(config.max_position_embeddings) ||
        reference.decode_inputs.size() != 2 || reference.greedy_tokens.size() != 3 ||
        reference.greedy_margins.size() != 3) {
        return Status::InvalidArgument("Numerical reference has invalid trajectory lengths");
    }
    if (reference.decode_inputs != std::vector<int64_t>{17, 23} ||
        (last_prefill_row_only && reference.prompt.size() + 2 != static_cast<size_t>(config.max_position_embeddings)) ||
        (!last_prefill_row_only && reference.prompt.size() != 3)) {
        return Status::InvalidArgument("Numerical reference differs from the frozen input trajectory");
    }
    constexpr std::array<int64_t, 3> pattern{1, 7, 3};
    for (size_t index = 0; index < reference.prompt.size(); ++index) {
        if (reference.prompt[index] != pattern[index % pattern.size()]) {
            return Status::InvalidArgument("Numerical reference has an unexpected prompt token");
        }
    }
    for (const auto* tokens: {&reference.prompt, &reference.decode_inputs, &reference.greedy_tokens}) {
        for (const int64_t token: *tokens) {
            if (token < 0 || token >= config.vocab_size || !std::in_range<uint32_t>(token)) {
                return Status::InvalidArgument("Numerical reference token is outside the model vocabulary");
            }
        }
    }
    if (!std::ranges::all_of(reference.greedy_margins, [](float margin) { return margin >= 0.0F; })) {
        return Status::InvalidArgument("Numerical reference has an invalid greedy margin");
    }
    for (size_t index = 0; index < reference.stages.size(); ++index) {
        const auto& stage = reference.stages[index];
        const size_t sequence = index == 0 ? reference.prompt.size() : 1;
        const size_t committed = reference.prompt.size() + index;
        const std::vector<int64_t> kv_shape{
                config.num_hidden_layers, static_cast<int64_t>(committed),
                config.num_key_value_heads, config.head_dim != 0 ? config.head_dim : config.hidden_size / config.num_attention_heads};
        if (stage.position != committed || stage.tokens.size() != sequence ||
            stage.logit_rows.empty() || stage.keys.shape != kv_shape || stage.values.shape != kv_shape ||
            stage.logits.shape != std::vector<int64_t>{static_cast<int64_t>(stage.logit_rows.size()), config.vocab_size}) {
            return Status::InvalidArgument("Numerical reference stage shape/watermark mismatch: " + std::string(kStageNames[index]));
        }
        const size_t rows = index == 0 && !last_prefill_row_only ? sequence : 1;
        if (stage.logit_rows.size() != rows) {
            return Status::InvalidArgument("Numerical reference is missing required logit rows");
        }
        for (size_t row = 0; row < rows; ++row) {
            const auto expected_row = static_cast<int64_t>(index == 0 && last_prefill_row_only ? sequence - 1 : row);
            if (stage.logit_rows[row] != expected_row) {
                return Status::InvalidArgument("Numerical reference logit row order is invalid");
            }
        }
    }
    return Status::Ok();
}

inline std::string Coordinate(const std::vector<int64_t>& shape, size_t index,
                              std::span<const int64_t> logit_rows = {}) {
    if (shape.size() == 2) {
        const size_t column = index % static_cast<size_t>(shape[1]);
        const size_t row = index / static_cast<size_t>(shape[1]);
        return "row=" + std::to_string(logit_rows.empty() ? static_cast<int64_t>(row) : logit_rows[row]) +
               ",column=" + std::to_string(column);
    }
    const size_t dim = index % static_cast<size_t>(shape[3]);
    index /= static_cast<size_t>(shape[3]);
    const size_t head = index % static_cast<size_t>(shape[2]);
    index /= static_cast<size_t>(shape[2]);
    const size_t position = index % static_cast<size_t>(shape[1]);
    return "layer=" + std::to_string(index / static_cast<size_t>(shape[1])) +
           ",position=" + std::to_string(position) + ",head=" + std::to_string(head) +
           ",dim=" + std::to_string(dim);
}

inline ::testing::AssertionResult CompareTensor(const TensorData<float>& actual,
                                                const TensorData<float>& expected,
                                                std::string_view stage, std::string_view name,
                                                float absolute_tolerance,
                                                std::span<const int64_t> logit_rows = {}) {
    if (actual.shape != expected.shape || (actual.shape.size() != 2 && actual.shape.size() != 4)) {
        return ::testing::AssertionFailure() << stage << " " << name << " shape mismatch";
    }
    const auto count = ElementCount(actual.shape);
    if (!count.ok() || actual.values.size() != *count || expected.values.size() != *count) {
        return ::testing::AssertionFailure() << stage << " " << name << " element count mismatch";
    }
    if (!logit_rows.empty() && logit_rows.size() != static_cast<size_t>(actual.shape[0])) {
        return ::testing::AssertionFailure() << stage << " " << name << " row count mismatch";
    }
    for (size_t index = 0; index < *count; ++index) {
        const double value = actual.values[index];
        const double reference = expected.values[index];
        const double error = std::abs(value - reference);
        if (!std::isfinite(value) || !std::isfinite(reference) ||
            error > static_cast<double>(absolute_tolerance) +
                            static_cast<double>(kRelativeTolerance) * std::abs(reference)) {
            return ::testing::AssertionFailure()
                   << stage << " " << name << " [" << Coordinate(actual.shape, index, logit_rows)
                   << "] actual=" << value << " expected=" << reference << " absolute_error=" << error;
        }
    }
    return ::testing::AssertionSuccess();
}

inline ::testing::AssertionResult CompareStage(const StageSnapshot& actual,
                                               const StageSnapshot& expected,
                                               std::string_view stage) {
    if (actual.position != expected.position) {
        return ::testing::AssertionFailure() << stage << " committed watermark mismatch: actual="
                                             << actual.position << " expected=" << expected.position;
    }
    if (actual.logit_rows != expected.logit_rows) {
        return ::testing::AssertionFailure() << stage << " logit row identity/order mismatch";
    }
    for (const auto& comparison: {
                 CompareTensor(actual.logits, expected.logits, stage, "logits", kLogitAbsoluteTolerance, actual.logit_rows),
                 CompareTensor(actual.keys, expected.keys, stage, "keys", kKvAbsoluteTolerance),
                 CompareTensor(actual.values, expected.values, stage, "values", kKvAbsoluteTolerance)}) {
        if (!comparison) {
            return comparison;
        }
    }
    if (actual.tokens != expected.tokens) {
        return ::testing::AssertionFailure() << stage << " Argmax token mismatch";
    }
    return ::testing::AssertionSuccess();
}

inline StatusOr<ExecutionValueId> FindLogits(const ExecutionPlan& plan) {
    if (plan.model_outputs().size() != 1) {
        return Status::InvalidArgument("Numerical observation requires one Argmax model output");
    }
    size_t found = 0;
    ExecutionValueId value{};
    for (const auto& step: plan.steps()) {
        if (step.kernel.op_type == OpType::kArgmax) {
            if (step.inputs.size() != 1 || step.outputs.size() != 1 || step.outputs[0] != plan.model_outputs()[0]) {
                return Status::InvalidArgument("Numerical observation requires unary Argmax as the model output");
            }
            value = step.inputs[0];
            ++found;
        }
    }
    if (found != 1) {
        return Status::InvalidArgument("Numerical observation requires one Argmax model output");
    }
    return value;
}

inline StatusOr<StageSnapshot> CaptureStage(const ExecutionContext& context,
                                            const ExecutionPlan& plan,
                                            std::span<const int64_t> rows,
                                            size_t sequence_length,
                                            KVCacheSessionView cache_view) {
    AM_ASSIGN_OR_RETURN(const auto logits_id, FindLogits(plan));
    const auto logits = FindOutput(context, plan, logits_id);
    const auto tokens = FindOutput(context, plan, plan.model_outputs()[0]);
    if (!cache_view.valid() || cache_view.kv_dtype() != DataType::Float32() ||
        !logits.is_valid() || logits.dtype() != DataType::Float32() || logits.rank() != 2 ||
        logits.dim(0) != static_cast<int64_t>(sequence_length) || logits.dim(1) <= 0 ||
        logits.stride(1) != 1 || logits.stride(0) != logits.dim(1) ||
        !tokens.is_valid() || tokens.dtype() != DataType::Int(64) || tokens.rank() != 1 ||
        tokens.dim(0) != static_cast<int64_t>(sequence_length) || tokens.stride(0) != 1) {
        return Status::InvalidArgument("Numerical observation requires dense FP32 logits and Int64 tokens");
    }
    StageSnapshot result;
    result.logit_rows.assign(rows.begin(), rows.end());
    result.logits.shape = {static_cast<int64_t>(rows.size()), logits.dim(1)};
    AM_ASSIGN_OR_RETURN(const size_t logits_count, ElementCount(result.logits.shape));
    result.logits.values.resize(logits_count);
    const size_t vocabulary = static_cast<size_t>(logits.dim(1));
    for (size_t row = 0; row < rows.size(); ++row) {
        if (rows[row] < 0 || static_cast<size_t>(rows[row]) >= sequence_length) {
            return Status::InvalidArgument("Numerical observation row exceeds the executed sequence");
        }
        std::memcpy(result.logits.values.data() + row * vocabulary,
                    static_cast<const std::byte*>(logits.data()) + static_cast<size_t>(rows[row]) * vocabulary * sizeof(float),
                    vocabulary * sizeof(float));
    }
    result.tokens.resize(sequence_length);
    std::memcpy(result.tokens.data(), tokens.data(), sequence_length * sizeof(int64_t));
    result.position = cache_view.current_pos();
    result.keys.shape = {static_cast<int64_t>(cache_view.num_layers()), static_cast<int64_t>(result.position),
                         static_cast<int64_t>(cache_view.num_kv_heads()), static_cast<int64_t>(cache_view.head_dim())};
    result.values.shape = result.keys.shape;
    AM_ASSIGN_OR_RETURN(const size_t kv_count, ElementCount(result.keys.shape));
    result.keys.values.reserve(kv_count);
    result.values.values.reserve(kv_count);
    for (size_t layer = 0; layer < cache_view.num_layers(); ++layer) {
        for (size_t position = 0; position < result.position; ++position) {
            for (size_t head = 0; head < cache_view.num_kv_heads(); ++head) {
                for (size_t dim = 0; dim < cache_view.head_dim(); ++dim) {
                    AM_ASSIGN_OR_RETURN(const auto key, cache_view.KeyData(layer, head, position, dim));
                    AM_ASSIGN_OR_RETURN(const auto value, cache_view.ValueData(layer, head, position, dim));
                    float key_value = 0;
                    float value_value = 0;
                    std::memcpy(&key_value, key, sizeof(float));
                    std::memcpy(&value_value, value, sizeof(float));
                    result.keys.values.push_back(key_value);
                    result.values.values.push_back(value_value);
                }
            }
        }
    }
    return result;
}

class SessionReservation {
public:
    SessionReservation(KVCacheManager& manager, KVCacheSessionView view) noexcept
        : manager_(manager), view_(view) {}
    SessionReservation(const SessionReservation&) = delete;
    SessionReservation& operator=(const SessionReservation&) = delete;
    ~SessionReservation() {
        if (view_.valid()) {
            (void) manager_.ReleaseSession(view_);
        }
    }
    KVCacheSessionView view() const noexcept {
        return view_;
    }
    Status Release() noexcept {
        return manager_.ReleaseSession(view_);
    }

private:
    KVCacheManager& manager_;
    KVCacheSessionView view_;
};

inline StatusOr<CapturedTrajectory> RunFixedTrajectory(const std::filesystem::path& model_dir,
                                                       const ReferenceTrajectory& reference,
                                                       CpuInferencePreset preset,
                                                       bool last_prefill_row_only = false) {
    AM_ASSIGN_OR_RETURN(auto loaded, ModelLoader::Load(model_dir));
    const HfModelConfig config = loaded->GetConfig();
    AM_RETURN_IF_ERROR(ValidateReference(reference, config, last_prefill_row_only));
    AM_ASSIGN_OR_RETURN(auto options, MakeCpuCompileOptions(preset));
    AM_ASSIGN_OR_RETURN(auto artifact, ModelCompiler::Compile(std::move(loaded), options));
    RuntimeOptions runtime_options;
    AM_ASSIGN_OR_RETURN(runtime_options.kv_cache, MakeFp32KVCacheOptions(config, reference.prompt.size() + 2));
    RuntimeBuilder builder;
    builder.WithOptions(runtime_options);
    AM_ASSIGN_OR_RETURN(auto runtime, builder.TryBuild());
    AM_ASSIGN_OR_RETURN(auto model, PrepareExecutableModel(runtime, std::move(artifact)));
    AM_ASSIGN_OR_RETURN(const auto prefill, model.plan(ExecPhase::kPrefill));
    AM_ASSIGN_OR_RETURN(const auto decode, model.plan(ExecPhase::kDecode));
    AM_ASSIGN_OR_RETURN(const auto prefill_weights, model.immutable_weight_bindings(ExecPhase::kPrefill));
    AM_ASSIGN_OR_RETURN(const auto decode_weights, model.immutable_weight_bindings(ExecPhase::kDecode));
    KVCacheManager* const manager = runtime.GetKVCacheManager();
    if (manager == nullptr) {
        return Status::Internal("Numerical trajectory requires a KV manager");
    }
    AM_ASSIGN_OR_RETURN(auto view, manager->ReserveForSession(reference.prompt.size(), 2));
    // Reservation is declared before contexts/storage so every return path
    // destroys borrowed contexts before releasing the Runtime-owned KV slot.
    SessionReservation reservation(*manager, view);
    const size_t workspace_bytes = std::max(prefill->total_workspace_bytes(), decode->total_workspace_bytes());
    const size_t workspace_alignment = std::max(prefill->workspace_alignment(), decode->workspace_alignment());
    AM_ASSIGN_OR_RETURN(auto allocator, runtime.TryGetAllocator(Device::CPU()));
    Buffer workspace_storage;
    if (workspace_bytes != 0) {
        AM_ASSIGN_OR_RETURN(workspace_storage, allocator->TryAllocate({.nbytes = workspace_bytes, .alignment = workspace_alignment}));
    }
    CpuWorkspaceArena workspace(workspace_bytes == 0 ? nullptr : workspace_storage.mutable_data(), workspace_bytes);
    WorkspaceArena* const workspace_ptr = workspace_bytes == 0 ? nullptr : &workspace;
    CapturedTrajectory snapshots;
    TestBuffer prompt(DataType::Int(64), {static_cast<int64_t>(reference.prompt.size())});
    TestBuffer positions(DataType::Int(64), {static_cast<int64_t>(reference.prompt.size())});
    std::memcpy(prompt.mutable_data(), reference.prompt.data(), reference.prompt.size() * sizeof(int64_t));
    for (size_t index = 0; index < reference.prompt.size(); ++index) {
        static_cast<int64_t*>(positions.mutable_data())[index] = static_cast<int64_t>(index);
    }
    {
        AM_ASSIGN_OR_RETURN(auto context, PrepareContext(runtime, *prefill, *prefill_weights, prompt, positions, workspace_ptr, view));
        AM_RETURN_IF_ERROR(Executor::Execute(*prefill, context));
        AM_ASSIGN_OR_RETURN(snapshots[0], CaptureStage(context, *prefill, reference.stages[0].logit_rows, reference.prompt.size(), view));
    }
    TestBuffer decode_token(DataType::Int(64), {1});
    TestBuffer decode_position(DataType::Int(64), {1});
    static_cast<int64_t*>(decode_token.mutable_data())[0] = reference.decode_inputs.front();
    static_cast<int64_t*>(decode_position.mutable_data())[0] = static_cast<int64_t>(reference.prompt.size());
    {
        AM_ASSIGN_OR_RETURN(auto context, PrepareContext(runtime, *decode, *decode_weights, decode_token, decode_position, workspace_ptr, view));
        for (size_t index = 0; index < reference.decode_inputs.size(); ++index) {
            static_cast<int64_t*>(decode_token.mutable_data())[0] = reference.decode_inputs[index];
            static_cast<int64_t*>(decode_position.mutable_data())[0] = static_cast<int64_t>(reference.prompt.size() + index);
            workspace.Reset();
            AM_RETURN_IF_ERROR(Executor::Execute(*decode, context));
            AM_ASSIGN_OR_RETURN(snapshots[index + 1], CaptureStage(context, *decode, reference.stages[index + 1].logit_rows, 1, view));
        }
    }
    AM_RETURN_IF_ERROR(reservation.Release());
    return snapshots;
}

} // namespace aethermind::test::numerical

#endif
