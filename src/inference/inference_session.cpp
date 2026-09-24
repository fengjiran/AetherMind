#include "aethermind/inference/inference_session.h"

#include "aethermind/base/tensor_view.h"
#include "aethermind/base/workspace_arena.h"
#include "aethermind/execution/execution_bindings.h"
#include "aethermind/execution/execution_context.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/execution/executor.h"
#include "aethermind/inference/executable_model.h"
#include "aethermind/operators/op_type.h"
#include "aethermind/runtime/kv_cache_manager.h"
#include "aethermind/runtime/runtime.h"
#include "inference_session_internal.h"
#include "utils/overflow_check.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace aethermind {
namespace {

struct ValidatedPhaseContract {
    uint32_t token_input = 0;
    uint32_t position_input = 0;
    uint32_t token_output = 0;
};

StatusOr<ValidatedPhaseContract> ValidatePlanContract(const ExecutionPlan& plan) {
    if (plan.model_inputs().size() != 2 || plan.model_outputs().size() != 1) {
        return Status::Unimplemented(
                "InferenceSession requires one token output and token_ids/position_ids inputs");
    }

    ValidatedPhaseContract contract;
    // The Llama graph builder appends token_ids then position_ids, and lowering
    // preserves this ordered model_inputs contract independently of debug names.
    contract.token_input = plan.model_inputs()[0].index;
    contract.position_input = plan.model_inputs()[1].index;
    if (contract.token_input == contract.position_input) {
        return Status::InvalidArgument(
                "InferenceSession token and position inputs must be distinct");
    }
    for (const uint32_t index:
         {contract.token_input, contract.position_input}) {
        if (index >= plan.values().size()) {
            return Status::InvalidArgument(
                    "InferenceSession model input references an invalid execution value");
        }
        const ExecutionValueDesc& value = plan.values()[index];
        const auto rank = value.spec.shape.rank();
        if (value.kind != ExecutionValueKind::kModelInput ||
            value.spec.dtype != DataType::Int(64) || !rank.has_value() || *rank != 1) {
            return Status::Unimplemented(
                    "InferenceSession model inputs must be rank-one Int64 tensors");
        }
    }

    const ExecutionValueId output = plan.model_outputs().front();
    if (output.index >= plan.values().size()) {
        return Status::InvalidArgument(
                "InferenceSession model output references an invalid execution value");
    }
    const ExecutionValueDesc& output_value = plan.values()[output.index];
    const auto output_rank = output_value.spec.shape.rank();
    if (output_value.kind != ExecutionValueKind::kActivation ||
        output_value.spec.dtype != DataType::Int(64) ||
        !output_rank.has_value() || *output_rank != 1) {
        return Status::Unimplemented(
                "InferenceSession requires a rank-one Int64 token output");
    }

    size_t argmax_count = 0;
    bool output_is_argmax = false;
    for (const ExecutionStep& step: plan.steps()) {
        if (step.selector.device_type != DeviceType::kCPU) {
            return Status::Unimplemented(
                    "InferenceSession currently supports CPU execution plans only");
        }
        if (step.kernel.op_type != OpType::kArgmax) {
            continue;
        }
        ++argmax_count;
        output_is_argmax = step.outputs.size() == 1 && step.outputs.front() == output;
    }
    if (argmax_count != 1 || !output_is_argmax) {
        return Status::Unimplemented(
                "InferenceSession requires its sole model output to be the Argmax result");
    }

    contract.token_output = output.index;
    return contract;
}


class SessionWorkspaceArena final : public WorkspaceArena {
public:
    void SetBuffer(void* base, size_t size) noexcept {
        base_ = static_cast<std::byte*>(base);
        size_ = size;
    }

    WorkspaceBinding Bind(const WorkspaceRequirement& requirement) noexcept override {
        if (requirement.bytes == 0 || base_ == nullptr ||
            !IsValidWorkspaceAlignment(requirement.alignment) ||
            requirement.offset > size_ ||
            requirement.bytes > size_ - requirement.offset) {
            return {};
        }

        std::byte* const data = base_ + requirement.offset;
        if (reinterpret_cast<uintptr_t>(data) % requirement.alignment != 0) {
            return {};
        }
        return {.data = data, .size = requirement.bytes};
    }

    void Reset() noexcept override {}

private:
    std::byte* base_ = nullptr;
    size_t size_ = 0;
};

class RequestResources {
public:
    explicit RequestResources(KVCacheManager& manager) noexcept
        : manager_(&manager) {}

    RequestResources(const RequestResources&) = delete;
    RequestResources& operator=(const RequestResources&) = delete;
    ~RequestResources() {
        static_cast<void>(Close());
    }

    Status Reserve(size_t prompt_len, size_t future_kv_appends) {
        auto reservation = manager_->ReserveForSession(prompt_len, future_kv_appends);
        if (!reservation.ok()) {
            return reservation.status();
        }
        cache_view_ = std::move(*reservation);
        has_reservation_ = true;
        return Status::Ok();
    }

    Status InitializeWorkspace(size_t bytes, size_t alignment) {
        if (bytes == 0) {
            return Status::Ok();
        }
        if (!IsValidWorkspaceAlignment(alignment)) {
            return Status::InvalidArgument(
                    "InferenceSession plan has an invalid workspace alignment");
        }

        size_t allocation_bytes = 0;
        if (CheckOverflowAdd(bytes, alignment - 1, &allocation_bytes)) {
            return Status::Overflow(
                    "InferenceSession workspace allocation size overflowed size_t");
        }
        storage_.reset(new (std::nothrow) std::byte[allocation_bytes]);
        if (storage_ == nullptr) {
            return Status::ResourceExhausted(
                    "InferenceSession could not allocate execution workspace");
        }

        const uintptr_t address = reinterpret_cast<uintptr_t>(storage_.get());
        const size_t remainder = static_cast<size_t>(address % alignment);
        const size_t padding = remainder == 0 ? 0 : alignment - remainder;
        workspace_base_ = storage_.get() + padding;
        workspace_size_ = bytes;
        workspace_arena_.SetBuffer(workspace_base_, workspace_size_);
        return Status::Ok();
    }

    WorkspaceArena* workspace_arena() noexcept {
        return workspace_size_ == 0 ? nullptr : &workspace_arena_;
    }

    KVCacheView cache_view() const noexcept {
        return cache_view_;
    }

    Status PrepareContext(Runtime& runtime,
                          const ExecutionPlan& plan,
                          const ExternalTensorBindings& immutable_bindings,
                          uint32_t token_input,
                          uint32_t position_input,
                          std::span<const int64_t> tokens,
                          std::span<const int64_t> positions) {
        if (tokens.empty() || tokens.size() != positions.size() ||
            tokens.size() > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
            return Status::InvalidArgument(
                    "InferenceSession token and position inputs must have equal non-zero lengths");
        }

        ExternalTensorBindings external = immutable_bindings;
        external.readable.reserve(external.readable.size() + 2);
        const std::array<int64_t, 1> shape{static_cast<int64_t>(tokens.size())};
        const std::array<int64_t, 1> strides{1};
        external.readable.push_back({
                .value = {.index = token_input},
                .tensor = TensorView(tokens.data(), DataType::Int(64), shape, strides,
                                     alignof(int64_t)),
        });
        external.readable.push_back({
                .value = {.index = position_input},
                .tensor = TensorView(positions.data(), DataType::Int(64), shape, strides,
                                     alignof(int64_t)),
        });

        auto prepared = PrepareExecutionBindings(
                plan, external, runtime.GetAllocator(Device::CPU()));
        if (!prepared.ok()) {
            return prepared.status();
        }
        auto context = ExecutionContext::Create(
                plan, std::move(*prepared), workspace_arena(), cache_view_);
        if (!context.ok()) {
            return context.status();
        }
        context_.emplace(std::move(*context));
        return Status::Ok();
    }

    ExecutionContext* context() noexcept {
        return context_.has_value() ? &*context_ : nullptr;
    }

    void ClearContext() noexcept {
        if (context_.has_value()) {
            context_->Clear();
            context_.reset();
        }
    }

    Status Close() noexcept {
        ClearContext();
        if (!has_reservation_) {
            return Status::Ok();
        }
        const Status status = manager_->ReleaseSession(cache_view_);
        if (status.ok()) {
            has_reservation_ = false;
        }
        return status;
    }

    StatusOr<std::vector<uint32_t>> Finish() {
        const Status status = Close();
        if (!status.ok()) {
            return status;
        }
        return std::move(generated_tokens);
    }

    std::vector<int64_t> prefill_tokens{};
    std::vector<int64_t> prefill_positions{};
    std::vector<int64_t> decode_tokens{};
    std::vector<int64_t> decode_positions{};
    std::vector<uint32_t> generated_tokens{};

private:
    KVCacheManager* manager_ = nullptr;
    KVCacheView cache_view_{};
    bool has_reservation_ = false;
    std::unique_ptr<std::byte[]> storage_{};
    std::byte* workspace_base_ = nullptr;
    size_t workspace_size_ = 0;
    SessionWorkspaceArena workspace_arena_{};
    std::optional<ExecutionContext> context_{};
};

StatusOr<uint32_t> ReadOutputToken(const ExecutionContext& context,
                                   ExecutionValueId output,
                                   size_t expected_length,
                                   size_t vocab_size) {
    const PreparedExecutionBindings* const prepared = context.prepared_bindings();
    if (prepared == nullptr || output.index >= prepared->values().size()) {
        return Status::Internal(
                "InferenceSession token output has no prepared binding");
    }

    const TensorView& tensor = prepared->values()[output.index].readable;
    if (!tensor.is_valid() || tensor.dtype() != DataType::Int(64) ||
        tensor.rank() != 1 ||
        tensor.dim(0) != static_cast<int64_t>(expected_length) ||
        !tensor.is_contiguous()) {
        return Status::Internal(
                "InferenceSession token output has an unexpected runtime shape or dtype");
    }

    const auto* const data = static_cast<const int64_t*>(tensor.data());
    const int64_t raw_token = data[expected_length - 1];
    if (raw_token < 0 || static_cast<uint64_t>(raw_token) >= vocab_size ||
        static_cast<uint64_t>(raw_token) > std::numeric_limits<uint32_t>::max()) {
        return Status::OutOfRange(
                "InferenceSession Argmax result does not fit the token ID contract");
    }
    return static_cast<uint32_t>(raw_token);
}


} // namespace

InferenceSession::InferenceSession(
        Runtime& runtime,
        std::shared_ptr<const ExecutableModel> model,
        PhaseContract prefill_contract,
        PhaseContract decode_contract) noexcept
    : runtime_(&runtime),
      model_(std::move(model)),
      prefill_contract_(prefill_contract),
      decode_contract_(decode_contract) {}

StatusOr<InferenceSession> InferenceSession::Create(
        Runtime& runtime,
        std::shared_ptr<const ExecutableModel> model) {
    if (model == nullptr) {
        return Status::InvalidArgument(
                "InferenceSession requires a non-null ExecutableModel");
    }
    if (!model->IsPreparedFor(runtime)) {
        return Status::FailedPrecondition(
                "InferenceSession Runtime must match the model preparation Runtime");
    }
    if (model->context_limit() == 0 || model->vocab_size() == 0) {
        return Status::FailedPrecondition(
                "InferenceSession ExecutableModel has no validated token limits");
    }

    auto prefill_plan = model->plan(ExecPhase::kPrefill);
    if (!prefill_plan.ok()) {
        return prefill_plan.status();
    }
    auto decode_plan = model->plan(ExecPhase::kDecode);
    if (!decode_plan.ok()) {
        return decode_plan.status();
    }

    auto prefill_contract = ValidatePlanContract(**prefill_plan);
    if (!prefill_contract.ok()) {
        return prefill_contract.status();
    }
    auto decode_contract = ValidatePlanContract(**decode_plan);
    if (!decode_contract.ok()) {
        return decode_contract.status();
    }
    const PhaseContract prefill{
            .token_input = prefill_contract->token_input,
            .position_input = prefill_contract->position_input,
            .token_output = prefill_contract->token_output,
    };
    const PhaseContract decode{
            .token_input = decode_contract->token_input,
            .position_input = decode_contract->position_input,
            .token_output = decode_contract->token_output,
    };
    return InferenceSession(runtime, std::move(model), prefill, decode);
}

StatusOr<std::vector<uint32_t>> InferenceSession::Generate(
        std::span<const uint32_t> prompt_tokens,
        const GenerationConfig& config) {
    if (runtime_ == nullptr || model_ == nullptr) {
        return Status::FailedPrecondition(
                "InferenceSession is moved from or has no Runtime");
    }
    if (prompt_tokens.empty()) {
        return Status::InvalidArgument(
                "InferenceSession prompt must contain at least one token");
    }

    const size_t vocab_size = model_->vocab_size();
    for (const uint32_t token: prompt_tokens) {
        if (static_cast<size_t>(token) >= vocab_size) {
            return Status::OutOfRange(
                    "InferenceSession prompt token is outside the model vocabulary");
        }
    }
    if (config.eos_token_id.has_value() &&
        static_cast<size_t>(*config.eos_token_id) >= vocab_size) {
        return Status::OutOfRange(
                "InferenceSession EOS token is outside the model vocabulary");
    }

    if (prompt_tokens.size() > model_->context_limit()) {
        return Status::OutOfRange(
                "InferenceSession prompt exceeds the model context limit");
    }
    if (config.max_new_tokens == 0) {
        return std::vector<uint32_t>{};
    }

    const size_t future_kv_appends = config.max_new_tokens - 1;
    size_t required_context_positions = 0;
    if (CheckOverflowAdd(prompt_tokens.size(), future_kv_appends,
                         &required_context_positions)) {
        return Status::Overflow(
                "InferenceSession requested token count overflowed size_t");
    }
    if (required_context_positions > model_->context_limit()) {
        return Status::OutOfRange(
                "InferenceSession prompt and generation exceed the model context limit");
    }

    KVCacheManager* const manager = runtime_->GetKVCacheManager();
    if (manager == nullptr) {
        return Status::FailedPrecondition(
                "InferenceSession Generate requires a Runtime KVCacheManager");
    }
    if (!runtime_->HasAllocatorProvider(DeviceType::kCPU)) {
        return Status::FailedPrecondition(
                "InferenceSession Generate requires a CPU activation allocator");
    }

    auto prefill_plan_result = model_->plan(ExecPhase::kPrefill);
    if (!prefill_plan_result.ok()) {
        return prefill_plan_result.status();
    }
    auto decode_plan_result = model_->plan(ExecPhase::kDecode);
    if (!decode_plan_result.ok()) {
        return decode_plan_result.status();
    }
    const ExecutionPlan& prefill_plan = **prefill_plan_result;
    const ExecutionPlan& decode_plan = **decode_plan_result;

    auto prefill_bindings_result =
            model_->immutable_weight_bindings(ExecPhase::kPrefill);
    if (!prefill_bindings_result.ok()) {
        return prefill_bindings_result.status();
    }
    auto decode_bindings_result =
            model_->immutable_weight_bindings(ExecPhase::kDecode);
    if (!decode_bindings_result.ok()) {
        return decode_bindings_result.status();
    }

    RequestResources request(*manager);
    AM_RETURN_IF_ERROR(request.Reserve(prompt_tokens.size(), future_kv_appends));

    request.prefill_tokens.resize(prompt_tokens.size());
    request.prefill_positions.resize(prompt_tokens.size());
    for (size_t index = 0; index < prompt_tokens.size(); ++index) {
        request.prefill_tokens[index] = static_cast<int64_t>(prompt_tokens[index]);
        request.prefill_positions[index] = static_cast<int64_t>(index);
    }
    request.generated_tokens.reserve(config.max_new_tokens);

    const size_t workspace_bytes =
            std::max(prefill_plan.total_workspace_bytes(),
                     decode_plan.total_workspace_bytes());
    const size_t workspace_alignment =
            std::max(prefill_plan.workspace_alignment(),
                     decode_plan.workspace_alignment());
    AM_RETURN_IF_ERROR(
            request.InitializeWorkspace(workspace_bytes, workspace_alignment));

    AM_RETURN_IF_ERROR(request.PrepareContext(
            *runtime_, prefill_plan, **prefill_bindings_result,
            prefill_contract_.token_input, prefill_contract_.position_input,
            request.prefill_tokens, request.prefill_positions));

    ExecutionContext* const prefill_context = request.context();
    if (prefill_context == nullptr) {
        return Status::Internal(
                "InferenceSession failed to create its Prefill context");
    }
    if (WorkspaceArena* const workspace = prefill_context->workspace_arena();
        workspace != nullptr) {
        workspace->Reset();
    }
    AM_RETURN_IF_ERROR(Executor::Execute(prefill_plan, *prefill_context));
    if (prefill_context->kv_cache_view().current_pos() != prompt_tokens.size()) {
        return Status::Internal(
                "InferenceSession Prefill did not commit the complete prompt");
    }

    const auto first_token = ReadOutputToken(
            *prefill_context, {.index = prefill_contract_.token_output},
            prompt_tokens.size(), vocab_size);
    if (!first_token.ok()) {
        return first_token.status();
    }
    request.generated_tokens.push_back(*first_token);
    request.ClearContext();

    const bool stop_on_eos = config.eos_token_id.has_value() &&
                             request.generated_tokens.back() == *config.eos_token_id;
    if (request.generated_tokens.size() == config.max_new_tokens || stop_on_eos) {
        return request.Finish();
    }

    request.decode_tokens.resize(1);
    request.decode_positions.resize(1);
    request.decode_tokens[0] = static_cast<int64_t>(request.generated_tokens.back());
    request.decode_positions[0] = static_cast<int64_t>(prompt_tokens.size());

    AM_RETURN_IF_ERROR(request.PrepareContext(
            *runtime_, decode_plan, **decode_bindings_result,
            decode_contract_.token_input, decode_contract_.position_input,
            request.decode_tokens, request.decode_positions));

    ExecutionContext* const decode_context = request.context();
    if (decode_context == nullptr) {
        return Status::Internal(
                "InferenceSession failed to create its Decode context");
    }
    AM_RETURN_IF_ERROR(inference::internal::RunDecodeLoop(
            decode_plan, *decode_context,
            {.index = decode_contract_.token_output},
            request.decode_tokens.data(), request.decode_positions.data(),
            vocab_size, config.max_new_tokens, config.eos_token_id,
            request.generated_tokens));
    return request.Finish();
}

namespace inference::internal {

Status RunDecodeLoop(const ExecutionPlan& plan,
                     ExecutionContext& context,
                     ExecutionValueId token_output,
                     int64_t* input_token,
                     int64_t* next_position,
                     size_t vocab_size,
                     size_t max_new_tokens,
                     std::optional<uint32_t> eos_token_id,
                     std::vector<uint32_t>& generated_tokens) {
    if (input_token == nullptr || next_position == nullptr ||
        generated_tokens.empty() ||
        generated_tokens.capacity() < max_new_tokens ||
        generated_tokens.size() > max_new_tokens ||
        max_new_tokens == 0 || vocab_size == 0 || *next_position < 0) {
        return Status::InvalidArgument(
                "InferenceSession Decode loop received invalid prepared state");
    }

    while (generated_tokens.size() < max_new_tokens) {
        if (eos_token_id.has_value() &&
            generated_tokens.back() == *eos_token_id) {
            return Status::Ok();
        }

        *input_token = static_cast<int64_t>(generated_tokens.back());
        if (WorkspaceArena* const workspace = context.workspace_arena();
            workspace != nullptr) {
            workspace->Reset();
        }
        AM_RETURN_IF_ERROR(Executor::Execute(plan, context));

        size_t expected_position = 0;
        if (CheckOverflowAdd(static_cast<size_t>(*next_position), size_t{1},
                             &expected_position)) {
            return Status::Overflow(
                    "InferenceSession Decode position overflowed size_t");
        }
        if (context.kv_cache_view().current_pos() != expected_position) {
            return Status::Internal(
                    "InferenceSession Decode did not commit exactly one KV position");
        }

        auto token = ReadOutputToken(context, token_output, 1, vocab_size);
        if (!token.ok()) {
            return token.status();
        }
        generated_tokens.push_back(*token);

        if (eos_token_id.has_value() && *token == *eos_token_id) {
            return Status::Ok();
        }
        if (generated_tokens.size() < max_new_tokens) {
            if (*next_position == std::numeric_limits<int64_t>::max()) {
                return Status::Overflow(
                        "InferenceSession Decode position exceeds Int64 range");
            }
            ++(*next_position);
        }
    }
    return Status::Ok();
}

} // namespace inference::internal
} // namespace aethermind
