#include "aethermind/inference/inference_session.h"
#include "aethermind/base/alignment.h"
#include "aethermind/base/workspace_arena.h"
#include "aethermind/execution/execution_bindings.h"
#include "aethermind/execution/execution_context.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/execution/executor.h"
#include "aethermind/inference/executable_model.h"
#include "aethermind/kv_cache/kv_cache_manager.h"
#include "aethermind/memory/allocator.h"
#include "aethermind/runtime/runtime.h"
#include "inference_internal.h"
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
#include <string>
#include <utility>
#include <vector>

namespace aethermind {
namespace {

class SessionWorkspaceArena final : public WorkspaceArena {
public:
    void SetBuffer(void* base, size_t size) noexcept {
        base_ = static_cast<std::byte*>(base);
        size_ = size;
    }

    WorkspaceBinding Bind(const WorkspaceRequirement& requirement) noexcept override {
        if (requirement.bytes == 0 || base_ == nullptr ||
            !IsValidAlignment(requirement.alignment) ||
            requirement.offset > size_ ||
            requirement.bytes > size_ - requirement.offset) {
            return {};
        }

        auto* const data = base_ + requirement.offset;
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
        UNUSED(Close());
    }

    Status Reserve(size_t prompt_len, size_t future_kv_appends) {
        auto reservation =
                manager_->ReserveForSession(prompt_len, future_kv_appends);
        if (!reservation.ok()) {
            return reservation.status();
        }

        cache_view_ = *reservation;
        has_reservation_ = true;
        return Status::Ok();
    }

    Status InitializeWorkspace(Allocator& allocator, size_t bytes, size_t alignment) {
        if (bytes == 0) {
            return Status::Ok();
        }

        if (!IsValidAlignment(alignment)) {
            return Status::InvalidArgument(
                    "InferenceSession plan has an invalid workspace alignment");
        }

        // The plan's alignment is a floor: the allocator may return stronger,
        // and every planned offset stays valid against a stronger base.
        auto storage = allocator.TryAllocate(
                {.nbytes = bytes,
                 .alignment = alignment});
        if (!storage.ok()) {
            return storage.status();
        }

        storage_ = std::move(storage).value();
        workspace_arena_.SetBuffer(storage_.mutable_data(), bytes);
        return Status::Ok();
    }

    WorkspaceArena* workspace_arena() noexcept {
        return storage_.is_initialized() ? &workspace_arena_ : nullptr;
    }

    StatusOr<ExecutionBindingPlan> PlanBindings(
            const ExecutionPlan& plan,
            const ExternalTensorBindings& immutable_bindings,
            uint32_t token_input,
            uint32_t position_input,
            std::span<const int64_t> tokens,
            std::span<const int64_t> positions) {
        AM_CHECK(!context_.has_value(),
                 "PrepareContext requires the previous execution context to be cleared");

        if (tokens.empty() || tokens.size() != positions.size() ||
            !std::in_range<int64_t>(tokens.size())) {
            return Status::InvalidArgument("InferenceSession token and position "
                                           "inputs must have equal non-zero lengths");
        }

        ExternalTensorBindings external = immutable_bindings;
        external.read_only.reserve(external.read_only.size() + 2);
        const std::array<int64_t, 1> shape{static_cast<int64_t>(tokens.size())};
        constexpr std::array<int64_t, 1> strides{1};
        external.read_only.push_back({
                .value = {.index = token_input},
                .tensor = TensorView(tokens.data(), DataType::Int(64), shape, strides,
                                     alignof(int64_t)),
        });

        external.read_only.push_back({
                .value = {.index = position_input},
                .tensor = TensorView(positions.data(), DataType::Int(64), shape, strides,
                                     alignof(int64_t)),
        });

        return PlanExecutionBindings(plan, external);
    }

    Status PrepareContext(Allocator& allocator,
                          const ExecutionPlan& plan,
                          const ExternalTensorBindings& immutable_bindings,
                          uint32_t token_input,
                          uint32_t position_input,
                          std::span<const int64_t> tokens,
                          std::span<const int64_t> positions) {
        AM_ASSIGN_OR_RETURN(auto binding_plan,
                            PlanBindings(plan, immutable_bindings, token_input, position_input, tokens, positions));
        return MaterializeContext(allocator, plan, std::move(binding_plan));
    }

    Status MaterializeContext(Allocator& allocator, const ExecutionPlan& plan,
                              ExecutionBindingPlan binding_plan) {
        AM_CHECK(!context_.has_value(), "MaterializeContext requires the previous context to be cleared");
        AM_ASSIGN_OR_RETURN(auto prepared,
                            MaterializeExecutionBindings(plan, std::move(binding_plan), allocator));
        auto context = ExecutionContext::Create(
                plan, std::move(prepared), workspace_arena(), cache_view_);
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
        context_.reset();
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
        if (auto status = Close(); !status.ok()) {
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
    KVCacheSessionView cache_view_{};
    bool has_reservation_ = false;
    // Owns the workspace backing storage; the arena borrows its memory and
    // context_ is cleared in the destructor before this releases it.
    Buffer storage_{};
    SessionWorkspaceArena workspace_arena_{};
    std::optional<ExecutionContext> context_{};
};

StatusOr<uint32_t> ReadOutputToken(const ExecutionContext& context,
                                   ExecutionValueId output,
                                   size_t expected_length,
                                   size_t vocab_size) {
    const auto* const prepared = context.prepared_bindings();
    if (prepared == nullptr || output.index >= prepared->values().size()) {
        return Status::Internal(
                "InferenceSession token output has no prepared binding");
    }

    const auto tensor = prepared->values()[output.index].ReadView();
    if (!tensor.is_valid() || tensor.dtype() != DataType::Int(64) ||
        tensor.rank() != 1 ||
        tensor.dim(0) != static_cast<int64_t>(expected_length) ||
        !tensor.is_contiguous()) {
        return Status::Internal(
                "InferenceSession token output has "
                "an unexpected runtime shape or dtype");
    }

    const auto* const data = static_cast<const int64_t*>(tensor.data());
    const int64_t raw_token = data[expected_length - 1];
    if (!std::in_range<uint32_t>(raw_token) ||
        static_cast<uint64_t>(raw_token) >= vocab_size) {
        return Status::OutOfRange(
                "InferenceSession Argmax result does not fit the token ID contract");
    }
    return static_cast<uint32_t>(raw_token);
}


StatusOr<RequestResourceUsage> RequestUsage(const Runtime& runtime,
                                            const ExecutableModel& model,
                                            size_t workspace, size_t prefill, size_t decode) {
    RequestResourceUsage usage{.workspace_bytes = workspace,
                               .prefill_activation_bytes = prefill,
                               .decode_activation_bytes = decode};
    if (CheckOverflowAdd(workspace, std::max(prefill, decode), &usage.peak_payload_bytes)) {
        return Status::Overflow("Request payload peak overflowed size_t");
    }
    const auto* manager = runtime.GetKVCacheManager();
    usage.kv_reserved_bytes = manager == nullptr ? 0 : manager->total_bytes();
    usage.model_managed_payload_bytes = model.resource_usage().managed_payload_bytes;
    usage.model_payload_complete = model.resource_usage().weights.backing_complete;
    if (CheckOverflowAdd(usage.kv_reserved_bytes, usage.model_managed_payload_bytes,
                         &usage.peak_with_persistent_payload_bytes) ||
        CheckOverflowAdd(usage.peak_with_persistent_payload_bytes, usage.peak_payload_bytes,
                         &usage.peak_with_persistent_payload_bytes)) {
        return Status::Overflow("Model and request payload peak overflowed size_t");
    }
    return usage;
}


} // namespace

InferenceSession::InferenceSession(Runtime& runtime,
                                   std::shared_ptr<const ExecutableModel> model,
                                   PhaseContract prefill_contract,
                                   PhaseContract decode_contract,
                                   size_t max_request_payload_bytes) noexcept
    : runtime_(&runtime),
      model_(std::move(model)),
      prefill_contract_(prefill_contract),
      decode_contract_(decode_contract),
      max_request_payload_bytes_(max_request_payload_bytes) {}

StatusOr<InferenceSession> InferenceSession::Create(
        Runtime& runtime, std::shared_ptr<const ExecutableModel> model) {
    return Create(runtime, std::move(model), {});
}

StatusOr<InferenceSession> InferenceSession::Create(
        Runtime& runtime,
        std::shared_ptr<const ExecutableModel> model,
        const InferenceSessionOptions& options) {
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

    auto prefill_contract =
            inference::internal::ValidatePlanContract(**prefill_plan);
    if (!prefill_contract.ok()) {
        return prefill_contract.status();
    }

    auto decode_contract =
            inference::internal::ValidatePlanContract(**decode_plan);
    if (!decode_contract.ok()) {
        return decode_contract.status();
    }

    return InferenceSession(runtime, std::move(model), *prefill_contract,
                            *decode_contract, options.max_request_payload_bytes);
}

StatusOr<RequestResourceUsage> InferenceSession::PlanResources(
        size_t prompt_tokens, const GenerationConfig& config) const try {
    if (runtime_ == nullptr || model_ == nullptr) {
        return Status::FailedPrecondition("InferenceSession is moved from or has no Runtime");
    }
    if (prompt_tokens == 0 || !std::in_range<int64_t>(prompt_tokens)) {
        return Status::InvalidArgument("Resource planning requires a representable nonempty prompt");
    }
    if (prompt_tokens > model_->context_limit() ||
        (config.eos_token_id.has_value() && *config.eos_token_id >= model_->vocab_size())) {
        return Status::OutOfRange("Resource planning exceeds model token limits");
    }
    if (config.max_new_tokens == 0) {
        return RequestUsage(*runtime_, *model_, 0, 0, 0);
    }
    size_t context = 0;
    if (CheckOverflowAdd(prompt_tokens, config.max_new_tokens - 1, &context)) {
        return Status::Overflow("Resource planning token count overflowed size_t");
    }
    if (context > model_->context_limit()) {
        return Status::OutOfRange("Resource planning exceeds model context");
    }
    auto* manager = runtime_->GetKVCacheManager();
    if (manager == nullptr) {
        return Status::FailedPrecondition("Resource planning requires a KVCacheManager");
    }
    AM_RETURN_IF_ERROR(manager->ValidateReservation(prompt_tokens, config.max_new_tokens - 1));
    AM_ASSIGN_OR_RETURN(const ExecutionPlan* prefill, model_->plan(ExecPhase::kPrefill));
    AM_ASSIGN_OR_RETURN(const ExecutionPlan* decode, model_->plan(ExecPhase::kDecode));
    AM_ASSIGN_OR_RETURN(const ExternalTensorBindings* prefill_weights, model_->immutable_weight_bindings(ExecPhase::kPrefill));
    AM_ASSIGN_OR_RETURN(const ExternalTensorBindings* decode_weights, model_->immutable_weight_bindings(ExecPhase::kDecode));
    RequestResources scratch(*manager);
    scratch.prefill_tokens.resize(prompt_tokens);
    scratch.prefill_positions.resize(prompt_tokens);
    AM_ASSIGN_OR_RETURN(auto prefill_layout, scratch.PlanBindings(*prefill, *prefill_weights,
                                                                  prefill_contract_.token_input, prefill_contract_.position_input,
                                                                  scratch.prefill_tokens, scratch.prefill_positions));
    size_t decode_bytes = 0;
    if (config.max_new_tokens > 1) {
        scratch.decode_tokens.resize(1);
        scratch.decode_positions.resize(1);
        AM_ASSIGN_OR_RETURN(auto decode_layout, scratch.PlanBindings(*decode, *decode_weights,
                                                                     decode_contract_.token_input, decode_contract_.position_input,
                                                                     scratch.decode_tokens, scratch.decode_positions));
        decode_bytes = decode_layout.activation_bytes();
    }
    return RequestUsage(*runtime_, *model_,
                        std::max(prefill->total_workspace_bytes(), decode->total_workspace_bytes()),
                        prefill_layout.activation_bytes(), decode_bytes);
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

StatusOr<std::vector<uint32_t>> InferenceSession::Generate(
        std::span<const uint32_t> prompt_tokens,
        const GenerationConfig& config) const try {
    if (runtime_ == nullptr || model_ == nullptr) {
        return Status::FailedPrecondition(
                "InferenceSession is moved from or has no Runtime");
    }

    if (prompt_tokens.empty()) {
        return Status::InvalidArgument(
                "InferenceSession prompt must contain at least one token");
    }

    const size_t vocab_size = model_->vocab_size();
    for (const auto token: prompt_tokens) {
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
    size_t required_context_pos = 0;
    if (CheckOverflowAdd(prompt_tokens.size(), future_kv_appends, &required_context_pos)) {
        return Status::Overflow(
                "InferenceSession requested token count overflowed size_t");
    }

    if (required_context_pos > model_->context_limit()) {
        return Status::OutOfRange(
                "InferenceSession prompt and generation exceed the model context limit");
    }

    KVCacheManager* const manager = runtime_->GetKVCacheManager();
    if (manager == nullptr) {
        return Status::FailedPrecondition(
                "InferenceSession Generate requires a Runtime KVCacheManager");
    }

    // One allocator serves both the workspace backing storage and the two
    // activation arenas, so resolve it once and report a missing provider here
    // rather than aborting inside the registry lookup.
    AM_ASSIGN_OR_RETURN(Allocator * allocator,
                        runtime_->TryGetAllocator(Device::CPU()));

    auto prefill_plan_result = model_->plan(ExecPhase::kPrefill);
    if (!prefill_plan_result.ok()) {
        return prefill_plan_result.status();
    }

    auto decode_plan_result = model_->plan(ExecPhase::kDecode);
    if (!decode_plan_result.ok()) {
        return decode_plan_result.status();
    }

    const auto& prefill_plan = **prefill_plan_result;
    const auto& decode_plan = **decode_plan_result;

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

    const size_t workspace_bytes = std::max(prefill_plan.total_workspace_bytes(),
                                            decode_plan.total_workspace_bytes());
    const bool request_cap = max_request_payload_bytes_ != 0;
    if (request_cap || runtime_->workspace_limit_bytes() != 0) {
        AM_RETURN_IF_ERROR(manager->ValidateReservation(prompt_tokens.size(), future_kv_appends));
    }
    if (runtime_->workspace_limit_bytes() != 0 && workspace_bytes > runtime_->workspace_limit_bytes()) {
        return Status::ResourceExhausted("Workspace payload exceeds limit: required=" +
                                         std::to_string(workspace_bytes) + ", limit=" + std::to_string(runtime_->workspace_limit_bytes()));
    }
    RequestResources request(*manager);
    if (!request_cap) {
        AM_RETURN_IF_ERROR(request.Reserve(prompt_tokens.size(), future_kv_appends));
    }

    request.prefill_tokens.resize(prompt_tokens.size());
    request.prefill_positions.resize(prompt_tokens.size());
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        request.prefill_tokens[i] = static_cast<int64_t>(prompt_tokens[i]);
        request.prefill_positions[i] = static_cast<int64_t>(i);
    }
    request.generated_tokens.reserve(config.max_new_tokens);

    std::optional<ExecutionBindingPlan> prefill_layout;
    std::optional<ExecutionBindingPlan> decode_layout;
    if (request_cap) {
        AM_ASSIGN_OR_RETURN(auto planned, request.PlanBindings(prefill_plan, **prefill_bindings_result,
                                                               prefill_contract_.token_input, prefill_contract_.position_input,
                                                               request.prefill_tokens, request.prefill_positions));
        prefill_layout.emplace(std::move(planned));
        size_t decode_bytes = 0;
        if (config.max_new_tokens > 1) {
            request.decode_tokens.resize(1);
            request.decode_positions.resize(1);
            AM_ASSIGN_OR_RETURN(auto decode_planned, request.PlanBindings(decode_plan, **decode_bindings_result,
                                                                          decode_contract_.token_input, decode_contract_.position_input,
                                                                          request.decode_tokens, request.decode_positions));
            decode_bytes = decode_planned.activation_bytes();
            decode_layout.emplace(std::move(decode_planned));
        }
        AM_ASSIGN_OR_RETURN(const auto usage, RequestUsage(*runtime_, *model_, workspace_bytes,
                                                           prefill_layout->activation_bytes(), decode_bytes));
        if (usage.peak_payload_bytes > max_request_payload_bytes_) {
            return Status::ResourceExhausted("Request payload exceeds limit: required=" +
                                             std::to_string(usage.peak_payload_bytes) + ", limit=" + std::to_string(max_request_payload_bytes_));
        }
        AM_RETURN_IF_ERROR(request.Reserve(prompt_tokens.size(), future_kv_appends));
    }
    const size_t workspace_alignment =
            std::max(prefill_plan.workspace_alignment(),
                     decode_plan.workspace_alignment());
    AM_RETURN_IF_ERROR(request.InitializeWorkspace(*allocator, workspace_bytes,
                                                   workspace_alignment));

    if (prefill_layout.has_value()) {
        AM_RETURN_IF_ERROR(request.MaterializeContext(*allocator, prefill_plan, std::move(*prefill_layout)));
    } else {
        AM_RETURN_IF_ERROR(request.PrepareContext(*allocator, prefill_plan, **prefill_bindings_result,
                                                  prefill_contract_.token_input, prefill_contract_.position_input,
                                                  request.prefill_tokens, request.prefill_positions));
    }

    ExecutionContext* const prefill_context = request.context();
    if (prefill_context == nullptr) {
        return Status::Internal(
                "InferenceSession failed to create its Prefill context");
    }

    AM_ASSIGN_OR_RETURN(const uint32_t first_token,
                        inference::internal::RunPrefill(
                                prefill_plan, *prefill_context,
                                {.index = prefill_contract_.token_output},
                                prompt_tokens.size(), vocab_size));
    request.generated_tokens.push_back(first_token);
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

    if (decode_layout.has_value()) {
        AM_RETURN_IF_ERROR(request.MaterializeContext(*allocator, decode_plan, std::move(*decode_layout)));
    } else {
        AM_RETURN_IF_ERROR(request.PrepareContext(*allocator, decode_plan, **decode_bindings_result,
                                                  decode_contract_.token_input, decode_contract_.position_input,
                                                  request.decode_tokens, request.decode_positions));
    }

    auto* const decode_context = request.context();
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
} catch (const std::bad_alloc&) {
    // Request-local owners have already unwound, clearing contexts before KV
    // release. An empty message avoids allocating while reporting this failure.
    return Status::ResourceExhausted({});
}

namespace inference::internal {

StatusOr<detail::PhaseContract> ValidatePlanContract(const ExecutionPlan& plan) {
    if (plan.model_inputs().size() != 2 || plan.model_outputs().size() != 1) {
        return Status::Unimplemented("InferenceSession requires one token "
                                     "output and token_ids/position_ids inputs");
    }

    detail::PhaseContract contract;
    // The Llama graph builder appends token_ids then position_ids, and lowering
    // preserves this ordered model_inputs contract independently of debug names.
    contract.token_input = plan.model_inputs()[0].index;
    contract.position_input = plan.model_inputs()[1].index;
    if (contract.token_input == contract.position_input) {
        return Status::InvalidArgument(
                "InferenceSession token and position inputs must be distinct");
    }

    for (const uint32_t idx: {contract.token_input, contract.position_input}) {
        if (idx >= plan.values().size()) {
            return Status::InvalidArgument("InferenceSession model input references"
                                           " an invalid execution value");
        }

        const auto& value = plan.values()[idx];
        if (value.kind != ExecutionValueKind::kModelInput) {
            return Status::InvalidArgument(
                    "InferenceSession model input must have kModelInput kind");
        }

        if (const auto rank = value.spec.shape.rank();
            value.spec.dtype != DataType::Int(64) || !rank.has_value() || *rank != 1) {
            return Status::Unimplemented(
                    "InferenceSession model inputs must be rank-one Int64 tensors");
        }
    }

    const auto output = plan.model_outputs().front();
    if (output.index >= plan.values().size()) {
        return Status::InvalidArgument(
                "InferenceSession model output references an invalid execution value");
    }

    const auto& output_value = plan.values()[output.index];
    if (const auto output_rank = output_value.spec.shape.rank();
        output_value.kind != ExecutionValueKind::kActivation ||
        output_value.spec.dtype != DataType::Int(64) ||
        !output_rank.has_value() || *output_rank != 1) {
        return Status::Unimplemented(
                "InferenceSession requires a rank-one Int64 token output");
    }

    size_t argmax_op_count = 0;
    bool output_is_argmax = false;
    for (const auto& step: plan.steps()) {
        if (step.selector.device_type != DeviceType::kCPU) {
            return Status::Unimplemented(
                    "InferenceSession currently supports CPU execution plans only");
        }

        if (step.kernel.op_type != OpType::kArgmax) {
            continue;
        }

        ++argmax_op_count;
        if (step.outputs.size() == 1 && step.outputs.front() == output) {
            output_is_argmax = true;
        }
    }

    if (argmax_op_count != 1 || !output_is_argmax) {
        return Status::Unimplemented(
                "InferenceSession requires its sole model output to be the Argmax result");
    }

    contract.token_output = output.index;
    return contract;
}

StatusOr<uint32_t> RunPrefill(const ExecutionPlan& plan,
                              ExecutionContext& context,
                              ExecutionValueId token_output,
                              size_t prompt_length,
                              size_t vocab_size) {
    if (prompt_length == 0 ||
        !std::in_range<int64_t>(prompt_length) ||
        vocab_size == 0) {
        return Status::InvalidArgument(
                "InferenceSession Prefill requires a positive Int64 prompt length "
                "and non-zero vocabulary size");
    }

    const auto& view = context.kv_cache_session_view();
    if (!context.HasKVCacheSessionView() || !view.awaiting_prefill() ||
        view.current_pos() != 0 || view.prompt_len() != prompt_length) {
        return Status::FailedPrecondition(
                "InferenceSession Prefill requires a fresh KV reservation "
                "matching the complete prompt length");
    }

    if (WorkspaceArena* const workspace = context.workspace_arena();
        workspace != nullptr) {
        workspace->Reset();
    }

    AM_RETURN_IF_ERROR(Executor::Execute(plan, context));
    if (view.current_pos() != prompt_length) {
        return Status::Internal(
                "InferenceSession Prefill did not commit the complete prompt");
    }

    return ReadOutputToken(context, token_output, prompt_length, vocab_size);
}

Status RunDecodeLoop(const ExecutionPlan& plan,
                     ExecutionContext& context,
                     ExecutionValueId token_output,
                     int64_t* input_token,
                     int64_t* next_pos,
                     size_t vocab_size,
                     size_t max_new_tokens,
                     std::optional<uint32_t> eos_token_id,
                     std::vector<uint32_t>& generated_tokens) {
    if (input_token == nullptr || next_pos == nullptr ||
        generated_tokens.empty() ||
        generated_tokens.capacity() < max_new_tokens ||
        generated_tokens.size() > max_new_tokens ||
        max_new_tokens == 0 || vocab_size == 0 || *next_pos < 0) {
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

        size_t expected_pos = 0;
        if (CheckOverflowAdd(static_cast<size_t>(*next_pos), size_t{1},
                             &expected_pos)) {
            return Status::Overflow(
                    "InferenceSession Decode position overflowed size_t");
        }

        if (context.kv_cache_session_view().current_pos() != expected_pos) {
            return Status::Internal(
                    "InferenceSession Decode did not "
                    "commit exactly one KV position");
        }

        auto token = ReadOutputToken(
                context, token_output, 1, vocab_size);
        if (!token.ok()) {
            return token.status();
        }
        generated_tokens.push_back(*token);

        if (eos_token_id.has_value() && *token == *eos_token_id) {
            return Status::Ok();
        }

        if (generated_tokens.size() < max_new_tokens) {
            if (*next_pos == std::numeric_limits<int64_t>::max()) {
                return Status::Overflow(
                        "InferenceSession Decode position exceeds Int64 range");
            }
            ++*next_pos;
        }
    }
    return Status::Ok();
}

} // namespace inference::internal
} // namespace aethermind
