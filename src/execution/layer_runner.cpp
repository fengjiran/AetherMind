#include "execution/layer_runner.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/kernel_invoker.h"
#include "aethermind/runtime/kv_cache_view.h"
#include "utils/overflow_check.h"

#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace aethermind {
namespace {

KernelContext BuildKernelContext(const ExecutionStep& step,
                                 const ExecutionContext& context,
                                 const KVCacheAppendBinding* kv_append,
                                 const KVCacheReadBinding* kv_read) noexcept {
    return KernelContext{
            .device_type = step.selector.device_type,
            .stream = nullptr,
            .workspace = context.workspace_arena(),
            .packed_weights = step.packed_weights
                                      ? step.packed_weights->storage().data()
                                      : nullptr,
            .kernel_params = nullptr,
            .kv_append = kv_append,
            .kv_read = kv_read,
            .attrs = step.kernel.attrs,
    };
}

StatusOr<ExecutionKVCacheStateIdentity> GetKVStateIdentity(
        const std::vector<ExecutionValueDesc>& values,
        ExecutionValueId id,
        std::string_view role) noexcept {
    if (id.index >= values.size()) {
        return Status::InvalidArgument("KV state " + std::string(role) +
                                       " references a value beyond the plan table");
    }

    const ExecutionValueDesc& value = values[id.index];
    if (value.kind != ExecutionValueKind::kState) {
        return Status::InvalidArgument("KV state " + std::string(role) +
                                       " must refer to a kState value");
    }

    const auto* identity = std::get_if<ExecutionKVCacheStateIdentity>(&value.state_binding);
    if (identity == nullptr) {
        return Status::InvalidArgument("KV state " + std::string(role) +
                                       " has no KV cache state identity");
    }
    return *identity;
}

Status ValidateKVCacheStateSpec(const TensorSpec& spec,
                                const KVCacheView& view,
                                std::string_view role) noexcept {
    if (spec.dtype != view.kv_dtype()) {
        return Status::InvalidArgument("KV state " + std::string(role) +
                                       " dtype does not match the KVCacheView element dtype");
    }

    if (const auto rank = spec.shape.rank(); !rank.has_value() || *rank != 3) {
        return Status::InvalidArgument("KV state " + std::string(role) +
                                       " must be rank 3 [kv_heads, cache_len, head_dim]");
    }

    if (!spec.shape[0].IsStatic() || !spec.shape[2].IsStatic()) {
        return Status::InvalidArgument("KV state " + std::string(role) +
                                       " requires static KV head and head dimension");
    }

    // Model graphs keep cache_len symbolic; KVCacheView owns the physical
    // capacity and supplies its runtime value.
    const bool cache_capacity_mismatch =
            spec.shape[1].IsStatic() &&
            spec.shape[1].GetStaticValue() != static_cast<int64_t>(view.max_tokens());
    if (spec.shape[0].GetStaticValue() != static_cast<int64_t>(view.num_kv_heads()) ||
        cache_capacity_mismatch ||
        spec.shape[2].GetStaticValue() != static_cast<int64_t>(view.head_dim())) {
        return Status::InvalidArgument("KV state " + std::string(role) +
                                       " geometry does not match the KVCacheView");
    }
    return Status::Ok();
}

Status ValidateKVCacheStatePair(const std::vector<ExecutionValueDesc>& values,
                                ExecutionValueId key_id,
                                ExecutionValueId value_id,
                                const KVCacheView& view,
                                std::string_view role) noexcept {
    if (key_id.index >= values.size() || value_id.index >= values.size()) {
        return Status::InvalidArgument("KV state " + std::string(role) +
                                       " references a value beyond the plan table");
    }

    const TensorSpec& key = values[key_id.index].spec;
    const TensorSpec& value = values[value_id.index].spec;
    if (key != value) {
        return Status::InvalidArgument("KV state " + std::string(role) +
                                       " key/value specs must match");
    }
    return ValidateKVCacheStateSpec(key, view, role);
}

StatusOr<uint32_t> ValidateKVCacheUpdateStateIdentity(
        const ExecutionStep& step,
        const std::vector<ExecutionValueDesc>& values,
        const KVCacheView& view) noexcept {
    if (step.inputs.size() != 4 || step.outputs.size() != 2) {
        return Status::InvalidArgument(
                "KVCacheUpdate requires its complete semantic K/V state ports");
    }
    AM_ASSIGN_OR_RETURN(const ExecutionKVCacheStateIdentity key_input,
                        GetKVStateIdentity(values, step.inputs[2], "key input"));
    AM_ASSIGN_OR_RETURN(const ExecutionKVCacheStateIdentity value_input,
                        GetKVStateIdentity(values, step.inputs[3], "value input"));
    AM_ASSIGN_OR_RETURN(const ExecutionKVCacheStateIdentity key_output,
                        GetKVStateIdentity(values, step.outputs[0], "key output"));
    AM_ASSIGN_OR_RETURN(const ExecutionKVCacheStateIdentity value_output,
                        GetKVStateIdentity(values, step.outputs[1], "value output"));

    if (key_input.slot != ExecutionKVCacheSlot::kKey ||
        value_input.slot != ExecutionKVCacheSlot::kValue ||
        key_output.slot != ExecutionKVCacheSlot::kKey ||
        value_output.slot != ExecutionKVCacheSlot::kValue) {
        return Status::InvalidArgument(
                "KVCacheUpdate state slots must be key then value");
    }

    if (key_input.decoder_layer_index != value_input.decoder_layer_index ||
        key_input != key_output || value_input != value_output) {
        return Status::InvalidArgument(
                "KVCacheUpdate state inputs and outputs must identify one decoder layer");
    }

    AM_RETURN_IF_ERROR(ValidateKVCacheStatePair(
            values, step.inputs[2], step.inputs[3], view, "KVCacheUpdate input"));
    AM_RETURN_IF_ERROR(ValidateKVCacheStatePair(
            values, step.outputs[0], step.outputs[1], view, "KVCacheUpdate output"));
    return key_input.decoder_layer_index;
}

StatusOr<uint32_t> ValidateAttentionStateIdentity(
        const ExecutionStep& step,
        const std::vector<ExecutionValueDesc>& values,
        const KVCacheView& view) noexcept {
    if (step.inputs.size() != 3) {
        return Status::InvalidArgument(
                "Attention requires complete key/value state ports");
    }

    AM_ASSIGN_OR_RETURN(const ExecutionKVCacheStateIdentity key,
                        GetKVStateIdentity(values, step.inputs[1], "attention key input"));
    AM_ASSIGN_OR_RETURN(const ExecutionKVCacheStateIdentity value,
                        GetKVStateIdentity(values, step.inputs[2], "attention value input"));
    if (key.slot != ExecutionKVCacheSlot::kKey ||
        value.slot != ExecutionKVCacheSlot::kValue ||
        key.decoder_layer_index != value.decoder_layer_index) {
        return Status::InvalidArgument(
                "Attention key/value state inputs must identify one decoder layer");
    }

    AM_RETURN_IF_ERROR(ValidateKVCacheStatePair(
            values, step.inputs[1], step.inputs[2], view, "Attention input"));
    return key.decoder_layer_index;
}

StatusOr<size_t> KVCacheUpdateSequenceLength(
        size_t step_index,
        const ExecutionStep& step,
        const PreparedExecutionBindings& prepared_bindings) noexcept {
    const StepTensorBinding& tensors = prepared_bindings.step(step_index);
    if (tensors.inputs.size() != 2 || step.kernel_input_ports.size() != 2 ||
        tensors.inputs[0].rank() != 2 || tensors.inputs[1].rank() != 2 ||
        tensors.inputs[0].shape() != tensors.inputs[1].shape() ||
        tensors.inputs[0].dim(0) <= 0) {
        return Status::InvalidArgument("KVCacheUpdate requires prepared K/V "
                                       "tensors with matching positive [T, hidden] shapes");
    }
    return static_cast<size_t>(tensors.inputs[0].dim(0));
}

} // namespace

struct KVAppendTransaction {
    enum class LayerProgress : uint8_t {
        kNotScheduled,
        kScheduled,
        kWritten,
    };

    bool has_append = false;
    size_t committed_begin = 0;
    size_t visible_end = 0;
    std::span<uint8_t> layer_progress{};
};

namespace {

Status PrepareKVAppendTransaction(const ExecutionPlan& plan,
                                  ExecutionContext& context,
                                  const PreparedExecutionBindings& prepared_bindings,
                                  KVAppendTransaction& transaction) noexcept {
    const KVCacheView& view = context.kv_cache_view();
    std::optional<size_t> sequence_length;
    if (context.HasKVCacheView()) {
        transaction.layer_progress = context.ResetKVLayerTransactionScratch();
        if (transaction.layer_progress.size() != view.num_layers()) {
            return Status::Internal("ExecutionContext KV transaction scratch "
                                    "does not match KVCacheView layers");
        }
    }

    for (size_t i = 0; i < plan.size(); ++i) {
        const ExecutionStep& step = plan.steps()[i];
        if (step.kernel.op_type != OpType::kKVCacheUpdate) {
            continue;
        }

        if (!context.HasKVCacheView()) {
            return Status::FailedPrecondition(
                    "KVCacheUpdate requires a valid KVCacheView");
        }

        AM_ASSIGN_OR_RETURN(const uint32_t layer,
                            ValidateKVCacheUpdateStateIdentity(step, plan.values(), view));
        if (layer >= transaction.layer_progress.size()) {
            return Status::OutOfRange(
                    "KVCacheUpdate state layer exceeds KVCacheView layer count");
        }

        if (transaction.layer_progress[layer] !=
            static_cast<uint8_t>(KVAppendTransaction::LayerProgress::kNotScheduled)) {
            return Status::InvalidArgument("KV append transaction has more than "
                                           "one update for one decoder layer");
        }

        transaction.layer_progress[layer] =
                static_cast<uint8_t>(KVAppendTransaction::LayerProgress::kScheduled);

        AM_ASSIGN_OR_RETURN(const size_t current_sequence_length,
                            KVCacheUpdateSequenceLength(i, step, prepared_bindings));
        if (sequence_length.has_value() && *sequence_length != current_sequence_length) {
            return Status::InvalidArgument("All KVCacheUpdate steps in one plan "
                                           "must have the same sequence length");
        }
        sequence_length = current_sequence_length;
    }

    transaction.committed_begin = context.HasKVCacheView() ? view.current_pos() : 0;
    transaction.visible_end = transaction.committed_begin;
    if (!sequence_length.has_value()) {
        return Status::Ok();
    }

    if (view.awaiting_prefill()) {
        if (*sequence_length != view.prompt_len() || transaction.committed_begin != 0) {
            return Status::FailedPrecondition(
                    "KV initial append must write "
                    "exactly the reserved Prefill length from position zero");
        }
    } else if (*sequence_length != 1) {
        return Status::InvalidArgument(
                "Phase 1 KV Decode append must contain exactly one token");
    }

    size_t end = 0;
    if (CheckOverflowAdd(transaction.committed_begin, *sequence_length, &end)) {
        return Status::Overflow("KV append transaction range overflowed size_t");
    }

    if (end > view.token_capacity() || end > view.max_tokens()) {
        return Status::OutOfRange("KV append transaction exceeds cache capacity");
    }

    for (size_t layer = 0; layer < transaction.layer_progress.size(); ++layer) {
        if (transaction.layer_progress[layer] !=
            static_cast<uint8_t>(KVAppendTransaction::LayerProgress::kScheduled)) {
            return Status::InvalidArgument("KV append transaction must update "
                                           "every KVCacheView layer exactly once");
        }

        if (const auto binding =
                    view.BindLayerForAppend(layer, transaction.committed_begin, end);
            !binding.ok()) {
            return binding.status();
        }
    }
    transaction.has_append = true;
    transaction.visible_end = end;
    return Status::Ok();
}

} // namespace

Status LayerRunner::Run(const ExecutionPlan& plan,
                        ExecutionContext& context) noexcept {
    const auto& steps = plan.steps();
    const auto& alias_plan = plan.state_alias_plan();
    const PreparedExecutionBindings* const prepared_bindings = context.prepared_bindings();
    if (prepared_bindings == nullptr) {
        return Status::FailedPrecondition("ExecutionContext requires a"
                                          " PreparedExecutionBindings before execution");
    }

    if (!prepared_bindings->IsCompatible(plan)) {
        return Status::InvalidArgument("ExecutionContext PreparedExecutionBindings "
                                       "is not compatible with ExecutionPlan");
    }

    KVAppendTransaction kv_transaction;
    AM_RETURN_IF_ERROR(PrepareKVAppendTransaction(
            plan, context, *prepared_bindings, kv_transaction));

    for (size_t i = 0; i < steps.size(); ++i) {
        if (const auto status = RunStep(i, steps[i], context, *prepared_bindings,
                                        alias_plan, plan.values(), kv_transaction);
            !status.ok()) {
            return status;
        }
    }

    if (kv_transaction.has_append) {
        KVCacheView view = context.kv_cache_view();
        AM_RETURN_IF_ERROR(view.CommitUntil(kv_transaction.visible_end));
    }
    return Status::Ok();
}

Status LayerRunner::RunStep(size_t step_index,
                            const ExecutionStep& step,
                            const ExecutionContext& context,
                            const PreparedExecutionBindings& prepared_bindings,
                            const StateAliasPlan& alias_plan,
                            const std::vector<ExecutionValueDesc>& values,
                            KVAppendTransaction& kv_transaction) noexcept {
    AM_RETURN_IF_ERROR(ValidateStateAliasesForStep(
            step_index, step, alias_plan, context, values));

    const auto workspace_binding =
            context.BindWorkspace(step.workspace_requirement);
    if (!workspace_binding.ok()) {
        return workspace_binding.status();
    }

    const StepTensorBinding& tensor_binding = prepared_bindings.step(step_index);
    if (tensor_binding.inputs.size() != step.kernel_input_ports.size() ||
        tensor_binding.outputs.size() != step.kernel_output_ports.size()) {
        return Status::InvalidArgument("Runtime tensor binding arity"
                                       " does not match ExecutionStep ports");
    }

    std::optional<KVCacheAppendBinding> append_binding;
    std::optional<KVCacheReadBinding> read_binding;
    if (step.kernel.op_type == OpType::kKVCacheUpdate) {
        AM_ASSIGN_OR_RETURN(const uint32_t layer,
                            ValidateKVCacheUpdateStateIdentity(
                                    step, values, context.kv_cache_view()));
        if (!kv_transaction.has_append) {
            return Status::Internal("KVCacheUpdate is absent from its append transaction");
        }

        const KVCacheView& view = context.kv_cache_view();
        AM_ASSIGN_OR_RETURN(append_binding,
                            view.BindLayerForAppend(layer, kv_transaction.committed_begin,
                                                    kv_transaction.visible_end));
    } else if (step.kernel.op_type == OpType::kAttention) {
        if (!context.HasKVCacheView()) {
            return Status::FailedPrecondition("Attention requires a valid KVCacheView");
        }

        if (tensor_binding.inputs.size() != 1 || tensor_binding.inputs[0].rank() != 2 ||
            tensor_binding.inputs[0].dim(0) <= 0) {
            return Status::InvalidArgument("Attention requires one prepared "
                                           "query TensorView with positive rank-2 sequence length");
        }

        const auto query_sequence_length = static_cast<size_t>(tensor_binding.inputs[0].dim(0));
        const KVCacheView& view = context.kv_cache_view();
        AM_ASSIGN_OR_RETURN(const uint32_t layer,
                            ValidateAttentionStateIdentity(step, values, view));
        if (layer >= kv_transaction.layer_progress.size()) {
            return Status::OutOfRange(
                    "Attention state layer exceeds KVCacheView layer count");
        }

        if (kv_transaction.has_append &&
            kv_transaction.layer_progress[layer] !=
                    static_cast<uint8_t>(KVAppendTransaction::LayerProgress::kWritten)) {
            return Status::FailedPrecondition("Attention cannot read this"
                                              " plan's uncommitted KV range before its layer update");
        }

        const size_t visible_end = kv_transaction.has_append
                                           ? kv_transaction.visible_end
                                           : kv_transaction.committed_begin;
        AM_ASSIGN_OR_RETURN(read_binding,
                            view.BindLayerForRead(
                                    layer, kv_transaction.committed_begin, visible_end));
        if (kv_transaction.has_append) {
            const size_t append_length =
                    kv_transaction.visible_end - kv_transaction.committed_begin;
            if (query_sequence_length != append_length) {
                return Status::FailedPrecondition("Attention query sequence length "
                                                  "must match this plan's KV append length");
            }

            read_binding->query_begin = kv_transaction.committed_begin;
            read_binding->query_end = kv_transaction.visible_end;
        } else {
            if (query_sequence_length > visible_end) {
                return Status::FailedPrecondition("Attention pure-read query "
                                                  "sequence length exceeds the visible KV range");
            }
            read_binding->query_begin = visible_end - query_sequence_length;
            read_binding->query_end = visible_end;
        }
    }

    KernelContext ctx = BuildKernelContext(step, context,
                                           append_binding ? &*append_binding : nullptr,
                                           read_binding ? &*read_binding : nullptr);
    ctx.workspace_binding = workspace_binding.value();

    Status status = InvokePreparedKernel(
            step.kernel, ctx, prepared_bindings.kernel_params(step_index));
    if (!status.ok()) {
        return status;
    }

    if (step.kernel.op_type == OpType::kKVCacheUpdate) {
        AM_ASSIGN_OR_RETURN(const uint32_t layer,
                            ValidateKVCacheUpdateStateIdentity(
                                    step, values, context.kv_cache_view()));
        kv_transaction.layer_progress[layer] = static_cast<uint8_t>(
                KVAppendTransaction::LayerProgress::kWritten);
    }
    return Status::Ok();
}

Status LayerRunner::ValidateStateAliasesForStep(
        size_t step_index,
        const ExecutionStep& step,
        const StateAliasPlan& alias_plan,
        const ExecutionContext& context,
        const std::vector<ExecutionValueDesc>& values) noexcept {
    const auto aliases = alias_plan.ForStep(step_index);
    if (aliases.empty()) {
        return Status::Ok();
    }

    // Phase 1: all state aliases are KV cache updates. The KVCacheView is the
    // shared physical storage that the operator reads and writes in place, so
    // its presence is the runtime invariant for must-alias state updates.
    if (!context.HasKVCacheView()) {
        return Status::InvalidArgument(
                "State alias requires a valid KVCacheView");
    }

    // Cross-check each alias against the bound view: the update is must-alias,
    // so input/output specs must agree, and the static geometry (dtype,
    // kv_heads, head_dim) must match the KV cache backing it.
    const KVCacheView& view = context.kv_cache_view();
    for (const ResolvedStateAlias& alias: aliases) {
        const TensorSpec& input_spec = values[step.inputs[alias.input_port].index].spec;
        const TensorSpec& output_spec = values[step.outputs[alias.output_port].index].spec;
        if (input_spec != output_spec) {
            return Status::InvalidArgument(
                    "State alias input and output specs must match for must-alias update");
        }

        AM_ASSIGN_OR_RETURN(const ExecutionKVCacheStateIdentity input_identity,
                            GetKVStateIdentity(values, step.inputs[alias.input_port],
                                               "alias input"));
        AM_ASSIGN_OR_RETURN(const ExecutionKVCacheStateIdentity output_identity,
                            GetKVStateIdentity(values, step.outputs[alias.output_port],
                                               "alias output"));
        if (input_identity != output_identity) {
            return Status::InvalidArgument(
                    "State alias input and output must retain the same state identity");
        }

        if (input_spec.dtype != view.kv_dtype()) {
            return Status::InvalidArgument(
                    "State alias dtype does not match the KVCacheView element dtype");
        }

        const SymbolicShape& shape = input_spec.shape;
        const auto rank = shape.rank();
        if (!rank.has_value() || *rank != 3) {
            return Status::InvalidArgument(
                    "State alias value must be rank 3 [kv_heads, cache_len, head_dim]");
        }

        // Keep the alias check aligned with the state spec contract: only the
        // physical cache capacity axis may remain symbolic.
        const ShapeSymbol& kv_heads = shape[0];
        const ShapeSymbol& cache_len = shape[1];
        const ShapeSymbol& head_dim = shape[2];
        if (!kv_heads.IsStatic() || !head_dim.IsStatic()) {
            return Status::InvalidArgument(
                    "State alias requires static KV head and head dimension");
        }

        if (static_cast<size_t>(kv_heads.GetStaticValue()) != view.num_kv_heads()) {
            return Status::InvalidArgument(
                    "State alias kv_heads does not match the KVCacheView head count");
        }

        if (static_cast<size_t>(head_dim.GetStaticValue()) != view.head_dim()) {
            return Status::InvalidArgument(
                    "State alias head_dim does not match the KVCacheView head dimension");
        }

        if (cache_len.IsStatic() &&
            static_cast<size_t>(cache_len.GetStaticValue()) != view.max_tokens()) {
            return Status::InvalidArgument(
                    "State alias cache_len does not match the KVCacheView capacity");
        }
    }

    // TODO: when non-KV-cache state aliases land (decode/streaming state) or
    // activation-port aliases are introduced, extend validation here with
    // StepTensorBinding pointer-comparison checks against aliases[i]
    // input_port/output_port. The `step` parameter is reserved for that path.

    return Status::Ok();
}

} // namespace aethermind
