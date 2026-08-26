#include "execution/layer_runner.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/kernel_invoker.h"
#include "aethermind/runtime/kv_cache_view.h"

namespace aethermind {
namespace {

KernelContext BuildKernelContext(const ExecutionStep& step,
                                 const RuntimeBindingContext& bindings) noexcept {
    return KernelContext{
            .device_type = step.selector.device_type,
            .stream = nullptr,
            .workspace = bindings.GetWorkspaceArena(),
            .packed_weights = step.packed_weights
                                      ? step.packed_weights->storage().data()
                                      : nullptr,
            .kernel_params = nullptr,
            .attrs = step.kernel.attrs,
    };
}

}// namespace

Status LayerRunner::Run(const ExecutionPlan& plan,
                        const RuntimeBindingContext& bindings) noexcept {
    const auto& steps = plan.steps();
    const auto& alias_plan = plan.state_alias_plan();
    const BindingTable* const binding_table = bindings.binding_table();
    if (binding_table == nullptr) {
        return Status::FailedPrecondition(
                "RuntimeBindingContext requires a BindingTable before execution");
    }

    if (!binding_table->IsCompatible(plan)) {
        return Status::InvalidArgument(
                "RuntimeBindingContext BindingTable is not compatible with ExecutionPlan");
    }

    for (size_t i = 0; i < steps.size(); ++i) {
        if (const auto status = RunStep(i, steps[i], bindings, *binding_table,
                                        alias_plan, plan.values());
            !status.ok()) {
            return status;
        }
    }
    return Status::Ok();
}

Status LayerRunner::RunStep(size_t step_index,
                            const ExecutionStep& step,
                            const RuntimeBindingContext& bindings,
                            const BindingTable& binding_table,
                            const StateAliasPlan& alias_plan,
                            const std::vector<ExecutionValueDesc>& values) noexcept {
    AM_RETURN_IF_ERROR(ValidateStateAliasesForStep(
            step_index, step, alias_plan, bindings, values));

    const auto workspace_binding = bindings.BindWorkspace(step.workspace_requirement);
    if (!workspace_binding.ok()) {
        return workspace_binding.status();
    }

    KernelContext ctx = BuildKernelContext(step, bindings);
    ctx.workspace_binding = workspace_binding.value();

    const StepTensorBinding& tensor_binding = binding_table.step(step_index);
    if (tensor_binding.inputs.size() != step.kernel_input_ports.size() ||
        tensor_binding.outputs.size() != step.kernel_output_ports.size()) {
        return Status::InvalidArgument("Runtime tensor binding arity does not match ExecutionStep ports");
    }
    return InvokeKernel(step.kernel, ctx, tensor_binding.inputs, tensor_binding.outputs);
}

Status LayerRunner::ValidateStateAliasesForStep(
        size_t step_index,
        const ExecutionStep& step,
        const StateAliasPlan& alias_plan,
        const RuntimeBindingContext& bindings,
        const std::vector<ExecutionValueDesc>& values) noexcept {
    const auto aliases = alias_plan.ForStep(step_index);
    if (aliases.empty()) {
        return Status::Ok();
    }

    // Phase 1: all state aliases are KV cache updates. The KVCacheView is the
    // shared physical storage that the operator reads and writes in place, so
    // its presence is the runtime invariant for must-alias state updates.
    if (!bindings.HasKVCacheView()) {
        return Status::InvalidArgument(
                "State alias requires a valid KVCacheView");
    }

    // Cross-check each alias against the bound view: the update is must-alias,
    // so input/output specs must agree, and the static geometry (dtype,
    // kv_heads, head_dim) must match the KV cache backing it.
    const KVCacheView& view = bindings.kv_cache_view();
    for (const ResolvedStateAlias& alias: aliases) {
        const TensorSpec& input_spec = values[step.inputs[alias.input_port].index].spec;
        const TensorSpec& output_spec = values[step.outputs[alias.output_port].index].spec;
        if (input_spec != output_spec) {
            return Status::InvalidArgument(
                    "State alias input and output specs must match for must-alias update");
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
        const ShapeSymbol& kv_heads = shape[0];
        const ShapeSymbol& head_dim = shape[2];
        if (kv_heads.IsStatic() &&
            static_cast<size_t>(kv_heads.GetStaticValue()) != view.num_kv_heads()) {
            return Status::InvalidArgument(
                    "State alias kv_heads does not match the KVCacheView head count");
        }
        if (head_dim.IsStatic() &&
            static_cast<size_t>(head_dim.GetStaticValue()) != view.head_dim()) {
            return Status::InvalidArgument(
                    "State alias head_dim does not match the KVCacheView head dimension");
        }
    }

    // TODO: when non-KV-cache state aliases land (decode/streaming state) or
    // activation-port aliases are introduced, extend validation here with
    // StepTensorBinding pointer-comparison checks against aliases[i]
    // input_port/output_port. The `step` parameter is reserved for that path.

    return Status::Ok();
}

}// namespace aethermind
