#include "execution/layer_runner.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/kernel_invoker.h"
#include "aethermind/shape_inference/shape_constraint_evaluator.h"

namespace aethermind {
namespace {

KernelContext BuildKernelContext(const ExecutionStep& step,
                                 const RuntimeBindingContext& bindings) noexcept {
    return KernelContext{
            .device_type = step.selector.device_type,
            .stream = nullptr,
            .workspace = bindings.GetWorkspaceArena(),
            .packed_weights = step.packed_weights,
            .kernel_params = nullptr,
            .attrs = step.kernel.attrs,
    };
}

}// namespace

Status LayerRunner::Run(const ExecutionPlan& plan,
                        const RuntimeBindingContext& bindings) noexcept {
    const auto& steps = plan.steps();
    const auto& alias_plan = plan.state_alias_plan();
    for (size_t i = 0; i < steps.size(); ++i) {
        if (const auto status = RunStep(i, steps[i], bindings, alias_plan);
            !status.ok()) {
            return status;
        }
    }
    return Status::Ok();
}

Status LayerRunner::RunStep(size_t step_index,
                            const ExecutionStep& step,
                            const RuntimeBindingContext& bindings,
                            const StateAliasPlan& alias_plan) noexcept {
    AM_RETURN_IF_ERROR(ValidateStateAliasesForStep(
            step_index, step, alias_plan, bindings));

    const auto workspace_binding = bindings.BindWorkspace(step.workspace_requirement);
    if (!workspace_binding.ok()) {
        return workspace_binding.status();
    }

    KernelContext ctx = BuildKernelContext(step, bindings);
    ctx.workspace_binding = workspace_binding.value();

    const auto tensor_binding = bindings.GetStepTensorBinding(step_index);
    if (!tensor_binding.ok()) {
        return tensor_binding.status();
    }
    if ((*tensor_binding)->inputs.size() != step.input_specs.size() ||
        (*tensor_binding)->outputs.size() != step.output_specs.size()) {
        return Status::InvalidArgument("Runtime tensor binding arity does not match ExecutionStep specs");
    }
    if (!step.runtime_checks.empty()) {
        AM_RETURN_IF_ERROR(ValidateShapeConstraints(step.runtime_checks,
                                                    (*tensor_binding)->inputs,
                                                    (*tensor_binding)->outputs));
    }

    return InvokeKernel(step.kernel, ctx,
                        (*tensor_binding)->inputs,
                        (*tensor_binding)->outputs);
}

Status LayerRunner::ValidateStateAliasesForStep(
        size_t step_index,
        const ExecutionStep& /*step*/,
        const StateAliasPlan& alias_plan,
        const RuntimeBindingContext& bindings) noexcept {
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

    // TODO: when non-KV-cache state aliases land (decode/streaming state) or
    // activation-port aliases are introduced, extend validation here with
    // StepTensorBinding pointer-comparison checks against aliases[i].
    // input_port/output_port. The `step` parameter is reserved for that path.
    // Note: cross-checking the view geometry against the step's operator
    // parameters (heads/head_dim) is not enforceable yet because
    // ExecutionStep does not carry op_params; revisit once steps expose the
    // typed parameters.

    return Status::Ok();
}

}// namespace aethermind
