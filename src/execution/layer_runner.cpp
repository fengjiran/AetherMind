#include "aethermind/execution/layer_runner.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/base/shape_constraint_evaluator.h"
#include "aethermind/execution/runtime_binding_context.h"

namespace aethermind {
namespace {

KernelContext BuildKernelContext(const ExecutionStep& step,
                                 RuntimeBindingContext& bindings) noexcept {
    const ResolvedKernel& resolved = step.op->GetResolvedKernel();
    return KernelContext{
            .device_type = step.selector.device_type,
            .stream = nullptr,
            .workspace = bindings.GetWorkspaceArena(),
            .packed_weights = step.packed_weights,
            .kernel_params = nullptr,
            .attrs = resolved.attrs,
    };
}

}// namespace

Status LayerRunner::Run(const ExecutionPlan& plan,
                        RuntimeBindingContext& bindings) noexcept {
    const auto& steps = plan.steps();
    for (size_t i = 0; i < steps.size(); ++i) {
        if (const auto status = RunStep(i, steps[i], bindings); !status.ok()) {
            return status;
        }
    }
    return Status::Ok();
}

Status LayerRunner::RunStep(size_t step_index,
                            const ExecutionStep& step,
                            RuntimeBindingContext& bindings) noexcept {
    if (step.op == nullptr) {
        return Status::InvalidArgument("Execution step operator cannot be null");
    }

    const auto workspace_binding = bindings.BindWorkspace(step.workspace_requirement);
    if (!workspace_binding.ok()) {
        return workspace_binding.status();
    }

    KernelContext ctx = BuildKernelContext(step, bindings);
    ctx.workspace_binding = workspace_binding.value();

    if (!step.runtime_checks.empty()) {
        const auto tensor_binding = bindings.GetStepTensorBinding(step_index);
        if (!tensor_binding.ok()) {
            return tensor_binding.status();
        }
        AM_RETURN_IF_ERROR(ValidateShapeConstraints(step.runtime_checks,
                                                    (*tensor_binding)->inputs,
                                                    (*tensor_binding)->outputs));
    }

    return step.op->Run(ctx, bindings, step_index);
}

}// namespace aethermind
