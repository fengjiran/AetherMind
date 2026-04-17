#include "aethermind/execution/layer_runner.h"

#include "aethermind/execution/runtime_binding_context.h"

namespace aethermind {

Status LayerRunner::Run(const ExecutionPlan& plan,
                        RuntimeBindingContext& bindings) noexcept {
    for (const ExecutionStep& step: plan.steps()) {
        if (const Status status = RunStep(step, bindings); !status.ok()) {
            return status;
        }
    }
    return Status::Ok();
}

Status LayerRunner::RunStep(const ExecutionStep& step,
                            RuntimeBindingContext& bindings) noexcept {
    if (step.fn == nullptr) {
        return Status::InvalidArgument("Execution step kernel function cannot be null");
    }

    const StatusOr<WorkspaceBinding> workspace_binding =
            bindings.BindWorkspace(step.workspace_requirement);
    if (!workspace_binding.ok()) {
        return workspace_binding.status();
    }

    return step.fn();
}

}// namespace aethermind
