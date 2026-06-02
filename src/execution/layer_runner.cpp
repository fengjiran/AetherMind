#include "aethermind/execution/layer_runner.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"

namespace aethermind {
namespace {

KernelContext BuildKernelContext(const ExecutionStep& step,
                                   RuntimeBindingContext& bindings) noexcept {
    const ResolvedKernel resolved = step.op->GetResolvedKernel();
    return KernelContext{
            .stream = nullptr,
            .workspace = bindings.GetWorkspaceArena(),
            .packed_params = step.packed_params,
            .attrs = resolved.attrs,
    };
}

}// namespace

Status LayerRunner::Run(const ExecutionPlan& plan,
                        RuntimeBindingContext& bindings) noexcept {
    for (const auto& step: plan.steps()) {
        if (const auto status = RunStep(step, bindings); !status.ok()) {
            return status;
        }
    }
    return Status::Ok();
}

Status LayerRunner::RunStep(const ExecutionStep& step,
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
    return step.op->Run(ctx);
}

}// namespace aethermind
