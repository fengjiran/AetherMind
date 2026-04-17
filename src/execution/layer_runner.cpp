#include "aethermind/execution/layer_runner.h"

#include "aethermind/backend/op_kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"

namespace aethermind {
namespace {

OpKernelContext BuildKernelContext(const ExecutionStep& step,
                                   RuntimeBindingContext& bindings) noexcept {
    const DeviceType device_type = step.invocation.selector.device_type != DeviceType::kUndefined
                                           ? step.invocation.selector.device_type
                                           : DeviceType::kCPU;

    return OpKernelContext{
            .device = Device(device_type),
            .stream = nullptr,
            .workspace = bindings.GetWorkspaceArena(),
            .tracing = nullptr,
            .caps = nullptr,
            .packed_params = step.packed_params,
            .attrs = step.attrs,
            .debug_name = step.debug_name,
            .backend_resources = {},
    };
}

}// namespace

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

    const OpKernelContext op_ctx = BuildKernelContext(step, bindings);
    return step.fn(step.invocation, op_ctx, workspace_binding.value());
}

}// namespace aethermind
