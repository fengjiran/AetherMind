#include "aethermind/execution/layer_runner.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_invocation.h"
#include "aethermind/execution/runtime_binding_context.h"

namespace aethermind {
namespace {

KernelContext BuildKernelContext(const ExecutionStep& step,
                                   RuntimeBindingContext& bindings) noexcept {
    const ResolvedKernel resolved = step.op->GetResolvedKernel();
    const auto device_type = step.selector.device_type != DeviceType::kUndefined
                                     ? step.selector.device_type
                                     : DeviceType::kCPU;

    return KernelContext{
            .device = Device(device_type),
            .stream = nullptr,
            .workspace = bindings.GetWorkspaceArena(),
            .tracing = nullptr,
            .caps = nullptr,
            .packed_params = step.packed_params,
            .attrs = resolved.attrs,
            .debug_name = step.debug_name != nullptr ? step.debug_name : resolved.debug_name,
            .backend_resources = {},
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

    const KernelContext op_ctx = BuildKernelContext(step, bindings);
    const KernelInvocation invocation{
            .op_type = step.op->Type(),
            .selector = step.selector,
    };
    return step.op->Run(invocation, op_ctx, workspace_binding.value());
}

}// namespace aethermind
