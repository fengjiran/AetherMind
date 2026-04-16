#include "aethermind/execution/execution_plan_builder.h"

namespace aethermind {

StatusOr<ResolvedKernel> ExecutionPlanBuilder::ResolveKernelForNode(
        const Backend& backend,
        const ExecutionPlanNodeSpec& node) noexcept {
    if (node.op_type == OpType::kUnknown) {
        return Status::InvalidArgument("ExecutionPlanNodeSpec.op_type cannot be kUnknown");
    }

    const KernelSelector selector{
            .device_type = node.device_type,
            .activation_dtype = node.activation_dtype,
            .weight_dtype = node.weight_dtype,
            .weight_format = node.weight_format,
            .isa = node.isa,
            .phase = node.phase,
    };

    const StatusOr<ResolvedKernel> resolved = backend.ResolveKernelInfo(node.op_type, selector);
    if (!resolved.ok()) {
        return resolved.status();
    }

    ResolvedKernel frozen = resolved.value();
    frozen.attrs = node.attrs;
    return frozen;
}

StatusOr<ExecutionPlan> ExecutionPlanBuilder::Build(
        RuntimeContext& runtime,
        const std::vector<ExecutionPlanNodeSpec>& nodes) {
    ExecutionPlan plan;
    for (const ExecutionPlanNodeSpec& node: nodes) {
        StatusOr<Backend*> backend = runtime.GetBackend(node.device_type);
        if (!backend.ok()) {
            return backend.status();
        }

        const StatusOr<ResolvedKernel> resolved = ResolveKernelForNode(*backend.value(), node);
        if (!resolved.ok()) {
            return resolved.status();
        }

        if (const Status status = plan.AddStep(resolved.value()); !status.ok()) {
            return status;
        }
    }

    return plan;
}

}// namespace aethermind
