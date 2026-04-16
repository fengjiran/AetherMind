#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/backend/dispatcher_bridge.h"

namespace aethermind {

StatusOr<ResolvedKernel> ExecutionPlanBuilder::ResolveKernelForNode(
        const Backend& backend,
        const ExecutionPlanNodeSpec& node) noexcept {
    const StatusOr<OpType> op_type = ToOpType(node.op_name);
    if (!op_type.ok()) {
        return op_type.status();
    }

    const KernelSelector selector = MakeKernelSelector(node.device_type,
                                                       node.activation_dtype,
                                                       node.weight_dtype,
                                                       node.weight_format,
                                                       node.isa,
                                                       node.phase);

    const StatusOr<ResolvedKernel> resolved = backend.ResolveKernelInfo(*op_type, selector);
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

        if (const Status status = plan.AddStep(node.op_name, resolved.value()); !status.ok()) {
            return status;
        }
    }

    return plan;
}

}// namespace aethermind
