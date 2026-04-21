#include "aethermind/execution/execution_plan_builder.h"

#include "aethermind/backend/packed_weights.h"
#include "aethermind/model/model_instance.h"

namespace aethermind {
namespace {

KernelSelector MakeSelectorForNode(const ExecutionPlanNodeSpec& node) noexcept {
    return KernelSelector{
            .device_type = node.device_type,
            .activation_dtype = node.activation_dtype,
            .weight_dtype = node.weight_dtype,
            .weight_format = node.weight_format,
            .isa = node.isa,
            .phase = node.phase,
    };
}

StatusOr<const void*> ResolvePackedParamsForNode(const ModelInstance* model_instance,
                                                 const ExecutionPlanNodeSpec& node) noexcept {
    if (node.weight_format != WeightFormat::kPacked) {
        return nullptr;
    }
    if (model_instance == nullptr) {
        return Status::NotFound("Packed-weight node requires a ModelInstance sidecar");
    }

    const auto selector = MakeSelectorForNode(node);
    const auto* packed_weights = model_instance->FindPackedWeights(node.op_type, selector);
    if (packed_weights == nullptr) {
        return Status::NotFound("Packed weights not found for ExecutionPlan node");
    }
    return packed_weights->storage().data();
}

StatusOr<ExecutionPlan> BuildExecutionPlan(RuntimeContext& runtime,
                                           const ModelInstance* model_instance,
                                           const std::vector<ExecutionPlanNodeSpec>& nodes) {
    std::vector<WorkspaceRequirement> workspace_requirements;
    workspace_requirements.reserve(nodes.size());
    for (const ExecutionPlanNodeSpec& node: nodes) {
        workspace_requirements.push_back(node.workspace_requirement);
    }

    if (const auto layout = PlanWorkspaceRequirements(std::span(workspace_requirements));
        !layout.ok()) {
        return layout.status();
    }

    ExecutionPlan plan;
    for (size_t index = 0; index < nodes.size(); ++index) {
        const auto& node = nodes[index];

        auto backend = runtime.GetBackend(node.device_type);
        if (!backend.ok()) {
            return backend.status();
        }

        const auto resolved =
                ExecutionPlanBuilder::ResolveKernelForNode(*backend.value(), node);
        if (!resolved.ok()) {
            return resolved.status();
        }

        const auto packed_params = ResolvePackedParamsForNode(model_instance, node);
        if (!packed_params.ok()) {
            return packed_params.status();
        }

        if (const Status status = plan.AddStep(ExecutionStep{
                    .op_type = resolved->op_type,
                    .invocation = {
                            .op_type = node.op_type,
                            .selector = MakeSelectorForNode(node),
                    },
                    .fn = resolved->fn,
                    .packed_params = packed_params.value(),
                    .workspace_requirement = workspace_requirements[index],
                    .attrs = resolved->attrs,
                    .debug_name = resolved->debug_name,
            });
            !status.ok()) {
            return status;
        }
    }

    return plan;
}

}// namespace

StatusOr<ResolvedKernel> ExecutionPlanBuilder::ResolveKernelForNode(
        const Backend& backend,
        const ExecutionPlanNodeSpec& node) noexcept {
    if (node.op_type == OpType::kUnknown) {
        return Status::InvalidArgument("ExecutionPlanNodeSpec.op_type cannot be kUnknown");
    }

    const auto selector = MakeSelectorForNode(node);
    const auto resolved = backend.ResolveKernelInfo(node.op_type, selector);
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
    return BuildExecutionPlan(runtime, nullptr, nodes);
}

StatusOr<ExecutionPlan> ExecutionPlanBuilder::Build(
        RuntimeContext& runtime,
        const ModelInstance& model_instance,
        const std::vector<ExecutionPlanNodeSpec>& nodes) {
    return BuildExecutionPlan(runtime, &model_instance, nodes);
}

}// namespace aethermind
