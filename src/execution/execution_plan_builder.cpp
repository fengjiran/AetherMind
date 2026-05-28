#include "aethermind/execution/execution_plan_builder.h"

#include "aethermind/backend/packed_weights.h"
#include "aethermind/model/model_instance.h"
#include "aethermind/operators/function_operator.h"
#include "aethermind/operators/operator_registry.h"
#include "aethermind/operators/rms_norm_op.h"

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

std::any MakeOperatorParamsForNode(const ExecutionPlanNodeSpec& node) {
    if (node.op_params.has_value()) {
        return node.op_params;
    }
    if (node.op_type == OpType::kRmsNorm) {
        return RmsNormOp::Params{};
    }
    return {};
}

StatusOr<OperatorPtr> CreateAndPrepareOperator(Backend& backend,
                                               const ExecutionPlanNodeSpec& node) {
    StatusOr<std::unique_ptr<Operator>> created = OperatorRegistry::Create(
            node.op_type,
            MakeOperatorParamsForNode(node));
    if (!created.ok()) {
        if (created.status().code() == StatusCode::kNotFound) {
            return OperatorPtr{};
        }
        return created.status();
    }

    if (!node.attrs.empty()) {
        return Status::InvalidArgument(
                "ExecutionPlanNodeSpec.attrs is only supported by raw kernel fallback; use op_params for registered operators");
    }

    std::unique_ptr<Operator> op = std::move(created).value();
    AM_RETURN_IF_ERROR(op->Validate());

    OperatorContext op_ctx{
            .backend = &backend,
            .kernel_registry = backend.TryGetKernelRegistryForDebug(),
            .workspace = nullptr,
            .selector = MakeSelectorForNode(node),
    };
    AM_RETURN_IF_ERROR(op->Prepare(op_ctx));

    return OperatorPtr(std::move(op));
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

        const auto prepared_operator = CreateAndPrepareOperator(*backend.value(), node);
        if (!prepared_operator.ok()) {
            return prepared_operator.status();
        }

        OperatorPtr op = prepared_operator.value();
        if (op == nullptr) {
            const auto resolved = ExecutionPlanBuilder::ResolveKernelForNode(*backend.value(), node);
            if (!resolved.ok()) {
                return resolved.status();
            }
            op = std::make_shared<FunctionOperator>(
                    resolved->op_type,
                    resolved->fn,
                    resolved->attrs,
                    resolved->debug_name);
        }

        const auto packed_params = ResolvePackedParamsForNode(model_instance, node);
        if (!packed_params.ok()) {
            return packed_params.status();
        }

        if (auto status = plan.AddStep(ExecutionStep{
                    .op_type = node.op_type,
                    .invocation = {
                            .op_type = node.op_type,
                            .selector = MakeSelectorForNode(node),
                    },
                    .op = std::move(op),
                    .packed_params = packed_params.value(),
                    .workspace_requirement = workspace_requirements[index],
                    .debug_name = nullptr,
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
