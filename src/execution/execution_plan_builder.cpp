#include "aethermind/execution/execution_plan_builder.h"

#include "aethermind/backend/packed_weights.h"
#include "aethermind/graph/compilation/graph_lowering.h"
#include "aethermind/model/model_instance.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/operators/operator_schema.h"

namespace aethermind {
namespace {

KernelSelector MakeSelectorForNode(const ExecutionPlanNodeSpec& node) noexcept {
    return KernelSelector{
            .device_type = node.device_type,
            .act_dtype = node.act_dtype,
            .weight_dtype = node.weight_dtype,
            .weight_format = node.weight_format,
            .isa = node.isa,
            .phase = node.phase,
    };
}

StatusOr<const void*> ResolvePackedWeightsForNode(const ModelInstance* model_instance,
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

struct PreparedNodeMetadata {
    std::vector<TensorSpec> compact_input_specs{};
    std::vector<TensorSpec> output_specs{};
    std::vector<ShapeConstraint> runtime_checks{};
};

// Validates caller-provided semantic metadata against the sole semantic
// authority, InferOperator. Empty fields are not inferred on the caller's
// behalf: they must match the explicit inference result exactly.
Status ValidateCallerMetadata(const ExecutionPlanNodeSpec& node,
                              std::span<const TensorSpec> compact_input_specs,
                              std::vector<TensorSpec>& outputs_out,
                              std::vector<ShapeConstraint>& checks_out) {
    if (std::holds_alternative<std::monostate>(node.op_params)) {
        return Status::InvalidArgument(
                "Untrusted ExecutionPlanNodeSpec adapter requires typed op_params; "
                "monostate is not accepted");
    }
    auto analyzed = InferOperator(node.op_type, node.op_params, compact_input_specs);
    if (!analyzed.ok()) {
        return analyzed.status();
    }
    if (analyzed->outputs != node.output_specs) {
        return Status::InvalidArgument(
                "ExecutionPlanNodeSpec.output_specs does not match InferOperator");
    }
    if (analyzed->runtime_checks != node.runtime_checks) {
        return Status::InvalidArgument(
                "ExecutionPlanNodeSpec.runtime_checks does not match InferOperator");
    }
    outputs_out = std::move(analyzed->outputs);
    checks_out = std::move(analyzed->runtime_checks);
    return Status::Ok();
}

StatusOr<PreparedNodeMetadata> PrepareNodeMetadata(const ExecutionPlanNodeSpec& node,
                                                   bool trusted) {
    if (node.op_type == OpType::kUnknown) {
        return Status::InvalidArgument("ExecutionPlanNodeSpec.op_type cannot be kUnknown");
    }
    if (std::holds_alternative<std::monostate>(node.op_params)) {
        return Status::InvalidArgument("ExecutionPlanNodeSpec requires typed op_params");
    }

    const auto schema = GetOperatorSchema(node.op_type);
    if (!schema.ok()) {
        return schema.status();
    }
    auto compact_input_specs = MakeCompactInputSpecs(*schema, node.input_specs);
    if (!compact_input_specs.ok()) {
        return compact_input_specs.status();
    }

    PreparedNodeMetadata metadata{
            .compact_input_specs = std::move(*compact_input_specs),
    };
    if (trusted) {
        // LoweredGraph carries the exact result already produced by graph
        // construction. Re-running inference here would create a second
        // semantic authority and is deliberately prohibited.
        metadata.output_specs = node.output_specs;
        metadata.runtime_checks = node.runtime_checks;
        return metadata;
    }

    AM_RETURN_IF_ERROR(ValidateCallerMetadata(node,
                                              metadata.compact_input_specs,
                                              metadata.output_specs,
                                              metadata.runtime_checks));
    return metadata;
}

StatusOr<ExecutionPlan> BuildExecutionPlan(RuntimeContext& runtime,
                                           const ModelInstance* model_instance,
                                           const std::vector<ExecutionPlanNodeSpec>& nodes,
                                           StateAliasPlan state_alias_plan,
                                           bool trusted) {
    std::vector<WorkspaceRequirement> workspace_requirements;
    workspace_requirements.reserve(nodes.size());
    for (const ExecutionPlanNodeSpec& node: nodes) {
        workspace_requirements.push_back(node.workspace_requirement);
    }
    if (const auto layout = PlanWorkspaceRequirements(std::span(workspace_requirements));
        !layout.ok()) {
        return layout.status();
    }

    std::vector<ExecutionStep> steps;
    steps.reserve(nodes.size());
    for (size_t index = 0; index < nodes.size(); ++index) {
        const ExecutionPlanNodeSpec& node = nodes[index];
        auto backend = runtime.GetBackend(node.device_type);
        if (!backend.ok()) {
            return backend.status();
        }

        auto metadata = PrepareNodeMetadata(node, trusted);
        if (!metadata.ok()) {
            return metadata.status();
        }
        auto kernel = ExecutionPlanBuilder::PrepareKernelForNode(*backend.value(), node);
        if (!kernel.ok()) {
            return kernel.status();
        }
        const auto packed_weights = ResolvePackedWeightsForNode(model_instance, node);
        if (!packed_weights.ok()) {
            return packed_weights.status();
        }

        steps.push_back({
                .selector = MakeSelectorForNode(node),
                .kernel = std::move(*kernel),
                .packed_weights = *packed_weights,
                .workspace_requirement = workspace_requirements[index],
                .input_specs = std::move(metadata->compact_input_specs),
                .output_specs = std::move(metadata->output_specs),
                .runtime_checks = std::move(metadata->runtime_checks),
        });
    }
    return ExecutionPlan::Create(std::move(steps), std::move(state_alias_plan));
}

}// namespace

StatusOr<ResolvedKernel> ExecutionPlanBuilder::PrepareKernelForNode(
        const Backend& backend,
        const ExecutionPlanNodeSpec& node) {
    if (node.op_type == OpType::kUnknown) {
        return Status::InvalidArgument("ExecutionPlanNodeSpec.op_type cannot be kUnknown");
    }
    if (std::holds_alternative<std::monostate>(node.op_params)) {
        return Status::InvalidArgument("ExecutionPlanNodeSpec requires typed op_params");
    }
    return backend.PrepareKernel(node.op_type, MakeSelectorForNode(node), node.op_params);
}

StatusOr<ExecutionPlan> ExecutionPlanBuilder::Build(
        RuntimeContext& runtime,
        const std::vector<ExecutionPlanNodeSpec>& nodes) {
    return BuildExecutionPlan(runtime, nullptr, nodes, StateAliasPlan{}, /*trusted=*/false);
}

StatusOr<ExecutionPlan> ExecutionPlanBuilder::Build(
        RuntimeContext& runtime,
        const ModelInstance& model_instance,
        const std::vector<ExecutionPlanNodeSpec>& nodes) {
    return BuildExecutionPlan(runtime, &model_instance, nodes, StateAliasPlan{}, /*trusted=*/false);
}

StatusOr<ExecutionPlan> ExecutionPlanBuilder::Build(
        RuntimeContext& runtime,
        const LoweredGraph& lowered) {
    auto alias_plan = ResolveStateAliases(lowered);
    if (!alias_plan.ok()) {
        return alias_plan.status();
    }
    return BuildExecutionPlan(runtime, nullptr, lowered.steps,
                              std::move(*alias_plan), /*trusted=*/true);
}

StatusOr<ExecutionPlan> ExecutionPlanBuilder::Build(
        RuntimeContext& runtime,
        const ModelInstance& model_instance,
        const LoweredGraph& lowered) {
    auto alias_plan = ResolveStateAliases(lowered);
    if (!alias_plan.ok()) {
        return alias_plan.status();
    }
    return BuildExecutionPlan(runtime, &model_instance, lowered.steps,
                              std::move(*alias_plan), /*trusted=*/true);
}

}// namespace aethermind
