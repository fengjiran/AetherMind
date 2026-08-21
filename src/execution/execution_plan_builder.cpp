#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/backend/packed_weights.h"
#include "aethermind/compiler/lowered_graph.h"
#include "aethermind/model/packed_weight_store.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/operators/operator_schema.h"
#include "execution/lowered_graph_adapter.h"

#include <concepts>
#include <ranges>

namespace aethermind {
namespace {

template<typename NodeSpec>
concept ExecutionNodeMetadata =
        std::same_as<std::remove_cvref_t<NodeSpec>, ExecutionPlanNodeSpec> ||
        std::same_as<std::remove_cvref_t<NodeSpec>, LoweredStepSpec>;

// Workspace size is a kernel implementation detail (e.g. tile buffer sizes)
// that the compiler artifact cannot compute, so LoweredStepSpec carries no
// requirement. Untrusted callers supply theirs explicitly; lowered steps get
// the default until kernel-prepare fills requirements on ResolvedKernel.
template<ExecutionNodeMetadata NodeSpec>
WorkspaceRequirement GetNodeWorkspaceRequirement(const NodeSpec& node) {
    if constexpr (std::same_as<std::remove_cvref_t<NodeSpec>, ExecutionPlanNodeSpec>) {
        return node.workspace_requirement;
    } else {
        return {};
    }
}

template<ExecutionNodeMetadata NodeSpec>
StatusOr<const void*> ResolvePackedWeightsForNode(const PackedWeightStore* packed_weight_store,
                                                  const NodeSpec& node) noexcept {
    if (node.selector.weight_format != WeightFormat::kPacked) {
        return nullptr;
    }
    if (packed_weight_store == nullptr) {
        return Status::NotFound("Packed-weight node requires a PackedWeightStore");
    }

    const auto* packed_weights = packed_weight_store->Find(node.op_type, node.selector);
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

template<ExecutionNodeMetadata NodeSpec>
StatusOr<PreparedNodeMetadata> PrepareCompactInputMetadata(const NodeSpec& node) {
    const auto schema = GetOperatorSchema(node.op_type);
    if (!schema.ok()) {
        return schema.status();
    }
    auto compact_input_specs = MakeCompactInputSpecs(*schema, node.input_specs);
    if (!compact_input_specs.ok()) {
        return compact_input_specs.status();
    }

    return PreparedNodeMetadata{
            .compact_input_specs = std::move(*compact_input_specs),
    };
}

StatusOr<PreparedNodeMetadata> PrepareNodeMetadata(const ExecutionPlanNodeSpec& node) {
    if (node.op_type == OpType::kUnknown) {
        return Status::InvalidArgument("ExecutionPlanNodeSpec.op_type cannot be kUnknown");
    }
    if (std::holds_alternative<std::monostate>(node.op_params)) {
        return Status::InvalidArgument("ExecutionPlanNodeSpec requires typed op_params");
    }

    auto metadata = PrepareCompactInputMetadata(node);
    if (!metadata.ok()) {
        return metadata.status();
    }

    AM_RETURN_IF_ERROR(ValidateCallerMetadata(node,
                                              metadata->compact_input_specs,
                                              metadata->output_specs,
                                              metadata->runtime_checks));
    return metadata;
}

StatusOr<PreparedNodeMetadata> PrepareNodeMetadata(const LoweredStepSpec& node) {
    if (node.op_type == OpType::kUnknown ||
        std::holds_alternative<std::monostate>(node.op_params)) {
        return Status::Internal("Finalized LoweredStepSpec is missing semantic metadata");
    }

    auto metadata = PrepareCompactInputMetadata(node);
    if (!metadata.ok()) {
        return Status::Internal("Finalized LoweredStepSpec has invalid compact input metadata: " +
                                metadata.status().message());
    }
    // LoweredGraph is validated by compiler and checked again at the
    // execution trust boundary. Re-running InferOperator here would create a
    // second semantic authority, so only the ExecutionPlan's owning copies
    // are made below.
    metadata->output_specs = node.output_specs;
    metadata->runtime_checks = node.runtime_checks;
    return metadata;
}

template<typename NodeRange>
    requires std::ranges::random_access_range<NodeRange> &&
             ExecutionNodeMetadata<std::ranges::range_value_t<NodeRange>>
StatusOr<ExecutionPlan> BuildExecutionPlan(RuntimeContext& runtime,
                                           const PackedWeightStore* packed_weight_store,
                                           NodeRange&& nodes,
                                           StateAliasPlan state_alias_plan) {
    std::vector<WorkspaceRequirement> workspace_requirements;
    workspace_requirements.reserve(std::ranges::size(nodes));
    for (const auto& node: nodes) {
        workspace_requirements.push_back(GetNodeWorkspaceRequirement(node));
    }
    if (const auto layout = PlanWorkspaceRequirements(std::span(workspace_requirements));
        !layout.ok()) {
        return layout.status();
    }

    std::vector<ExecutionStep> steps;
    steps.reserve(std::ranges::size(nodes));
    for (size_t index = 0; index < std::ranges::size(nodes); ++index) {
        const auto& node = nodes[index];
        auto backend = runtime.GetBackend(node.selector.device_type);
        if (!backend.ok()) {
            return backend.status();
        }

        auto metadata = PrepareNodeMetadata(node);
        if (!metadata.ok()) {
            return metadata.status();
        }
        auto kernel = backend.value()->PrepareKernel(node.op_type, node.selector, node.op_params);
        if (!kernel.ok()) {
            return kernel.status();
        }
        const auto packed_weights = ResolvePackedWeightsForNode(packed_weight_store, node);
        if (!packed_weights.ok()) {
            return packed_weights.status();
        }

        steps.push_back({
                .selector = node.selector,
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
    return backend.PrepareKernel(node.op_type, node.selector, node.op_params);
}

StatusOr<ExecutionPlan> ExecutionPlanBuilder::Build(
        RuntimeContext& runtime,
        const std::vector<ExecutionPlanNodeSpec>& nodes) {
    return BuildExecutionPlan(runtime, nullptr, nodes, StateAliasPlan{});
}

StatusOr<ExecutionPlan> ExecutionPlanBuilder::Build(
        RuntimeContext& runtime,
        const PackedWeightStore& packed_weight_store,
        const std::vector<ExecutionPlanNodeSpec>& nodes) {
    return BuildExecutionPlan(runtime, &packed_weight_store, nodes, StateAliasPlan{});
}

StatusOr<ExecutionPlan> ExecutionPlanBuilder::Build(
        RuntimeContext& runtime,
        const LoweredGraph& lowered) {
    auto alias_plan = ResolveStateAliasesForExecution(lowered);
    if (!alias_plan.ok()) {
        return alias_plan.status();
    }
    const auto node_specs = std::views::transform(
            lowered.steps(),
            [](const LoweredStep& step) -> const LoweredStepSpec& {
                return step.spec;
            });
    return BuildExecutionPlan(runtime, nullptr, node_specs, std::move(*alias_plan));
}

StatusOr<ExecutionPlan> ExecutionPlanBuilder::Build(
        RuntimeContext& runtime,
        const PackedWeightStore& packed_weight_store,
        const LoweredGraph& lowered) {
    auto alias_plan = ResolveStateAliasesForExecution(lowered);
    if (!alias_plan.ok()) {
        return alias_plan.status();
    }
    const auto node_specs = std::views::transform(
            lowered.steps(),
            [](const LoweredStep& step) -> const LoweredStepSpec& {
                return step.spec;
            });
    return BuildExecutionPlan(runtime, &packed_weight_store, node_specs, std::move(*alias_plan));
}

}// namespace aethermind
