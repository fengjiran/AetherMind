#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/backend/packed_weights.h"
#include "aethermind/compiler/lowered_graph.h"
#include "aethermind/model/packed_weight_store.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/operators/operator_schema.h"
#include <algorithm>
#include <concepts>
#include <ranges>
#include <tuple>

namespace aethermind {
namespace {

template<typename NodeSpec>
concept ExecutionNodeMetadata =
        std::same_as<std::remove_cvref_t<NodeSpec>, ExecutionPlanNodeSpec> ||
        std::same_as<std::remove_cvref_t<NodeSpec>, LoweredStepSpec>;

bool HasLegacyWorkspaceRequirement(const WorkspaceRequirement& requirement) noexcept {
    const WorkspaceRequirement default_requirement;
    return requirement.bytes != default_requirement.bytes ||
           requirement.alignment != default_requirement.alignment ||
           requirement.lifetime != default_requirement.lifetime ||
           requirement.reusable != default_requirement.reusable;
}

bool SameWorkspaceRequirement(const WorkspaceRequirement& lhs,
                              const WorkspaceRequirement& rhs) noexcept {
    return lhs.bytes == rhs.bytes &&
           lhs.alignment == rhs.alignment &&
           lhs.lifetime == rhs.lifetime &&
           lhs.reusable == rhs.reusable;
}

Status ValidatePreparedWorkspaceRequirement(const ResolvedKernel& kernel) {
    if (kernel.workspace_requirement.offset != 0) {
        return Status::Internal(
                "Backend prepared a workspace requirement with a non-zero offset");
    }
    if (!IsValidWorkspaceAlignment(kernel.workspace_requirement.alignment)) {
        return Status::Internal(
                "Backend prepared a workspace requirement with an invalid alignment");
    }
    return Status::Ok();
}

Status ValidateCallerWorkspaceRequirement(const ExecutionPlanNodeSpec& node,
                                          const ResolvedKernel& kernel) {
    if (!HasLegacyWorkspaceRequirement(node.workspace_requirement)) {
        return Status::Ok();
    }
    if (!SameWorkspaceRequirement(node.workspace_requirement,
                                  kernel.workspace_requirement)) {
        return Status::InvalidArgument(
                "ExecutionPlanNodeSpec.workspace_requirement must match the "
                "backend-prepared kernel workspace requirement");
    }
    return Status::Ok();
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

struct PreparedExecutionStep {
    ResolvedKernel kernel{};
    const void* packed_weights = nullptr;
    PreparedNodeMetadata metadata{};
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
        return Status::Internal(
                "Finalized LoweredStepSpec is missing semantic metadata");
    }

    auto metadata = PrepareCompactInputMetadata(node);
    if (!metadata.ok()) {
        return Status::Internal(
                "Finalized LoweredStepSpec has invalid compact input metadata: " +
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
    std::vector<PreparedExecutionStep> prepared_steps;
    prepared_steps.reserve(std::ranges::size(nodes));

    std::vector<WorkspaceRequirement> workspace_requirements;
    workspace_requirements.reserve(std::ranges::size(nodes));
    for (const auto& node: nodes) {
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
        AM_RETURN_IF_ERROR(ValidatePreparedWorkspaceRequirement(*kernel));
        if constexpr (std::same_as<std::remove_cvref_t<decltype(node)>,
                                   ExecutionPlanNodeSpec>) {
            AM_RETURN_IF_ERROR(ValidateCallerWorkspaceRequirement(node, *kernel));
        }

        const auto packed_weights = ResolvePackedWeightsForNode(packed_weight_store, node);
        if (!packed_weights.ok()) {
            return packed_weights.status();
        }

        workspace_requirements.push_back(kernel->workspace_requirement);
        prepared_steps.push_back({
                .kernel = std::move(*kernel),
                .packed_weights = *packed_weights,
                .metadata = std::move(*metadata),
        });
    }

    if (const auto layout = PlanWorkspaceRequirements(std::span(workspace_requirements));
        !layout.ok()) {
        return layout.status();
    }

    std::vector<ExecutionStep> steps;
    steps.reserve(std::ranges::size(nodes));
    for (size_t index = 0; index < std::ranges::size(nodes); ++index) {
        const auto& node = nodes[index];
        PreparedExecutionStep& prepared = prepared_steps[index];
        prepared.kernel.workspace_requirement = workspace_requirements[index];

        steps.push_back({
                .selector = node.selector,
                .kernel = std::move(prepared.kernel),
                .packed_weights = prepared.packed_weights,
                .workspace_requirement = workspace_requirements[index],
                .input_specs = std::move(prepared.metadata.compact_input_specs),
                .output_specs = std::move(prepared.metadata.output_specs),
                .runtime_checks = std::move(prepared.metadata.runtime_checks),
        });
    }
    return ExecutionPlan::Create(std::move(steps), std::move(state_alias_plan));
}

}// namespace

StatusOr<StateAliasPlan> ResolveStateAliasesForExecution(const LoweredGraph& lowered) {
    // The compiler finalizer establishes these invariants. Recheck at the
    // trust boundary so an invalid artifact is surfaced as Internal rather
    // than passed to runtime state binding.
    AM_RETURN_IF_ERROR(ValidateLoweredGraph(lowered));

    StateAliasPlan plan;
    plan.aliases.reserve(lowered.state_aliases().size());
    for (const auto& alias: lowered.state_aliases()) {
        plan.aliases.push_back({
                .step_index = alias.step_index,
                .input_port = alias.input_port,
                .output_port = alias.output_port,
        });
    }

    std::ranges::sort(
            plan.aliases,
            [](const ResolvedStateAlias& lhs,
               const ResolvedStateAlias& rhs) noexcept {
                return std::tie(lhs.step_index, lhs.input_port, lhs.output_port) <
                       std::tie(rhs.step_index, rhs.input_port, rhs.output_port);
            });
    return plan;
}

StatusOr<ResolvedKernel> ExecutionPlanBuilder::PrepareKernelForNode(
        const Backend& backend,
        const ExecutionPlanNodeSpec& node) {
    if (node.op_type == OpType::kUnknown) {
        return Status::InvalidArgument("ExecutionPlanNodeSpec.op_type cannot be kUnknown");
    }

    if (std::holds_alternative<std::monostate>(node.op_params)) {
        return Status::InvalidArgument("ExecutionPlanNodeSpec requires typed op_params");
    }
    auto kernel = backend.PrepareKernel(node.op_type, node.selector, node.op_params);
    if (!kernel.ok()) {
        return kernel.status();
    }
    AM_RETURN_IF_ERROR(ValidatePreparedWorkspaceRequirement(*kernel));
    AM_RETURN_IF_ERROR(ValidateCallerWorkspaceRequirement(node, *kernel));
    return kernel;
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
        const LoweredGraph& lowered_graph) {
    auto alias_plan = ResolveStateAliasesForExecution(lowered_graph);
    if (!alias_plan.ok()) {
        return alias_plan.status();
    }

    const auto node_specs = std::views::transform(
            lowered_graph.steps(),
            [](const LoweredStep& step) -> const LoweredStepSpec& {
                return step.spec;
            });
    return BuildExecutionPlan(runtime, nullptr,
                              node_specs, std::move(*alias_plan));
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
