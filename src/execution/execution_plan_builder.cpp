#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/backend/packed_weights.h"
#include "aethermind/compiler/lowered_graph.h"
#include "aethermind/model/packed_weight_store.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/operators/operator_schema.h"

#include <algorithm>
#include <optional>
#include <span>
#include <tuple>

namespace aethermind {
namespace {

bool HasLegacyWorkspaceRequirement(const WorkspaceRequirement& requirement) noexcept {
    constexpr WorkspaceRequirement default_requirement;
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

// Only a non-default caller-supplied requirement is an assertion worth
// reconciling against the backend-prepared kernel requirement; a default
// value means "no assertion".
std::optional<WorkspaceRequirement> MakeCallerWorkspaceAssertion(
        const WorkspaceRequirement& requirement) noexcept {
    if (!HasLegacyWorkspaceRequirement(requirement)) {
        return std::nullopt;
    }
    return requirement;
}

Status ValidateCallerWorkspaceRequirement(const WorkspaceRequirement& caller,
                                          const ResolvedKernel& kernel) {
    if (!SameWorkspaceRequirement(caller, kernel.workspace_requirement)) {
        return Status::InvalidArgument(
                "ExecutionPlanNodeSpec.workspace_requirement must match the "
                "backend-prepared kernel workspace requirement");
    }
    return Status::Ok();
}

StatusOr<const void*> ResolvePackedWeights(const PackedWeightStore* packed_weight_store,
                                           OpType op_type,
                                           const KernelSelector& selector) noexcept {
    if (selector.weight_format != WeightFormat::kPacked) {
        return nullptr;
    }

    if (packed_weight_store == nullptr) {
        return Status::NotFound("Packed-weight node requires a PackedWeightStore");
    }

    const auto* packed_weights = packed_weight_store->Find(op_type, selector);
    if (packed_weights == nullptr) {
        return Status::NotFound("Packed weights not found for ExecutionPlan node");
    }
    if (packed_weights->storage().data() == nullptr) {
        return Status::NotFound(
                "Packed weights storage is empty for ExecutionPlan node");
    }
    return packed_weights->storage().data();
}

// One node carried through preparation (PrepareUntrustedNode/PrepareTrustedNode)
// and then filled in-place by AssembleExecutionPlan. Trust-specific preparation
// (untrusted re-inference vs. trusted verbatim copy) has already happened, so
// assembly carries no knowledge of which source produced the node. The kernel
// and packed-weights fields are filled by assembly after backend resolution.
struct PreparedNode {
    OpType op_type;
    KernelSelector selector;
    OpParams op_params;
    // Kernel-facing compact views of the inputs, the operator outputs, and the
    // deferred runtime shape checks. These together are the semantic metadata
    // moved into the ExecutionStep.
    std::vector<TensorSpec> compact_input_specs{};
    std::vector<TensorSpec> output_specs{};
    std::vector<ShapeConstraint> runtime_checks{};
    // Only the untrusted path records a caller-supplied workspace assertion to
    // reconcile against the backend-prepared kernel requirement.
    std::optional<WorkspaceRequirement> caller_workspace_assertion{};
    // Filled by AssembleExecutionPlan after kernel resolution.
    ResolvedKernel kernel{};
    const void* packed_weights = nullptr;
};

// Validates caller-provided semantic metadata against the sole semantic
// authority, InferOperator. Empty fields are not inferred on the caller's
// behalf: they must match the explicit inference result exactly. Typed
// op_params are guaranteed by PrepareUntrustedNode before this runs.
Status ValidateCallerMetadata(const ExecutionPlanNodeSpec& node,
                              std::span<const TensorSpec> compact_input_specs,
                              std::vector<TensorSpec>& outputs_out,
                              std::vector<ShapeConstraint>& checks_out) {
    auto analyzed = InferOperator(
            node.op_type, node.op_params, compact_input_specs);
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

StatusOr<std::vector<TensorSpec>> PrepareCompactInputSpecs(
        OpType op_type, std::span<const TensorSpec> input_specs) {
    const auto schema = GetOperatorSchema(op_type);
    if (!schema.ok()) {
        return schema.status();
    }

    auto compact_input_specs = MakeCompactInputSpecs(*schema, input_specs);
    if (!compact_input_specs.ok()) {
        return compact_input_specs.status();
    }

    return std::move(*compact_input_specs);
}

StatusOr<PreparedNode> PrepareUntrustedNode(const ExecutionPlanNodeSpec& node) {
    if (node.op_type == OpType::kUnknown) {
        return Status::InvalidArgument(
                "ExecutionPlanNodeSpec.op_type cannot be kUnknown");
    }

    if (std::holds_alternative<std::monostate>(node.op_params)) {
        return Status::InvalidArgument(
                "ExecutionPlanNodeSpec requires typed op_params");
    }

    auto compact_input_specs = PrepareCompactInputSpecs(
            node.op_type, node.input_specs);
    if (!compact_input_specs.ok()) {
        return compact_input_specs.status();
    }

    std::vector<TensorSpec> output_specs;
    std::vector<ShapeConstraint> runtime_checks;
    AM_RETURN_IF_ERROR(ValidateCallerMetadata(
            node, *compact_input_specs, output_specs, runtime_checks));

    // Cross-check the selector dtypes against the operator's specs using the
    // same derivation rule as graph lowering. The backend resolves kernels by
    // selector only, so a mismatched or undefined selector dtype would run a
    // kernel for a different dtype than the bound specs (type confusion).
    const auto schema = GetOperatorSchema(node.op_type);
    if (!schema.ok()) {
        return schema.status();
    }

    const auto selector_dtypes = DeriveSelectorDTypes(
            *schema, node.input_specs, node.output_specs);
    if (!selector_dtypes.ok()) {
        return Status::InvalidArgument(
                "ExecutionPlanNodeSpec selector dtypes cannot be derived: " +
                selector_dtypes.status().message());
    }

    if (node.selector.act_dtype != selector_dtypes->act_dtype) {
        return Status::InvalidArgument(
                "ExecutionPlanNodeSpec.selector.act_dtype " +
                ToString(node.selector.act_dtype) +
                " does not match the operator activation dtype " +
                ToString(selector_dtypes->act_dtype) + " for " +
                ToString(node.op_type));
    }

    if (node.selector.weight_dtype != selector_dtypes->weight_dtype) {
        return Status::InvalidArgument(
                "ExecutionPlanNodeSpec.selector.weight_dtype " +
                ToString(node.selector.weight_dtype) +
                " does not match the operator weight dtype " +
                ToString(selector_dtypes->weight_dtype) + " for " +
                ToString(node.op_type));
    }

    return PreparedNode{
            .op_type = node.op_type,
            .selector = node.selector,
            .op_params = node.op_params,
            .compact_input_specs = std::move(*compact_input_specs),
            .output_specs = std::move(output_specs),
            .runtime_checks = std::move(runtime_checks),
            .caller_workspace_assertion =
                    MakeCallerWorkspaceAssertion(node.workspace_requirement),
    };
}

StatusOr<PreparedNode> PrepareTrustedNode(const LoweredStepSpec& node) {
    if (node.op_type == OpType::kUnknown ||
        std::holds_alternative<std::monostate>(node.op_params)) {
        return Status::Internal(
                "Finalized LoweredStepSpec is missing semantic metadata");
    }

    auto compact_input_specs = PrepareCompactInputSpecs(
            node.op_type, node.input_specs);
    if (!compact_input_specs.ok()) {
        return Status::Internal(
                "Finalized LoweredStepSpec has invalid compact input metadata: " +
                compact_input_specs.status().message());
    }
    // LoweredGraph is validated by compiler and checked again at the
    // execution trust boundary. Re-running InferOperator here would create a
    // second semantic authority, so only the ExecutionPlan's owning copies
    // are made below.
    return PreparedNode{
            .op_type = node.op_type,
            .selector = node.selector,
            .op_params = node.op_params,
            .compact_input_specs = std::move(*compact_input_specs),
            .output_specs = node.output_specs,
            .runtime_checks = node.runtime_checks,
    };
}

StatusOr<ResolvedKernel> PrepareKernelChecked(
        const Backend& backend,
        OpType op_type,
        const KernelSelector& selector,
        const OpParams& op_params,
        const std::optional<WorkspaceRequirement>& caller_assertion) {
    auto kernel = backend.PrepareKernel(op_type, selector, op_params);
    if (!kernel.ok()) {
        return kernel.status();
    }

    AM_RETURN_IF_ERROR(ValidatePreparedWorkspaceRequirement(*kernel));
    if (caller_assertion.has_value()) {
        AM_RETURN_IF_ERROR(ValidateCallerWorkspaceRequirement(*caller_assertion, *kernel));
    }
    return kernel;
}

StatusOr<ExecutionPlan> AssembleExecutionPlan(
        RuntimeContext& runtime,
        const PackedWeightStore* packed_weight_store,
        std::vector<PreparedNode> nodes,
        StateAliasPlan state_alias_plan) {
    std::vector<WorkspaceRequirement> workspace_requirements;
    workspace_requirements.reserve(nodes.size());
    for (auto& node: nodes) {
        auto backend = runtime.GetBackend(node.selector.device_type);
        if (!backend.ok()) {
            return backend.status();
        }

        auto kernel = PrepareKernelChecked(*backend.value(),
                                           node.op_type, node.selector, node.op_params,
                                           node.caller_workspace_assertion);
        if (!kernel.ok()) {
            return kernel.status();
        }

        const auto packed_weights =
                ResolvePackedWeights(packed_weight_store, node.op_type, node.selector);
        if (!packed_weights.ok()) {
            return packed_weights.status();
        }

        workspace_requirements.push_back(kernel->workspace_requirement);
        node.kernel = std::move(*kernel);
        node.packed_weights = *packed_weights;
    }

    const auto layout = PlanWorkspaceRequirements(std::span(workspace_requirements));
    if (!layout.ok()) {
        return layout.status();
    }

    std::vector<ExecutionStep> steps;
    steps.reserve(nodes.size());
    for (size_t index = 0; index < nodes.size(); ++index) {
        auto& node = nodes[index];
        node.kernel.workspace_requirement = workspace_requirements[index];

        steps.push_back({
                .selector = node.selector,
                .kernel = std::move(node.kernel),
                .packed_weights = node.packed_weights,
                .workspace_requirement = workspace_requirements[index],
                .input_specs = std::move(node.compact_input_specs),
                .output_specs = std::move(node.output_specs),
                .runtime_checks = std::move(node.runtime_checks),
        });
    }
    return ExecutionPlan::Create(std::move(steps), std::move(state_alias_plan), *layout);
}

StatusOr<std::vector<PreparedNode>> PrepareUntrustedNodes(
        const std::vector<ExecutionPlanNodeSpec>& nodes) {
    std::vector<PreparedNode> prepared;
    prepared.reserve(nodes.size());
    for (const auto& node: nodes) {
        auto prepared_node = PrepareUntrustedNode(node);
        if (!prepared_node.ok()) {
            return prepared_node.status();
        }
        prepared.push_back(std::move(*prepared_node));
    }
    return prepared;
}

StatusOr<std::vector<PreparedNode>> PrepareTrustedNodes(const LoweredGraph& lowered) {
    std::vector<PreparedNode> prepared;
    prepared.reserve(lowered.steps().size());
    for (const auto& [spec, _]: lowered.steps()) {
        auto prepared_node = PrepareTrustedNode(spec);
        if (!prepared_node.ok()) {
            return prepared_node.status();
        }
        prepared.push_back(std::move(*prepared_node));
    }
    return prepared;
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
        return Status::InvalidArgument(
                "ExecutionPlanNodeSpec.op_type cannot be kUnknown");
    }

    if (std::holds_alternative<std::monostate>(node.op_params)) {
        return Status::InvalidArgument(
                "ExecutionPlanNodeSpec requires typed op_params");
    }
    return PrepareKernelChecked(backend, node.op_type, node.selector, node.op_params,
                                MakeCallerWorkspaceAssertion(node.workspace_requirement));
}

StatusOr<ExecutionPlan> ExecutionPlanBuilder::Build(
        RuntimeContext& runtime,
        const std::vector<ExecutionPlanNodeSpec>& nodes) {
    auto prepared = PrepareUntrustedNodes(nodes);
    if (!prepared.ok()) {
        return prepared.status();
    }
    return AssembleExecutionPlan(runtime, nullptr,
                                 std::move(*prepared), {});
}

StatusOr<ExecutionPlan> ExecutionPlanBuilder::Build(
        RuntimeContext& runtime,
        const PackedWeightStore& packed_weight_store,
        const std::vector<ExecutionPlanNodeSpec>& nodes) {
    auto prepared = PrepareUntrustedNodes(nodes);
    if (!prepared.ok()) {
        return prepared.status();
    }
    return AssembleExecutionPlan(runtime, &packed_weight_store,
                                 std::move(*prepared), {});
}

StatusOr<ExecutionPlan> ExecutionPlanBuilder::Build(
        RuntimeContext& runtime,
        const LoweredGraph& lowered_graph) {
    auto state_alias_plan = ResolveStateAliasesForExecution(lowered_graph);
    if (!state_alias_plan.ok()) {
        return state_alias_plan.status();
    }

    auto prepared = PrepareTrustedNodes(lowered_graph);
    if (!prepared.ok()) {
        return prepared.status();
    }
    return AssembleExecutionPlan(runtime, nullptr,
                                 std::move(*prepared), std::move(*state_alias_plan));
}

StatusOr<ExecutionPlan> ExecutionPlanBuilder::Build(
        RuntimeContext& runtime,
        const PackedWeightStore& packed_weight_store,
        const LoweredGraph& lowered) {
    auto state_alias_plan = ResolveStateAliasesForExecution(lowered);
    if (!state_alias_plan.ok()) {
        return state_alias_plan.status();
    }

    auto prepared = PrepareTrustedNodes(lowered);
    if (!prepared.ok()) {
        return prepared.status();
    }
    return AssembleExecutionPlan(runtime, &packed_weight_store,
                                 std::move(*prepared), std::move(*state_alias_plan));
}

}// namespace aethermind
