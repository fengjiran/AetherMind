#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/backend/packed_weights.h"
#include "aethermind/model/graph/compilation/graph_lowering.h"
#include "aethermind/model/graph/operator_schema.h"
#include "aethermind/model/model_instance.h"
#include "aethermind/operators/function_operator.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/operators/operator_registry.h"

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

struct PreparedOperator {
    OperatorPtr op{};
    std::vector<TensorSpec> compact_input_specs{};
    // Trusted path: copied verbatim from node.output_specs / node.runtime_checks.
    // Untrusted path: copied from InferOperator's InferenceResult after
    // strict-equality validation against caller-provided metadata.
    std::vector<TensorSpec> output_specs{};
    std::vector<ShapeConstraint> runtime_checks{};
};

// Validates caller-provided semantic metadata (op_params, output_specs,
// runtime_checks) against the single semantic authority InferOperator.
// Used by the untrusted raw ExecutionPlanNodeSpec adapter path.
//
// Rejects:
// - monostate op_params (caller must provide typed params)
// - InferOperator failures (including variant and parameter validation)
// - Any mismatch between caller metadata and inferred metadata (strict
//   equality; empty caller fields are NOT treated as "infer for me").
Status ValidateCallerMetadata(const ExecutionPlanNodeSpec& node,
                              std::span<const TensorSpec> compact_input_specs,
                              std::vector<TensorSpec>& outputs_out,
                              std::vector<ShapeConstraint>& checks_out) {
    if (std::holds_alternative<std::monostate>(node.op_params)) {
        return Status::InvalidArgument(
                "Untrusted ExecutionPlanNodeSpec adapter requires typed op_params; "
                "monostate is not accepted");
    }
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

StatusOr<PreparedOperator> CreateAndPrepareOperator(Backend& backend,
                                                    const ExecutionPlanNodeSpec& node,
                                                    bool trusted) {
    // Untrusted path: reject monostate op_params early, before
    // OperatorRegistry::Create returns a generic "Wrong params type"
    // error. The explicit message makes the contract visible to callers.
    if (!trusted && std::holds_alternative<std::monostate>(node.op_params)) {
        return Status::InvalidArgument(
                "Untrusted ExecutionPlanNodeSpec adapter requires typed op_params; "
                "monostate is not accepted");
    }

    // Trusted path: Operator creation is attempted only to obtain an
    // executable handle for kernel dispatch. Semantic validation was
    // already performed during graph construction (ModelGraph::AddNode ->
    // InferOperator) and carried through lowering; the trusted builder
    // MUST NOT re-invoke InferOperator.
    //
    // Untrusted path: caller-authored ExecutionPlanNodeSpec is treated as
    // potentially stale/forged. We re-invoke InferOperator and require strict equality
    // with caller-provided metadata before entering the common trusted
    // builder tail.
    //
    StatusOr<std::unique_ptr<Operator>> created = OperatorRegistry::Create(
            node.op_type, node.op_params);
    if (!created.ok()) {
        if (created.status().code() != StatusCode::kNotFound) {
            return created.status();
        }
        // Raw kernel fallback: Operator not registered. The ExecutionStep
        // will wrap a FunctionOperator. Semantic validation is still
        // required so FunctionOperator's no-op inference cannot bypass
        // the single semantic authority.
        PreparedOperator fallback;
        fallback.compact_input_specs = node.input_specs;
        if (trusted) {
            // Trusted path: LoweredGraph already carried verbatim metadata
            // from InferOperator during graph construction. Use it
            // directly without re-invoking the semantic authority, so
            // FunctionOperator-based ExecutionSteps stay consistent with
            // the registered-Operator path. The contract is "create if
            // registered, otherwise resolve raw fallback"; returning
            // FailedPrecondition here would violate it.
            fallback.output_specs = node.output_specs;
            fallback.runtime_checks = node.runtime_checks;
        } else {
            // Untrusted raw fallback: validate caller metadata via
            // InferOperator using the full input_specs as the compact view
            // (no schema available to derive a compact subset because the
            // Operator is not registered; InferOperator internally invokes
            // GetOperatorSchema, which must succeed for the OpType).
            std::vector<TensorSpec> outputs;
            std::vector<ShapeConstraint> checks;
            AM_RETURN_IF_ERROR(ValidateCallerMetadata(node, node.input_specs, outputs, checks));
            fallback.output_specs = std::move(outputs);
            fallback.runtime_checks = std::move(checks);
        }
        return fallback;
    }

    if (!node.attrs.empty()) {
        return Status::InvalidArgument(
                "ExecutionPlanNodeSpec.attrs is only supported by raw kernel fallback; use op_params for registered operators");
    }

    std::unique_ptr<Operator> op = std::move(created).value();

    PreparedOperator prepared;

    // Empty input_specs skips compact derivation (test fixtures that only
    // exercise kernel resolution). Both paths handle this identically.
    if (node.input_specs.empty()) {
        if (trusted) {
            prepared.output_specs = node.output_specs;
            prepared.runtime_checks = node.runtime_checks;
        } else {
            std::vector<TensorSpec> empty_inputs;
            AM_RETURN_IF_ERROR(ValidateCallerMetadata(node, empty_inputs,
                                                      prepared.output_specs,
                                                      prepared.runtime_checks));
        }
    } else {
        // Derive compact specs from the complete schema-port-ordered input_specs.
        StatusOr<OperatorSchema> schema_or = GetOperatorSchema(node.op_type);
        AM_RETURN_IF_ERROR(schema_or.status());
        auto compact_or = MakeCompactInputSpecs(*schema_or, node.input_specs);
        AM_RETURN_IF_ERROR(compact_or.status());
        prepared.compact_input_specs = std::move(*compact_or);

        if (trusted) {
            // Carry graph-derived metadata forward without re-inference.
            prepared.output_specs = node.output_specs;
            prepared.runtime_checks = node.runtime_checks;
        } else {
            // Untrusted: validate caller metadata against InferOperator.
            AM_RETURN_IF_ERROR(ValidateCallerMetadata(node, prepared.compact_input_specs,
                                                      prepared.output_specs,
                                                      prepared.runtime_checks));
        }
    }

    OperatorContext op_ctx{
            .backend = &backend,
            .kernel_registry = backend.TryGetKernelRegistryForDebug(),
            .workspace = nullptr,
            .selector = MakeSelectorForNode(node),
    };
    AM_RETURN_IF_ERROR(op->Prepare(op_ctx));
    prepared.op = OperatorPtr(std::move(op));

    return prepared;
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

    if (const auto layout = PlanWorkspaceRequirements(
                std::span(workspace_requirements));
        !layout.ok()) {
        return layout.status();
    }

    std::vector<ExecutionStep> steps;
    steps.reserve(nodes.size());
    for (size_t index = 0; index < nodes.size(); ++index) {
        const auto& node = nodes[index];

        auto backend = runtime.GetBackend(node.device_type);
        if (!backend.ok()) {
            return backend.status();
        }

        auto prepared_operator =
                CreateAndPrepareOperator(*backend.value(), node, trusted);
        if (!prepared_operator.ok()) {
            return prepared_operator.status();
        }

        PreparedOperator prepared = prepared_operator.value();
        OperatorPtr op = std::move(prepared.op);
        if (op == nullptr) {
            const auto resolved =
                    ExecutionPlanBuilder::ResolveKernelForNode(*backend.value(), node);
            if (!resolved.ok()) {
                return resolved.status();
            }
            op = std::make_shared<FunctionOperator>(
                    resolved->op_type,
                    resolved->fn,
                    std::span<const std::byte>(resolved->attrs),
                    resolved->debug_name);
        }

        const auto packed_weights =
                ResolvePackedWeightsForNode(model_instance, node);
        if (!packed_weights.ok()) {
            return packed_weights.status();
        }

        steps.push_back({
                .selector = MakeSelectorForNode(node),
                .op = std::move(op),
                .packed_weights = packed_weights.value(),
                .workspace_requirement = workspace_requirements[index],
                .input_specs = std::move(prepared.compact_input_specs),
                .output_specs = std::move(prepared.output_specs),
                .runtime_checks = std::move(prepared.runtime_checks),
                .debug_name = nullptr,
        });
    }

    return ExecutionPlan::Create(std::move(steps), std::move(state_alias_plan));
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
    frozen.attrs.assign(node.attrs.begin(), node.attrs.end());
    return frozen;
}

StatusOr<ExecutionPlan> ExecutionPlanBuilder::Build(
        RuntimeContext& runtime,
        const std::vector<ExecutionPlanNodeSpec>& nodes) {
    return BuildExecutionPlan(runtime, nullptr, nodes,
                              StateAliasPlan{}, /*trusted=*/false);
}

StatusOr<ExecutionPlan> ExecutionPlanBuilder::Build(
        RuntimeContext& runtime,
        const ModelInstance& model_instance,
        const std::vector<ExecutionPlanNodeSpec>& nodes) {
    return BuildExecutionPlan(runtime, &model_instance, nodes,
                              StateAliasPlan{}, /*trusted=*/false);
}

StatusOr<ExecutionPlan> ExecutionPlanBuilder::Build(
        RuntimeContext& runtime,
        const LoweredGraph& lowered) {
    StatusOr<StateAliasPlan> alias_plan = ResolveStateAliases(lowered);
    AM_RETURN_IF_ERROR(alias_plan.status());
    return BuildExecutionPlan(runtime, nullptr, lowered.steps,
                              std::move(alias_plan).value(), /*trusted=*/true);
}

StatusOr<ExecutionPlan> ExecutionPlanBuilder::Build(
        RuntimeContext& runtime,
        const ModelInstance& model_instance,
        const LoweredGraph& lowered) {
    StatusOr<StateAliasPlan> alias_plan = ResolveStateAliases(lowered);
    AM_RETURN_IF_ERROR(alias_plan.status());
    return BuildExecutionPlan(runtime, &model_instance, lowered.steps,
                              std::move(alias_plan).value(), /*trusted=*/true);
}

}// namespace aethermind
