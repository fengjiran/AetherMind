#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/backend/packed_weights.h"
#include "aethermind/compiler/lowered_graph.h"
#include "aethermind/model/packed_weight_store.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/operators/operator_schema.h"

#include <optional>
#include <span>
#include <string>
#include <utility>

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

std::optional<WorkspaceRequirement> MakeCallerWorkspaceAssertion(
        const WorkspaceRequirement& requirement) noexcept {
    return HasLegacyWorkspaceRequirement(requirement) ? std::optional{requirement} : std::nullopt;
}

Status ValidateCallerWorkspaceRequirement(const WorkspaceRequirement& caller,
                                          const ResolvedKernel& kernel) {
    if (!SameWorkspaceRequirement(caller, kernel.workspace_requirement)) {
        return Status::InvalidArgument(
                "ExecutionPlanNodeSpec.workspace_requirement must match the"
                " backend-prepared kernel workspace requirement");
    }
    return Status::Ok();
}

std::vector<uint32_t> MakeKernelInputPorts(const OperatorSchema& schema) {
    std::vector<uint32_t> ports;
    ports.reserve(schema.input_ports.size());
    for (size_t i = 0; i < schema.input_ports.size(); ++i) {
        if (schema.input_ports[i].kind != OperatorPortKind::kState) {
            ports.push_back(static_cast<uint32_t>(i));
        }
    }
    return ports;
}

std::vector<uint32_t> MakeInferenceInputPorts(const OperatorSchema& schema) {
    std::vector<uint32_t> ports;
    ports.reserve(schema.input_ports.size());
    for (size_t i = 0; i < schema.input_ports.size(); ++i) {
        if (schema.input_ports[i].contributes_tensor_spec) {
            ports.push_back(static_cast<uint32_t>(i));
        }
    }
    return ports;
}

std::vector<uint32_t> MakeKernelOutputPorts(const OperatorSchema& schema) {
    std::vector<uint32_t> ports;
    ports.reserve(schema.output_ports.size());
    for (size_t i = 0; i < schema.output_ports.size(); ++i) {
        if (schema.output_ports[i].kind != OperatorPortKind::kState) {
            ports.push_back(static_cast<uint32_t>(i));
        }
    }
    return ports;
}

std::vector<TensorSpec> GatherSpecs(const std::vector<TensorSpec>& semantic_specs,
                                    std::span<const uint32_t> ports) {
    std::vector<TensorSpec> compact;
    compact.reserve(ports.size());
    for (const uint32_t port: ports) {
        compact.push_back(semantic_specs[port]);
    }
    return compact;
}

StatusOr<std::vector<ShapeConstraint>> RemapRuntimeChecks(
        const std::vector<ShapeConstraint>& checks,
        std::span<const uint32_t> inference_input_ports,
        std::span<const uint32_t> kernel_input_ports,
        std::span<const uint32_t> semantic_output_ports,
        size_t semantic_output_count) {
    std::vector<size_t> input_to_kernel(inference_input_ports.size(), SIZE_MAX);
    for (size_t i = 0; i < inference_input_ports.size(); ++i) {
        const uint32_t semantic_port = inference_input_ports[i];
        for (size_t j = 0; j < kernel_input_ports.size(); ++j) {
            if (kernel_input_ports[j] == semantic_port) {
                input_to_kernel[i] = j;
                break;
            }
        }
    }

    std::vector<size_t> output_to_kernel(semantic_output_count, SIZE_MAX);
    for (size_t i = 0; i < semantic_output_ports.size(); ++i) {
        const uint32_t semantic_port = semantic_output_ports[i];
        output_to_kernel[semantic_port] = i;
    }

    auto remap_port = [&input_to_kernel, &output_to_kernel](TensorPort& port) -> Status {
        if (port.direction == TensorPortType::kInput) {
            if (port.tensor_idx >= input_to_kernel.size() ||
                input_to_kernel[port.tensor_idx] == SIZE_MAX) {
                return Status::Internal(
                        "Runtime check references a non-kernel inference input port");
            }

            port.tensor_idx = input_to_kernel[port.tensor_idx];
            return Status::Ok();
        }

        if (port.tensor_idx >= output_to_kernel.size() ||
            output_to_kernel[port.tensor_idx] == SIZE_MAX) {
            return Status::Internal(
                    "Runtime check references a non-kernel semantic output port");
        }

        port.tensor_idx = output_to_kernel[port.tensor_idx];
        return Status::Ok();
    };

    auto remap_dim = [&remap_port](DimLocator& locator) {
        return remap_port(locator.tensor_port);
    };

    std::vector<ShapeConstraint> remapped = checks;
    for (auto& check: remapped) {
        auto visitor = overloaded{
                [&](DimEqualConstraint& c) {
                    AM_RETURN_IF_ERROR(remap_dim(c.lhs));
                    return remap_dim(c.rhs);
                },
                [&](DimBroadcastableConstraint& c) {
                    AM_RETURN_IF_ERROR(remap_dim(c.lhs));
                    return remap_dim(c.rhs);
                },
                [&](VolumeEqualConstraint& c) {
                    for (auto& dim: c.lhs_dims) {
                        AM_RETURN_IF_ERROR(remap_dim(dim));
                    }

                    for (auto& dim: c.rhs_dims) {
                        AM_RETURN_IF_ERROR(remap_dim(dim));
                    }
                    return Status::Ok();
                },
                [&](RankEqualConstraint& c) {
                    return remap_port(c.port);
                },
                [&](RankAtLeastConstraint& c) {
                    return remap_port(c.port);
                },
                [&](DimPositiveConstraint& c) {
                    return remap_dim(c.dim);
                }};
        AM_RETURN_IF_ERROR(std::visit(visitor, check.condition));
    }
    return remapped;
}

struct PreparedNode {
    OpType op_type = OpType::kUnknown;
    KernelSelector selector{};
    OpParams op_params{};
    std::vector<ExecutionValueId> inputs{};
    std::vector<ExecutionValueId> outputs{};
    std::vector<TensorSpec> semantic_input_specs{};
    std::vector<TensorSpec> semantic_output_specs{};
    std::vector<uint32_t> kernel_input_ports{};
    std::vector<uint32_t> kernel_output_ports{};
    std::vector<TensorSpec> compact_input_specs{};
    std::vector<TensorSpec> compact_output_specs{};
    std::vector<ShapeConstraint> runtime_checks{};
    std::optional<WorkspaceRequirement> caller_workspace_assertion{};
    // Filled by AssembleExecutionPlan after backend resolution.
    ResolvedKernel kernel{};
    // Populated during graph preparation when selector.weight_format is kPacked;
    // resolved against the PackedWeightStore during assembly.
    std::optional<WeightArtifactKey> packed_key{};
    std::shared_ptr<const PackedWeights> packed_weights{};
};

struct PreparedExecutionGraph {
    std::vector<ExecutionValueDesc> values{};
    std::vector<ExecutionValueId> model_inputs{};
    std::vector<ExecutionValueId> model_outputs{};
    std::vector<PreparedNode> nodes{};
};

Status ValidateCallerMetadata(const ExecutionPlanNodeSpec& node,
                              std::span<const TensorSpec> compact_input_specs,
                              std::vector<TensorSpec>& outputs_out,
                              std::vector<ShapeConstraint>& checks_out) {
    auto analyzed = InferOperator(
            node.op_type, node.op_params, compact_input_specs);
    if (!analyzed.ok()) return analyzed.status();
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

StatusOr<PreparedNode> PrepareNode(OpType op_type,
                                   KernelSelector selector,
                                   OpParams op_params,
                                   const std::vector<TensorSpec>& semantic_inputs,
                                   const std::vector<TensorSpec>& semantic_outputs,
                                   const std::vector<ShapeConstraint>& runtime_checks,
                                   std::optional<WorkspaceRequirement> caller_assertion,
                                   bool untrusted) {
    if (op_type == OpType::kUnknown || std::holds_alternative<std::monostate>(op_params)) {
        return untrusted
                       ? Status::InvalidArgument(
                                 "ExecutionPlanNodeSpec requires an op_type and typed op_params")
                       : Status::Internal(
                                 "Finalized LoweredStepSpec is missing semantic metadata");
    }

    const auto schema = GetOperatorSchema(op_type);
    if (!schema.ok()) {
        return untrusted ? schema.status()
                         : Status::Internal("Finalized LoweredStepSpec has no registered schema");
    }

    if (semantic_inputs.size() != schema->input_ports.size() ||
        semantic_outputs.size() != schema->output_ports.size()) {
        return untrusted
                       ? Status::InvalidArgument("ExecutionPlanNodeSpec semantic "
                                                 "port arity differs from schema")
                       : Status::Internal("Finalized LoweredStepSpec semantic port"
                                          " arity differs from schema");
    }

    const auto inference_input_ports = MakeInferenceInputPorts(*schema);
    const auto kernel_inputs = MakeKernelInputPorts(*schema);
    const auto kernel_outputs = MakeKernelOutputPorts(*schema);
    auto inference_inputs = MakeCompactInputSpecs(*schema, semantic_inputs);
    if (!inference_inputs.ok()) {
        return untrusted
                       ? inference_inputs.status()
                       : Status::Internal("Finalized LoweredStepSpec has invalid"
                                          " inference input metadata");
    }

    if (untrusted) {
        std::vector<TensorSpec> inferred_outputs;
        std::vector<ShapeConstraint> inferred_checks;
        AM_RETURN_IF_ERROR(ValidateCallerMetadata(
                ExecutionPlanNodeSpec{
                        .op_type = op_type,
                        .selector = selector,
                        .workspace_requirement = caller_assertion.value_or(WorkspaceRequirement{}),
                        .input_specs = semantic_inputs,
                        .output_specs = semantic_outputs,
                        .runtime_checks = runtime_checks,
                        .op_params = op_params},
                *inference_inputs, inferred_outputs, inferred_checks));
    }

    const auto selector_dtypes =
            DeriveSelectorDTypes(*schema, semantic_inputs, semantic_outputs);
    if (!selector_dtypes.ok()) {
        return untrusted
                       ? Status::InvalidArgument(
                                 "ExecutionPlanNodeSpec selector dtypes cannot be derived")
                       : Status::Internal(
                                 "Finalized LoweredStepSpec selector dtypes cannot be derived");
    }

    if (selector.act_dtype != selector_dtypes->act_dtype) {
        return untrusted
                       ? Status::InvalidArgument("ExecutionPlanNodeSpec.selector.act_dtype "
                                                 "does not match semantic specs")
                       : Status::Internal("Finalized LoweredStepSpec.selector.act_dtype "
                                          "does not match semantic specs");
    }

    if (selector.weight_dtype != selector_dtypes->weight_dtype) {
        return untrusted
                       ? Status::InvalidArgument("ExecutionPlanNodeSpec.selector.weight_dtype"
                                                 " does not match semantic specs")
                       : Status::Internal("Finalized LoweredStepSpec.selector.weight_dtype"
                                          " does not match semantic specs");
    }

    auto remapped_checks =
            RemapRuntimeChecks(runtime_checks, inference_input_ports,
                               kernel_inputs, kernel_outputs,
                               semantic_outputs.size());
    if (!remapped_checks.ok()) {
        return untrusted
                       ? remapped_checks.status()
                       : Status::Internal(remapped_checks.status().message());
    }

    return PreparedNode{
            .op_type = op_type,
            .selector = selector,
            .op_params = std::move(op_params),
            .semantic_input_specs = semantic_inputs,
            .semantic_output_specs = semantic_outputs,
            .kernel_input_ports = kernel_inputs,
            .kernel_output_ports = kernel_outputs,
            .compact_input_specs = GatherSpecs(semantic_inputs, kernel_inputs),
            .compact_output_specs = GatherSpecs(semantic_outputs, kernel_outputs),
            .runtime_checks = std::move(*remapped_checks),
            .caller_workspace_assertion = caller_assertion,
    };
}

StatusOr<ExecutionValueKind> KindFromPayload(const GraphValuePayload& payload) {
    if (std::holds_alternative<ModelInputValue>(payload)) {
        return ExecutionValueKind::kModelInput;
    }

    if (std::holds_alternative<ActivationValue>(payload)) {
        return ExecutionValueKind::kActivation;
    }

    if (std::holds_alternative<WeightValue>(payload)) {
        return ExecutionValueKind::kWeight;
    }

    if (std::holds_alternative<ConstantValue>(payload)) {
        return ExecutionValueKind::kConstant;
    }

    if (std::holds_alternative<StateValue>(payload)) {
        return ExecutionValueKind::kState;
    }
    return Status::Internal(
            "Finalized LoweredGraph contains a value with unknown payload kind");
}

StatusOr<PreparedExecutionGraph> PrepareUntrustedGraph(
        const std::vector<ExecutionPlanNodeSpec>& nodes) {
    PreparedExecutionGraph graph;
    graph.nodes.reserve(nodes.size());
    for (const auto& [op_type, selector, workspace_requirement,
                      input_specs, output_specs,
                      runtime_checks, op_params]: nodes) {
        auto prepared =
                PrepareNode(op_type, selector, op_params, input_specs,
                            output_specs, runtime_checks,
                            MakeCallerWorkspaceAssertion(workspace_requirement), true);
        if (!prepared.ok()) {
            return prepared.status();
        }

        const auto schema = GetOperatorSchema(op_type);
        if (!schema.ok()) {
            return schema.status();
        }

        for (size_t i = 0; i < input_specs.size(); ++i) {
            const OperatorPortKind port_kind = schema->input_ports[i].kind;
            const ExecutionValueKind kind =
                    port_kind == OperatorPortKind::kWeight     ? ExecutionValueKind::kWeight
                    : port_kind == OperatorPortKind::kConstant ? ExecutionValueKind::kConstant
                    : port_kind == OperatorPortKind::kState    ? ExecutionValueKind::kState
                                                               : ExecutionValueKind::kModelInput;
            const ExecutionValueId id{.index = static_cast<uint32_t>(graph.values.size())};
            graph.values.push_back({.spec = input_specs[i],
                                    .kind = kind});
            prepared->inputs.push_back(id);
            if (kind == ExecutionValueKind::kModelInput) {
                graph.model_inputs.push_back(id);
            }
        }

        for (size_t i = 0; i < output_specs.size(); ++i) {
            const ExecutionValueKind kind = schema->output_ports[i].kind == OperatorPortKind::kState
                                                    ? ExecutionValueKind::kState
                                                    : ExecutionValueKind::kActivation;
            const ExecutionValueId id{.index = static_cast<uint32_t>(graph.values.size())};
            graph.values.push_back({.spec = output_specs[i],
                                    .kind = kind});
            prepared->outputs.push_back(id);
            if (kind == ExecutionValueKind::kActivation) {
                graph.model_outputs.push_back(id);
            }
        }

        if (selector.weight_format == WeightFormat::kPacked) {
            // Untrusted nodes carry no WeightBinding, so the empty binding is
            // the only identity available. Multiple kPacked untrusted nodes
            // therefore collide on one key and are rejected at Store time.
            prepared->packed_key = WeightArtifactKey{.binding = {},
                                                     .selector = selector};
        }
        graph.nodes.push_back(std::move(*prepared));
    }
    return graph;
}

StatusOr<PreparedExecutionGraph> PrepareTrustedGraph(const LoweredGraph& lowered) {
    AM_RETURN_IF_ERROR(ValidateLoweredGraph(lowered));
    PreparedExecutionGraph graph;
    graph.values.reserve(lowered.values().size());
    for (const auto& value: lowered.values()) {
        auto kind = KindFromPayload(value.payload);
        if (!kind.ok()) {
            return kind.status();
        }
        graph.values.push_back({.spec = value.spec,
                                .kind = *kind,
                                .name = value.name});
    }

    graph.model_inputs.reserve(lowered.model_inputs().size());
    for (const auto id: lowered.model_inputs()) {
        graph.model_inputs.push_back({.index = id.index});
    }

    graph.model_outputs.reserve(lowered.model_outputs().size());
    for (const auto id: lowered.model_outputs()) {
        graph.model_outputs.push_back({.index = id.index});
    }

    graph.nodes.reserve(lowered.steps().size());
    for (const auto& [spec, binding]: lowered.steps()) {
        auto prepared = PrepareNode(spec.op_type, spec.selector,
                                    spec.op_params, spec.input_specs,
                                    spec.output_specs, spec.runtime_checks,
                                    std::nullopt, false);
        if (!prepared.ok()) {
            return prepared.status();
        }

        prepared->inputs.reserve(binding.input_values.size());
        for (const GraphValueId id: binding.input_values) {
            prepared->inputs.push_back({.index = id.index});
        }

        prepared->outputs.reserve(binding.output_values.size());
        for (const GraphValueId id: binding.output_values) {
            prepared->outputs.push_back({.index = id.index});
        }

        if (spec.selector.weight_format == WeightFormat::kPacked) {
            // Derive the artifact key from the step's kWeight input value so
            // layers/roles/qkv do not collide on one (op_type, selector).
            const auto schema = GetOperatorSchema(spec.op_type);
            if (!schema.ok()) {
                return schema.status();
            }

            std::optional<WeightArtifactKey> packed_key;
            for (size_t port = 0;
                 port < schema->input_ports.size() && port < prepared->inputs.size();
                 ++port) {
                if (schema->input_ports[port].kind != OperatorPortKind::kWeight) {
                    continue;
                }
                const GraphValuePayload& payload =
                        lowered.values()[prepared->inputs[port].index].payload;
                const WeightValue* weight = std::get_if<WeightValue>(&payload);
                if (weight == nullptr) {
                    return Status::Internal(
                            "kWeight input value has no WeightValue payload");
                }

                packed_key = WeightArtifactKey{.binding = weight->binding,
                                               .selector = spec.selector};
                break;
            }

            if (!packed_key.has_value()) {
                return Status::Internal(
                        "kPacked execution step has no kWeight input value");
            }
            prepared->packed_key = std::move(packed_key);
        }
        graph.nodes.push_back(std::move(*prepared));
    }
    return graph;
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

    if (kernel->op_type != op_type) {
        return Status::Internal(
                "Backend prepared a kernel for an unexpected op_type");
    }

    AM_RETURN_IF_ERROR(ValidatePreparedWorkspaceRequirement(*kernel));
    if (caller_assertion.has_value()) {
        AM_RETURN_IF_ERROR(ValidateCallerWorkspaceRequirement(*caller_assertion, *kernel));
    }
    return kernel;
}

StatusOr<ExecutionPlan> AssembleExecutionPlan(RuntimeContext& runtime,
                                              const PackedWeightStore* packed_weight_store,
                                              PreparedExecutionGraph graph,
                                              StateAliasPlan state_alias_plan) {
    std::vector<WorkspaceRequirement> workspace_requirements;
    workspace_requirements.reserve(graph.nodes.size());
    for (auto& node: graph.nodes) {
        auto backend = runtime.GetBackend(node.selector.device_type);
        if (!backend.ok()) {
            return backend.status();
        }

        auto kernel = PrepareKernelChecked(
                *backend.value(), node.op_type, node.selector,
                node.op_params, node.caller_workspace_assertion);
        if (!kernel.ok()) {
            return kernel.status();
        }

        if (node.packed_key.has_value()) {
            if (packed_weight_store == nullptr) {
                return Status::NotFound(
                        "Packed-weight node requires a PackedWeightStore");
            }

            const WeightArtifactKey exact_key{
                    .binding = node.packed_key->binding,
                    .selector = node.packed_key->selector,
                    .recipe = kernel->expected_packing_recipe};
            auto packed_weights = packed_weight_store->Find(exact_key);
            if (packed_weights == nullptr ||
                packed_weights->storage().data() == nullptr) {
                return Status::NotFound(
                        "Packed weights not found for ExecutionPlan node");
            }
            node.packed_weights = std::move(packed_weights);
        }

        workspace_requirements.push_back(kernel->workspace_requirement);
        node.kernel = std::move(*kernel);
    }

    const auto layout = PlanWorkspaceRequirements(
            std::span(workspace_requirements));
    if (!layout.ok()) {
        return layout.status();
    }

    std::vector<ExecutionStep> steps;
    steps.reserve(graph.nodes.size());
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        auto& node = graph.nodes[i];
        node.kernel.workspace_requirement = workspace_requirements[i];
        steps.push_back({.selector = node.selector,
                         .kernel = std::move(node.kernel),
                         .packed_weights = node.packed_weights,
                         .workspace_requirement = workspace_requirements[i],
                         .inputs = std::move(node.inputs),
                         .outputs = std::move(node.outputs),
                         .kernel_input_ports = std::move(node.kernel_input_ports),
                         .kernel_output_ports = std::move(node.kernel_output_ports),
                         .semantic_input_specs = std::move(node.semantic_input_specs),
                         .semantic_output_specs = std::move(node.semantic_output_specs),
                         .input_specs = std::move(node.compact_input_specs),
                         .output_specs = std::move(node.compact_output_specs),
                         .runtime_checks = std::move(node.runtime_checks)});
    }
    return ExecutionPlan::Create(std::move(graph.values), std::move(graph.model_inputs),
                                 std::move(graph.model_outputs), std::move(steps),
                                 std::move(state_alias_plan), *layout);
}

}// namespace

StatusOr<StateAliasPlan> ResolveStateAliasesForExecution(const LoweredGraph& lowered) {
    AM_RETURN_IF_ERROR(ValidateLoweredGraph(lowered));
    StateAliasPlan plan;
    plan.aliases.reserve(lowered.state_aliases().size());
    for (const auto& alias: lowered.state_aliases()) {
        plan.aliases.push_back({.step_index = alias.step_index,
                                .input_port = alias.input_port,
                                .output_port = alias.output_port});
    }
    return plan;
}

StatusOr<ResolvedKernel> ExecutionPlanBuilder::PrepareKernelForNode(
        const Backend& backend,
        const ExecutionPlanNodeSpec& node) {
    if (node.op_type == OpType::kUnknown ||
        std::holds_alternative<std::monostate>(node.op_params)) {
        return Status::InvalidArgument(
                "ExecutionPlanNodeSpec requires an op_type and typed op_params");
    }
    return PrepareKernelChecked(backend, node.op_type, node.selector, node.op_params,
                                MakeCallerWorkspaceAssertion(node.workspace_requirement));
}

StatusOr<ExecutionPlan> ExecutionPlanBuilder::Build(
        RuntimeContext& runtime,
        const std::vector<ExecutionPlanNodeSpec>& nodes) {
    auto graph = PrepareUntrustedGraph(nodes);
    if (!graph.ok()) {
        return graph.status();
    }
    return AssembleExecutionPlan(runtime, nullptr,
                                 std::move(*graph), {});
}

StatusOr<ExecutionPlan> ExecutionPlanBuilder::Build(
        RuntimeContext& runtime,
        const PackedWeightStore& packed_weight_store,
        const std::vector<ExecutionPlanNodeSpec>& nodes) {
    auto graph = PrepareUntrustedGraph(nodes);
    if (!graph.ok()) {
        return graph.status();
    }
    return AssembleExecutionPlan(runtime, &packed_weight_store,
                                 std::move(*graph), {});
}

StatusOr<ExecutionPlan> ExecutionPlanBuilder::Build(
        RuntimeContext& runtime,
        const LoweredGraph& lowered_graph) {
    auto aliases = ResolveStateAliasesForExecution(lowered_graph);
    if (!aliases.ok()) {
        return aliases.status();
    }

    auto graph = PrepareTrustedGraph(lowered_graph);
    if (!graph.ok()) {
        return graph.status();
    }
    return AssembleExecutionPlan(runtime, nullptr,
                                 std::move(*graph), std::move(*aliases));
}

StatusOr<ExecutionPlan> ExecutionPlanBuilder::Build(
        RuntimeContext& runtime,
        const PackedWeightStore& packed_weight_store,
        const LoweredGraph& lowered) {
    auto aliases = ResolveStateAliasesForExecution(lowered);
    if (!aliases.ok()) {
        return aliases.status();
    }

    auto graph = PrepareTrustedGraph(lowered);
    if (!graph.ok()) {
        return graph.status();
    }
    return AssembleExecutionPlan(runtime, &packed_weight_store,
                                 std::move(*graph), std::move(*aliases));
}

}// namespace aethermind
