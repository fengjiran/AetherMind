#include "aethermind/compiler/graph_lowering.h"
#include "aethermind/graph/graph.h"
#include "aethermind/operators/operator_schema.h"
#include "compiler/lowered_graph_draft.h"

#include <algorithm>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>

namespace aethermind {
namespace {

Status SetCandidateDType(const TensorSpec& spec,
                         std::optional<DataType>& candidate,
                         std::string_view role) {
    if (spec.dtype.IsUndefined()) {
        return Status::Internal(
                "LowerModelGraph: undefined " + std::string(role) + " dtype");
    }

    if (candidate.has_value() && *candidate != spec.dtype) {
        return Status::Internal(
                "LowerModelGraph: inconsistent " + std::string(role) +
                " dtypes in one operator");
    }
    candidate = spec.dtype;
    return Status::Ok();
}

Status CollectInputSelectorDTypes(const OperatorInputPort& port,
                                  const TensorSpec& spec,
                                  std::optional<DataType>& act_dtype,
                                  std::optional<DataType>& weight_dtype) {
    if (!port.contributes_tensor_spec) {
        return Status::Ok();
    }

    if (port.kind == OperatorPortKind::kActivation) {
        return SetCandidateDType(spec, act_dtype, "activation");
    }

    if (port.kind == OperatorPortKind::kWeight) {
        return SetCandidateDType(spec, weight_dtype, "weight");
    }
    return Status::Ok();
}

Status CollectOutputActivationDType(const OperatorSchema& schema,
                                    const GraphNode& node,
                                    std::span<const GraphValue> values,
                                    std::optional<DataType>& act_dtype) {
    if (act_dtype.has_value()) {
        return Status::Ok();
    }

    bool found_activation_output = false;
    for (size_t i = 0; i < schema.output_ports.size(); ++i) {
        if (schema.output_ports[i].kind == OperatorPortKind::kActivation) {
            found_activation_output = true;
            AM_RETURN_IF_ERROR(SetCandidateDType(values[node.outputs[i].index].spec,
                                                 act_dtype,
                                                 "activation"));
        }
    }

    if (!found_activation_output) {
        return Status::Internal(
                "LowerModelGraph: operator schema has no contributing activation dtype source");
    }
    return Status::Ok();
}

Status AddLoweringTimeStateAliases(const OperatorSchema& schema,
                                   const GraphNode& node,
                                   size_t step_index,
                                   std::vector<LoweredStateAlias>& aliases) {
    for (const StateAliasPortPair& pair: schema.state_alias_ports) {
        auto input_port = FindInputPortIndex(schema, pair.input_port);
        AM_RETURN_IF_ERROR(input_port.status());
        auto output_port = FindOutputPortIndex(schema, pair.output_port);
        AM_RETURN_IF_ERROR(output_port.status());
        aliases.push_back({
                .step_index = step_index,
                .input_port = input_port.value(),
                .output_port = output_port.value(),
                .input = node.inputs[input_port.value()],
                .output = node.outputs[output_port.value()],
        });
    }
    return Status::Ok();
}

bool IsValidValueId(const LoweredGraph& lowered, GraphValueId id) {
    return id.index < lowered.values().size();
}

Status ValidateValueId(const LoweredGraph& lowered,
                       GraphValueId id,
                       std::string_view context) {
    if (!IsValidValueId(lowered, id)) {
        return Status::Internal("ValidateLoweredGraph: invalid GraphValueId for " +
                                std::string(context));
    }
    return Status::Ok();
}

Status ValidateStateAlias(const LoweredGraph& lowered,
                          const LoweredStateAlias& alias,
                          size_t alias_index,
                          std::set<std::tuple<size_t, uint32_t, uint32_t>>& pairs,
                          std::set<std::pair<size_t, uint32_t>>& inputs,
                          std::set<std::pair<size_t, uint32_t>>& outputs) {
    const std::string prefix = "ValidateLoweredGraph: state alias " +
                               std::to_string(alias_index) + ": ";
    if (alias.step_index >= lowered.steps().size()) {
        return Status::Internal(prefix + "step index out of range");
    }
    const LoweredStep& step = lowered.steps()[alias.step_index];
    const auto schema = GetOperatorSchema(step.spec.op_type);
    if (!schema.ok()) {
        return Status::Internal(prefix + "step has no registered operator schema");
    }
    if (alias.input_port >= schema->input_ports.size() ||
        alias.output_port >= schema->output_ports.size()) {
        return Status::Internal(prefix + "schema port index out of range");
    }
    if (schema->input_ports[alias.input_port].kind != OperatorPortKind::kState ||
        schema->output_ports[alias.output_port].kind != OperatorPortKind::kState) {
        return Status::Internal(prefix + "alias ports must both be state ports");
    }

    const bool declared = std::any_of(
            schema->state_alias_ports.begin(), schema->state_alias_ports.end(),
            [&schema, &alias](const StateAliasPortPair& pair) {
                const auto input = FindInputPortIndex(*schema, pair.input_port);
                const auto output = FindOutputPortIndex(*schema, pair.output_port);
                return input.ok() && output.ok() && input.value() == alias.input_port &&
                       output.value() == alias.output_port;
            });
    if (!declared) {
        return Status::Internal(prefix + "alias is not declared by the operator schema");
    }

    if (alias.input_port >= step.binding.input_values.size() ||
        step.binding.input_values[alias.input_port] != alias.input ||
        alias.output_port >= step.binding.output_values.size() ||
        step.binding.output_values[alias.output_port] != alias.output) {
        return Status::Internal(prefix + "binding does not match its recorded GraphValueIds");
    }
    AM_RETURN_IF_ERROR(ValidateValueId(lowered, alias.input, "state alias input"));
    AM_RETURN_IF_ERROR(ValidateValueId(lowered, alias.output, "state alias output"));

    const auto* input_state = std::get_if<StateValue>(&lowered.values()[alias.input.index].payload);
    const auto* output_state = std::get_if<StateValue>(&lowered.values()[alias.output.index].payload);
    if (input_state == nullptr || output_state == nullptr) {
        return Status::Internal(prefix + "aliased values must both carry StateValue payloads");
    }
    if (input_state->binding != output_state->binding) {
        return Status::Internal(prefix + "state bindings must describe the same state slot");
    }

    const auto pair = std::make_tuple(alias.step_index, alias.input_port, alias.output_port);
    if (!pairs.insert(pair).second) {
        return Status::Internal(prefix + "duplicate alias declaration");
    }
    if (!inputs.insert({alias.step_index, alias.input_port}).second ||
        !outputs.insert({alias.step_index, alias.output_port}).second) {
        return Status::Internal(prefix + "conflicts with another alias port");
    }
    return Status::Ok();
}

}// namespace

Status ValidateLoweredGraph(const LoweredGraph& lowered) {
    std::unordered_set<uint32_t> nodes;
    for (size_t step_index = 0; step_index < lowered.steps().size(); ++step_index) {
        const LoweredStep& step = lowered.steps()[step_index];
        const auto schema = GetOperatorSchema(step.spec.op_type);
        if (!schema.ok()) {
            return Status::Internal(
                    "ValidateLoweredGraph: step " + std::to_string(step_index) +
                    " has no registered operator schema");
        }

        if (!nodes.insert(step.binding.node.index).second) {
            return Status::Internal(
                    "ValidateLoweredGraph: duplicate GraphNodeId in steps");
        }

        if (step.binding.input_values.size() != schema->input_ports.size() ||
            step.spec.input_specs.size() != schema->input_ports.size()) {
            return Status::Internal(
                    "ValidateLoweredGraph: step " + std::to_string(step_index) +
                    " input binding/spec arity differs from schema");
        }

        if (step.binding.output_values.size() != schema->output_ports.size() ||
            step.spec.output_specs.size() != schema->output_ports.size()) {
            return Status::Internal(
                    "ValidateLoweredGraph: step " + std::to_string(step_index) +
                    " output binding/spec arity differs from schema");
        }

        if (step.spec.selector.act_dtype.IsUndefined() ||
            step.spec.selector.weight_dtype.IsUndefined()) {
            return Status::Internal(
                    "ValidateLoweredGraph: step " + std::to_string(step_index) +
                    " has undefined selector dtype");
        }

        for (size_t port = 0; port < step.binding.input_values.size(); ++port) {
            const GraphValueId id = step.binding.input_values[port];
            AM_RETURN_IF_ERROR(ValidateValueId(lowered, id, "step input"));
            if (step.spec.input_specs[port] != lowered.values()[id.index].spec) {
                return Status::Internal(
                        "ValidateLoweredGraph: input spec does not match value metadata");
            }
        }

        for (size_t port = 0; port < step.binding.output_values.size(); ++port) {
            const GraphValueId id = step.binding.output_values[port];
            AM_RETURN_IF_ERROR(ValidateValueId(lowered, id, "step output"));
            if (step.spec.output_specs[port] != lowered.values()[id.index].spec) {
                return Status::Internal(
                        "ValidateLoweredGraph: output spec does not match value metadata");
            }
        }
    }

    for (const GraphValueId id: lowered.model_inputs()) {
        AM_RETURN_IF_ERROR(ValidateValueId(lowered, id, "model input"));
        if (!std::holds_alternative<ModelInputValue>(lowered.values()[id.index].payload)) {
            return Status::Internal(
                    "ValidateLoweredGraph: model input does not carry ModelInputValue");
        }
    }

    for (const GraphValueId id: lowered.model_outputs()) {
        AM_RETURN_IF_ERROR(ValidateValueId(lowered, id, "model output"));
    }

    std::set<std::tuple<size_t, uint32_t, uint32_t>> pairs;
    std::set<std::pair<size_t, uint32_t>> inputs;
    std::set<std::pair<size_t, uint32_t>> outputs;
    for (size_t index = 0; index < lowered.state_aliases().size(); ++index) {
        AM_RETURN_IF_ERROR(ValidateStateAlias(
                lowered, lowered.state_aliases()[index], index, pairs, inputs, outputs));
    }
    return Status::Ok();
}

StatusOr<LoweredGraph> LowerModelGraph(const ModelGraph& graph,
                                       const GraphLoweringConfig& config) {
    const auto order = graph.ValidateAndTopologicalOrder();
    AM_RETURN_IF_ERROR(order.status());

    compiler_internal::LoweredGraphDraft lowered;
    lowered.steps.reserve(order->size());
    lowered.model_inputs.reserve(graph.GetInputs().size());
    lowered.model_outputs.reserve(graph.GetOutputs().size());

    const auto values = graph.GetValues();
    lowered.values.reserve(values.size());
    for (const auto& value: values) {
        lowered.values.push_back({
                .spec = value.spec,
                .payload = value.payload,
                .quantization = value.quantization,
                .name = value.name,
        });
    }

    for (const GraphInput& input: graph.GetInputs()) {
        lowered.model_inputs.push_back(input.value);
    }

    for (const GraphOutput& output: graph.GetOutputs()) {
        lowered.model_outputs.push_back(output.value);
    }

    for (const auto node_id: *order) {
        const auto& node = graph.GetNode(node_id);
        const auto schema = GetOperatorSchema(node.op_type);
        if (!schema.ok()) {
            return Status::Internal(
                    "LowerModelGraph: validated graph has no registered operator schema");
        }

        LoweredNodeSpec spec{
                .op_type = node.op_type,
                .selector = config.selector,
                .op_params = node.op_params,
        };
        LoweredStepBinding binding{.node = node_id};
        binding.input_values.reserve(node.inputs.size());
        binding.output_values.reserve(node.outputs.size());
        spec.input_specs.reserve(schema->input_ports.size());
        spec.output_specs.reserve(schema->output_ports.size());

        std::optional<DataType> act_dtype;
        std::optional<DataType> weight_dtype;
        for (size_t i = 0; i < schema->input_ports.size(); ++i) {
            const auto value_id = node.inputs[i];
            const auto& value = values[value_id.index];
            binding.input_values.push_back(value_id);
            spec.input_specs.push_back(value.spec);
            AM_RETURN_IF_ERROR(CollectInputSelectorDTypes(
                    schema->input_ports[i], value.spec, act_dtype, weight_dtype));
        }

        for (size_t i = 0; i < schema->output_ports.size(); ++i) {
            const auto value_id = node.outputs[i];
            binding.output_values.push_back(value_id);
            const auto& value = values[value_id.index];
            spec.output_specs.push_back(value.spec);
        }

        AM_RETURN_IF_ERROR(CollectOutputActivationDType(*schema, node, values, act_dtype));
        if (!act_dtype.has_value()) {
            return Status::Internal(
                    "LowerModelGraph: no activation dtype available for selector");
        }

        spec.selector.act_dtype = *act_dtype;
        spec.selector.weight_dtype = weight_dtype.value_or(*act_dtype);
        spec.runtime_checks = node.runtime_checks;

        AM_RETURN_IF_ERROR(AddLoweringTimeStateAliases(
                *schema, node, lowered.steps.size(), lowered.state_aliases));
        lowered.steps.push_back({.spec = std::move(spec), .binding = std::move(binding)});
    }

    return std::move(lowered).Finalize();
}

}// namespace aethermind
