#include "aethermind/compiler/graph_lowering.h"
#include "aethermind/graph/graph.h"
#include "aethermind/operators/operator_schema.h"

#include <optional>
#include <string>
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
                "LowerModelGraph: operator schema has no "
                "contributing activation dtype source");
    }
    return Status::Ok();
}

Status AddLoweringStateAliases(const OperatorSchema& schema,
                               const GraphNode& node,
                               size_t step_index,
                               std::vector<LoweredStateAlias>& aliases) {
    for (const auto& pair: schema.state_alias_ports) {
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

}// namespace

StatusOr<LoweredGraph> LowerModelGraph(const ModelGraph& graph,
                                       const GraphLoweringConfig& config) {
    const auto order = graph.ValidateAndTopologicalOrder();
    AM_RETURN_IF_ERROR(order.status());

    LoweredGraph::Builder lowered;
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

        LoweredStepSpec spec{
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

        AM_RETURN_IF_ERROR(AddLoweringStateAliases(
                *schema, node, lowered.steps.size(), lowered.state_aliases));
        lowered.steps.push_back({.spec = std::move(spec), .binding = std::move(binding)});
    }

    return std::move(lowered).Build();
}

}// namespace aethermind
