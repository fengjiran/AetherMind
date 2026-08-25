#include "aethermind/compiler/graph_lowering.h"
#include "aethermind/graph/graph.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/operators/operator_schema.h"

#include <algorithm>
#include <string>
#include <utility>

namespace aethermind {
namespace {

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
                .op_params = node.op_params};
        spec.input_specs.reserve(schema->input_ports.size());
        spec.output_specs.reserve(schema->output_ports.size());

        LoweredStepBinding binding{.node = node_id};
        binding.input_values.reserve(node.inputs.size());
        binding.output_values.reserve(node.outputs.size());

        for (size_t i = 0; i < schema->input_ports.size(); ++i) {
            const auto value_id = node.inputs[i];
            const auto& value = values[value_id.index];
            binding.input_values.push_back(value_id);
            spec.input_specs.push_back(value.spec);
        }

        for (size_t i = 0; i < schema->output_ports.size(); ++i) {
            const auto value_id = node.outputs[i];
            binding.output_values.push_back(value_id);
            const auto& value = values[value_id.index];
            spec.output_specs.push_back(value.spec);
        }

        // Selector dtypes come from the shared derivation rule so lowering and
        // the untrusted execution path cannot drift; a failure here is an
        // internal artifact inconsistency, not a caller error.
        const auto selector_dtypes = DeriveSelectorDTypes(
                *schema, spec.input_specs, spec.output_specs);
        if (!selector_dtypes.ok()) {
            return Status::Internal(
                    "LowerModelGraph: cannot derive selector dtypes: " +
                    selector_dtypes.status().message());
        }
        spec.selector.act_dtype = selector_dtypes->act_dtype;
        spec.selector.weight_dtype = selector_dtypes->weight_dtype;
        const bool has_weight_input = std::ranges::any_of(
                schema->input_ports,
                [](const OperatorInputPort& port) {
                    return port.kind == OperatorPortKind::kWeight;
                });
        if (has_weight_input && config.enable_packed_weights) {
            spec.selector.weight_format = WeightFormat::kPacked;
        } else if (!has_weight_input) {
            // Packing describes weight storage only; a weightless step must
            // never claim kPacked (the validated invariant kPacked -> exactly
            // one kWeight input would reject it downstream).
            spec.selector.weight_format = WeightFormat::kPlain;
        }
        spec.runtime_checks = node.runtime_checks;

        AM_RETURN_IF_ERROR(AddLoweringStateAliases(
                *schema, node, lowered.steps.size(), lowered.state_aliases));
        lowered.steps.push_back({.spec = std::move(spec), .binding = std::move(binding)});
    }

    return std::move(lowered).Build();
}

}// namespace aethermind
