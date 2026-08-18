#include "aethermind/graph/lowering/graph_lowering.h"
#include "aethermind/operators/operator_schema.h"

#include <algorithm>
#include <optional>
#include <string_view>
#include <utility>

namespace aethermind {
namespace {

void MaybeSetSelectorDTypes(const OperatorInputPort& port,
                            const TensorSpec& spec,
                            std::optional<DataType>& act_dtype,
                            std::optional<DataType>& weight_dtype) {
    if (!port.contributes_tensor_spec) {
        return;
    }

    if (port.kind == OperatorPortKind::kActivation && !act_dtype.has_value()) {
        act_dtype = spec.dtype;
        return;
    }

    if (port.kind == OperatorPortKind::kWeight && !weight_dtype.has_value()) {
        weight_dtype = spec.dtype;
    }
}

void MaybeSetActivationDTypeFromOutputs(const OperatorSchema& schema,
                                        const GraphNode& node,
                                        const std::span<const GraphValue> values,
                                        std::optional<DataType>& act_dtype) {
    if (act_dtype.has_value()) {
        return;
    }

    // Lowering stores one activation dtype selector per op. Operator schemas are expected to keep
    // all activation outputs at the same dtype, so the first activation output is representative.
    for (size_t output_index = 0; output_index < schema.output_ports.size(); ++output_index) {
        if (schema.output_ports[output_index].kind != OperatorPortKind::kActivation) {
            continue;
        }
        act_dtype = values[node.outputs[output_index].index].spec.dtype;
        return;
    }
}

Status AddLoweringTimeStateAliases(const OperatorSchema& schema,
                                   const GraphNode& node,
                                   size_t step_index,
                                   LoweredGraph& lowered) {
    // State alias pairs are declared in the operator schema; each pair yields
    // one lowering-time alias record with known step/port coordinates (see
    // LoweredStateAlias). Operators without state aliases produce no records,
    // so new stateful operators need no changes here.
    for (const StateAliasPortPair& pair: schema.state_alias_ports) {
        StatusOr<uint32_t> in = FindInputPortIndex(schema, pair.input_port);
        AM_RETURN_IF_ERROR(in.status());
        StatusOr<uint32_t> out = FindOutputPortIndex(schema, pair.output_port);
        AM_RETURN_IF_ERROR(out.status());

        lowered.state_aliases.push_back(LoweredStateAlias{
                .step_index = step_index,
                .input_port = in.value(),
                .output_port = out.value(),
                .input = node.inputs[in.value()],
                .output = node.outputs[out.value()],
        });
    }
    return Status::Ok();
}

}// namespace

StatusOr<LoweredGraph> LowerModelGraph(const ModelGraph& graph,
                                       const GraphLoweringConfig& config) {
    StatusOr<std::vector<GraphNodeId>> order_or = graph.ValidateAndTopologicalOrder();
    AM_RETURN_IF_ERROR(order_or.status());

    LoweredGraph lowered;
    lowered.steps.reserve(order_or->size());
    lowered.model_inputs.reserve(graph.GetInputs().size());
    lowered.model_outputs.reserve(graph.GetOutputs().size());

    const std::span<const GraphValue> values = graph.GetValues();
    lowered.values.reserve(values.size());
    for (const auto& value: values) {
        lowered.values.push_back(LoweredValueDesc{
                .spec = value.spec,
                .payload = value.payload,
                .quantization = value.quantization,
                .name = value.name,
        });
    }

    for (const auto& input: graph.GetInputs()) {
        lowered.model_inputs.push_back(input.value);
    }

    for (const auto& output: graph.GetOutputs()) {
        lowered.model_outputs.push_back(output.value);
    }

    for (const auto node_id: *order_or) {
        const auto& node = graph.GetNode(node_id);
        auto schema_or = GetOperatorSchema(node.op_type);
        AM_RETURN_IF_ERROR(schema_or.status());
        const auto& schema = *schema_or;

        ExecutionPlanNodeSpec step{
                .op_type = node.op_type,
                .selector = config.selector,
                .op_params = node.op_params,
        };

        LoweredStepBinding binding{.node = node_id};
        binding.input_values.reserve(node.inputs.size());
        binding.output_values.reserve(node.outputs.size());
        step.output_specs.reserve(schema.output_ports.size());

        std::optional<DataType> act_dtype;
        std::optional<DataType> weight_dtype;
        for (size_t input_index = 0; input_index < schema.input_ports.size(); ++input_index) {
            const GraphValueId value_id = node.inputs[input_index];
            const GraphValue& value = values[value_id.index];
            binding.input_values.push_back(value_id);
            MaybeSetSelectorDTypes(
                    schema.input_ports[input_index], value.spec, act_dtype, weight_dtype);
            step.input_specs.push_back(value.spec);
        }

        for (size_t output_index = 0; output_index < schema.output_ports.size(); ++output_index) {
            const GraphValueId value_id = node.outputs[output_index];
            binding.output_values.push_back(value_id);
            step.output_specs.push_back(values[value_id.index].spec);
        }
        step.runtime_checks = node.runtime_checks;
        MaybeSetActivationDTypeFromOutputs(schema, node, values, act_dtype);

        step.selector.act_dtype = act_dtype.value_or(DataType{});
        step.selector.weight_dtype = weight_dtype.value_or(step.selector.act_dtype);

        AM_RETURN_IF_ERROR(AddLoweringTimeStateAliases(
                schema, node, lowered.steps.size(), lowered));
        lowered.steps.push_back(
                LoweredStep{.spec = std::move(step), .binding = std::move(binding)});
    }

    return lowered;
}

StatusOr<StateAliasPlan> ResolveStateAliases(const LoweredGraph& lowered) {
    StateAliasPlan plan;
    plan.aliases.reserve(lowered.state_aliases.size());

    for (const LoweredStateAlias& alias: lowered.state_aliases) {
        if (alias.step_index >= lowered.steps.size()) {
            return Status::InvalidArgument(
                    "ResolveStateAliases: alias step index out of range");
        }
        const LoweredStepBinding& binding = lowered.steps[alias.step_index].binding;
        if (alias.input_port >= binding.input_values.size() ||
            binding.input_values[alias.input_port] != alias.input) {
            return Status::InvalidArgument(
                    "ResolveStateAliases: aliased input GraphValueId not at "
                    "recorded input port");
        }
        if (alias.output_port >= binding.output_values.size() ||
            binding.output_values[alias.output_port] != alias.output) {
            return Status::InvalidArgument(
                    "ResolveStateAliases: aliased output GraphValueId not at "
                    "recorded output port");
        }
        plan.aliases.push_back(
                {.step_index = alias.step_index,
                 .input_port = alias.input_port,
                 .output_port = alias.output_port});
    }

    // Lowering collects aliases in topological step order, so the plan is
    // already sorted; sort defensively so ForStep()'s binary-search contract
    // holds for manually constructed plans as well.
    std::ranges::sort(plan.aliases,
                      [](const ResolvedStateAlias& a, const ResolvedStateAlias& b) noexcept {
                          return a.step_index < b.step_index;
                      });

    return plan;
}

}// namespace aethermind
