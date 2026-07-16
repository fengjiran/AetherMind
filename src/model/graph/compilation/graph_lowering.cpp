#include "aethermind/model/graph/compilation/graph_lowering.h"
#include "aethermind/model/graph/operator_schema.h"
#include "utils/variant_utils.h"

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
    for (const auto& port: schema.output_ports) {
        if (port.kind != OperatorPortKind::kActivation) {
            continue;
        }
        act_dtype = values[node.outputs[port.index].index].spec.dtype;
        return;
    }
}

Status AddKVCacheLoweringTimeAliases(const OperatorSchema& schema,
                                     const GraphNode& node,
                                     LoweredGraph& lowered) {
    if (node.op_type != OpType::kKVCacheUpdate) {
        return Status::Ok();
    }

    StatusOr<uint32_t> k_in = FindInputPortIndex(schema, kv_cache_ports::kCacheIn);
    AM_RETURN_IF_ERROR(k_in.status());
    StatusOr<uint32_t> v_in = FindInputPortIndex(schema, kv_cache_ports::vCacheIn);
    AM_RETURN_IF_ERROR(v_in.status());
    StatusOr<uint32_t> k_out = FindOutputPortIndex(schema, kv_cache_ports::kCacheOut);
    AM_RETURN_IF_ERROR(k_out.status());
    StatusOr<uint32_t> v_out = FindOutputPortIndex(schema, kv_cache_ports::vCacheOut);
    AM_RETURN_IF_ERROR(v_out.status());

    lowered.state_aliases.push_back(LoweredStateAlias{
            .input = node.inputs[k_in.value()],
            .output = node.outputs[k_out.value()],
    });

    lowered.state_aliases.push_back(LoweredStateAlias{
            .input = node.inputs[v_in.value()],
            .output = node.outputs[v_out.value()],
    });
    return Status::Ok();
}

}// namespace

StatusOr<LoweredGraph> LowerModelGraph(const ModelGraph& graph,
                                       const GraphLoweringConfig& config) {
    StatusOr<std::vector<GraphNodeId>> order_or = graph.ValidateAndTopologicalOrder();
    AM_RETURN_IF_ERROR(order_or.status());

    LoweredGraph lowered;
    lowered.steps.reserve(order_or->size());
    lowered.step_bindings.reserve(order_or->size());
    lowered.model_inputs.reserve(graph.GetInputs().size());
    lowered.model_outputs.reserve(graph.GetOutputs().size());

    for (const auto& input: graph.GetInputs()) {
        lowered.model_inputs.push_back(input.value);
    }

    for (const auto& output: graph.GetOutputs()) {
        lowered.model_outputs.push_back(output.value);
    }

    const std::span<const GraphValue> values = graph.GetValues();
    for (const GraphNodeId node_id: *order_or) {
        const GraphNode& node = graph.GetNode(node_id);
        StatusOr<OperatorSchema> schema_or = GetOperatorSchema(node.op_type);
        AM_RETURN_IF_ERROR(schema_or.status());
        const OperatorSchema& schema = *schema_or;

        ExecutionPlanNodeSpec step{
                .op_type = node.op_type,
                .device_type = config.device_type,
                .weight_format = config.weight_format,
                .isa = config.isa,
                .phase = config.phase,
                .attrs = node.attrs.bytes,
                .op_params = node.op_params,
        };

        LoweredStepBinding binding{.node = node_id};
        binding.input_values.reserve(node.inputs.size());
        binding.output_values.reserve(node.outputs.size());
        step.output_specs.reserve(schema.output_ports.size());

        std::optional<DataType> act_dtype;
        std::optional<DataType> weight_dtype;
        for (const auto& port: schema.input_ports) {
            const GraphValueId value_id = node.inputs[port.index];
            const GraphValue& value = values[value_id.index];
            binding.input_values.push_back(value_id);
            MaybeSetSelectorDTypes(port, value.spec, act_dtype, weight_dtype);
            if (port.contributes_tensor_spec) {
                step.input_specs.push_back(value.spec);
            }
            // Record ConstantValue payloads so backend lowering can resolve
            // inline data or named external constants without revisiting the
            // graph. Other payload kinds (weight/state/input/activation) are
            // resolved through their bindings at execution-planning time.
            std::visit(overloaded{
                               [&](const ConstantValue& cv) {
                                   binding.constant_bindings.push_back(
                                           LoweredConstantBinding{
                                                   .input_port = port.index,
                                                   .binding = cv.binding,
                                           });
                               },
                               [](const auto&) {},
                       },
                       value.payload);
        }

        for (const auto& port: schema.output_ports) {
            const GraphValueId value_id = node.outputs[port.index];
            binding.output_values.push_back(value_id);
            step.output_specs.push_back(values[value_id.index].spec);
        }
        MaybeSetActivationDTypeFromOutputs(schema, node, values, act_dtype);

        step.act_dtype = act_dtype.value_or(DataType{});
        step.weight_dtype = weight_dtype.value_or(DataType{});

        AM_RETURN_IF_ERROR(AddKVCacheLoweringTimeAliases(schema, node, lowered));
        lowered.steps.push_back(std::move(step));
        lowered.step_bindings.push_back(std::move(binding));
    }

    return lowered;
}

StatusOr<StateAliasPlan> ResolveStateAliases(const LoweredGraph& lowered) {
    StateAliasPlan plan;
    plan.aliases.reserve(lowered.state_aliases.size());

    for (const LoweredStateAlias& alias: lowered.state_aliases) {
        bool found = false;
        for (size_t s = 0; s < lowered.step_bindings.size(); ++s) {
            const LoweredStepBinding& binding = lowered.step_bindings[s];
            auto input_it =
                    std::ranges::find(binding.input_values, alias.input);
            auto output_it =
                    std::ranges::find(binding.output_values, alias.output);

            if (input_it != binding.input_values.end() && output_it != binding.output_values.end()) {
                plan.aliases.push_back(ResolvedStateAlias{
                        .step_index = s,
                        .input_port = static_cast<uint32_t>(input_it - binding.input_values.begin()),
                        .output_port = static_cast<uint32_t>(output_it - binding.output_values.begin()),
                });
                found = true;
                break;
            }
        }

        if (!found) {
            return Status::InvalidArgument(
                    "ResolveStateAliases: aliased GraphValueId not found in "
                    "any step binding");
        }
    }

    // Sort by step_index so that ForStep() can use binary search.
    std::ranges::sort(plan.aliases,
                      [](const ResolvedStateAlias& a, const ResolvedStateAlias& b) noexcept {
                          return a.step_index < b.step_index;
                      });

    return plan;
}

}// namespace aethermind
