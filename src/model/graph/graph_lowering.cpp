#include "aethermind/model/graph/graph_lowering.h"

#include "aethermind/model/graph/operator_schema.h"
#include "aethermind/operators/embedding_op.h"
#include "aethermind/operators/rmsnorm_op.h"

#include <any>
#include <optional>
#include <string_view>
#include <utility>
#include <variant>

namespace aethermind {
namespace {

StatusOr<uint32_t> FindInputPortIndex(const OperatorSchema& schema, std::string_view name) {
    for (const OperatorInputPort& port: schema.input_ports) {
        if (std::string_view(port.name) == name) {
            return port.index;
        }
    }
    return Status::InvalidArgument("Operator schema input port not found during graph lowering");
}

StatusOr<uint32_t> FindOutputPortIndex(const OperatorSchema& schema, std::string_view name) {
    for (const OperatorOutputPort& port: schema.output_ports) {
        if (std::string_view(port.name) == name) {
            return port.index;
        }
    }
    return Status::InvalidArgument("Operator schema output port not found during graph lowering");
}

std::any ToExecutionOpParams(const OpParams& params) {
    return std::visit(
            [](const auto& typed_params) -> std::any {
                using Params = std::decay_t<decltype(typed_params)>;
                if constexpr (std::is_same_v<Params, std::monostate>) {
                    return {};
                } else if constexpr (std::is_same_v<Params, EmbeddingParams>) {
                    return std::any{EmbeddingOp::Params{}};
                } else if constexpr (std::is_same_v<Params, RmsNormParams>) {
                    return std::any{RmsNormOp::Params{.eps = typed_params.eps}};
                } else {
                    return std::any{typed_params};
                }
            },
            params);
}

bool IsActivationDTypePort(OperatorPortKind kind) noexcept {
    return kind == OperatorPortKind::kActivation;
}

void MaybeSetSelectorDTypes(const OperatorInputPort& port,
                            const TensorSpec& spec,
                            std::optional<DataType>& activation_dtype,
                            std::optional<DataType>& weight_dtype) {
    if (!port.contributes_tensor_spec) {
        return;
    }

    if (IsActivationDTypePort(port.kind) && !activation_dtype.has_value()) {
        activation_dtype = spec.dtype;
        return;
    }

    if (port.kind == OperatorPortKind::kWeight && !weight_dtype.has_value()) {
        weight_dtype = spec.dtype;
    }
}

void MaybeSetActivationDTypeFromOutputs(const OperatorSchema& schema,
                                        const GraphNode& node,
                                        std::span<const GraphValue> values,
                                        std::optional<DataType>& activation_dtype) {
    if (activation_dtype.has_value()) {
        return;
    }

    for (const OperatorOutputPort& port: schema.output_ports) {
        if (port.kind != OperatorPortKind::kActivation) {
            continue;
        }
        activation_dtype = values[node.outputs[port.index].index].spec.dtype;
        return;
    }
}

Status AddKVCacheAliases(const OperatorSchema& schema,
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
    AM_RETURN_IF_ERROR(graph.Validate());

    StatusOr<std::vector<GraphNodeId>> order_or = graph.TopologicalOrder();
    AM_RETURN_IF_ERROR(order_or.status());

    LoweredGraph lowered;
    lowered.steps.reserve(order_or->size());
    lowered.step_bindings.reserve(order_or->size());
    lowered.model_inputs.reserve(graph.GetInputs().size());
    lowered.model_outputs.reserve(graph.GetOutputs().size());

    for (const ModelGraph::Input& input: graph.GetInputs()) {
        lowered.model_inputs.push_back(input.value);
    }
    for (const ModelGraph::Output& output: graph.GetOutputs()) {
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
                .op_params = ToExecutionOpParams(node.op_params),
        };

        LoweredStepBinding binding{.node = node_id};
        binding.input_values.reserve(node.inputs.size());
        binding.output_values.reserve(node.outputs.size());

        std::optional<DataType> activation_dtype;
        std::optional<DataType> weight_dtype;
        for (const OperatorInputPort& port: schema.input_ports) {
            const GraphValueId value_id = node.inputs[port.index];
            const GraphValue& value = values[value_id.index];
            binding.input_values.push_back(value_id);
            MaybeSetSelectorDTypes(port, value.spec, activation_dtype, weight_dtype);
            if (port.contributes_tensor_spec) {
                step.input_specs.push_back(value.spec);
            }
        }

        for (const OperatorOutputPort& port: schema.output_ports) {
            binding.output_values.push_back(node.outputs[port.index]);
        }
        MaybeSetActivationDTypeFromOutputs(schema, node, values, activation_dtype);

        step.activation_dtype = activation_dtype.value_or(DataType{});
        step.weight_dtype = weight_dtype.value_or(DataType{});

        AM_RETURN_IF_ERROR(AddKVCacheAliases(schema, node, lowered));
        lowered.steps.push_back(std::move(step));
        lowered.step_bindings.push_back(std::move(binding));
    }

    return lowered;
}

}// namespace aethermind
