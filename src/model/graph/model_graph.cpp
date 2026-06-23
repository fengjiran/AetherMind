#include "aethermind/model/graph/model_graph.h"
#include "aethermind/model/graph/operator_schema.h"
#include "utils/logging.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <string>
#include <utility>

namespace aethermind {
namespace {

bool IsValidNodeId(GraphNodeId id, std::span<const GraphNode> nodes) noexcept {
    return id.index < nodes.size();
}

bool IsValidValueId(GraphValueId id, std::span<const GraphValue> values) noexcept {
    return id.index < values.size();
}

uint32_t NextNodeIndex(const std::vector<GraphNode>& nodes) {
    AM_CHECK(nodes.size() < std::numeric_limits<uint32_t>::max(), "Graph node id space exhausted");
    return static_cast<uint32_t>(nodes.size());
}

uint32_t NextValueIndex(const std::vector<GraphValue>& values) {
    AM_CHECK(values.size() < std::numeric_limits<uint32_t>::max(), "Graph value id space exhausted");
    return static_cast<uint32_t>(values.size());
}

bool PayloadMatchesPort(const GraphValuePayload& payload, OperatorPortKind kind) {
    switch (kind) {
        case OperatorPortKind::kModelInput:
            return std::holds_alternative<ModelInputValue>(payload);
        case OperatorPortKind::kActivation:
            return std::holds_alternative<ActivationValue>(payload);
        case OperatorPortKind::kWeight:
            return std::holds_alternative<WeightValue>(payload);
        case OperatorPortKind::kState:
            return std::holds_alternative<StateValue>(payload);
    }
    return false;
}

Status ValidateStateBinding(const StateBinding& binding) {
    if (binding.logical_id.empty()) {
        return Status::InvalidArgument("State binding logical_id must not be empty");
    }
    if (binding.kind == StateKind::kUnknown) {
        return Status::InvalidArgument("State binding kind must not be unknown");
    }
    if (binding.kind == StateKind::kKvCache && binding.slot.empty()) {
        return Status::InvalidArgument("KV cache state binding slot must not be empty");
    }
    return Status::Ok();
}

bool SameStateFamily(const StateBinding& lhs, const StateBinding& rhs) {
    return lhs.logical_id == rhs.logical_id && lhs.kind == rhs.kind && lhs.slot == rhs.slot;
}

bool CarriesProducerDependency(const GraphValue& value) {
    return std::holds_alternative<ActivationValue>(value.payload) ||
           (std::holds_alternative<StateValue>(value.payload) && value.producer.has_value());
}

bool NodeListsOutput(const GraphNode& node, GraphValueId value) {
    return std::ranges::find(node.outputs, value) != node.outputs.end();
}

bool IsFinitePositive(double value) noexcept {
    return std::isfinite(value) && value > 0.0;
}

template<typename Params>
Status RequireParams(const OpParams& params, const char* message) {
    if (!std::holds_alternative<Params>(params)) {
        return Status::InvalidArgument(message);
    }
    return Status::Ok();
}

Status ValidateRmsNormParams(const OpParams& params) {
    const auto* typed = std::get_if<RmsNormParams>(&params);
    if (typed == nullptr) {
        return Status::InvalidArgument("RmsNorm node requires RmsNormParams");
    }
    if (!std::isfinite(typed->eps) || typed->eps <= 0.0F) {
        return Status::InvalidArgument("RmsNormParams eps must be finite and positive");
    }
    return Status::Ok();
}

Status ValidateRoPEParams(const OpParams& params) {
    const auto* typed = std::get_if<RoPEParams>(&params);
    if (typed == nullptr) {
        return Status::InvalidArgument("RoPE node requires RoPEParams");
    }
    if (typed->head_dim <= 0 || typed->num_attention_heads <= 0 ||
        typed->num_key_value_heads <= 0 || typed->max_position_embeddings <= 0) {
        return Status::InvalidArgument("RoPEParams dimensions must be positive");
    }
    if (!IsFinitePositive(typed->theta)) {
        return Status::InvalidArgument("RoPEParams theta must be finite and positive");
    }
    if (typed->scaling_type == HfRopeScalingType::kUnknown) {
        return Status::InvalidArgument("RoPEParams scaling type must be known");
    }
    if (typed->scaling_type == HfRopeScalingType::kNone) {
        if (typed->scaling_factor.has_value()) {
            return Status::InvalidArgument("RoPEParams default scaling must not set a scaling factor");
        }
        return Status::Ok();
    }
    if (!typed->scaling_factor.has_value() || !IsFinitePositive(*typed->scaling_factor)) {
        return Status::InvalidArgument("RoPEParams scaled modes require a finite positive scaling factor");
    }
    return Status::Ok();
}

Status ValidateOpParams(OpType op_type, const OpParams& params) {
    switch (op_type) {
        case OpType::kEmbedding:
            return RequireParams<EmbeddingParams>(params, "Embedding node requires EmbeddingParams");
        case OpType::kRmsNorm:
            return ValidateRmsNormParams(params);
        case OpType::kLinear:
            return RequireParams<LinearParams>(params, "Linear node requires LinearParams");
        case OpType::kRoPE:
            return ValidateRoPEParams(params);
        case OpType::kMatMul:
            return RequireParams<MatMulParams>(params, "MatMul node requires MatMulParams");
        case OpType::kSoftmax:
            return RequireParams<SoftmaxParams>(params, "Softmax node requires SoftmaxParams");
        case OpType::kAdd:
            return RequireParams<AddParams>(params, "Add node requires AddParams");
        case OpType::kSiluMul:
            return RequireParams<SiluMulParams>(params, "SiluMul node requires SiluMulParams");
        case OpType::kKVCacheUpdate:
            return RequireParams<KVCacheUpdateParams>(params, "KVCacheUpdate node requires KVCacheUpdateParams");
        case OpType::kAttention:
            return RequireParams<AttentionParams>(params, "Attention node requires AttentionParams");
        case OpType::kArgmax:
            return RequireParams<ArgmaxParams>(params, "Argmax node requires ArgmaxParams");
        case OpType::kSilu:
        case OpType::kElementwiseMul:
            return Status::InvalidArgument("Op type is not registered for ModelGraph typed params");
        case OpType::kUnknown:
            return Status::InvalidArgument("Unknown op type cannot have validated graph params");
    }
    return Status::InvalidArgument("Unsupported op type cannot have validated graph params");
}

}// namespace

ModelGraph::ModelGraph(HfModelConfig config) noexcept : config_(std::move(config)) {}

ModelGraph::ModelGraph(HfModelConfig config, std::vector<GraphNode> nodes,
                       std::vector<GraphValue> values) noexcept
    : config_(std::move(config)), nodes_(std::move(nodes)), values_(std::move(values)) {}

GraphValueId ModelGraph::AddInput(TensorSpec spec, std::string name) {
    GraphValueId value{NextValueIndex(values_)};
    values_.push_back(GraphValue{.payload = ModelInputValue{},
                                 .spec = std::move(spec),
                                 .debug_name = name});
    inputs_.push_back(Input{.value = value, .name = std::move(name)});
    return value;
}

GraphValueId ModelGraph::AddWeight(TensorSpec spec, ModelWeightBinding binding, std::string debug_name) {
    GraphValueId value{NextValueIndex(values_)};
    values_.push_back(GraphValue{.payload = WeightValue{.binding = binding},
                                 .spec = std::move(spec),
                                 .debug_name = std::move(debug_name)});
    return value;
}

GraphValueId ModelGraph::AddState(TensorSpec spec, StateBinding binding, std::string debug_name) {
    GraphValueId value{NextValueIndex(values_)};
    StateValue state_value{.binding = std::move(binding)};
    values_.push_back(GraphValue{.payload = std::move(state_value),
                                 .spec = std::move(spec),
                                 .debug_name = std::move(debug_name)});
    return value;
}

ModelGraph::AddedNode ModelGraph::AddNode(OpType op_type,
                                          std::optional<uint32_t> decoder_layer_index,
                                          std::vector<GraphValueId> inputs,
                                          std::vector<NodeOutputDecl> outputs,
                                          OpParams op_params,
                                          ModelGraphAttrs attrs,
                                          std::string debug_name) {
    GraphNodeId node_id{NextNodeIndex(nodes_)};
    std::vector<GraphValueId> output_ids;
    output_ids.reserve(outputs.size());
    for (auto& output: outputs) {
        GraphValuePayload payload = std::holds_alternative<std::monostate>(output.payload)
                                            ? GraphValuePayload{ActivationValue{}}
                                            : output.payload;
        GraphValueId value{NextValueIndex(values_)};
        values_.push_back(GraphValue{
                .payload = payload,
                .spec = std::move(output.spec),
                .producer = node_id,
                .debug_name = std::move(output.debug_name),
        });
        output_ids.push_back(value);
    }

    nodes_.push_back(GraphNode{
            .op_type = op_type,
            .decoder_layer_index = decoder_layer_index,
            .debug_name = std::move(debug_name),
            .inputs = std::move(inputs),
            .outputs = output_ids,
            .attrs = std::move(attrs),
            .op_params = op_params,
    });

    return AddedNode{.node = node_id, .outputs = std::move(output_ids)};
}

void ModelGraph::MarkOutput(GraphValueId value, std::string name) {
    outputs_.push_back(Output{.value = value, .name = std::move(name)});
}

Status ModelGraph::Validate() const {
    // -- Validate graph inputs and outputs --
    for (const auto& input: inputs_) {
        if (!IsValidValueId(input.value, values_)) {
            return Status::InvalidArgument("Graph input references an invalid value id");
        }
    }

    for (const auto& output: outputs_) {
        if (!IsValidValueId(output.value, values_)) {
            return Status::InvalidArgument("Graph output references an invalid value id");
        }

        const GraphValue& value = values_[output.value.index];
        if (!std::holds_alternative<ActivationValue>(value.payload)) {
            return Status::InvalidArgument("Graph output must be an activation value");
        }

        if (!value.producer.has_value()) {
            return Status::InvalidArgument("Graph output must be produced by a node");
        }
    }

    // -- Validate each value: no uninitialized payload, producers are consistent --
    for (size_t value_index = 0; value_index < values_.size(); ++value_index) {
        const GraphValue& value = values_[value_index];
        if (std::holds_alternative<std::monostate>(value.payload)) {
            return Status::InvalidArgument("Graph value has monostate payload");
        }

        if (std::holds_alternative<ActivationValue>(value.payload)) {
            if (!value.producer.has_value() || !IsValidNodeId(*value.producer, nodes_)) {
                return Status::InvalidArgument("Activation value has no valid producer");
            }

            // Verify the producer's output list includes this value.
            if (!NodeListsOutput(nodes_[value.producer->index], GraphValueId{static_cast<uint32_t>(value_index)})) {
                return Status::InvalidArgument("Activation producer does not list produced value");
            }
        } else if (std::holds_alternative<StateValue>(value.payload)) {
            const StateBinding& binding = std::get<StateValue>(value.payload).binding;
            AM_RETURN_IF_ERROR(ValidateStateBinding(binding));

            // State values may be external entry points (no producer) or produced by
            // a state-update node. When produced, the producer must list them.
            if (value.producer.has_value()) {
                if (!IsValidNodeId(*value.producer, nodes_)) {
                    return Status::InvalidArgument("State value has an invalid producer");
                }
                if (!NodeListsOutput(nodes_[value.producer->index], GraphValueId{static_cast<uint32_t>(value_index)})) {
                    return Status::InvalidArgument("State producer does not list produced value");
                }
            }
        } else if (value.producer.has_value()) {
            return Status::InvalidArgument("External graph value must not have a producer");
        }
    }

    // -- Validate each node against the operator schema --
    for (size_t node_index = 0; node_index < nodes_.size(); ++node_index) {
        const GraphNode& node = nodes_[node_index];
        StatusOr<OperatorSchema> schema_or = GetOperatorSchema(node.op_type);
        if (!schema_or.ok()) {
            return schema_or.status();
        }

        AM_RETURN_IF_ERROR(ValidateOpParams(node.op_type, node.op_params));
        if (!node.attrs.bytes.empty()) {
            return Status::InvalidArgument("Registered ModelGraph operators must use typed op params, not attrs");
        }

        const OperatorSchema& schema = *schema_or;
        if (node.inputs.size() != schema.input_ports.size()) {
            return Status::InvalidArgument("Graph node input count does not match operator schema");
        }

        if (node.outputs.size() != schema.output_ports.size()) {
            return Status::InvalidArgument("Graph node output count does not match operator schema");
        }

        for (size_t input_index = 0; input_index < node.inputs.size(); ++input_index) {
            const GraphValueId input = node.inputs[input_index];
            if (!IsValidValueId(input, values_)) {
                return Status::InvalidArgument("Graph node input references an invalid value id");
            }

            if (!PayloadMatchesPort(values_[input.index].payload, schema.input_ports[input_index].kind)) {
                return Status::InvalidArgument("Graph node input payload kind does not match operator schema");
            }
        }

        for (size_t output_index = 0; output_index < node.outputs.size(); ++output_index) {
            const GraphValueId output = node.outputs[output_index];
            if (!IsValidValueId(output, values_)) {
                return Status::InvalidArgument("Graph node output references an invalid value id");
            }

            const GraphValue& output_value = values_[output.index];
            if (!output_value.producer.has_value() || output_value.producer->index != node_index) {
                return Status::InvalidArgument("Graph node output producer mismatch");
            }

            if (!PayloadMatchesPort(output_value.payload, schema.output_ports[output_index].kind)) {
                return Status::InvalidArgument("Graph node output payload kind does not match operator schema");
            }
        }

        for (const GraphValueId output: node.outputs) {
            if (std::ranges::find(node.inputs, output) != node.inputs.end()) {
                return Status::InvalidArgument("Graph node output must not reuse an input value");
            }
        }

        if (node.op_type == OpType::kKVCacheUpdate) {
            const GraphValue& state_input = values_[node.inputs[2].index];
            const GraphValue& state_output = values_[node.outputs[0].index];
            const StateBinding& input_binding = std::get<StateValue>(state_input.payload).binding;
            const StateBinding& output_binding = std::get<StateValue>(state_output.payload).binding;
            if (!SameStateFamily(input_binding, output_binding)) {
                return Status::InvalidArgument("KVCacheUpdate state input and output must share a state family");
            }
        }
    }

    // -- Final pass: verify the graph is acyclic via topological sort --
    if (StatusOr<std::vector<GraphNodeId>> order = TopologicalOrder(); !order.ok()) {
        return order.status();
    }
    return Status::Ok();
}

StatusOr<std::vector<GraphNodeId>> ModelGraph::TopologicalOrder() const {
    // Kahn's algorithm over activation edges. Weight and input edges are
    // excluded because they do not carry execution-order dependencies.
    std::vector<std::vector<GraphNodeId>> outgoing(nodes_.size());
    std::vector<size_t> indegree(nodes_.size(), 0);
    for (size_t node_index = 0; node_index < nodes_.size(); ++node_index) {
        for (const GraphNode& node = nodes_[node_index]; const GraphValueId input: node.inputs) {
            if (!IsValidValueId(input, values_)) {
                return Status::InvalidArgument("TopologicalOrder found invalid input value id");
            }

            const GraphValue& value = values_[input.index];
            if (!CarriesProducerDependency(value)) {
                continue;
            }

            if (!value.producer.has_value() || !IsValidNodeId(*value.producer, nodes_)) {
                return Status::InvalidArgument("TopologicalOrder found activation with invalid producer");
            }

            if (value.producer->index == node_index) {
                return Status::InvalidArgument("TopologicalOrder found self-cycle");
            }
            outgoing[value.producer->index].push_back(GraphNodeId{static_cast<uint32_t>(node_index)});
            ++indegree[node_index];
        }
    }

    std::deque<GraphNodeId> ready;
    for (size_t node_index = 0; node_index < indegree.size(); ++node_index) {
        if (indegree[node_index] == 0) {
            ready.push_back(GraphNodeId{static_cast<uint32_t>(node_index)});
        }
    }

    std::vector<GraphNodeId> order;
    order.reserve(nodes_.size());
    while (!ready.empty()) {
        GraphNodeId node = ready.front();
        ready.pop_front();
        order.push_back(node);
        for (const GraphNodeId consumer: outgoing[node.index]) {
            --indegree[consumer.index];
            if (indegree[consumer.index] == 0) {
                ready.push_back(consumer);
            }
        }
    }

    if (order.size() != nodes_.size()) {
        return Status::InvalidArgument("Graph contains a cycle");
    }
    return order;
}

std::span<const GraphNode> ModelGraph::GetNodes() const noexcept {
    return nodes_;
}

std::span<const GraphValue> ModelGraph::GetValues() const noexcept {
    return values_;
}

std::span<const ModelGraph::Input> ModelGraph::GetInputs() const noexcept {
    return inputs_;
}

std::span<const ModelGraph::Output> ModelGraph::GetOutputs() const noexcept {
    return outputs_;
}

const GraphNode& ModelGraph::GetNode(GraphNodeId id) const {
    AM_CHECK(id.index < nodes_.size(), "Invalid GraphNodeId");
    return nodes_[id.index];
}

const GraphValue& ModelGraph::GetValue(GraphValueId id) const {
    AM_CHECK(id.index < values_.size(), "Invalid GraphValueId");
    return values_[id.index];
}

const HfModelConfig& ModelGraph::GetConfig() const noexcept {
    return config_;
}

// Builds a per-value list of nodes that consume that value as an input.
StatusOr<std::vector<std::vector<GraphNodeId>>> BuildConsumerIndex(const ModelGraph& graph) {
    const std::span<const GraphValue> values = graph.GetValues();
    const std::span<const GraphNode> nodes = graph.GetNodes();
    std::vector<std::vector<GraphNodeId>> index(values.size());
    for (size_t node_index = 0; node_index < nodes.size(); ++node_index) {
        for (const GraphNode& node = nodes[node_index]; const GraphValueId input: node.inputs) {
            if (input.index >= values.size()) {
                return Status::InvalidArgument("Consumer index found invalid input value id");
            }
            index[input.index].push_back(GraphNodeId{static_cast<uint32_t>(node_index)});
        }
    }
    return index;
}

// Returns the consumers of a value from a pre-built index.
// Returns an empty span if the value id is out of range.
std::span<const GraphNodeId> GetConsumers(const std::vector<std::vector<GraphNodeId>>& index,
                                          GraphValueId value) {
    if (value.index >= index.size()) {
        return {};
    }
    return index[value.index];
}

}// namespace aethermind
