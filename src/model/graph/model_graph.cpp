#include "aethermind/model/graph/model_graph.h"
#include "aethermind/model/graph/operator_schema.h"
#include "utils/logging.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
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

const KVCacheStateBinding* AsKVCacheBinding(const StateBinding& binding) noexcept {
    return std::get_if<KVCacheStateBinding>(&binding);
}

Status ValidateStateBinding(const StateBinding& binding) {
    if (const auto* kv_cache = AsKVCacheBinding(binding); kv_cache != nullptr) {
        switch (kv_cache->slot) {
            case KVCacheSlot::kKey:
            case KVCacheSlot::kValue:
                return Status::Ok();
        }
        return Status::InvalidArgument("KV cache state binding slot must be known");
    }
    return Status::Ok();
}

bool SameStateFamily(const StateBinding& lhs, const StateBinding& rhs) {
    return lhs == rhs;
}

bool SameStateCollection(const StateBinding& lhs, const StateBinding& rhs) {
    const auto* lhs_kv_cache = AsKVCacheBinding(lhs);
    const auto* rhs_kv_cache = AsKVCacheBinding(rhs);
    if (lhs_kv_cache == nullptr || rhs_kv_cache == nullptr) {
        return false;
    }
    return lhs_kv_cache->decoder_layer_index == rhs_kv_cache->decoder_layer_index;
}

Status RequireStateCollection(const StateBinding& lhs, const StateBinding& rhs, const char* message) {
    if (!SameStateCollection(lhs, rhs)) {
        return Status::InvalidArgument(message);
    }
    return Status::Ok();
}

Status RequireStateLayerMatchesNode(const StateBinding& binding,
                                    std::optional<uint32_t> decoder_layer_index,
                                    const char* message) {
    const auto* kv_cache = AsKVCacheBinding(binding);
    if (kv_cache == nullptr) {
        return Status::InvalidArgument(message);
    }

    if (!decoder_layer_index.has_value() || kv_cache->decoder_layer_index != *decoder_layer_index) {
        return Status::InvalidArgument(message);
    }
    return Status::Ok();
}

Status RequireStateSlot(const StateBinding& binding, KVCacheSlot expected_slot, const char* message) {
    if (const auto* kv_cache = AsKVCacheBinding(binding);
        kv_cache == nullptr || kv_cache->slot != expected_slot) {
        return Status::InvalidArgument(message);
    }
    return Status::Ok();
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

GraphValueId ModelGraph::AddWeight(TensorSpec spec, WeightBinding binding, std::string debug_name) {
    GraphValueId value{NextValueIndex(values_)};
    values_.push_back(GraphValue{.payload = WeightValue{.binding = binding},
                                 .spec = std::move(spec),
                                 .debug_name = std::move(debug_name)});
    return value;
}

GraphValueId ModelGraph::AddState(TensorSpec spec, StateBinding binding, std::string debug_name) {
    GraphValueId value{NextValueIndex(values_)};
    values_.push_back(GraphValue{.payload = StateValue{.binding = std::move(binding)},
                                 .spec = std::move(spec),
                                 .debug_name = std::move(debug_name)});
    return value;
}

ModelGraph::AddedNode ModelGraph::AddNode(OpType op_type,
                                          std::optional<uint32_t> decoder_layer_index,
                                          std::vector<GraphValueId> inputs,
                                          std::vector<NodeOutputDesc> outputs,
                                          const OpParams& op_params,
                                          ModelGraphAttrs attrs,
                                          std::string debug_name) {
    GraphNodeId node_id{NextNodeIndex(nodes_)};
    std::vector<GraphValueId> output_ids;
    output_ids.reserve(outputs.size());
    for (auto& output: outputs) {
        auto payload = std::holds_alternative<std::monostate>(output.payload)
                               ? GraphValuePayload{ActivationValue{}}
                               : output.payload;
        output_ids.push_back(GraphValueId{NextValueIndex(values_)});
        values_.push_back(GraphValue{
                .payload = payload,
                .spec = std::move(output.spec),
                .producer = node_id,
                .debug_name = std::move(output.debug_name),
        });
    }

    nodes_.push_back(GraphNode{
            .op_type = op_type,
            .decoder_layer_index = decoder_layer_index,
            .inputs = std::move(inputs),
            .outputs = output_ids,
            .attrs = std::move(attrs),
            .op_params = op_params,
            .debug_name = std::move(debug_name)});

    return AddedNode{.node = node_id, .outputs = std::move(output_ids)};
}

void ModelGraph::MarkOutput(GraphValueId value, std::string name) {
    outputs_.push_back(Output{.value = value, .name = std::move(name)});
}

Status ModelGraph::Validate() const {
    return ValidateAndTopologicalOrder().status();
}

StatusOr<std::vector<GraphNodeId>> ModelGraph::ValidateAndTopologicalOrder() const {
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
            if (!NodeListsOutput(nodes_[value.producer->index],
                                 GraphValueId{static_cast<uint32_t>(value_index)})) {
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

                if (!NodeListsOutput(nodes_[value.producer->index],
                                     GraphValueId{static_cast<uint32_t>(value_index)})) {
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
            return Status::InvalidArgument(
                    "Registered ModelGraph operators must use typed op params, not attrs");
        }

        const OperatorSchema& schema = *schema_or;
        if (node.inputs.size() != schema.input_ports.size()) {
            return Status::InvalidArgument(
                    "Graph node input count does not match operator schema");
        }

        if (node.outputs.size() != schema.output_ports.size()) {
            return Status::InvalidArgument(
                    "Graph node output count does not match operator schema");
        }

        for (size_t input_index = 0; input_index < node.inputs.size(); ++input_index) {
            const GraphValueId input = node.inputs[input_index];
            if (!IsValidValueId(input, values_)) {
                return Status::InvalidArgument(
                        "Graph node input references an invalid value id");
            }

            if (!PayloadMatchesPort(values_[input.index].payload,
                                    schema.input_ports[input_index].kind)) {
                return Status::InvalidArgument(
                        "Graph node input payload kind does not match operator schema");
            }
        }

        for (size_t output_index = 0; output_index < node.outputs.size(); ++output_index) {
            const GraphValueId output = node.outputs[output_index];
            if (!IsValidValueId(output, values_)) {
                return Status::InvalidArgument(
                        "Graph node output references an invalid value id");
            }

            const GraphValue& output_value = values_[output.index];
            if (!output_value.producer.has_value() || output_value.producer->index != node_index) {
                return Status::InvalidArgument("Graph node output producer mismatch");
            }

            if (!PayloadMatchesPort(output_value.payload, schema.output_ports[output_index].kind)) {
                return Status::InvalidArgument(
                        "Graph node output payload kind does not match operator schema");
            }
        }

        for (const GraphValueId output: node.outputs) {
            if (std::ranges::find(node.inputs, output) != node.inputs.end()) {
                return Status::InvalidArgument(
                        "Graph node output must not reuse an input value");
            }
        }

        if (node.op_type == OpType::kKVCacheUpdate) {
            StatusOr<uint32_t> k_in_idx_or = FindInputPortIndex(schema, kv_cache_ports::kCacheIn);
            AM_RETURN_IF_ERROR(k_in_idx_or.status());
            const uint32_t k_in_idx = k_in_idx_or.value();

            StatusOr<uint32_t> v_in_idx_or = FindInputPortIndex(schema, kv_cache_ports::vCacheIn);
            AM_RETURN_IF_ERROR(v_in_idx_or.status());
            const uint32_t v_in_idx = v_in_idx_or.value();

            StatusOr<uint32_t> k_out_idx_or = FindOutputPortIndex(schema, kv_cache_ports::kCacheOut);
            AM_RETURN_IF_ERROR(k_out_idx_or.status());
            const uint32_t k_out_idx = k_out_idx_or.value();

            StatusOr<uint32_t> v_out_idx_or = FindOutputPortIndex(schema, kv_cache_ports::vCacheOut);
            AM_RETURN_IF_ERROR(v_out_idx_or.status());
            const uint32_t v_out_idx = v_out_idx_or.value();

            const GraphValue& k_state_input = values_[node.inputs[k_in_idx].index];
            const GraphValue& k_state_output = values_[node.outputs[k_out_idx].index];
            const StateBinding& k_input_binding = std::get<StateValue>(k_state_input.payload).binding;
            const StateBinding& k_output_binding = std::get<StateValue>(k_state_output.payload).binding;
            AM_RETURN_IF_ERROR(RequireStateSlot(
                    k_input_binding,
                    KVCacheSlot::kKey,
                    "KVCacheUpdate K state input must use slot k"));
            AM_RETURN_IF_ERROR(RequireStateSlot(
                    k_output_binding,
                    KVCacheSlot::kKey,
                    "KVCacheUpdate K state output must use slot k"));
            AM_RETURN_IF_ERROR(RequireStateLayerMatchesNode(
                    k_input_binding,
                    node.decoder_layer_index,
                    "KVCacheUpdate K state layer must match the node layer"));
            if (!SameStateFamily(k_input_binding, k_output_binding)) {
                return Status::InvalidArgument(
                        "KVCacheUpdate K state input and output must share a state family");
            }

            const GraphValue& v_state_input = values_[node.inputs[v_in_idx].index];
            const GraphValue& v_state_output = values_[node.outputs[v_out_idx].index];
            const StateBinding& v_input_binding = std::get<StateValue>(v_state_input.payload).binding;
            const StateBinding& v_output_binding = std::get<StateValue>(v_state_output.payload).binding;
            AM_RETURN_IF_ERROR(RequireStateSlot(
                    v_input_binding,
                    KVCacheSlot::kValue,
                    "KVCacheUpdate V state input must use slot v"));
            AM_RETURN_IF_ERROR(RequireStateSlot(
                    v_output_binding,
                    KVCacheSlot::kValue,
                    "KVCacheUpdate V state output must use slot v"));
            AM_RETURN_IF_ERROR(RequireStateLayerMatchesNode(
                    v_input_binding,
                    node.decoder_layer_index,
                    "KVCacheUpdate V state layer must match the node layer"));
            AM_RETURN_IF_ERROR(RequireStateCollection(
                    k_input_binding,
                    v_input_binding,
                    "KVCacheUpdate K and V state inputs must share a state collection"));
            if (!SameStateFamily(v_input_binding, v_output_binding)) {
                return Status::InvalidArgument(
                        "KVCacheUpdate V state input and output must share a state family");
            }
        } else if (node.op_type == OpType::kAttention) {
            StatusOr<uint32_t> k_cache_idx_or = FindInputPortIndex(schema, kv_cache_ports::kCache);
            AM_RETURN_IF_ERROR(k_cache_idx_or.status());
            const uint32_t k_cache_idx = k_cache_idx_or.value();

            StatusOr<uint32_t> v_cache_idx_or = FindInputPortIndex(schema, kv_cache_ports::vCache);
            AM_RETURN_IF_ERROR(v_cache_idx_or.status());
            const uint32_t v_cache_idx = v_cache_idx_or.value();

            const StateBinding& k_cache_binding = std::get<StateValue>(values_[node.inputs[k_cache_idx].index].payload).binding;
            const StateBinding& v_cache_binding = std::get<StateValue>(values_[node.inputs[v_cache_idx].index].payload).binding;
            AM_RETURN_IF_ERROR(RequireStateSlot(
                    k_cache_binding,
                    KVCacheSlot::kKey,
                    "Attention K cache input must use slot k"));
            AM_RETURN_IF_ERROR(RequireStateSlot(
                    v_cache_binding,
                    KVCacheSlot::kValue,
                    "Attention V cache input must use slot v"));
            AM_RETURN_IF_ERROR(RequireStateLayerMatchesNode(
                    k_cache_binding,
                    node.decoder_layer_index,
                    "Attention K cache layer must match the node layer"));
            AM_RETURN_IF_ERROR(RequireStateLayerMatchesNode(
                    v_cache_binding,
                    node.decoder_layer_index,
                    "Attention V cache layer must match the node layer"));
            AM_RETURN_IF_ERROR(RequireStateCollection(
                    k_cache_binding,
                    v_cache_binding,
                    "Attention K and V cache inputs must share a state collection"));
        }
    }

    // -- Final pass: verify the graph is acyclic and return the order --
    return TopologicalOrder();
}

StatusOr<std::vector<GraphNodeId>> ModelGraph::TopologicalOrder() const {
    // Kahn's algorithm over activation and produced-state edges.
    // Weight and input edges are excluded because they do not carry
    // execution-order dependencies.
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
