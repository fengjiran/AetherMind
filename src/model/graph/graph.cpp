#include "aethermind/model/graph/graph.h"
#include "aethermind/model/graph/operator_schema.h"
#include "aethermind/operators/op_type.h"
#include "aethermind/operators/operator_semantics.h"
#include "utils/logging.h"

#include <algorithm>
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
            return std::holds_alternative<ActivationValue>(payload) ||
                   std::holds_alternative<ConstantValue>(payload);
        case OperatorPortKind::kWeight:
            return std::holds_alternative<WeightValue>(payload);
        case OperatorPortKind::kConstant:
            return std::holds_alternative<ConstantValue>(payload);
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

// Expected ParameterSlot for a weight consumed by op_type.
// Returns nullopt for ops that don't have a weight-specific slot constraint.
std::optional<ParameterSlot> ExpectedWeightSlotForOp(OpType op_type) noexcept {
    switch (op_type) {
        case OpType::kEmbedding:
            return ParameterSlot::kEmbeddingTable;
        case OpType::kRmsNorm:
            return ParameterSlot::kScale;
        case OpType::kLinear:
            return ParameterSlot::kKernel;
        default:
            return std::nullopt;
    }
}

// Whether a TransformerWeightRole is per-layer (requires decoder_layer_index)
// or model-level (forbids it).
bool TransformerRoleRequiresLayer(TransformerWeightRole role) noexcept {
    switch (role) {
        case TransformerWeightRole::kTokenEmbedding:
        case TransformerWeightRole::kFinalNorm:
        case TransformerWeightRole::kLmHead:
            return false;
        case TransformerWeightRole::kInputNorm:
        case TransformerWeightRole::kPostAttentionNorm:
        case TransformerWeightRole::kAttentionQ:
        case TransformerWeightRole::kAttentionK:
        case TransformerWeightRole::kAttentionV:
        case TransformerWeightRole::kAttentionO:
        case TransformerWeightRole::kMlpGate:
        case TransformerWeightRole::kMlpUp:
        case TransformerWeightRole::kMlpDown:
        case TransformerWeightRole::kMoERouter:
            return true;
    }
    return true;
}

// Validates WeightBinding self-consistency: slot vs semantic_role pairing,
// and semantic_role vs decoder_layer_index constraints. When semantic_role is
// monostate (generic computation graph), only the slot field is trusted as-is.
Status ValidateWeightBindingSelfConsistency(const WeightBinding& binding) {
    if (std::holds_alternative<std::monostate>(binding.semantic_role)) {
        return Status::Ok();
    }

    const auto role = std::get<TransformerWeightRole>(binding.semantic_role);
    if (binding.slot != SlotForTransformerRole(role)) {
        return Status::InvalidArgument(
                "WeightBinding slot does not match the semantic_role");
    }

    const bool requires_layer = TransformerRoleRequiresLayer(role);
    if (requires_layer && !binding.decoder_layer_index.has_value()) {
        return Status::InvalidArgument(
                "WeightBinding semantic_role requires decoder_layer_index");
    }

    if (!requires_layer && binding.decoder_layer_index.has_value()) {
        return Status::InvalidArgument(
                "WeightBinding semantic_role must not carry decoder_layer_index");
    }
    return Status::Ok();
}

// Shared KVCacheUpdate / Attention state-binding consistency check.
// Called by AddNode before mutation and by Validate after construction,
// ensuring the same rules apply at both boundaries. Output validation
// covers KVCacheUpdate output state bindings (family, slot, layer).
// output_state_bindings is one entry per output port; ports that are not
// State ports should pass nullopt for position-based alignment.
Status ValidateNodeStateBindings(
        OpType op_type,
        std::optional<uint32_t> decoder_layer_index,
        const OperatorSchema& schema,
        std::span<const GraphValue> values,
        std::span<const GraphValueId> node_inputs,
        std::span<const std::optional<StateBinding>> output_state_bindings) {
    if (op_type == OpType::kKVCacheUpdate) {
        auto port_or = FindInputPortIndex(schema, kv_cache_ports::kCacheIn);
        AM_RETURN_IF_ERROR(port_or.status());
        const uint32_t k_in_idx = *port_or;
        port_or = FindInputPortIndex(schema, kv_cache_ports::vCacheIn);
        AM_RETURN_IF_ERROR(port_or.status());
        const uint32_t v_in_idx = *port_or;
        port_or = FindOutputPortIndex(schema, kv_cache_ports::kCacheOut);
        AM_RETURN_IF_ERROR(port_or.status());
        const uint32_t k_out_idx = *port_or;
        port_or = FindOutputPortIndex(schema, kv_cache_ports::vCacheOut);
        AM_RETURN_IF_ERROR(port_or.status());
        const uint32_t v_out_idx = *port_or;

        const StateBinding& k_in_binding =
                std::get<StateValue>(values[node_inputs[k_in_idx].index].payload).binding;
        AM_RETURN_IF_ERROR(RequireStateSlot(
                k_in_binding, KVCacheSlot::kKey,
                "KVCacheUpdate K state input must use slot k"));
        AM_RETURN_IF_ERROR(RequireStateLayerMatchesNode(
                k_in_binding, decoder_layer_index,
                "KVCacheUpdate K state layer must match the node layer"));

        if (k_out_idx < output_state_bindings.size() && output_state_bindings[k_out_idx].has_value()) {
            const StateBinding& k_out_binding = *output_state_bindings[k_out_idx];
            AM_RETURN_IF_ERROR(RequireStateSlot(
                    k_out_binding, KVCacheSlot::kKey,
                    "KVCacheUpdate K state output must use slot k"));
            if (!SameStateFamily(k_in_binding, k_out_binding)) {
                return Status::InvalidArgument(
                        "KVCacheUpdate K state input and output must share a state family");
            }
        }

        const StateBinding& v_in_binding =
                std::get<StateValue>(values[node_inputs[v_in_idx].index].payload).binding;
        AM_RETURN_IF_ERROR(RequireStateSlot(
                v_in_binding, KVCacheSlot::kValue,
                "KVCacheUpdate V state input must use slot v"));
        AM_RETURN_IF_ERROR(RequireStateLayerMatchesNode(
                v_in_binding, decoder_layer_index,
                "KVCacheUpdate V state layer must match the node layer"));
        AM_RETURN_IF_ERROR(RequireStateCollection(
                k_in_binding, v_in_binding,
                "KVCacheUpdate K and V state inputs must share a state collection"));

        if (v_out_idx < output_state_bindings.size() && output_state_bindings[v_out_idx].has_value()) {
            const StateBinding& v_out_binding = *output_state_bindings[v_out_idx];
            AM_RETURN_IF_ERROR(RequireStateSlot(
                    v_out_binding, KVCacheSlot::kValue,
                    "KVCacheUpdate V state output must use slot v"));
            if (!SameStateFamily(v_in_binding, v_out_binding)) {
                return Status::InvalidArgument(
                        "KVCacheUpdate V state input and output must share a state family");
            }
        }
    } else if (op_type == OpType::kAttention) {
        auto port_or = FindInputPortIndex(schema, kv_cache_ports::kCache);
        AM_RETURN_IF_ERROR(port_or.status());
        const uint32_t k_cache_idx = *port_or;
        port_or = FindInputPortIndex(schema, kv_cache_ports::vCache);
        AM_RETURN_IF_ERROR(port_or.status());
        const uint32_t v_cache_idx = *port_or;

        const StateBinding& k_cache_binding =
                std::get<StateValue>(values[node_inputs[k_cache_idx].index].payload).binding;
        const StateBinding& v_cache_binding =
                std::get<StateValue>(values[node_inputs[v_cache_idx].index].payload).binding;
        AM_RETURN_IF_ERROR(RequireStateSlot(
                k_cache_binding, KVCacheSlot::kKey,
                "Attention K cache input must use slot k"));
        AM_RETURN_IF_ERROR(RequireStateSlot(
                v_cache_binding, KVCacheSlot::kValue,
                "Attention V cache input must use slot v"));
        AM_RETURN_IF_ERROR(RequireStateLayerMatchesNode(
                k_cache_binding, decoder_layer_index,
                "Attention K cache layer must match the node layer"));
        AM_RETURN_IF_ERROR(RequireStateLayerMatchesNode(
                v_cache_binding, decoder_layer_index,
                "Attention V cache layer must match the node layer"));
        AM_RETURN_IF_ERROR(RequireStateCollection(
                k_cache_binding, v_cache_binding,
                "Attention K and V cache inputs must share a state collection"));
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

// Compares ShapeConstraint condition and error_context exactly.
// Does NOT use global operator== (which intentionally excludes error_context).
bool ShapeConstraintsEquivalent(const ShapeConstraint& lhs, const ShapeConstraint& rhs) {
    return lhs.condition == rhs.condition && lhs.error_context == rhs.error_context;
}

}// namespace

ModelGraph::ModelGraph(HfModelConfig config) noexcept : config_(std::move(config)) {}

ModelGraph::ModelGraph(HfModelConfig config, std::vector<GraphNode> nodes,
                       std::vector<GraphValue> values) noexcept
    : config_(std::move(config)), nodes_(std::move(nodes)), values_(std::move(values)) {}

GraphValueId ModelGraph::AddInput(TensorSpec spec, std::string name) {
    GraphValueId value_id{NextValueIndex(values_)};
    values_.push_back(GraphValue{.payload = ModelInputValue{},
                                 .spec = std::move(spec),
                                 .debug_name = name});
    inputs_.push_back(GraphInput{.value = value_id, .name = std::move(name)});
    return value_id;
}

GraphValueId ModelGraph::AddWeight(TensorSpec spec, WeightBinding binding, std::string debug_name) {
    GraphValueId value_id{NextValueIndex(values_)};
    values_.push_back(GraphValue{.payload = WeightValue{.binding = binding},
                                 .spec = std::move(spec),
                                 .debug_name = std::move(debug_name)});
    return value_id;
}

GraphValueId ModelGraph::AddConstant(TensorSpec spec, ConstantBinding binding, std::string debug_name) {
    GraphValueId value_id{NextValueIndex(values_)};
    values_.push_back(GraphValue{.payload = ConstantValue{.binding = std::move(binding)},
                                 .spec = std::move(spec),
                                 .debug_name = std::move(debug_name)});
    return value_id;
}

GraphValueId ModelGraph::AddState(TensorSpec spec, StateBinding binding, std::string debug_name) {
    GraphValueId value_id{NextValueIndex(values_)};
    values_.push_back(GraphValue{.payload = StateValue{.binding = binding},
                                 .spec = std::move(spec),
                                 .debug_name = std::move(debug_name)});
    return value_id;
}

StatusOr<AddedNode> ModelGraph::AddNode(OpType op_type,
                                        std::optional<uint32_t> decoder_layer_index,
                                        std::vector<GraphValueId> inputs,
                                        std::vector<NodeOutputDesc> outputs_desc,
                                        const OpParams& op_params,
                                        ModelGraphAttrs attrs,
                                        std::string debug_name) {
    const uint32_t node_idx = NextNodeIndex(nodes_);
    auto ctx = [&](const std::string& detail) {
        return "node " + std::to_string(node_idx) + " (" +
               std::string(ToString(op_type)) +
               (debug_name.empty() ? "" : " " + debug_name) + "): " + detail;
    };

    if (!attrs.bytes.empty()) {
        return Status::InvalidArgument(ctx("must use typed op params, not attrs"));
    }

    auto schema_or = GetOperatorSchema(op_type);
    if (!schema_or.ok()) {
        return Status::InvalidArgument(ctx(schema_or.status().message()));
    }
    const auto& schema = *schema_or;

    {
        auto status = ValidateOperatorParams(op_type, op_params);
        if (!status.ok()) {
            return Status::InvalidArgument(ctx(status.message()));
        }
    }

    if (inputs.size() != schema.input_ports.size()) {
        return Status::InvalidArgument(ctx("input count " + std::to_string(inputs.size()) +
                                           " != schema " + std::to_string(schema.input_ports.size())));
    }
    if (outputs_desc.size() != schema.output_ports.size()) {
        return Status::InvalidArgument(ctx("output count " + std::to_string(outputs_desc.size()) +
                                           " != schema " + std::to_string(schema.output_ports.size())));
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!IsValidValueId(inputs[i], values_)) {
            return Status::InvalidArgument(
                    ctx("input[" + std::to_string(i) + "] " +
                        (i < schema.input_ports.size() ? schema.input_ports[i].name : "<?>") +
                        " invalid value id"));
        }
        if (!PayloadMatchesPort(values_[inputs[i].index].payload, schema.input_ports[i].kind)) {
            return Status::InvalidArgument(
                    ctx("input[" + std::to_string(i) + "] " + schema.input_ports[i].name +
                        " payload kind mismatch"));
        }
    }

    for (size_t i = 0; i < outputs_desc.size(); ++i) {
        auto normalized = std::holds_alternative<std::monostate>(outputs_desc[i].payload)
                                  ? GraphValuePayload{ActivationValue{}}
                                  : outputs_desc[i].payload;
        if (!PayloadMatchesPort(normalized, schema.output_ports[i].kind)) {
            return Status::InvalidArgument(
                    ctx("output[" + std::to_string(i) + "] " + schema.output_ports[i].name +
                        " payload kind mismatch"));
        }
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        if (schema.input_ports[i].kind == OperatorPortKind::kWeight) {
            const auto& binding = std::get<WeightValue>(values_[inputs[i].index].payload).binding;
            if (auto status = ValidateWeightBindingSelfConsistency(binding); !status.ok()) {
                return Status::InvalidArgument(
                        ctx("input[" + std::to_string(i) + "] " + schema.input_ports[i].name +
                            " weight: " + status.message()));
            }
            if (auto exp = ExpectedWeightSlotForOp(op_type);
                exp.has_value() && binding.slot != *exp) {
                return Status::InvalidArgument(
                        ctx("input[" + std::to_string(i) + "] " + schema.input_ports[i].name +
                            " weight slot mismatch"));
            }
        }
        if (schema.input_ports[i].kind == OperatorPortKind::kState) {
            const auto& binding = std::get<StateValue>(values_[inputs[i].index].payload).binding;
            if (auto status = ValidateStateBinding(binding); !status.ok()) {
                return Status::InvalidArgument(
                        ctx("input[" + std::to_string(i) + "] " + schema.input_ports[i].name +
                            " state: " + status.message()));
            }
        }
    }

    // Validate state-binding consistency for ops with state I/O
    // (KVCacheUpdate / Attention slot/family/layer rules).
    // Output state bindings are extracted from outputs_desc before
    // output values are materialized.
    {
        std::vector<std::optional<StateBinding>> output_bindings;
        output_bindings.reserve(outputs_desc.size());
        for (const auto& desc: outputs_desc) {
            if (const auto* sv = std::get_if<StateValue>(&desc.payload)) {
                output_bindings.push_back(sv->binding);
            } else {
                output_bindings.push_back(std::nullopt);
            }
        }
        auto status = ValidateNodeStateBindings(
                op_type, decoder_layer_index, schema, values_, inputs, output_bindings);
        if (!status.ok()) {
            return Status::InvalidArgument(ctx(status.message()));
        }
    }

    std::vector<TensorSpec> all_input_specs;
    all_input_specs.reserve(inputs.size());
    for (const auto& id: inputs) {
        all_input_specs.push_back(values_[id.index].spec);
    }

    auto inference_or = AnalyzeOperator(op_type, op_params, all_input_specs);
    if (!inference_or.ok()) {
        return Status::InvalidArgument(ctx("AnalyzeOperator: " + inference_or.status().message()));
    }
    const auto& inference = *inference_or;

    if (inference.outputs.size() != schema.output_ports.size()) {
        return Status::InvalidArgument(
                ctx("AnalyzeOperator inferred " + std::to_string(inference.outputs.size()) +
                    " outputs, schema expects " + std::to_string(schema.output_ports.size())));
    }

    // Stage all values and node before any observable mutation.
    uint32_t base_value_idx = NextValueIndex(values_);
    std::vector<GraphValueId> output_ids;
    output_ids.reserve(outputs_desc.size());
    for (size_t i = 0; i < outputs_desc.size(); ++i) {
        uint64_t idx64 = static_cast<uint64_t>(base_value_idx) + i;
        AM_CHECK(idx64 < std::numeric_limits<uint32_t>::max(), "Graph value id space exhausted");
        output_ids.push_back(GraphValueId{static_cast<uint32_t>(idx64)});
    }

    GraphNodeId node_id{node_idx};
    std::vector<GraphValue> staged_values;
    staged_values.reserve(outputs_desc.size());
    for (size_t i = 0; i < outputs_desc.size(); ++i) {
        auto payload = std::holds_alternative<std::monostate>(outputs_desc[i].payload)
                               ? GraphValuePayload{ActivationValue{}}
                               : outputs_desc[i].payload;
        staged_values.push_back(GraphValue{
                .payload = payload,
                .spec = inference.outputs[i],
                .producer = node_id,
                .quantization = outputs_desc[i].quantization,
                .debug_name = std::move(outputs_desc[i].debug_name),
        });
    }

    values_.reserve(values_.size() + staged_values.size());
    nodes_.reserve(nodes_.size() + 1);

    for (auto& val: staged_values) {
        values_.push_back(std::move(val));
    }
    nodes_.push_back(GraphNode{
            .op_type = op_type,
            .decoder_layer_index = decoder_layer_index,
            .inputs = std::move(inputs),
            .outputs = output_ids,
            .attrs = std::move(attrs),
            .op_params = op_params,
            .debug_name = std::move(debug_name),
            .runtime_checks = inference.runtime_checks,
    });

    return AddedNode{.node = node_id, .outputs = std::move(output_ids)};
}

std::vector<GraphNodeId> ModelGraph::FindNodesByOpType(OpType op_type) const {
    std::vector<GraphNodeId> result;
    for (uint32_t i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i].op_type == op_type) {
            result.emplace_back(i);
        }
    }
    return result;
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
        if (!std::holds_alternative<ActivationValue>(value.payload) &&
            !std::holds_alternative<ConstantValue>(value.payload)) {
            return Status::InvalidArgument(
                    "Graph output must be an activation or constant value");
        }

        if (std::holds_alternative<ActivationValue>(value.payload) && !value.producer.has_value()) {
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

            if (!NodeListsOutput(nodes_[value.producer->index],
                                 GraphValueId{static_cast<uint32_t>(value_index)})) {
                return Status::InvalidArgument("Activation producer does not list produced value");
            }
        } else if (std::holds_alternative<StateValue>(value.payload)) {
            const StateBinding& binding = std::get<StateValue>(value.payload).binding;
            AM_RETURN_IF_ERROR(ValidateStateBinding(binding));

            if (value.producer.has_value()) {
                if (!IsValidNodeId(*value.producer, nodes_)) {
                    return Status::InvalidArgument("State value has an invalid producer");
                }

                if (!NodeListsOutput(nodes_[value.producer->index],
                                     GraphValueId{static_cast<uint32_t>(value_index)})) {
                    return Status::InvalidArgument(
                            "State producer does not list produced value");
                }
            }
        } else if (std::holds_alternative<ConstantValue>(value.payload)) {
            if (value.producer.has_value()) {
                return Status::InvalidArgument("Constant value must not have a producer");
            }
        } else if (std::holds_alternative<WeightValue>(value.payload)) {
            if (value.producer.has_value()) {
                return Status::InvalidArgument("Weight value must not have a producer");
            }
            const WeightBinding& binding = std::get<WeightValue>(value.payload).binding;
            AM_RETURN_IF_ERROR(ValidateWeightBindingSelfConsistency(binding));
        } else if (value.producer.has_value()) {
            return Status::InvalidArgument(
                    "External graph value must not have a producer");
        }
    }

    // -- Validate each node against the operator schema --
    for (size_t node_index = 0; node_index < nodes_.size(); ++node_index) {
        const GraphNode& node = nodes_[node_index];

        // Stable node-context formatter — defined before the first check so
        // every node-specific diagnostic carries the stored node identity.
        auto n_ctx = [&](const std::string& detail) {
            return "node " + std::to_string(node_index) + " (" +
                   std::string(ToString(node.op_type)) +
                   (node.debug_name.empty() ? "" : " " + node.debug_name) + "): " + detail;
        };

        StatusOr<OperatorSchema> schema_or = GetOperatorSchema(node.op_type);
        if (!schema_or.ok()) {
            return Status::InvalidArgument(
                    n_ctx("GetOperatorSchema: " + schema_or.status().message()));
        }

        {
            auto status = ValidateOperatorParams(node.op_type, node.op_params);
            if (!status.ok()) {
                return Status::InvalidArgument(n_ctx(status.message()));
            }
        }
        if (!node.attrs.bytes.empty()) {
            return Status::InvalidArgument(
                    n_ctx("registered ModelGraph operators must use typed op params, not attrs"));
        }

        const OperatorSchema& schema = *schema_or;

        if (node.inputs.size() != schema.input_ports.size()) {
            return Status::InvalidArgument(
                    n_ctx("input count " + std::to_string(node.inputs.size()) +
                          " != schema " + std::to_string(schema.input_ports.size())));
        }

        if (node.outputs.size() != schema.output_ports.size()) {
            return Status::InvalidArgument(
                    n_ctx("output count " + std::to_string(node.outputs.size()) +
                          " != schema " + std::to_string(schema.output_ports.size())));
        }

        for (size_t input_index = 0; input_index < node.inputs.size(); ++input_index) {
            const GraphValueId input = node.inputs[input_index];
            if (!IsValidValueId(input, values_)) {
                return Status::InvalidArgument(
                        n_ctx("input[" + std::to_string(input_index) + "] " +
                              schema.input_ports[input_index].name + " invalid value id"));
            }

            if (!PayloadMatchesPort(values_[input.index].payload,
                                    schema.input_ports[input_index].kind)) {
                return Status::InvalidArgument(
                        n_ctx("input[" + std::to_string(input_index) + "] " +
                              schema.input_ports[input_index].name + " payload kind mismatch"));
            }

            if (schema.input_ports[input_index].kind == OperatorPortKind::kWeight) {
                const auto& binding = std::get<WeightValue>(values_[input.index].payload).binding;
                if (const auto& expected_slot = ExpectedWeightSlotForOp(node.op_type);
                    expected_slot.has_value() && binding.slot != *expected_slot) {
                    return Status::InvalidArgument(
                            n_ctx("input[" + std::to_string(input_index) + "] " +
                                  schema.input_ports[input_index].name + " weight slot mismatch"));
                }
            }
        }

        {
            std::vector<TensorSpec> all_input_specs;
            all_input_specs.reserve(node.inputs.size());
            for (const auto& id: node.inputs) {
                all_input_specs.push_back(values_[id.index].spec);
            }

            auto inference_or = AnalyzeOperator(node.op_type, node.op_params, all_input_specs);
            if (!inference_or.ok()) {
                return Status::InvalidArgument(
                        n_ctx("semantic re-analysis failed: " + inference_or.status().message()));
            }
            const auto& derived = *inference_or;

            if (derived.outputs.size() != node.outputs.size()) {
                return Status::InvalidArgument(
                        n_ctx("output count mismatch: derived " +
                              std::to_string(derived.outputs.size()) + " != stored " +
                              std::to_string(node.outputs.size())));
            }

            for (size_t oi = 0; oi < node.outputs.size(); ++oi) {
                const GraphValueId output = node.outputs[oi];
                if (!IsValidValueId(output, values_)) {
                    return Status::InvalidArgument(
                            n_ctx("output[" + std::to_string(oi) + "] " +
                                  schema.output_ports[oi].name + " invalid value id"));
                }

                const GraphValue& output_value = values_[output.index];
                if (!output_value.producer.has_value() || output_value.producer->index != node_index) {
                    return Status::InvalidArgument(
                            n_ctx("output[" + std::to_string(oi) + "] " +
                                  schema.output_ports[oi].name + " producer mismatch"));
                }

                if (!PayloadMatchesPort(output_value.payload, schema.output_ports[oi].kind)) {
                    return Status::InvalidArgument(
                            n_ctx("output[" + std::to_string(oi) + "] " +
                                  schema.output_ports[oi].name + " payload kind mismatch"));
                }

                if (!(output_value.spec == derived.outputs[oi])) {
                    return Status::InvalidArgument(
                            n_ctx("output[" + std::to_string(oi) + "] " +
                                  schema.output_ports[oi].name + " spec mismatch"));
                }
            }

            if (derived.runtime_checks.size() != node.runtime_checks.size()) {
                return Status::InvalidArgument(
                        n_ctx("runtime check count mismatch: derived " +
                              std::to_string(derived.runtime_checks.size()) + " != stored " +
                              std::to_string(node.runtime_checks.size())));
            }

            for (size_t ci = 0; ci < node.runtime_checks.size(); ++ci) {
                if (!ShapeConstraintsEquivalent(derived.runtime_checks[ci],
                                                node.runtime_checks[ci])) {
                    return Status::InvalidArgument(
                            n_ctx("stale runtime check [" + std::to_string(ci) + "] derived={" +
                                  derived.runtime_checks[ci].error_context + "} stored={" +
                                  node.runtime_checks[ci].error_context + "}"));
                }
            }
        }

        for (const GraphValueId output: node.outputs) {
            if (std::ranges::find(node.inputs, output) != node.inputs.end()) {
                return Status::InvalidArgument(
                        n_ctx("output must not reuse an input value"));
            }
        }

        if (node.op_type == OpType::kKVCacheUpdate || node.op_type == OpType::kAttention) {
            std::vector<std::optional<StateBinding>> output_bindings;
            output_bindings.reserve(node.outputs.size());
            for (const GraphValueId ov: node.outputs) {
                if (const auto* sv = std::get_if<StateValue>(&values_[ov.index].payload)) {
                    output_bindings.push_back(sv->binding);
                } else {
                    output_bindings.push_back(std::nullopt);
                }
            }
            auto status = ValidateNodeStateBindings(
                    node.op_type, node.decoder_layer_index, schema,
                    values_, node.inputs, output_bindings);
            if (!status.ok()) {
                return Status::InvalidArgument(n_ctx(status.message()));
            }
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
