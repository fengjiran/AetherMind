#ifndef AETHERMIND_GRAPH_GRAPH_H
#define AETHERMIND_GRAPH_GRAPH_H

/// @file graph.h
/// @brief Core IR container for the AetherMind computation graph.
///
/// ModelGraph is a directed acyclic graph of operator nodes connected by
/// tensor values. This file defines the graph container API; primitive
/// data types (identifiers, payload kinds, node/value structs) live in
/// graph_types.h.

#include "aethermind/base/status.h"
#include "aethermind/graph/graph_types.h"
#include "macros.h"
#include "utils/logging.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace aethermind {

/// @brief Directed acyclic graph of operators and tensor values.
///
/// Owns node and value storage. Mutations go through the public API,
/// which maintains internal consistency. Validation and topological
/// ordering are available as read-only queries.
class ModelGraph {
public:
    ModelGraph() = default;

    /// @brief Test-only escape hatch that does not initialize inputs/outputs.
    ///
    /// Callers must ensure the graph is otherwise valid.
    ///
    /// @param nodes Pre-built node storage.
    /// @param values Pre-built value storage.
    ModelGraph(std::vector<GraphNode> nodes,
               std::vector<GraphValue> values) noexcept;


    /// @brief Returns all operator nodes owned by the graph.
    ///
    /// @return Span over internal node storage; valid until the next
    ///         mutating operation on the graph.
    AM_NODISCARD std::span<const GraphNode> GetNodes() const noexcept {
        return nodes_;
    }

    /// @brief Returns all tensor values owned by the graph.
    ///
    /// @return Span over internal value storage; valid until the next
    ///         mutating operation on the graph.
    AM_NODISCARD std::span<const GraphValue> GetValues() const noexcept {
        return values_;
    }

    /// @brief Returns all registered external inputs.
    ///
    /// @return Span over internal input storage; valid until the next
    ///         mutating operation on the graph.
    AM_NODISCARD std::span<const GraphInput> GetInputs() const noexcept {
        return inputs_;
    }

    /// @brief Returns all registered graph outputs.
    ///
    /// @return Span over internal output storage; valid until the next
    ///         mutating operation on the graph.
    AM_NODISCARD std::span<const GraphOutput> GetOutputs() const noexcept {
        return outputs_;
    }

    /// @brief Registers an external input tensor.
    ///
    /// @param spec Tensor specification (shape, dtype, etc.).
    /// @param name Optional human-readable name for the input port.
    /// @return Value id of the newly registered input.
    AM_NODISCARD GraphValueId AddInput(TensorSpec spec, std::string name = {});

    /// @brief Registers a model weight tensor.
    ///
    /// @param spec Tensor specification (shape, dtype, etc.).
    /// @param binding Weight binding metadata.
    /// @param name Optional human-readable name for the weight.
    /// @return Value id of the newly registered weight.
    AM_NODISCARD GraphValueId AddWeight(TensorSpec spec, WeightBinding binding,
                                        std::string name = {});

    /// @brief Registers a compile-time constant value.
    ///
    /// @param spec Tensor specification (shape, dtype, etc.).
    /// @param binding Constant binding metadata.
    /// @param name Optional human-readable name for the constant.
    /// @return Value id of the newly registered constant.
    AM_NODISCARD GraphValueId AddConstant(TensorSpec spec, ConstantBinding binding,
                                          std::string name = {});

    /// @brief Registers a persistent state tensor.
    ///
    /// @param spec Tensor specification (shape, dtype, etc.).
    /// @param binding State binding metadata.
    /// @param name Optional human-readable name for the state.
    /// @return Value id of the newly registered state.
    AM_NODISCARD GraphValueId AddState(TensorSpec spec, StateBinding binding,
                                       std::string name = {});

    /// @brief Adds an operator node with the given input and output declarations.
    ///
    /// Validates inputs, schema, params, and output metadata, then calls
    /// InferOperator to derive output TensorSpecs and runtime checks before
    /// any observable mutation. Output payloads supplied as monostate are
    /// normalized to ActivationValue.
    ///
    /// @param op_type Operator type to add.
    /// @param decoder_layer_index Optional decoder layer index for
    ///                            layer-scoped operators.
    /// @param inputs Input value ids consumed by the operator.
    /// @param outputs_desc Output declarations for the operator.
    /// @param op_params Operator-specific parameters.
    /// @param attrs Graph-level attributes.
    /// @param name Optional human-readable name for the node.
    /// @return The new node id and its output value ids on success, or an
    ///         error Status on failure; the graph is unchanged.
    StatusOr<AddedNode> AddNode(OpType op_type,
                                std::optional<uint32_t> decoder_layer_index,
                                std::vector<GraphValueId> inputs,
                                std::vector<NodeOutputDesc> outputs_desc,
                                const OpParams& op_params = std::monostate{},
                                ModelGraphAttrs attrs = {},
                                std::string name = {});

    /// @brief Returns the node at the given id.
    ///
    /// @param id Node id to look up.
    /// @return Reference to the requested node.
    /// @note Aborts via AM_CHECK if `id` is out of range.
    AM_NODISCARD const GraphNode& GetNode(GraphNodeId id) const {
        AM_CHECK(id.index < nodes_.size(), "Invalid GraphNodeId");
        return nodes_[id.index];
    }

    /// @brief Returns the value at the given id.
    ///
    /// @param id Value id to look up.
    /// @return Reference to the requested value.
    /// @note Aborts via AM_CHECK if `id` is out of range.
    AM_NODISCARD const GraphValue& GetValue(GraphValueId id) const {
        AM_CHECK(id.index < values_.size(), "Invalid GraphValueId");
        return values_[id.index];
    }

    /// @brief Designates a value as a graph output.
    ///
    /// The user-facing port name is the referenced GraphValue::name; callers
    /// must set it via AddNode/AddInput/AddWeight/AddConstant/AddState
    /// before invoking MarkOutput.
    ///
    /// @param value Value id to mark as an output.
    void MarkOutput(GraphValueId value) {
        outputs_.push_back({.value = value});
    }

    /// @brief Attaches a semantic quantization scheme to a value.
    ///
    /// Applies to any payload kind (weights, activations, constants). Per
    /// design §15, this only records the model-level scheme; backend packed
    /// weight formats are produced during lowering.
    ///
    /// @param value Target value id.
    /// @param quantization Quantization scheme to attach.
    /// @throws std::out_of_range if `value` is invalid.
    void SetQuantization(GraphValueId value, QuantizationSpec quantization) {
        values_.at(value.index).quantization = quantization;
    }

    /// @brief Returns all nodes whose op_type matches the given value.
    ///
    /// Results are returned in ascending node-index order. Performs a linear
    /// scan; suitable for graph-pass usage where query frequency is low.
    /// Callers that need the node contents can resolve each id via GetNode(id).
    ///
    /// @param op_type Operator type to search for.
    /// @return Node ids matching `op_type`, in ascending index order.
    /// @note Linear scan; avoid calling in hot paths.
    AM_NODISCARD std::vector<GraphNodeId> FindNodesByOpType(OpType op_type) const;

    /// @brief Checks graph invariants.
    ///
    /// Validates valid value ids, schema compliance, producer consistency,
    /// and acyclicity.
    ///
    /// @return Success if all invariants hold, otherwise an error Status.
    Status Validate() const;

    /// @brief Returns nodes in topological order.
    ///
    /// Follows activation edges and produced state edges.
    ///
    /// @return Topologically ordered node ids on success, or an error
    ///         Status if the graph contains a cycle.
    StatusOr<std::vector<GraphNodeId>> TopologicalOrder() const;

    /// @brief Combines Validate() and TopologicalOrder() into a single pass.
    ///
    /// Performs full semantic validation and returns the topological order
    /// on success, avoiding the redundant traversal that results from calling
    /// Validate() followed by TopologicalOrder().
    ///
    /// @return Topologically ordered node ids on success, or an error
    ///         Status if validation fails or the graph contains a cycle.
    StatusOr<std::vector<GraphNodeId>> ValidateAndTopologicalOrder() const;

private:
    std::vector<GraphNode> nodes_{};
    std::vector<GraphValue> values_{};
    std::vector<GraphInput> inputs_{};
    std::vector<GraphOutput> outputs_{};
};

/// @brief Builds a reverse mapping from each value to the nodes that
///        consume it.
///
/// @param graph The graph to analyze.
/// @return Consumer index where each entry corresponds to a value id and
///         contains the ids of nodes that consume it.
StatusOr<std::vector<std::vector<GraphNodeId>>> BuildConsumerIndex(const ModelGraph& graph);

/// @brief Returns the consumers of a value from a pre-built consumer index.
///
/// @param index Consumer index previously built by BuildConsumerIndex.
/// @param value Value id to look up.
/// @return Span of node ids that consume the given value.
AM_NODISCARD std::span<const GraphNodeId> GetConsumers(
        const std::vector<std::vector<GraphNodeId>>& index,
        GraphValueId value);

}// namespace aethermind

#endif
