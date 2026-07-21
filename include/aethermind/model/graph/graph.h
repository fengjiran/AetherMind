#ifndef AETHERMIND_MODEL_GRAPH_GRAPH_H
#define AETHERMIND_MODEL_GRAPH_GRAPH_H

/// Core IR container for the AetherMind computation graph.
///
/// ModelGraph is a directed acyclic graph of operator nodes connected by
/// tensor values. This file defines the graph container API. Primitive
/// data types (identifiers, payload kinds, node/value structs) live in
/// model_graph_types.h.
#include "aethermind/base/status.h"
#include "aethermind/model/formats/hf/hf_model_config.h"
#include "aethermind/model/graph/graph_types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace aethermind {

/// Directed acyclic graph of operators and tensor values.
///
/// Owns node and value storage. Mutations go through the public API
/// which maintains internal consistency. Validation and topological
/// ordering are available as read-only queries.
class ModelGraph {
public:
    ModelGraph() = default;
    explicit ModelGraph(HfModelConfig config) noexcept;

    // Test-only escape hatch. Does not initialize inputs_/outputs_;
    // callers must ensure the graph is otherwise valid.
    ModelGraph(HfModelConfig config, std::vector<GraphNode> nodes,
               std::vector<GraphValue> values) noexcept;


    AM_NODISCARD std::span<const GraphNode> GetNodes() const noexcept {
        return nodes_;
    }

    AM_NODISCARD std::span<const GraphValue> GetValues() const noexcept {
        return values_;
    }

    AM_NODISCARD std::span<const GraphInput> GetInputs() const noexcept {
        return inputs_;
    }

    AM_NODISCARD std::span<const GraphOutput> GetOutputs() const noexcept {
        return outputs_;
    }

    /// Registers an external input tensor and returns its value id.
    AM_NODISCARD GraphValueId AddInput(TensorSpec spec, std::string name = {});

    /// Registers a model weight tensor and returns its value id.
    AM_NODISCARD GraphValueId AddWeight(TensorSpec spec, WeightBinding binding,
                                        std::string name = {});

    /// Registers a compile-time constant value and returns its value id.
    AM_NODISCARD GraphValueId AddConstant(TensorSpec spec, ConstantBinding binding,
                                          std::string name = {});

    /// Registers a persistent state tensor and returns its value id.
    AM_NODISCARD GraphValueId AddState(TensorSpec spec, StateBinding binding,
                                       std::string name = {});

    /// Adds an operator node with the given input and output declarations.
    ///
    /// Validates inputs, schema, params, and output metadata, then calls
    /// AnalyzeOperator to derive output TensorSpecs and runtime checks before
    /// any observable mutation. Output payloads supplied as monostate are
    /// normalized to ActivationValue.
    ///
    /// On success returns the new node id and its output value ids. On failure
    /// returns an error Status; the graph is unchanged.
    AM_NODISCARD StatusOr<AddedNode> AddNode(OpType op_type,
                                             std::optional<uint32_t> decoder_layer_index,
                                             std::vector<GraphValueId> inputs,
                                             std::vector<NodeOutputDesc> outputs_desc,
                                             const OpParams& op_params = std::monostate{},
                                             ModelGraphAttrs attrs = {},
                                             std::string name = {});

    AM_NODISCARD const GraphNode& GetNode(GraphNodeId id) const {
        AM_CHECK(id.index < nodes_.size(), "Invalid GraphNodeId");
        return nodes_[id.index];
    }

    AM_NODISCARD const GraphValue& GetValue(GraphValueId id) const {
        AM_CHECK(id.index < values_.size(), "Invalid GraphValueId");
        return values_[id.index];
    }

    AM_NODISCARD const HfModelConfig& GetConfig() const noexcept {
        return config_;
    }

    /// Designates a value as a graph output with a user-facing name.
    void MarkOutput(GraphValueId value, std::string name) {
        outputs_.push_back({.value = value, .name = std::move(name)});
    }

    /// Attaches a semantic quantization scheme to a value. Applies to any
    /// payload kind (weights, activations, constants). Per design §15,
    /// this only records the model-level scheme; backend packed weight
    /// formats are produced during lowering.
    void SetQuantization(GraphValueId value, QuantizationSpec quantization) {
        values_.at(value.index).quantization = quantization;
    }

    /// Returns the ids of all nodes whose op_type matches `op_type`, in
    /// ascending node-index order. Performs a linear scan; suitable for
    /// graph-pass usage where query frequency is low. Callers that need the
    /// node contents can resolve each id via GetNode(id).
    AM_NODISCARD std::vector<GraphNodeId> FindNodesByOpType(OpType op_type) const;

    /// Checks graph invariants: valid value ids, schema compliance,
    /// producer consistency, and acyclicity.
    AM_NODISCARD Status Validate() const;

    /// Returns nodes in topological order following activation edges and
    /// produced state edges.
    /// Returns an error if the graph contains a cycle.
    AM_NODISCARD StatusOr<std::vector<GraphNodeId>> TopologicalOrder() const;

    /// Combines Validate() and TopologicalOrder() into a single pass.
    /// Performs full semantic validation and returns the topological order
    /// on success, avoiding the redundant traversal that results from
    /// calling Validate() followed by TopologicalOrder().
    AM_NODISCARD StatusOr<std::vector<GraphNodeId>> ValidateAndTopologicalOrder() const;

private:
    HfModelConfig config_{};
    std::vector<GraphNode> nodes_{};
    std::vector<GraphValue> values_{};
    std::vector<GraphInput> inputs_{};
    std::vector<GraphOutput> outputs_{};
};

/// Builds a reverse mapping from each value to the nodes that consume it.
AM_NODISCARD StatusOr<std::vector<std::vector<GraphNodeId>>> BuildConsumerIndex(const ModelGraph& graph);

/// Returns the consumers of a value from a pre-built consumer index.
AM_NODISCARD std::span<const GraphNodeId> GetConsumers(
        const std::vector<std::vector<GraphNodeId>>& index,
        GraphValueId value);

}// namespace aethermind

#endif
