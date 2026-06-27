#ifndef AETHERMIND_MODEL_GRAPH_MODEL_GRAPH_H
#define AETHERMIND_MODEL_GRAPH_MODEL_GRAPH_H

/// Core IR containers for the AetherMind computation graph.
///
/// ModelGraph is a directed acyclic graph of operator nodes connected by
/// tensor values. This file defines graph-level identifiers, value payload
/// types, node/value structs, and the graph container API.
#include "aethermind/base/status.h"
#include "aethermind/model/formats/hf/hf_model_config.h"
#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/op_type.h"
#include "aethermind/shape_inference/tensor_spec.h"
#include "macros.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace aethermind {

/// Identifies the logical role of a weight tensor in a model architecture.
enum class WeightRole : uint8_t {
    kTokenEmbedding,
    kAttentionQ,
    kAttentionK,
    kAttentionV,
    kAttentionO,
    kMlpGate,
    kMlpUp,
    kMlpDown,
    kInputNorm,
    kPostAttentionNorm,
    kFinalNorm,
    kLmHead,
};

struct WeightBinding {
    std::optional<uint32_t> decoder_layer_index{};
    WeightRole role{};
};

/// Compile-time constant binding: carries a small inline payload or resolves
/// a named external constant. Large constants should be referenced by `name`
/// rather than copied into `inline_data`.
///
/// `inline_data` uses `shared_ptr<const ...>` so that `ModelGraph` copy
/// (e.g. `GraphRewriteSession::Commit`) only bumps a refcount instead of
/// deep-copying the payload. Interning is transparent: callers that do not
/// need inline data leave the pointer null.
struct ConstantBinding {
    std::string name{};
    std::shared_ptr<const std::vector<std::byte>> inline_data{};

    AM_NODISCARD friend bool operator==(const ConstantBinding& lhs,
                                        const ConstantBinding& rhs) noexcept {
        if (lhs.name != rhs.name) return false;
        if (lhs.inline_data == rhs.inline_data) return true;// same pointer or both null
        if (!lhs.inline_data || !rhs.inline_data) return false;
        return *lhs.inline_data == *rhs.inline_data;
    }
};

/// Semantic quantization scheme: describes the model-level quantization
/// (int4/int8, group size, scale dtype, zero-point policy) without
/// prescribing backend-specific packed weight formats.
enum class QuantizationKind : uint8_t {
    kNone,
    kInt8,
    kInt4,
};

struct QuantizationSpec {
    QuantizationKind kind = QuantizationKind::kNone;
    uint32_t group_size = 0;
    DataType scale_dtype{};
    bool has_zero_point = false;

    AM_NODISCARD friend bool operator==(const QuantizationSpec& lhs,
                                        const QuantizationSpec& rhs) noexcept = default;
};

enum class KVCacheSlot : uint8_t {
    kKey,
    kValue,
};

struct KVCacheStateBinding {
    uint32_t decoder_layer_index = 0;
    KVCacheSlot slot = KVCacheSlot::kKey;

    AM_NODISCARD friend constexpr bool operator==(const KVCacheStateBinding& lhs,
                                                  const KVCacheStateBinding& rhs) noexcept = default;
};

struct DecodeStateBinding {
    AM_NODISCARD friend constexpr bool operator==(const DecodeStateBinding& lhs,
                                                  const DecodeStateBinding& rhs) noexcept = default;
};

struct StreamingStateBinding {
    AM_NODISCARD friend constexpr bool operator==(const StreamingStateBinding& lhs,
                                                  const StreamingStateBinding& rhs) noexcept = default;
};

using StateBinding = std::variant<KVCacheStateBinding,
                                  DecodeStateBinding,
                                  StreamingStateBinding>;

struct ModelGraphAttrs {
    std::vector<std::byte> bytes{};
};

struct GraphNodeId {
    uint32_t index = 0;

    AM_NODISCARD friend constexpr bool operator==(GraphNodeId lhs, GraphNodeId rhs) noexcept = default;
};

struct GraphValueId {
    uint32_t index = 0;

    AM_NODISCARD friend constexpr bool operator==(GraphValueId lhs, GraphValueId rhs) noexcept = default;
};

/// Payload tag for values that enter the graph as external inputs (e.g. token ids).
struct ModelInputValue {};

/// Payload tag for values produced by operator nodes during execution.
struct ActivationValue {};

/// Payload tag for model weight values with a logical binding.
struct WeightValue {
    WeightBinding binding{};
};

/// Payload tag for compile-time constant values (e.g. scalar constants,
/// fixed masks, RoPE sin/cos tables). Carries a logical ConstantBinding.
struct ConstantValue {
    ConstantBinding binding{};
};

/// Payload tag for persistent state values that survive across execution steps.
struct StateValue {
    StateBinding binding{};
};

using GraphValuePayload = std::variant<std::monostate,
                                       ModelInputValue,
                                       ActivationValue,
                                       WeightValue,
                                       ConstantValue,
                                       StateValue>;

struct GraphValue {
    GraphValuePayload payload{std::monostate{}};
    TensorSpec spec{};
    std::optional<GraphNodeId> producer{};
    QuantizationSpec quantization{};
    std::string debug_name;
};

struct GraphNode {
    OpType op_type = OpType::kUnknown;
    std::optional<uint32_t> decoder_layer_index{};
    std::vector<GraphValueId> inputs{};
    std::vector<GraphValueId> outputs{};
    ModelGraphAttrs attrs{};
    OpParams op_params{};
    std::string debug_name;
};

/// Directed acyclic graph of operators and tensor values.
///
/// Owns node and value storage. Mutations go through the public API
/// which maintains internal consistency. Validation and topological
/// ordering are available as read-only queries.
class ModelGraph {
public:
    struct Input {
        GraphValueId value{};
        std::string name{};
    };

    struct Output {
        GraphValueId value{};
        std::string name{};
    };

    struct NodeOutputDesc {
        TensorSpec spec{};
        GraphValuePayload payload{std::monostate{}};
        std::string debug_name{};
    };

    struct AddedNode {
        GraphNodeId node{};
        std::vector<GraphValueId> outputs{};
    };

    ModelGraph() = default;
    explicit ModelGraph(HfModelConfig config) noexcept;

    // Test-only escape hatch. Does not initialize inputs_/outputs_;
    // callers must ensure the graph is otherwise valid.
    ModelGraph(HfModelConfig config, std::vector<GraphNode> nodes,
               std::vector<GraphValue> values) noexcept;

    /// Registers an external input tensor and returns its value id.
    AM_NODISCARD GraphValueId AddInput(TensorSpec spec, std::string name);

    /// Registers a model weight tensor and returns its value id.
    AM_NODISCARD GraphValueId AddWeight(TensorSpec spec, WeightBinding binding,
                                        std::string debug_name = "");

    /// Registers a compile-time constant value and returns its value id.
    AM_NODISCARD GraphValueId AddConstant(TensorSpec spec, ConstantBinding binding,
                                          std::string debug_name = "");

    /// Attaches a semantic quantization scheme to a value. Applies to any
    /// payload kind (weights, activations, constants). Per design §15,
    /// this only records the model-level scheme; backend packed weight
    /// formats are produced during lowering.
    void SetQuantization(GraphValueId value, QuantizationSpec quantization);

    /// Registers a persistent state tensor and returns its value id.
    AM_NODISCARD GraphValueId AddState(TensorSpec spec, StateBinding binding,
                                       std::string debug_name = "");

    /// Adds an operator node with the given input and output declarations.
    ///
    /// Output declarations with a monostate payload are implicitly treated
    /// as activation outputs. Returns the new node id and its output value ids.
    /// The caller is responsible for ensuring that input value ids are valid
    /// and that the node schema matches the operator's registered schema.
    AM_NODISCARD AddedNode AddNode(OpType op_type,
                                   std::optional<uint32_t> decoder_layer_index,
                                   std::vector<GraphValueId> inputs,
                                   std::vector<NodeOutputDesc> outputs,
                                   const OpParams& op_params = std::monostate{},
                                   ModelGraphAttrs attrs = {},
                                   std::string debug_name = "");

    /// Designates a value as a graph output with a user-facing name.
    void MarkOutput(GraphValueId value, std::string name);

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
    AM_NODISCARD std::span<const GraphNode> GetNodes() const noexcept;
    AM_NODISCARD std::span<const GraphValue> GetValues() const noexcept;
    AM_NODISCARD std::span<const Input> GetInputs() const noexcept;
    AM_NODISCARD std::span<const Output> GetOutputs() const noexcept;
    AM_NODISCARD const GraphNode& GetNode(GraphNodeId id) const;
    AM_NODISCARD const GraphValue& GetValue(GraphValueId id) const;
    AM_NODISCARD const HfModelConfig& GetConfig() const noexcept;

private:
    HfModelConfig config_{};
    std::vector<GraphNode> nodes_{};
    std::vector<GraphValue> values_{};
    std::vector<Input> inputs_{};
    std::vector<Output> outputs_{};
};

/// Builds a reverse mapping from each value to the nodes that consume it.
AM_NODISCARD StatusOr<std::vector<std::vector<GraphNodeId>>> BuildConsumerIndex(const ModelGraph& graph);

/// Returns the consumers of a value from a pre-built consumer index.
AM_NODISCARD std::span<const GraphNodeId> GetConsumers(
        const std::vector<std::vector<GraphNodeId>>& index,
        GraphValueId value);

}// namespace aethermind

#endif
