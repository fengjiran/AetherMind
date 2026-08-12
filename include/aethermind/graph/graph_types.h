#ifndef AETHERMIND_GRAPH_GRAPH_TYPES_H
#define AETHERMIND_GRAPH_GRAPH_TYPES_H

/// @file graph_types.h
/// @brief Core IR data types for the AetherMind computation graph.
///
/// This header defines value payload types, node/value structs, and
/// identifiers used across graph construction, rewriting, dumping, and
/// lowering. It intentionally excludes the ModelGraph container class so
/// that downstream consumers that only need the data structures can include
/// this lightweight header without pulling in the full graph API.
#include "aethermind/operators/op_params.h"
#include "aethermind/operators/op_type.h"
#include "aethermind/shape_inference/shape_constraint.h"
#include "aethermind/shape_inference/tensor_spec.h"
#include "utils/variant_utils.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace aethermind {

/// @brief Describes the parameter's functional position in a low-level operator.
enum class ParameterSlot : uint8_t {
    kKernel,
    kBias,
    kScale,
    kShift,
    kEmbeddingTable,
};

/// @brief Describes the parameter's semantic role in Transformer-family models.
enum class TransformerWeightRole : uint8_t {
    kTokenEmbedding = 0,
    kInputNorm,
    kAttentionQ,
    kAttentionK,
    kAttentionV,
    kAttentionO,
    kMlpGate,
    kMlpUp,
    kMlpDown,
    kPostAttentionNorm,
    kFinalNorm,
    kLmHead,
    kMoERouter,
};

/// @brief Either no role or a TransformerWeightRole.
using ModelSemanticRole = std::variant<std::monostate, TransformerWeightRole>;

/// @brief Returns the ParameterSlot implied by a TransformerWeightRole.
///
/// This is the single source of truth for the slot↔role mapping;
/// direct-weight binding construction and Validate() both rely on it, so callers
/// building a TransformerWeightRole binding need not (and should not)
/// specify the slot separately.
///
/// @param role Transformer weight role to map.
/// @return ParameterSlot corresponding to `role`.
constexpr ParameterSlot SlotForTransformerRole(TransformerWeightRole role) noexcept {
    switch (role) {
        case TransformerWeightRole::kTokenEmbedding:
            return ParameterSlot::kEmbeddingTable;
        case TransformerWeightRole::kInputNorm:
        case TransformerWeightRole::kPostAttentionNorm:
        case TransformerWeightRole::kFinalNorm:
            return ParameterSlot::kScale;
        case TransformerWeightRole::kAttentionQ:
        case TransformerWeightRole::kAttentionK:
        case TransformerWeightRole::kAttentionV:
        case TransformerWeightRole::kAttentionO:
        case TransformerWeightRole::kMlpGate:
        case TransformerWeightRole::kMlpUp:
        case TransformerWeightRole::kMlpDown:
        case TransformerWeightRole::kLmHead:
        case TransformerWeightRole::kMoERouter:
            return ParameterSlot::kKernel;
    }
    return ParameterSlot::kKernel;
}

/// @brief Binding metadata for one directly-addressable model weight.
///
/// `semantic_role` is optional for generic graphs. When it holds a
/// TransformerWeightRole, its slot and layer scope are validated against the
/// Transformer role contract.
struct DirectWeightBinding {
    ParameterSlot slot = ParameterSlot::kKernel;
    ModelSemanticRole semantic_role{};

    AM_NODISCARD friend bool operator==(
            const DirectWeightBinding& lhs,
            const DirectWeightBinding& rhs) noexcept = default;
};

/// @brief Logical QKV kernel recipe.
///
/// The recipe is fixed to concat(AttentionQ, AttentionK, AttentionV, axis=0)
/// in Q, K, V order. Tensor shape belongs to TensorSpec and split boundaries
/// belong to QkvLinearParams, so neither is duplicated here.
struct QkvWeightBinding {
    AM_NODISCARD friend constexpr bool operator==(
            const QkvWeightBinding&,
            const QkvWeightBinding&) noexcept = default;
};

/// @brief Logical Gate-Up kernel recipe.
///
/// The recipe is fixed to concat(MlpGate, MlpUp, axis=0) in Gate, Up order.
/// Tensor shape belongs to TensorSpec and split boundaries belong to the
/// consuming operator parameters.
struct GateUpWeightBinding {
    AM_NODISCARD friend constexpr bool operator==(
            const GateUpWeightBinding&,
            const GateUpWeightBinding&) noexcept = default;
};

/// @brief Variant over direct and fixed-composite logical weight recipes.
using WeightBindingSpec = std::variant<DirectWeightBinding,
                                       QkvWeightBinding,
                                       GateUpWeightBinding>;

/// @brief Logical binding metadata for a weight value.
///
/// Tensor dtype and shape belong to GraphValue::spec. Composite recipes do
/// not duplicate dimensions or backend packing information.
struct WeightBinding {
    std::optional<uint32_t> decoder_layer_index{};
    WeightBindingSpec spec{DirectWeightBinding{}};

    AM_NODISCARD friend bool operator==(const WeightBinding& lhs,
                                        const WeightBinding& rhs) noexcept = default;
};

/// @brief Creates a generic or semantically-tagged direct-weight binding.
AM_NODISCARD constexpr WeightBinding MakeDirectWeightBinding(
        ParameterSlot slot,
        ModelSemanticRole semantic_role = {},
        std::optional<uint32_t> decoder_layer_index = std::nullopt) noexcept {
    return {.decoder_layer_index = decoder_layer_index,
            .spec = DirectWeightBinding{.slot = slot,
                                        .semantic_role = semantic_role}};
}

/// @brief Creates a direct Transformer weight binding with its canonical slot.
AM_NODISCARD constexpr WeightBinding MakeTransformerWeightBinding(
        std::optional<uint32_t> decoder_layer_index,
        TransformerWeightRole role) noexcept {
    return MakeDirectWeightBinding(
            SlotForTransformerRole(role), ModelSemanticRole{role}, decoder_layer_index);
}

/// @brief Creates the fixed Q, K, V kernel recipe for one decoder layer.
AM_NODISCARD constexpr WeightBinding MakeQkvWeightBinding(
        uint32_t decoder_layer_index) noexcept {
    return {.decoder_layer_index = decoder_layer_index,
            .spec = QkvWeightBinding{}};
}

/// @brief Creates the fixed Gate, Up kernel recipe for one decoder layer.
AM_NODISCARD constexpr WeightBinding MakeGateUpWeightBinding(
        uint32_t decoder_layer_index) noexcept {
    return {.decoder_layer_index = decoder_layer_index,
            .spec = GateUpWeightBinding{}};
}

/// @brief Returns the parameter slot implied by a logical weight recipe.
AM_NODISCARD constexpr ParameterSlot GetParameterSlot(
        const WeightBinding& binding) noexcept {
    auto visitor = overloaded{
            [](const DirectWeightBinding& direct) constexpr noexcept {
                return direct.slot;
            },
            [](const QkvWeightBinding&) constexpr noexcept {
                return ParameterSlot::kKernel;
            },
            [](const GateUpWeightBinding&) constexpr noexcept {
                return ParameterSlot::kKernel;
            },
    };
    return std::visit(visitor, binding.spec);
}

/// @brief Returns the direct binding, or null when `binding` is composite.
AM_NODISCARD constexpr const DirectWeightBinding* TryGetDirectWeightBinding(
        const WeightBinding& binding) noexcept {
    return std::get_if<DirectWeightBinding>(&binding.spec);
}

/// @brief Returns the Transformer role of a direct binding when present.
AM_NODISCARD constexpr std::optional<TransformerWeightRole>
TryGetTransformerWeightRole(const WeightBinding& binding) noexcept {
    const DirectWeightBinding* direct = TryGetDirectWeightBinding(binding);
    if (direct == nullptr) {
        return std::nullopt;
    }

    if (const auto* role = std::get_if<TransformerWeightRole>(&direct->semantic_role)) {
        return *role;
    }
    return std::nullopt;
}

/// @brief Returns true when `binding` describes a fixed composite recipe.
AM_NODISCARD constexpr bool IsCompositeWeightBinding(
        const WeightBinding& binding) noexcept {
    return !std::holds_alternative<DirectWeightBinding>(binding.spec);
}

/// @brief Compile-time constant binding: carries a small inline payload or
///        resolves a named external constant.
///
/// Large constants should be referenced by `name` rather than copied into
/// `inline_data`.
///
/// `inline_data` uses `shared_ptr<const ...>` so that `ModelGraph` copy
/// (e.g. `GraphRewriteSession::Commit`) only bumps a refcount instead of
/// deep-copying the payload. Interning is transparent: callers that do not
/// need inline data leave the pointer null.
struct ConstantBinding {
    std::shared_ptr<const std::vector<std::byte>> inline_data{};
    std::string name{};

    AM_NODISCARD friend bool operator==(const ConstantBinding& lhs,
                                        const ConstantBinding& rhs) noexcept {
        if (lhs.name != rhs.name) return false;
        if (lhs.inline_data == rhs.inline_data) return true;// same pointer or both null
        if (!lhs.inline_data || !rhs.inline_data) return false;
        return *lhs.inline_data == *rhs.inline_data;
    }
};

/// @brief Semantic quantization scheme: describes the model-level quantization
///        (int4/int8, group size, scale dtype, zero-point policy) without
///        prescribing backend-specific packed weight formats.
enum class QuantizationKind : uint8_t {
    kNone,
    kInt8,
    kInt4,
};

/// @brief Concrete quantization parameters attached to a value.
struct QuantizationSpec {
    QuantizationKind kind = QuantizationKind::kNone;
    uint32_t group_size = 0;
    DataType scale_dtype{};
    bool has_zero_point = false;

    AM_NODISCARD friend bool operator==(
            const QuantizationSpec& lhs,
            const QuantizationSpec& rhs) noexcept = default;
};

enum class KVCacheSlot : uint8_t {
    kKey,
    kValue,
};

struct KVCacheStateBinding {
    uint32_t decoder_layer_index = 0;
    KVCacheSlot slot = KVCacheSlot::kKey;

    AM_NODISCARD friend constexpr bool operator==(
            const KVCacheStateBinding& lhs,
            const KVCacheStateBinding& rhs) noexcept = default;
};

struct DecodeStateBinding {
    AM_NODISCARD friend constexpr bool operator==(
            const DecodeStateBinding& lhs,
            const DecodeStateBinding& rhs) noexcept = default;
};

struct StreamingStateBinding {
    AM_NODISCARD friend constexpr bool operator==(
            const StreamingStateBinding& lhs,
            const StreamingStateBinding& rhs) noexcept = default;
};

/// @brief Variant over all persistent state binding kinds.
using StateBinding = std::variant<KVCacheStateBinding,
                                  DecodeStateBinding,
                                  StreamingStateBinding>;

/// @brief Opaque byte bag for graph-level and node-level attributes.
struct ModelGraphAttrs {
    std::vector<std::byte> bytes{};
};

struct GraphNodeId {
    uint32_t index = 0;

    AM_NODISCARD friend constexpr bool operator==(
            GraphNodeId lhs, GraphNodeId rhs) noexcept = default;
};

struct GraphValueId {
    uint32_t index = 0;

    AM_NODISCARD friend constexpr bool operator==(
            GraphValueId lhs, GraphValueId rhs) noexcept = default;
};

/// @brief Payload tag for values that enter the graph as external inputs
///        (e.g. token ids).
struct ModelInputValue {};

/// @brief Payload tag for values produced by operator nodes during execution.
struct ActivationValue {};

/// @brief Payload tag for model weight values with a logical binding.
struct WeightValue {
    WeightBinding binding{};
};

/// @brief Payload tag for compile-time constant values (e.g. scalar constants,
///        fixed masks, RoPE sin/cos tables). Carries a logical ConstantBinding.
struct ConstantValue {
    ConstantBinding binding{};
};

/// @brief Payload tag for persistent state values that survive across
///        execution steps.
struct StateValue {
    StateBinding binding{};
};

/// @brief Variant over all graph value payload kinds.
using GraphValuePayload = std::variant<std::monostate,
                                       ModelInputValue,
                                       ActivationValue,
                                       WeightValue,
                                       ConstantValue,
                                       StateValue>;

/// @brief A tensor value in the graph: payload + spec + producer link.
///
/// Producers are set by ModelGraph::AddNode for activations; inputs,
/// weights, constants, and states have no producer.
struct GraphValue {
    GraphValuePayload payload{std::monostate{}};
    TensorSpec spec{};
    std::optional<GraphNodeId> producer{};
    QuantizationSpec quantization{};
    std::string name;
};

/// @brief An operator node in the graph.
///
/// Inputs and outputs reference GraphValue ids. runtime_checks carries
/// deferred shape constraints derived during AddNode and verified by
/// ValidateAndTopologicalOrder.
struct GraphNode {
    OpType op_type = OpType::kUnknown;
    std::optional<uint32_t> decoder_layer_index{};
    std::vector<GraphValueId> inputs{};
    std::vector<GraphValueId> outputs{};
    ModelGraphAttrs attrs{};
    OpParams op_params{};

    /// Runtime shape constraints derived by operator semantic analysis.
    /// Written by ModelGraph::AddNode and verified by ValidateAndTopologicalOrder.
    std::vector<ShapeConstraint> runtime_checks{};
    std::string name;
};

/// @brief Describes one output of a graph node being constructed via
///        ModelGraph::AddNode.
///
/// The caller specifies the payload kind, optional quantization scheme,
/// and debug name.
///
/// Output TensorSpecs are derived by operator semantic analysis
/// (InferOperator) and stored on GraphValue. Runtime shape constraints
/// are stored on GraphNode.runtime_checks.
struct NodeOutputDesc {
    GraphValuePayload payload{std::monostate{}};
    QuantizationSpec quantization{};
    std::string name{};
};

/// @brief Describes an existing graph value with its authoritative TensorSpec.
///
/// Returned by GraphRewriteSession::GetValueOutputMetadata and consumed by
/// ConstEvaluator::Plan — both need the TensorSpec to reason about shapes,
/// dtypes, and byte counts. For new node construction (ModelGraph::AddNode),
/// use NodeOutputDesc (spec is derived by the analyzer, not caller-supplied).
struct GraphValueDesc {
    TensorSpec spec{};
    GraphValuePayload payload{std::monostate{}};
    QuantizationSpec quantization{};
    std::string name{};
};

/// @brief Result of ModelGraph::AddNode: the assigned node id and its
///        output value ids.
struct AddedNode {
    GraphNodeId node{};
    std::vector<GraphValueId> outputs{};
};

/// @brief Result of AddRoPE: the rotated q and k value ids.
struct RoPEOutputs {
    GraphValueId q{};
    GraphValueId k{};
};

/// @brief A pair of (key, value) value ids representing a KV cache slot.
///
/// Returned by AddKVCacheUpdate and threaded through transformer builders as
/// the flowing cache state (attention block input/output, cross-layer cache).
struct KVCachePair {
    GraphValueId k{};
    GraphValueId v{};
};

/// @brief Input port binding: a value that enters the graph from outside.
///
/// The debug name lives on the referenced GraphValue; this struct only
/// carries the value id so that GetInputs() can enumerate the model input
/// ports.
struct GraphInput {
    GraphValueId value{};
};

/// @brief Output port binding: a value exported by the graph.
///
/// The port name is the referenced GraphValue::name; this struct only
/// carries the value id.
struct GraphOutput {
    GraphValueId value{};
};

}// namespace aethermind

#endif
