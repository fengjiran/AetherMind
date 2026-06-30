#ifndef AETHERMIND_MODEL_GRAPH_GRAPH_TYPES_H
#define AETHERMIND_MODEL_GRAPH_GRAPH_TYPES_H

/// Core IR data types for the AetherMind computation graph.
///
/// This header defines value payload types, node/value structs, and
/// identifiers used across graph construction, rewriting, dumping, and
/// lowering. It intentionally excludes the ModelGraph container class so
/// that downstream consumers that only need the data structures can include
/// this lightweight header without pulling in the full graph API.
#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/op_type.h"
#include "aethermind/shape_inference/tensor_spec.h"
#include "macros.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
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

/// Describes one output of a graph node being constructed via
/// ModelGraph::AddNode. The caller specifies the spec, payload kind,
/// optional quantization scheme, and debug name.
struct NodeOutputDesc {
    TensorSpec spec{};
    GraphValuePayload payload{std::monostate{}};
    QuantizationSpec quantization{};
    std::string debug_name{};
};

/// Result of ModelGraph::AddNode: the assigned node id and its output value ids.
struct AddedNode {
    GraphNodeId node{};
    std::vector<GraphValueId> outputs{};
};

/// Input port binding: a named value that enters the graph from outside.
struct GraphInput {
    GraphValueId value{};
    std::string name{};
};

/// Output port binding: a named value leaves the graph to the runtime.
struct GraphOutput {
    GraphValueId value{};
    std::string name{};
};

} // namespace aethermind

#endif
