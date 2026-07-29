#ifndef AETHERMIND_MODEL_GRAPH_OPERATOR_SCHEMA_H
#define AETHERMIND_MODEL_GRAPH_OPERATOR_SCHEMA_H

#include "aethermind/base/status.h"
#include "aethermind/operators/op_type.h"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace aethermind {

/// Classifies the role of an operator port. Used by schema validation to
/// distinguish value kinds that have different producer/spec rules.
enum class OperatorPortKind : uint8_t {
    kModelInput,///< Provided by the model entry point (e.g. token ids, position ids).
    kActivation,///< Produced and consumed within the graph as a tensor activation.
    kWeight,    ///< Static weight tensor bound at model load time.
    kConstant,  ///< Compile-time-foldable constant value.
    kState,     ///< Persistent state carried across steps (e.g. KV cache).
};

/// Describes an input port of an operator. Each port identifies a position
/// in the operator's input list and its expected value kind.
struct OperatorInputPort {
    uint32_t index = 0;
    OperatorPortKind kind = OperatorPortKind::kActivation;
    /// Controls whether this port participates in output tensor spec inference.
    /// Ports with contributes_tensor_spec = false (e.g. state inputs whose
    /// layout is determined by the operator itself, not propagated) are
    /// excluded from spec derivation.
    bool contributes_tensor_spec = true;
    std::string name{};
};

/// Describes an output port of an operator. Each port identifies one of the
/// output values produced by the operator and its value kind.
struct OperatorOutputPort {
    uint32_t index = 0;
    OperatorPortKind kind = OperatorPortKind::kActivation;
    std::string name{};
};

/// Structural and semantic properties of an operator that are independent
/// of the specific op type. These traits guide pass scheduling, value
/// liveness analysis, and compile-time evaluation.
struct OperatorTraits {
    /// True if the operator modifies persistent state (e.g. KV cache update).
    bool has_side_effects = true;
    /// True if repeated invocation with identical inputs produces identical
    /// outputs. Non-deterministic operators (e.g. dropout) are excluded from
    /// some fusion and CSE passes.
    bool deterministic = false;
    /// True if the operator can be constant-folded during graph lowering.
    /// Requires has_side_effects = false and deterministic = true.
    bool compile_time_evaluable = false;
};

/// Associates an OpType with its expected port layout and semantic traits.
/// Schemas are defined statically in operator_schema.cpp and used by graph
/// validation (graph.cpp) to verify node structure at build time.
struct OperatorSchema {
    OpType op_type = OpType::kUnknown;
    std::vector<OperatorInputPort> input_ports{};
    std::vector<OperatorOutputPort> output_ports{};
    OperatorTraits traits{};
};

/// Port name constants for KV cache operators.
/// Schema definitions in operator_schema.cpp and validation logic in
/// model_graph.cpp both reference these constants so that the port name
/// contract is enforced at compile time.
namespace kv_cache_ports {
inline constexpr std::string_view kCacheIn = "k_cache_in";
inline constexpr std::string_view vCacheIn = "v_cache_in";
inline constexpr std::string_view kCacheOut = "k_cache_out";
inline constexpr std::string_view vCacheOut = "v_cache_out";
inline constexpr std::string_view kCache = "k_cache";
inline constexpr std::string_view vCache = "v_cache";
}// namespace kv_cache_ports

/// Finds the index of an input port by name. Returns kInvalidArgument if not found.
AM_NODISCARD StatusOr<uint32_t> FindInputPortIndex(const OperatorSchema& schema,
                                                   std::string_view name) noexcept;

/// Finds the index of an output port by name. Returns kInvalidArgument if not found.
AM_NODISCARD StatusOr<uint32_t> FindOutputPortIndex(const OperatorSchema& schema,
                                                    std::string_view name) noexcept;

AM_NODISCARD bool HasStatefulOutput(const OperatorSchema& schema) noexcept;
AM_NODISCARD bool IsPureOperator(const OperatorSchema& schema) noexcept;
AM_NODISCARD bool IsCompileTimeEvaluable(const OperatorSchema& schema) noexcept;

/// Looks up the schema for `op_type` from the static registry.
/// Returns OK with the schema for a registered op, or kNotFound for an
/// unregistered op type (e.g. OpType::kUnknown).
AM_NODISCARD StatusOr<OperatorSchema> GetOperatorSchema(OpType op_type);
AM_NODISCARD std::span<const OperatorSchema> GetOperatorSchemas() noexcept;

}// namespace aethermind

#endif
