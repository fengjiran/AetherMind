#ifndef AETHERMIND_OPERATORS_OPERATOR_SCHEMA_H
#define AETHERMIND_OPERATORS_OPERATOR_SCHEMA_H

/// @file operator_schema.h
/// @brief Semantic ABI for operator port order, value roles, and traits.
///
/// Port order and names are part of the graph/operator contract and must
/// remain synchronized with inference and model graph construction. A port's
/// position in its schema vector is its index; there is no separate index
/// field to keep in sync.

#include "aethermind/base/status.h"
#include "aethermind/operators/op_type.h"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace aethermind {

/// @brief Classifies the semantic role of an operator port.
///
/// Used by schema validation to
/// distinguish value kinds that have different producer/spec rules.
enum class OperatorPortKind : uint8_t {
    kModelInput, ///< Provided by the model entry point (e.g. token ids, position ids).
    kActivation, ///< Produced and consumed within the graph as a tensor activation.
    kWeight,     ///< Static weight tensor bound at model load time.
    kConstant,   ///< Compile-time-foldable constant value.
    kState,      ///< Persistent state carried across steps (e.g. KV cache).
};

/// @brief Describes one input position in an operator's semantic ABI.
///
/// The port's vector position is its index: `input_ports[0]` is input 0.
/// Consumers index node inputs by this position, so the declaration order
/// of a schema is the semantic ABI order.
struct OperatorInputPort {
    OperatorPortKind kind = OperatorPortKind::kActivation;
    /// @brief Controls whether this port participates in output tensor-spec inference.
    /// Ports with contributes_tensor_spec = false (e.g. state inputs whose
    /// layout is determined by the operator itself, not propagated) are
    /// excluded from spec derivation.
    bool contributes_tensor_spec = true;
    std::string name{};
};

/// @brief Describes one output position in an operator's semantic ABI.
///
/// The port's vector position is its index: `output_ports[0]` is output 0.
struct OperatorOutputPort {
    OperatorPortKind kind = OperatorPortKind::kActivation;
    std::string name{};
};

/// @brief Structural properties used by graph optimization and validation.
///
/// These properties are independent
/// of the specific op type. These traits guide pass scheduling, value
/// liveness analysis, and compile-time evaluation.
struct OperatorTraits {
    /// @brief Whether the operator modifies persistent state.
    bool has_side_effects = true;
    /// @brief Whether identical inputs always produce identical outputs.
    ///
    /// outputs. Non-deterministic operators (e.g. dropout) are excluded from
    /// some fusion and CSE passes.
    bool deterministic = false;
    /// @brief Whether graph lowering may evaluate the operator at compile time.
    /// Requires has_side_effects = false and deterministic = true.
    bool compile_time_evaluable = false;
};

/// @brief One lowering-time state alias: a state input port and a state output
/// port that must map to the same physical runtime state buffer.
///
/// Ports are referenced by name; the schema is the sole authority for their
/// indexes, resolved via FindInputPortIndex/FindOutputPortIndex at use time.
struct StateAliasPortPair {
    std::string input_port{};
    std::string output_port{};
};

/// @brief Associates an operator type with its port layout and semantic traits.
///
/// Schemas are defined statically in operator_schema.cpp and used by graph
/// validation (graph.cpp) to verify node structure at build time.
struct OperatorSchema {
    OpType op_type = OpType::kUnknown;
    std::vector<OperatorInputPort> input_ports{};
    std::vector<OperatorOutputPort> output_ports{};
    OperatorTraits traits{};
    /// @brief Lowering-time state alias pairs declared by this operator.
    ///
    /// Each pair names a state input port and a state output port that must
    /// map to the same physical runtime state buffer (e.g. the KV-cache in/out
    /// ports of KVCacheUpdate). Lowering generates one alias record per pair
    /// from the schema, so new stateful operators need no lowering changes.
    /// Empty for operators without persistent state.
    std::vector<StateAliasPortPair> state_alias_ports{};
};

/// @brief Port-name constants shared by KV-cache schemas and validation.
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
} // namespace kv_cache_ports

/// @brief Finds an input port index by name.
///
/// @param schema Schema whose input ports are searched.
/// @param name Exact port name to find.
/// @return Port index, or `kInvalidArgument` when no input port matches.
StatusOr<uint32_t> FindInputPortIndex(const OperatorSchema& schema,
                                      std::string_view name) noexcept;

/// @brief Finds an output port index by name.
///
/// @param schema Schema whose output ports are searched.
/// @param name Exact port name to find.
/// @return Port index, or `kInvalidArgument` when no output port matches.
StatusOr<uint32_t> FindOutputPortIndex(const OperatorSchema& schema,
                                       std::string_view name) noexcept;

/// @brief Tests whether a schema exposes at least one state output.
///
/// @param schema Schema to inspect.
/// @return True when an output port has kind `kState`.
AM_NODISCARD bool HasStatefulOutput(const OperatorSchema& schema) noexcept;
/// @brief Tests whether an operator is deterministic and side-effect free.
///
/// @param schema Schema to inspect.
/// @return True when the schema describes a pure operator.
AM_NODISCARD bool IsPureOperator(const OperatorSchema& schema) noexcept;
/// @brief Tests whether graph lowering may evaluate an operator at compile time.
///
/// @param schema Schema to inspect.
/// @return True when the operator is pure and marked compile-time evaluable.
AM_NODISCARD bool IsCompileTimeEvaluable(const OperatorSchema& schema) noexcept;

/// @brief Looks up an operator schema in the static registry.
///
/// Returns OK with the schema for a registered op, or kNotFound for an
/// unregistered op type (e.g. OpType::kUnknown).
///
/// @param op_type Operator type to look up.
/// @return Registered schema, or `kNotFound` when the type is unknown.
StatusOr<OperatorSchema> GetOperatorSchema(OpType op_type);
/// @brief Returns all schemas in semantic ABI order.
///
/// @return Borrowed view backed by immutable static storage.
AM_NODISCARD std::span<const OperatorSchema> GetOperatorSchemas() noexcept;

} // namespace aethermind

#endif
