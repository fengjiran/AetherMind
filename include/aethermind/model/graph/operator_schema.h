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

enum class OperatorPortKind : uint8_t {
    kModelInput,
    kActivation,
    kWeight,
    kConstant,
    kState,
};

struct OperatorInputPort {
    uint32_t index = 0;
    std::string name{};
    OperatorPortKind kind = OperatorPortKind::kActivation;
    bool contributes_tensor_spec = true;
};

struct OperatorOutputPort {
    uint32_t index = 0;
    std::string name{};
    OperatorPortKind kind = OperatorPortKind::kActivation;
};

struct OperatorSchema {
    OpType op_type = OpType::kUnknown;
    std::vector<OperatorInputPort> input_ports{};
    std::vector<OperatorOutputPort> output_ports{};
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

AM_NODISCARD StatusOr<OperatorSchema> GetOperatorSchema(OpType op_type);
AM_NODISCARD std::span<const OperatorSchema> GetOperatorSchemas() noexcept;

}// namespace aethermind

#endif
