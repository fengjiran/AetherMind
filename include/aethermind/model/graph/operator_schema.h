#ifndef AETHERMIND_MODEL_GRAPH_OPERATOR_SCHEMA_H
#define AETHERMIND_MODEL_GRAPH_OPERATOR_SCHEMA_H

#include "aethermind/base/status.h"
#include "aethermind/operators/op_type.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace aethermind {

enum class OperatorPortKind : uint8_t {
    kModelInput,
    kActivation,
    kWeight,
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

AM_NODISCARD StatusOr<OperatorSchema> GetOperatorSchema(OpType op_type);
AM_NODISCARD std::span<const OperatorSchema> GetOperatorSchemas() noexcept;

}// namespace aethermind

#endif
