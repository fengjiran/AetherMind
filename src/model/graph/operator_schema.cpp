#include "aethermind/model/graph/operator_schema.h"

#include <array>

namespace aethermind {
namespace {

OperatorInputPort Input(uint32_t index, const char* name, OperatorPortKind kind) {
    return OperatorInputPort{.index = index, .name = name, .kind = kind};
}

OperatorOutputPort Output(uint32_t index, const char* name) {
    return OperatorOutputPort{.index = index, .name = name, .kind = OperatorPortKind::kActivation};
}

const std::array<OperatorSchema, 9> kOperatorSchemas{
        OperatorSchema{
                .op_type = OpType::kEmbedding,
                .input_ports = {Input(0, "tokens", OperatorPortKind::kModelInput),
                                Input(1, "weight", OperatorPortKind::kWeight)},
                .output_ports = {Output(0, "output")},
        },
        OperatorSchema{
                .op_type = OpType::kRmsNorm,
                .input_ports = {Input(0, "input", OperatorPortKind::kActivation),
                                Input(1, "weight", OperatorPortKind::kWeight)},
                .output_ports = {Output(0, "output")},
        },
        OperatorSchema{
                .op_type = OpType::kLinear,
                .input_ports = {Input(0, "input", OperatorPortKind::kActivation),
                                Input(1, "weight", OperatorPortKind::kWeight)},
                .output_ports = {Output(0, "output")},
        },
        OperatorSchema{
                .op_type = OpType::kRoPE,
                .input_ports = {Input(0, "q", OperatorPortKind::kActivation),
                                Input(1, "k", OperatorPortKind::kActivation)},
                .output_ports = {Output(0, "q_rope"), Output(1, "k_rope")},
        },
        OperatorSchema{
                .op_type = OpType::kMatMul,
                .input_ports = {Input(0, "lhs", OperatorPortKind::kActivation),
                                Input(1, "rhs", OperatorPortKind::kActivation)},
                .output_ports = {Output(0, "output")},
        },
        OperatorSchema{
                .op_type = OpType::kSoftmax,
                .input_ports = {Input(0, "input", OperatorPortKind::kActivation)},
                .output_ports = {Output(0, "output")},
        },
        OperatorSchema{
                .op_type = OpType::kAdd,
                .input_ports = {Input(0, "lhs", OperatorPortKind::kActivation),
                                Input(1, "rhs", OperatorPortKind::kActivation)},
                .output_ports = {Output(0, "output")},
        },
        OperatorSchema{
                .op_type = OpType::kSiluMul,
                .input_ports = {Input(0, "gate", OperatorPortKind::kActivation),
                                Input(1, "up", OperatorPortKind::kActivation)},
                .output_ports = {Output(0, "output")},
        },
        OperatorSchema{
                .op_type = OpType::kArgmax,
                .input_ports = {Input(0, "logits", OperatorPortKind::kActivation)},
                .output_ports = {Output(0, "output")},
        },
};

}// namespace

StatusOr<OperatorSchema> GetOperatorSchema(OpType op_type) {
    for (const OperatorSchema& schema: kOperatorSchemas) {
        if (schema.op_type == op_type) {
            return schema;
        }
    }
    return Status::InvalidArgument("No ModelGraph operator schema registered for op");
}

std::span<const OperatorSchema> GetOperatorSchemas() noexcept {
    return kOperatorSchemas;
}

}// namespace aethermind
