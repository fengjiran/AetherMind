#include "aethermind/model/graph/operator_schema.h"

#include <array>

namespace aethermind {
namespace {

OperatorInputPort Input(uint32_t index, const char* name, OperatorPortKind kind) {
    return OperatorInputPort{.index = index, .name = name, .kind = kind};
}

OperatorInputPort Input(uint32_t index, const char* name, OperatorPortKind kind, bool contributes_tensor_spec) {
    return OperatorInputPort{.index = index, .name = name, .kind = kind, .contributes_tensor_spec = contributes_tensor_spec};
}

OperatorOutputPort Output(uint32_t index, const char* name) {
    return OperatorOutputPort{.index = index, .name = name, .kind = OperatorPortKind::kActivation};
}

OperatorOutputPort Output(uint32_t index, const char* name, OperatorPortKind kind) {
    return OperatorOutputPort{.index = index, .name = name, .kind = kind};
}

const std::array<OperatorSchema, 11> kOperatorSchemas{
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
                .output_ports = {Output(0, "q_rope"),
                                 Output(1, "k_rope")},
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
        OperatorSchema{
                .op_type = OpType::kKVCacheUpdate,
                .input_ports = {Input(0, "k", OperatorPortKind::kActivation),
                                Input(1, "v", OperatorPortKind::kActivation),
                                Input(2, "kv_cache_in", OperatorPortKind::kState, false)},
                .output_ports = {Output(0, "kv_cache_out", OperatorPortKind::kState)},
        },
        OperatorSchema{
                .op_type = OpType::kAttention,
                .input_ports = {Input(0, "q", OperatorPortKind::kActivation),
                                Input(1, "kv_cache", OperatorPortKind::kState, false)},
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
