#include "aethermind/model/graph/operator_schema.h"

#include <array>

namespace aethermind {
namespace {

OperatorInputPort Input(uint32_t index, std::string_view name, OperatorPortKind kind) {
    return OperatorInputPort{.index = index, .name = std::string(name), .kind = kind};
}

OperatorInputPort Input(uint32_t index, std::string_view name, OperatorPortKind kind,
                        bool contributes_tensor_spec) {
    return OperatorInputPort{.index = index,
                             .name = std::string(name),
                             .kind = kind,
                             .contributes_tensor_spec = contributes_tensor_spec};
}

OperatorOutputPort Output(uint32_t index, std::string_view name) {
    return OperatorOutputPort{.index = index,
                              .name = std::string(name),
                              .kind = OperatorPortKind::kActivation};
}

OperatorOutputPort Output(uint32_t index, std::string_view name, OperatorPortKind kind) {
    return OperatorOutputPort{.index = index, .name = std::string(name), .kind = kind};
}

template<typename Port>
StatusOr<uint32_t> FindPortIndex(std::span<const Port> ports,
                                 std::string_view name,
                                 const char* not_found_message) noexcept {
    for (const Port& port: ports) {
        if (std::string_view(port.name) == name) {
            return port.index;
        }
    }
    return Status::InvalidArgument(not_found_message);
}

const std::array<OperatorSchema, 13> kOperatorSchemas{
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
                                Input(1, "k", OperatorPortKind::kActivation),
                                Input(2, "position_ids", OperatorPortKind::kModelInput)},
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
                .op_type = OpType::kSilu,
                .input_ports = {Input(0, "input", OperatorPortKind::kActivation)},
                .output_ports = {Output(0, "output")},
        },
        OperatorSchema{
                .op_type = OpType::kElementwiseMul,
                .input_ports = {Input(0, "lhs", OperatorPortKind::kActivation),
                                Input(1, "rhs", OperatorPortKind::kActivation)},
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
                                Input(2, kv_cache_ports::kCacheIn, OperatorPortKind::kState, false),
                                Input(3, kv_cache_ports::vCacheIn, OperatorPortKind::kState, false)},
                .output_ports = {Output(0, kv_cache_ports::kCacheOut, OperatorPortKind::kState),
                                 Output(1, kv_cache_ports::vCacheOut, OperatorPortKind::kState)},
        },
        OperatorSchema{
                .op_type = OpType::kAttention,
                .input_ports = {Input(0, "q", OperatorPortKind::kActivation),
                                Input(1, kv_cache_ports::kCache, OperatorPortKind::kState, false),
                                Input(2, kv_cache_ports::vCache, OperatorPortKind::kState, false)},
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

StatusOr<uint32_t> FindInputPortIndex(const OperatorSchema& schema, std::string_view name) noexcept {
    return FindPortIndex<OperatorInputPort>(schema.input_ports,
                                            name,
                                            "Operator schema input port not found");
}

StatusOr<uint32_t> FindOutputPortIndex(const OperatorSchema& schema, std::string_view name) noexcept {
    return FindPortIndex<OperatorOutputPort>(schema.output_ports,
                                             name,
                                             "Operator schema output port not found");
}

}// namespace aethermind
