#include "aethermind/model/graph/operator_schema.h"

#include <array>

namespace aethermind {
namespace {

// Static registry defining the expected port layout and semantic traits for
// every registered operator. Graph validation (graph.cpp) reads these schemas
// to verify node arity, port kinds, and payload consistency at build time.
// Add a new entry here when introducing a new OpType.

OperatorInputPort Input(uint32_t index, std::string_view name, OperatorPortKind kind) {
    return {.index = index, .kind = kind, .name = std::string(name)};
}

OperatorInputPort Input(uint32_t index, std::string_view name, OperatorPortKind kind,
                        bool contributes_tensor_spec) {
    return {.index = index,
            .kind = kind,
            .contributes_tensor_spec = contributes_tensor_spec,
            .name = std::string(name)};
}

OperatorOutputPort Output(uint32_t index, std::string_view name) {
    return {.index = index,
            .kind = OperatorPortKind::kActivation,
            .name = std::string(name)};
}

OperatorOutputPort Output(uint32_t index, std::string_view name, OperatorPortKind kind) {
    return {.index = index, .kind = kind, .name = std::string(name)};
}

constexpr OperatorTraits RuntimeOnly() noexcept {
    return {.has_side_effects = false,
            .deterministic = true,
            .compile_time_evaluable = false};
}

constexpr OperatorTraits CompileTimeEvaluable() noexcept {
    return {.has_side_effects = false,
            .deterministic = true,
            .compile_time_evaluable = true};
}

constexpr OperatorTraits Stateful() noexcept {
    return {.has_side_effects = true,
            .deterministic = false,
            .compile_time_evaluable = false};
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
                .traits = RuntimeOnly(),
        },
        OperatorSchema{
                .op_type = OpType::kRmsNorm,
                .input_ports = {Input(0, "input", OperatorPortKind::kActivation),
                                Input(1, "weight", OperatorPortKind::kWeight)},
                .output_ports = {Output(0, "output")},
                .traits = RuntimeOnly(),
        },
        OperatorSchema{
                .op_type = OpType::kLinear,
                .input_ports = {Input(0, "input", OperatorPortKind::kActivation),
                                Input(1, "weight", OperatorPortKind::kWeight)},
                .output_ports = {Output(0, "output")},
                .traits = RuntimeOnly(),
        },
        OperatorSchema{
                .op_type = OpType::kRoPE,
                .input_ports = {Input(0, "q", OperatorPortKind::kActivation),
                                Input(1, "k", OperatorPortKind::kActivation),
                                Input(2, "position_ids", OperatorPortKind::kModelInput)},
                .output_ports = {Output(0, "q_rope"),
                                 Output(1, "k_rope")},
                .traits = RuntimeOnly(),
        },
        OperatorSchema{
                .op_type = OpType::kMatMul,
                .input_ports = {Input(0, "lhs", OperatorPortKind::kActivation),
                                Input(1, "rhs", OperatorPortKind::kActivation)},
                .output_ports = {Output(0, "output")},
                .traits = RuntimeOnly(),
        },
        OperatorSchema{
                .op_type = OpType::kSoftmax,
                .input_ports = {Input(0, "input", OperatorPortKind::kActivation)},
                .output_ports = {Output(0, "output")},
                .traits = RuntimeOnly(),
        },
        OperatorSchema{
                .op_type = OpType::kAdd,
                .input_ports = {Input(0, "lhs", OperatorPortKind::kActivation),
                                Input(1, "rhs", OperatorPortKind::kActivation)},
                .output_ports = {Output(0, "output")},
                .traits = CompileTimeEvaluable(),
        },
        OperatorSchema{
                .op_type = OpType::kSiluMul,
                .input_ports = {Input(0, "gate", OperatorPortKind::kActivation),
                                Input(1, "up", OperatorPortKind::kActivation)},
                .output_ports = {Output(0, "output")},
                .traits = CompileTimeEvaluable(),
        },
        OperatorSchema{
                .op_type = OpType::kSilu,
                .input_ports = {Input(0, "input", OperatorPortKind::kActivation)},
                .output_ports = {Output(0, "output")},
                .traits = CompileTimeEvaluable(),
        },
        OperatorSchema{
                .op_type = OpType::kElementwiseMul,
                .input_ports = {Input(0, "lhs", OperatorPortKind::kActivation),
                                Input(1, "rhs", OperatorPortKind::kActivation)},
                .output_ports = {Output(0, "output")},
                .traits = CompileTimeEvaluable(),
        },
        OperatorSchema{
                .op_type = OpType::kArgmax,
                .input_ports = {Input(0, "logits", OperatorPortKind::kActivation)},
                .output_ports = {Output(0, "output")},
                .traits = RuntimeOnly(),
        },
        OperatorSchema{
                .op_type = OpType::kKVCacheUpdate,
                .input_ports = {Input(0, "k", OperatorPortKind::kActivation),
                                Input(1, "v", OperatorPortKind::kActivation),
                                Input(2, kv_cache_ports::kCacheIn, OperatorPortKind::kState, false),
                                Input(3, kv_cache_ports::vCacheIn, OperatorPortKind::kState, false)},
                .output_ports = {Output(0, kv_cache_ports::kCacheOut, OperatorPortKind::kState),
                                 Output(1, kv_cache_ports::vCacheOut, OperatorPortKind::kState)},
                .traits = Stateful(),
        },
        OperatorSchema{
                .op_type = OpType::kAttention,
                .input_ports = {Input(0, "q", OperatorPortKind::kActivation),
                                Input(1, kv_cache_ports::kCache, OperatorPortKind::kState, false),
                                Input(2, kv_cache_ports::vCache, OperatorPortKind::kState, false)},
                .output_ports = {Output(0, "output")},
                .traits = RuntimeOnly(),
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

bool HasStatefulOutput(const OperatorSchema& schema) noexcept {
    for (const OperatorOutputPort& output_port: schema.output_ports) {
        if (output_port.kind == OperatorPortKind::kState) {
            return true;
        }
    }
    return false;
}

// A pure operator has no observable side effects and produces deterministic
// outputs. These properties are prerequisites for CSE, fusion, and reordering.
bool IsPureOperator(const OperatorSchema& schema) noexcept {
    return !schema.traits.has_side_effects && schema.traits.deterministic;
}

// A compile-time evaluable operator is a pure operator whose result depends
// only on constant-foldable inputs. The lowering pass may replace such nodes
// with pre-computed constants.
bool IsCompileTimeEvaluable(const OperatorSchema& schema) noexcept {
    return IsPureOperator(schema) && schema.traits.compile_time_evaluable;
}

}// namespace aethermind
