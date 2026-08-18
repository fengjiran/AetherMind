#include "aethermind/operators/operator_schema.h"

#include <algorithm>
#include <array>

namespace aethermind {
namespace {

OperatorInputPort Input(std::string_view name, OperatorPortKind kind) {
    return {.kind = kind, .name = std::string(name)};
}

OperatorInputPort Input(std::string_view name, OperatorPortKind kind,
                        bool contributes_tensor_spec) {
    return {.kind = kind,
            .contributes_tensor_spec = contributes_tensor_spec,
            .name = std::string(name)};
}

OperatorOutputPort Output(std::string_view name) {
    return {.kind = OperatorPortKind::kActivation,
            .name = std::string(name)};
}

OperatorOutputPort Output(std::string_view name, OperatorPortKind kind) {
    return {.kind = kind, .name = std::string(name)};
}

StateAliasPortPair StateAlias(std::string_view input, std::string_view output) {
    return {.input_port = std::string(input), .output_port = std::string(output)};
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
            return static_cast<uint32_t>(&port - ports.data());
        }
    }
    return Status::InvalidArgument(not_found_message);
}

// Static registry defining the expected port layout and semantic traits for
// every registered operator. Graph validation (graph.cpp) reads these schemas
// to verify node arity, port kinds, and payload consistency at build time.
// Add a new entry here when introducing a new OpType.
const std::array<OperatorSchema, 19> kOperatorSchemas{
        OperatorSchema{
                .op_type = OpType::kEmbedding,
                .input_ports = {Input("tokens", OperatorPortKind::kModelInput),
                                Input("weight", OperatorPortKind::kWeight)},
                .output_ports = {Output("output")},
                .traits = RuntimeOnly(),
        },
        OperatorSchema{
                .op_type = OpType::kRmsNorm,
                .input_ports = {Input("input", OperatorPortKind::kActivation),
                                Input("weight", OperatorPortKind::kWeight)},
                .output_ports = {Output("output")},
                .traits = RuntimeOnly(),
        },
        OperatorSchema{
                .op_type = OpType::kLinear,
                .input_ports = {Input("input", OperatorPortKind::kActivation),
                                Input("weight", OperatorPortKind::kWeight)},
                .output_ports = {Output("output")},
                .traits = RuntimeOnly(),
        },
        OperatorSchema{
                .op_type = OpType::kQkvLinear,
                .input_ports = {Input("input", OperatorPortKind::kActivation),
                                Input("qkv_weight", OperatorPortKind::kWeight)},
                .output_ports = {Output("q"),
                                 Output("k"),
                                 Output("v")},
                // Fused runtime projection: pure and deterministic, but not
                // compile-time evaluable (no fused constant evaluator exists).
                .traits = RuntimeOnly(),
        },
        OperatorSchema{
                .op_type = OpType::kGateUpLinear,
                .input_ports = {Input("input", OperatorPortKind::kActivation),
                                Input("gate_up_weight", OperatorPortKind::kWeight)},
                .output_ports = {Output("gate"),
                                 Output("up")},
                .traits = RuntimeOnly(),
        },
        OperatorSchema{
                .op_type = OpType::kRoPE,
                .input_ports = {Input("q", OperatorPortKind::kActivation),
                                Input("k", OperatorPortKind::kActivation),
                                Input("position_ids", OperatorPortKind::kModelInput)},
                .output_ports = {Output("q_rope"),
                                 Output("k_rope")},
                .traits = RuntimeOnly(),
        },
        OperatorSchema{
                .op_type = OpType::kMatMul,
                .input_ports = {Input("lhs", OperatorPortKind::kActivation),
                                Input("rhs", OperatorPortKind::kActivation)},
                .output_ports = {Output("output")},
                .traits = RuntimeOnly(),
        },
        OperatorSchema{
                .op_type = OpType::kSoftmax,
                .input_ports = {Input("input", OperatorPortKind::kActivation)},
                .output_ports = {Output("output")},
                .traits = RuntimeOnly(),
        },
        OperatorSchema{
                .op_type = OpType::kAdd,
                .input_ports = {Input("lhs", OperatorPortKind::kActivation),
                                Input("rhs", OperatorPortKind::kActivation)},
                .output_ports = {Output("output")},
                .traits = CompileTimeEvaluable(),
        },
        OperatorSchema{
                .op_type = OpType::kAddRmsNorm,
                .input_ports = {Input("input", OperatorPortKind::kActivation),
                                Input("residual", OperatorPortKind::kActivation),
                                Input("weight", OperatorPortKind::kWeight)},
                .output_ports = {Output("output"),
                                 Output("new_residual")},
                // Fused add + RMSNorm: output = RmsNorm(input + residual, weight);
                // new_residual = input + residual feeds the next block's residual.
                // Follows the stricter trait of RmsNorm (RuntimeOnly) rather than
                // Add (CompileTimeEvaluable).
                .traits = RuntimeOnly(),
        },
        OperatorSchema{
                .op_type = OpType::kSiluMul,
                .input_ports = {Input("gate", OperatorPortKind::kActivation),
                                Input("up", OperatorPortKind::kActivation)},
                .output_ports = {Output("output")},
                .traits = CompileTimeEvaluable(),
        },
        OperatorSchema{
                .op_type = OpType::kSilu,
                .input_ports = {Input("input", OperatorPortKind::kActivation)},
                .output_ports = {Output("output")},
                .traits = CompileTimeEvaluable(),
        },
        OperatorSchema{
                .op_type = OpType::kElementwiseMul,
                .input_ports = {Input("lhs", OperatorPortKind::kActivation),
                                Input("rhs", OperatorPortKind::kActivation)},
                .output_ports = {Output("output")},
                .traits = CompileTimeEvaluable(),
        },
        OperatorSchema{
                .op_type = OpType::kArgmax,
                .input_ports = {Input("logits", OperatorPortKind::kActivation)},
                .output_ports = {Output("output")},
                .traits = RuntimeOnly(),
        },
        OperatorSchema{
                .op_type = OpType::kKVCacheUpdate,
                .input_ports = {Input("k", OperatorPortKind::kActivation),
                                Input("v", OperatorPortKind::kActivation),
                                Input(kv_cache_ports::kCacheIn, OperatorPortKind::kState, false),
                                Input(kv_cache_ports::vCacheIn, OperatorPortKind::kState, false)},
                .output_ports = {Output(kv_cache_ports::kCacheOut, OperatorPortKind::kState),
                                 Output(kv_cache_ports::vCacheOut, OperatorPortKind::kState)},
                .traits = Stateful(),
                .state_alias_ports = {StateAlias(kv_cache_ports::kCacheIn, kv_cache_ports::kCacheOut),
                                      StateAlias(kv_cache_ports::vCacheIn, kv_cache_ports::vCacheOut)},
        },
        OperatorSchema{
                .op_type = OpType::kAttention,
                .input_ports = {Input("q", OperatorPortKind::kActivation),
                                Input(kv_cache_ports::kCache, OperatorPortKind::kState, false),
                                Input(kv_cache_ports::vCache, OperatorPortKind::kState, false)},
                .output_ports = {Output("output")},
                .traits = RuntimeOnly(),
        },
        OperatorSchema{
                .op_type = OpType::kReshape,
                .input_ports = {Input("input", OperatorPortKind::kActivation)},
                .output_ports = {Output("output")},
                // Semantic-only: pure and deterministic (so DCE-removable),
                // but not advertised as compile-time evaluable (no constant
                // evaluator exists; ReshapeParams is not yet a constant-foldable
                // surface).
                .traits = RuntimeOnly(),
        },
        OperatorSchema{
                .op_type = OpType::kPermute,
                .input_ports = {Input("input", OperatorPortKind::kActivation)},
                .output_ports = {Output("output")},
                // Semantic-only: pure and deterministic (so DCE-removable),
                // but not advertised as compile-time evaluable (no constant
                // evaluator exists; PermuteParams is not yet a constant-foldable
                // surface).
                .traits = RuntimeOnly(),
        },
        OperatorSchema{
                .op_type = OpType::kReorder,
                .input_ports = {Input("input", OperatorPortKind::kActivation)},
                .output_ports = {Output("output")},
                // Semantic-only: pure and deterministic (so DCE-removable when
                // dead), but not compile-time evaluable. Reorder records a
                // physical materialization intent (canonical contiguous) that
                // prevents algebraic identity erasure of live nodes.
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
    return Status::NotFound("No ModelGraph operator schema registered for op");
}

std::span<const OperatorSchema> GetOperatorSchemas() noexcept {
    return kOperatorSchemas;
}

StatusOr<uint32_t> FindInputPortIndex(const OperatorSchema& schema,
                                      std::string_view name) noexcept {
    return FindPortIndex<OperatorInputPort>(
            schema.input_ports,
            name,
            "Operator schema input port not found");
}

StatusOr<uint32_t> FindOutputPortIndex(const OperatorSchema& schema,
                                       std::string_view name) noexcept {
    return FindPortIndex<OperatorOutputPort>(
            schema.output_ports,
            name,
            "Operator schema output port not found");
}

bool HasStatefulOutput(const OperatorSchema& schema) noexcept {
    return std::ranges::any_of(schema.output_ports, [&](const auto& port) {
        return port.kind == OperatorPortKind::kState;
    });
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
