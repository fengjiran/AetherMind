#include "aethermind/model/graph/graph_op_builder.h"

#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace aethermind {
namespace {

StatusOr<GraphValueId> OnlyOneOutput(const AddedNode& added_node) {
    if (added_node.outputs.size() != 1U) {
        return Status::InvalidArgument(
                "Expected graph op builder helper to create exactly one output, got " +
                std::to_string(added_node.outputs.size()));
    }
    return added_node.outputs.front();
}

NodeOutputDesc ActivationOutput() {
    return NodeOutputDesc{.payload = ActivationValue{}};
}

}// namespace

GraphValueId AddInput(ModelGraph& graph, TensorSpec spec, std::string debug_name) {
    return graph.AddInput(std::move(spec), std::move(debug_name));
}

GraphValueId AddState(ModelGraph& graph,
                      TensorSpec spec,
                      StateBinding binding,
                      std::string debug_name) {
    return graph.AddState(std::move(spec), binding, std::move(debug_name));
}

StatusOr<GraphValueId> AddLinear(ModelGraph& graph,
                                 GraphValueId input,
                                 int64_t out_features,
                                 DataType weight_dtype,
                                 WeightBinding binding,
                                 std::string debug_name) {
    if (out_features <= 0) {
        return Status::InvalidArgument("Linear out_features must be positive");
    }

    // Snapshot input specs BEFORE graph mutation — graph.AddWeight/AddNode may
    // reallocate the graph's value storage and invalidate references into it.
    const TensorSpec input_spec = graph.GetValue(input).spec;
    if (!input_spec.shape.IsRanked()) {
        return Status::InvalidArgument("Linear input shape must be ranked");
    }
    const std::vector<ShapeSymbol> input_shape = *input_spec.shape.shape();
    if (input_shape.empty()) {
        return Status::InvalidArgument("Linear input rank must be at least 1");
    }

    const ShapeSymbol in_features_symbol = input_shape.back();
    if (!in_features_symbol.IsStatic()) {
        return Status::InvalidArgument("Linear input last dimension must be static");
    }
    if (in_features_symbol.GetStaticValue() <= 0) {
        return Status::InvalidArgument("Linear input last dimension must be positive");
    }

    const ShapeSymbol out_features_symbol = ShapeSymbol::CreateFromValue(out_features);
    const GraphValueId weight = graph.AddWeight(
            {.dtype = weight_dtype,
             .shape = {out_features_symbol, in_features_symbol}},
            binding,
            debug_name);

    AM_ASSIGN_OR_RETURN(AddedNode node, graph.AddNode(
                                                OpType::kLinear,
                                                binding.decoder_layer_index,
                                                {input, weight},
                                                {ActivationOutput()},
                                                LinearParams{},
                                                {},
                                                std::move(debug_name)));
    return OnlyOneOutput(node);
}

StatusOr<GraphValueId> AddRmsNorm(ModelGraph& graph,
                                  GraphValueId input,
                                  DataType weight_dtype,
                                  WeightBinding binding,
                                  float eps,
                                  std::string debug_name) {
    const TensorSpec input_spec = graph.GetValue(input).spec;
    if (!input_spec.shape.IsRanked()) {
        return Status::InvalidArgument("RmsNorm input shape must be ranked");
    }
    const std::vector<ShapeSymbol> input_shape = *input_spec.shape.shape();
    if (input_shape.empty()) {
        return Status::InvalidArgument("RmsNorm input rank must be at least 1");
    }

    const ShapeSymbol in_features_symbol = input_shape.back();
    if (!in_features_symbol.IsStatic()) {
        return Status::InvalidArgument("RmsNorm input last dimension must be static");
    }
    if (in_features_symbol.GetStaticValue() <= 0) {
        return Status::InvalidArgument("RmsNorm input last dimension must be positive");
    }

    const GraphValueId weight = graph.AddWeight(
            {.dtype = weight_dtype,
             .shape = {in_features_symbol}},
            binding,
            debug_name);

    AM_ASSIGN_OR_RETURN(AddedNode node, graph.AddNode(
                                                OpType::kRmsNorm,
                                                binding.decoder_layer_index,
                                                {input, weight},
                                                {ActivationOutput()},
                                                RmsNormParams{.eps = eps},
                                                {},
                                                std::move(debug_name)));
    return OnlyOneOutput(node);
}

StatusOr<GraphValueId> AddEmbedding(ModelGraph& graph,
                                    GraphValueId token_ids,
                                    int64_t vocab_size,
                                    int64_t embedding_dim,
                                    DataType weight_dtype,
                                    WeightBinding binding,
                                    std::string debug_name) {
    if (vocab_size <= 0) {
        return Status::InvalidArgument("Embedding vocab_size must be positive");
    }
    if (embedding_dim <= 0) {
        return Status::InvalidArgument("Embedding embedding_dim must be positive");
    }

    const TensorSpec token_spec = graph.GetValue(token_ids).spec;
    if (!token_spec.shape.IsRanked()) {
        return Status::InvalidArgument("Embedding token_ids shape must be ranked");
    }

    const GraphValueId weight = graph.AddWeight(
            {.dtype = weight_dtype,
             .shape = {ShapeSymbol::CreateFromValue(vocab_size),
                       ShapeSymbol::CreateFromValue(embedding_dim)}},
            binding,
            debug_name);

    AM_ASSIGN_OR_RETURN(AddedNode node, graph.AddNode(
                                                OpType::kEmbedding,
                                                std::nullopt,
                                                {token_ids, weight},
                                                {ActivationOutput()},
                                                EmbeddingParams{},
                                                {},
                                                std::move(debug_name)));
    return OnlyOneOutput(node);
}

StatusOr<RoPEOutputs> AddRoPE(ModelGraph& graph,
                              std::optional<uint32_t> decoder_layer_index,
                              GraphValueId q,
                              GraphValueId k,
                              GraphValueId position_ids,
                              RoPEParams params,
                              std::string debug_name) {
    AM_ASSIGN_OR_RETURN(AddedNode node, graph.AddNode(
                                                OpType::kRoPE,
                                                decoder_layer_index,
                                                {q, k, position_ids},
                                                {ActivationOutput(),
                                                 ActivationOutput()},
                                                params,
                                                {},
                                                std::move(debug_name)));
    if (node.outputs.size() != 2U) {
        return Status::InvalidArgument(
                "Expected RoPE helper to create exactly two outputs, got " +
                std::to_string(node.outputs.size()));
    }
    return RoPEOutputs{.q = node.outputs[0], .k = node.outputs[1]};
}

StatusOr<KVCachePair> AddKVCacheUpdate(ModelGraph& graph,
                                       std::optional<uint32_t> decoder_layer_index,
                                       GraphValueId k_new,
                                       GraphValueId v_new,
                                       GraphValueId k_cache,
                                       GraphValueId v_cache,
                                       std::string debug_name) {
    const GraphValue& k_cache_value = graph.GetValue(k_cache);
    const auto* k_cache_state = std::get_if<StateValue>(&k_cache_value.payload);
    if (k_cache_state == nullptr) {
        return Status::InvalidArgument("K cache input must be a StateValue");
    }
    StateBinding k_binding = k_cache_state->binding;// NOLINT

    const GraphValue& v_cache_value = graph.GetValue(v_cache);
    const auto* v_cache_state = std::get_if<StateValue>(&v_cache_value.payload);
    if (v_cache_state == nullptr) {
        return Status::InvalidArgument("V cache input must be a StateValue");
    }
    StateBinding v_binding = v_cache_state->binding;// NOLINT

    AM_ASSIGN_OR_RETURN(AddedNode node, graph.AddNode(
                                                OpType::kKVCacheUpdate,
                                                decoder_layer_index,
                                                {k_new, v_new, k_cache, v_cache},
                                                {NodeOutputDesc{.payload = StateValue{.binding = k_binding}},
                                                 NodeOutputDesc{.payload = StateValue{.binding = v_binding}}},
                                                KVCacheUpdateParams{},
                                                {},
                                                std::move(debug_name)));
    if (node.outputs.size() != 2U) {
        return Status::InvalidArgument(
                "Expected KV cache update helper to create exactly two outputs, got " +
                std::to_string(node.outputs.size()));
    }
    return KVCachePair{.k = node.outputs[0], .v = node.outputs[1]};
}

StatusOr<GraphValueId> AddAttention(ModelGraph& graph,
                                    std::optional<uint32_t> decoder_layer_index,
                                    GraphValueId q,
                                    GraphValueId k,
                                    GraphValueId v,
                                    AttentionParams params,
                                    std::string debug_name) {
    AM_ASSIGN_OR_RETURN(AddedNode node, graph.AddNode(OpType::kAttention,
                                                      decoder_layer_index,
                                                      {q, k, v},
                                                      {ActivationOutput()},
                                                      params,
                                                      {},
                                                      std::move(debug_name)));
    return OnlyOneOutput(node);
}

StatusOr<GraphValueId> AddElementwiseAdd(ModelGraph& graph,
                                         std::optional<uint32_t> decoder_layer_index,
                                         GraphValueId lhs,
                                         GraphValueId rhs,
                                         std::string debug_name) {
    const TensorSpec lhs_spec = graph.GetValue(lhs).spec;
    const TensorSpec rhs_spec = graph.GetValue(rhs).spec;
    if (lhs_spec.dtype != rhs_spec.dtype) {
        return Status::InvalidArgument(
                "Add requires matching dtypes for lhs and rhs operands");
    }
    AM_ASSIGN_OR_RETURN(AddedNode node, graph.AddNode(OpType::kAdd,
                                                      decoder_layer_index,
                                                      {lhs, rhs},
                                                      {ActivationOutput()},
                                                      AddParams{},
                                                      {},
                                                      std::move(debug_name)));
    return OnlyOneOutput(node);
}

StatusOr<GraphValueId> AddSiluMul(ModelGraph& graph,
                                  std::optional<uint32_t> decoder_layer_index,
                                  GraphValueId gate,
                                  GraphValueId up,
                                  std::string debug_name) {
    const TensorSpec gate_spec = graph.GetValue(gate).spec;
    if (gate_spec != graph.GetValue(up).spec) {
        return Status::InvalidArgument("SiluMul gate and up specs must match");
    }

    AM_ASSIGN_OR_RETURN(AddedNode node, graph.AddNode(OpType::kSiluMul,
                                                      decoder_layer_index,
                                                      {gate, up},
                                                      {ActivationOutput()},
                                                      SiluMulParams{},
                                                      {},
                                                      std::move(debug_name)));
    return OnlyOneOutput(node);
}

StatusOr<GraphValueId> AddSilu(ModelGraph& graph,
                               std::optional<uint32_t> decoder_layer_index,
                               GraphValueId input,
                               std::string debug_name) {
    AM_ASSIGN_OR_RETURN(AddedNode node, graph.AddNode(OpType::kSilu,
                                                      decoder_layer_index,
                                                      {input},
                                                      {ActivationOutput()},
                                                      SiluParams{},
                                                      {},
                                                      std::move(debug_name)));
    return OnlyOneOutput(node);
}

StatusOr<GraphValueId> AddElementwiseMul(ModelGraph& graph,
                                         std::optional<uint32_t> decoder_layer_index,
                                         GraphValueId lhs,
                                         GraphValueId rhs,
                                         std::string debug_name) {
    const TensorSpec lhs_spec = graph.GetValue(lhs).spec;
    if (lhs_spec != graph.GetValue(rhs).spec) {
        return Status::InvalidArgument("ElementwiseMul lhs and rhs specs must match");
    }

    AM_ASSIGN_OR_RETURN(AddedNode node, graph.AddNode(OpType::kElementwiseMul,
                                                      decoder_layer_index,
                                                      {lhs, rhs},
                                                      {ActivationOutput()},
                                                      ElementwiseMulParams{},
                                                      {},
                                                      std::move(debug_name)));
    return OnlyOneOutput(node);
}

StatusOr<GraphValueId> AddArgmax(ModelGraph& graph,
                                 std::optional<uint32_t> decoder_layer_index,
                                 GraphValueId input,
                                 int64_t axis,
                                 std::string debug_name) {
    AM_ASSIGN_OR_RETURN(AddedNode node, graph.AddNode(OpType::kArgmax,
                                                      decoder_layer_index,
                                                      {input},
                                                      {ActivationOutput()},
                                                      ArgmaxParams{.axis = axis},
                                                      {},
                                                      std::move(debug_name)));
    return OnlyOneOutput(node);
}

}// namespace aethermind
