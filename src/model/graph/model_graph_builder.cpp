#include "aethermind/model/graph/model_graph_builder.h"
#include "aethermind/model/formats/hf/hf_model_validator.h"
#include "aethermind/operators/embedding_op.h"
#include "aethermind/operators/rmsnorm_op.h"

#include <cstddef>
#include <cstdint>
#include <utility>

namespace aethermind {
namespace {

constexpr size_t kEmbeddingNodeCount = 1;
// Decoder-only dense Llama layer nodes, in execution order:
// input RMSNorm, Q/K/V projection, RoPE, attention score MatMul, Softmax,
// attention value MatMul, output projection, residual Add, post-attention RMSNorm,
// MLP gate/up projection, SiLU multiply, MLP down projection, residual Add.
constexpr size_t kDenseLlamaLayerNodeCount = 16;
constexpr size_t kTailNodeCount = 3;

TensorSpec SymbolicTensorSpec(DataType dtype, std::vector<ShapeSymbol> dims) {
    return TensorSpec{.dtype = dtype, .shape = SymbolicShape(std::move(dims))};
}

std::vector<ShapeSymbol> SymbolicDims(const std::vector<int64_t>& shape) {
    std::vector<ShapeSymbol> dims;
    dims.reserve(shape.size());
    for (const int64_t dim: shape) {
        dims.push_back(ShapeSymbol::CreateFromValue(dim));
    }
    return dims;
}

TensorSpec WeightTensor(const RawWeightView& weight) {
    return SymbolicTensorSpec(weight.dtype, SymbolicDims(weight.shape));
}

TensorSpec ActivationTensor(DataType dtype, ShapeSymbol seq_len, int64_t feature_dim) {
    return SymbolicTensorSpec(dtype, {seq_len, ShapeSymbol::CreateFromValue(feature_dim)});
}

ModelWeightBinding Bind(ModelWeightRole role, uint32_t layer_index = 0) noexcept {
    return ModelWeightBinding{.role = role, .layer_index = layer_index};
}

GraphNode MakeWeightedNode(OpType op_type,
                           uint32_t layer_index,
                           std::vector<TensorSpec> inputs,
                           TensorSpec output,
                           ModelWeightBinding weight) {
    return GraphNode{
            .op_type = op_type,
            .layer_index = layer_index,
            .inputs = std::move(inputs),
            .outputs = {std::move(output)},
            .weights = {weight},
    };
}

GraphNode MakePureNode(OpType op_type,
                       uint32_t layer_index,
                       std::vector<TensorSpec> inputs,
                       std::vector<TensorSpec> outputs) {
    return GraphNode{
            .op_type = op_type,
            .layer_index = layer_index,
            .inputs = std::move(inputs),
            .outputs = std::move(outputs),
    };
}

GraphNode MakeRmsNormNode(uint32_t layer_index,
                          ModelWeightRole role,
                          std::vector<TensorSpec> inputs,
                          TensorSpec output,
                          float eps) {
    GraphNode node = MakeWeightedNode(OpType::kRmsNorm,
                                      layer_index,
                                      std::move(inputs),
                                      std::move(output),
                                      Bind(role, layer_index));
    node.op_params = RmsNormOp::Params{.eps = eps};
    return node;
}

void AppendDenseLlamaLayerNodes(std::vector<GraphNode>& nodes,
                                uint32_t layer_index,
                                const DecoderLayerRawWeights& layer,
                                const TensorSpec& hidden,
                                const TensorSpec& kv_hidden,
                                const TensorSpec& intermediate,
                                const TensorSpec& scores,
                                float rms_norm_eps) {
    const size_t layer_begin = nodes.size();

    nodes.push_back(MakeRmsNormNode(layer_index,
                                    ModelWeightRole::kInputNorm,
                                    {hidden, WeightTensor(layer.norm.input_rmsnorm)},
                                    hidden,
                                    rms_norm_eps));
    nodes.push_back(MakeWeightedNode(OpType::kLinear,
                                     layer_index,
                                     {hidden, WeightTensor(layer.attn.q_proj)},
                                     hidden,
                                     Bind(ModelWeightRole::kAttentionQ, layer_index)));
    nodes.push_back(MakeWeightedNode(OpType::kLinear,
                                     layer_index,
                                     {hidden, WeightTensor(layer.attn.k_proj)},
                                     kv_hidden,
                                     Bind(ModelWeightRole::kAttentionK, layer_index)));
    nodes.push_back(MakeWeightedNode(OpType::kLinear,
                                     layer_index,
                                     {hidden, WeightTensor(layer.attn.v_proj)},
                                     kv_hidden,
                                     Bind(ModelWeightRole::kAttentionV, layer_index)));
    nodes.push_back(MakePureNode(OpType::kRoPE,
                                 layer_index,
                                 {hidden, kv_hidden},
                                 {hidden, kv_hidden}));
    nodes.push_back(MakePureNode(OpType::kMatMul,
                                 layer_index,
                                 {hidden, kv_hidden},
                                 {scores}));
    nodes.push_back(MakePureNode(OpType::kSoftmax,
                                 layer_index,
                                 {scores},
                                 {scores}));
    nodes.push_back(MakePureNode(OpType::kMatMul,
                                 layer_index,
                                 {scores, kv_hidden},
                                 {kv_hidden}));
    nodes.push_back(MakeWeightedNode(OpType::kLinear,
                                     layer_index,
                                     {hidden, WeightTensor(layer.attn.o_proj)},
                                     hidden,
                                     Bind(ModelWeightRole::kAttentionO, layer_index)));
    nodes.push_back(MakePureNode(OpType::kAdd,
                                 layer_index,
                                 {hidden, hidden},
                                 {hidden}));
    nodes.push_back(MakeRmsNormNode(layer_index,
                                    ModelWeightRole::kPostAttentionNorm,
                                    {hidden, WeightTensor(layer.norm.post_attn_rmsnorm)},
                                    hidden,
                                    rms_norm_eps));
    nodes.push_back(MakeWeightedNode(OpType::kLinear,
                                     layer_index,
                                     {hidden, WeightTensor(layer.mlp.gate_proj)},
                                     intermediate,
                                     Bind(ModelWeightRole::kMlpGate, layer_index)));
    nodes.push_back(MakeWeightedNode(OpType::kLinear,
                                     layer_index,
                                     {hidden, WeightTensor(layer.mlp.up_proj)},
                                     intermediate,
                                     Bind(ModelWeightRole::kMlpUp, layer_index)));
    nodes.push_back(MakePureNode(OpType::kSiluMul,
                                 layer_index,
                                 {intermediate, intermediate},
                                 {intermediate}));
    nodes.push_back(MakeWeightedNode(OpType::kLinear,
                                     layer_index,
                                     {intermediate, WeightTensor(layer.mlp.down_proj)},
                                     hidden,
                                     Bind(ModelWeightRole::kMlpDown, layer_index)));
    nodes.push_back(MakePureNode(OpType::kAdd,
                                 layer_index,
                                 {hidden, hidden},
                                 {hidden}));

    AM_CHECK(nodes.size() - layer_begin == kDenseLlamaLayerNodeCount,
             "Dense Llama layer node count changed unexpectedly");
}

Status ValidateInputs(const HfModelConfig& config, const ResolvedModelWeights& weights) {
    AM_RETURN_IF_ERROR(HfModelValidator::ValidateConfig(config));
    return HfModelValidator::ValidateResolvedModel(config, weights);
}

}// namespace

StatusOr<ModelGraph> ModelGraphBuilder::BuildLlamaDense(const HfModelConfig& config,
                                                        const ResolvedModelWeights& weights) {
    AM_RETURN_IF_ERROR(ValidateInputs(config, weights));
    const DataType act_dtype = !config.weight_dtype_hint.IsUndefined() ? config.weight_dtype_hint
                                                                       : DataType::Float32();
    const ShapeSymbol seq_len = ShapeSymbol::Create();
    const int64_t hidden_size = config.hidden_size;
    const int64_t head_dim = config.head_dim != 0 ? config.head_dim
                                                  : config.hidden_size / config.num_attention_heads;
    const int64_t kv_hidden_size = config.num_key_value_heads * head_dim;

    const TensorSpec token_ids = SymbolicTensorSpec(DataType::Int(64), {seq_len});
    const TensorSpec hidden = ActivationTensor(act_dtype, seq_len, hidden_size);
    const TensorSpec kv_hidden = ActivationTensor(act_dtype, seq_len, kv_hidden_size);
    const TensorSpec intermediate = ActivationTensor(act_dtype, seq_len, config.intermediate_size);
    const TensorSpec scores = SymbolicTensorSpec(act_dtype, {seq_len, seq_len});
    const TensorSpec logits = ActivationTensor(act_dtype, seq_len, config.vocab_size);
    const float rms_norm_eps = static_cast<float>(config.rms_norm_eps);

    std::vector<GraphNode> nodes;
    nodes.reserve(kEmbeddingNodeCount +
                  static_cast<size_t>(config.num_hidden_layers) * kDenseLlamaLayerNodeCount +
                  kTailNodeCount);

    GraphNode embedding = MakeWeightedNode(OpType::kEmbedding,
                                           0,
                                           {token_ids, WeightTensor(weights.embed_tokens)},
                                           hidden,
                                           Bind(ModelWeightRole::kTokenEmbedding));
    embedding.op_params = EmbeddingOp::Params{};
    nodes.push_back(std::move(embedding));

    for (uint32_t layer_index = 0; layer_index < static_cast<uint32_t>(config.num_hidden_layers); ++layer_index) {
        const DecoderLayerRawWeights& layer = weights.layers[layer_index];
        AppendDenseLlamaLayerNodes(nodes,
                                   layer_index,
                                   layer,
                                   hidden,
                                   kv_hidden,
                                   intermediate,
                                   scores,
                                   rms_norm_eps);
    }

    nodes.push_back(MakeRmsNormNode(0,
                                    ModelWeightRole::kFinalNorm,
                                    {hidden, WeightTensor(weights.final_norm)},
                                    hidden,
                                    rms_norm_eps));

    const RawWeightView& lm_head_weight = weights.lm_head.has_value() ? *weights.lm_head : weights.embed_tokens;
    nodes.push_back(MakeWeightedNode(OpType::kLinear,
                                     0,
                                     {hidden, WeightTensor(lm_head_weight)},
                                     logits,
                                     Bind(ModelWeightRole::kLmHead)));
    nodes.push_back(MakePureNode(OpType::kArgmax,
                                 0,
                                 {logits},
                                 {token_ids}));

    return ModelGraph(config, std::move(nodes));
}

}// namespace aethermind
