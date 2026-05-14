#include "aethermind/model/formats/hf/hf_tensor_name_resolver.h"

#include <string>
#include <string_view>

namespace aethermind {

namespace {

StatusOr<RawTensorView> FindRequiredTensor(const RawTensorTable& tensors, const std::string& tensor_name) {
    const auto it = tensors.find(tensor_name);
    if (it == tensors.end()) {
        return Status::InvalidArgument("HF tensor resolver missing required tensor '" + tensor_name + "'");
    }
    return it->second;
}

std::string LayerTensorName(int64_t layer_index, std::string_view suffix) {
    return "model.layers." + std::to_string(layer_index) + std::string(suffix);
}

}// namespace

namespace hf {

StatusOr<ResolvedTensorIndex> Resolve(const ModelConfig& config,
                                     const RawTensorTable& tensors) {
    if (config.num_hidden_layers <= 0) {
        return Status::InvalidArgument("Model config field 'num_hidden_layers' must be positive");
    }

    ResolvedTensorIndex index;

    auto embed_tokens = FindRequiredTensor(tensors, "model.embed_tokens.weight");
    if (!embed_tokens.ok()) {
        return embed_tokens.status();
    }
    index.embed_tokens = *embed_tokens;

    auto final_norm = FindRequiredTensor(tensors, "model.norm.weight");
    if (!final_norm.ok()) {
        return final_norm.status();
    }
    index.final_norm = *final_norm;

    if (const auto lm_head = tensors.find("lm_head.weight"); lm_head != tensors.end()) {
        index.lm_head = lm_head->second;
    }

    index.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
    for (int64_t layer_index = 0; layer_index < config.num_hidden_layers; ++layer_index) {
        ResolvedDecoderLayerRaw layer;

        auto input_rmsnorm = FindRequiredTensor(tensors, LayerTensorName(layer_index, ".input_layernorm.weight"));
        if (!input_rmsnorm.ok()) {
            return input_rmsnorm.status();
        }
        layer.norm.input_rmsnorm = *input_rmsnorm;

        auto post_attn_rmsnorm = FindRequiredTensor(tensors, LayerTensorName(layer_index, ".post_attention_layernorm.weight"));
        if (!post_attn_rmsnorm.ok()) {
            return post_attn_rmsnorm.status();
        }
        layer.norm.post_attn_rmsnorm = *post_attn_rmsnorm;

        auto q_proj = FindRequiredTensor(tensors, LayerTensorName(layer_index, ".self_attn.q_proj.weight"));
        if (!q_proj.ok()) {
            return q_proj.status();
        }
        layer.attn.q_proj = *q_proj;

        auto k_proj = FindRequiredTensor(tensors, LayerTensorName(layer_index, ".self_attn.k_proj.weight"));
        if (!k_proj.ok()) {
            return k_proj.status();
        }
        layer.attn.k_proj = *k_proj;

        auto v_proj = FindRequiredTensor(tensors, LayerTensorName(layer_index, ".self_attn.v_proj.weight"));
        if (!v_proj.ok()) {
            return v_proj.status();
        }
        layer.attn.v_proj = *v_proj;

        auto o_proj = FindRequiredTensor(tensors, LayerTensorName(layer_index, ".self_attn.o_proj.weight"));
        if (!o_proj.ok()) {
            return o_proj.status();
        }
        layer.attn.o_proj = *o_proj;

        auto gate_proj = FindRequiredTensor(tensors, LayerTensorName(layer_index, ".mlp.gate_proj.weight"));
        if (!gate_proj.ok()) {
            return gate_proj.status();
        }
        layer.ffn.gate_proj = *gate_proj;

        auto up_proj = FindRequiredTensor(tensors, LayerTensorName(layer_index, ".mlp.up_proj.weight"));
        if (!up_proj.ok()) {
            return up_proj.status();
        }
        layer.ffn.up_proj = *up_proj;

        auto down_proj = FindRequiredTensor(tensors, LayerTensorName(layer_index, ".mlp.down_proj.weight"));
        if (!down_proj.ok()) {
            return down_proj.status();
        }
        layer.ffn.down_proj = *down_proj;

        index.layers.push_back(layer);
    }

    return index;
}

}// namespace hf

}// namespace aethermind
