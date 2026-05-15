#include "aethermind/model/formats/hf/hf_weight_resolver.h"

#include <string>
#include <string_view>
#include <utility>

namespace aethermind {

namespace {

StatusOr<RawWeightView> FindRequiredWeight(const RawWeightTable& weights, const std::string& weight_name) {
    const auto it = weights.find(weight_name);
    if (it == weights.end()) {
        return Status::InvalidArgument("HF weight resolver missing required weight '" + weight_name + "'");
    }
    return it->second;
}

std::string LayerWeightName(int64_t layer_index, std::string_view suffix) {
    return "model.layers." + std::to_string(layer_index) + std::string(suffix);
}

}// namespace

namespace hf {

StatusOr<ModelWeightIndex> ResolveWeights(const HfModelConfig& config,
                                          const RawWeightTable& weights) {
    if (config.num_hidden_layers <= 0) {
        return Status::InvalidArgument("Model config field 'num_hidden_layers' must be positive");
    }

    ModelWeightIndex index;
    auto embed_tokens = FindRequiredWeight(weights, "model.embed_tokens.weight");
    if (!embed_tokens.ok()) {
        return embed_tokens.status();
    }
    index.embed_tokens = std::move(*embed_tokens);

    auto final_norm = FindRequiredWeight(weights, "model.norm.weight");
    if (!final_norm.ok()) {
        return final_norm.status();
    }
    index.final_norm = std::move(*final_norm);

    if (const auto lm_head = weights.find("lm_head.weight"); lm_head != weights.end()) {
        index.lm_head = lm_head->second;
    }

    index.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
    for (int64_t i = 0; i < config.num_hidden_layers; ++i) {
        DecoderLayerRawWeights layer;

        auto input_rmsnorm = FindRequiredWeight(weights, LayerWeightName(i, ".input_layernorm.weight"));
        if (!input_rmsnorm.ok()) {
            return input_rmsnorm.status();
        }
        layer.norm.input_rmsnorm = std::move(*input_rmsnorm);

        auto post_attn_rmsnorm = FindRequiredWeight(weights, LayerWeightName(i, ".post_attention_layernorm.weight"));
        if (!post_attn_rmsnorm.ok()) {
            return post_attn_rmsnorm.status();
        }
        layer.norm.post_attn_rmsnorm = std::move(*post_attn_rmsnorm);

        auto q_proj = FindRequiredWeight(weights, LayerWeightName(i, ".self_attn.q_proj.weight"));
        if (!q_proj.ok()) {
            return q_proj.status();
        }
        layer.attn.q_proj = std::move(*q_proj);

        auto k_proj = FindRequiredWeight(weights, LayerWeightName(i, ".self_attn.k_proj.weight"));
        if (!k_proj.ok()) {
            return k_proj.status();
        }
        layer.attn.k_proj = std::move(*k_proj);

        auto v_proj = FindRequiredWeight(weights, LayerWeightName(i, ".self_attn.v_proj.weight"));
        if (!v_proj.ok()) {
            return v_proj.status();
        }
        layer.attn.v_proj = std::move(*v_proj);

        auto o_proj = FindRequiredWeight(weights, LayerWeightName(i, ".self_attn.o_proj.weight"));
        if (!o_proj.ok()) {
            return o_proj.status();
        }
        layer.attn.o_proj = std::move(*o_proj);

        auto gate_proj = FindRequiredWeight(weights, LayerWeightName(i, ".mlp.gate_proj.weight"));
        if (!gate_proj.ok()) {
            return gate_proj.status();
        }
        layer.ffn.gate_proj = std::move(*gate_proj);

        auto up_proj = FindRequiredWeight(weights, LayerWeightName(i, ".mlp.up_proj.weight"));
        if (!up_proj.ok()) {
            return up_proj.status();
        }
        layer.ffn.up_proj = std::move(*up_proj);

        auto down_proj = FindRequiredWeight(weights, LayerWeightName(i, ".mlp.down_proj.weight"));
        if (!down_proj.ok()) {
            return down_proj.status();
        }
        layer.ffn.down_proj = std::move(*down_proj);

        index.layers.push_back(std::move(layer));
    }

    return index;
}

}// namespace hf

}// namespace aethermind
