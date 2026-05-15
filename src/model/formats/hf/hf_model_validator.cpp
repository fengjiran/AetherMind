#include "aethermind/model/formats/hf/hf_model_validator.h"

#include <algorithm>
#include <array>
#include <string>
#include <string_view>

namespace aethermind {
namespace {

bool ContainsLlamaArchitecture(const std::vector<std::string>& architectures) {
    return std::ranges::any_of(architectures, [](const std::string& architecture) {
        return architecture.find("Llama") != std::string::npos;
    });
}

Status RequirePositive(int64_t value, std::string_view field_name) {
    if (value <= 0) {
        return Status::InvalidArgument(std::string("Model config field '") +
                                       std::string(field_name) + "' must be positive");
    }
    return Status::Ok();
}

Status RequireWeight(const RawWeightTable& weights, std::string_view weight_name) {
    if (!weights.contains(std::string(weight_name))) {
        return Status::InvalidArgument(std::string("Model weight set is missing required weight '") +
                                       std::string(weight_name) + "'");
    }
    return Status::Ok();
}

Status RequireLayerWeight(const RawWeightTable& weights,
                          int64_t layer_index,
                          std::string_view suffix) {
    return RequireWeight(weights, "model.layers." + std::to_string(layer_index) + std::string(suffix));
}

}// namespace

Status HfModelValidator::ValidateConfig(const ModelConfig& config) {
    if (config.model_type != "llama" && !ContainsLlamaArchitecture(config.architectures)) {
        return Status::InvalidArgument("Only Llama-family dense decoder-only models are supported");
    }

    AM_RETURN_IF_ERROR(RequirePositive(config.hidden_size, "hidden_size"));
    AM_RETURN_IF_ERROR(RequirePositive(config.intermediate_size, "intermediate_size"));
    AM_RETURN_IF_ERROR(RequirePositive(config.num_hidden_layers, "num_hidden_layers"));
    AM_RETURN_IF_ERROR(RequirePositive(config.num_attention_heads, "num_attention_heads"));
    AM_RETURN_IF_ERROR(RequirePositive(config.num_key_value_heads, "num_key_value_heads"));
    AM_RETURN_IF_ERROR(RequirePositive(config.vocab_size, "vocab_size"));

    if (config.rms_norm_eps <= 0.0) {
        return Status::InvalidArgument("Model config field 'rms_norm_eps' must be positive");
    }

    if (config.hidden_size % config.num_attention_heads != 0) {
        return Status::InvalidArgument("Model config hidden_size must be divisible by num_attention_heads");
    }

    if (config.num_key_value_heads > config.num_attention_heads) {
        return Status::InvalidArgument("Model config num_key_value_heads must not exceed num_attention_heads");
    }

    return Status::Ok();
}

Status HfModelValidator::ValidateWeightSet(const ModelConfig& config, const RawWeightTable& weights) {
    AM_RETURN_IF_ERROR(RequirePositive(config.num_hidden_layers, "num_hidden_layers"));
    AM_RETURN_IF_ERROR(RequireWeight(weights, "model.embed_tokens.weight"));
    AM_RETURN_IF_ERROR(RequireWeight(weights, "model.norm.weight"));

    constexpr std::array<std::string_view, 9> kRequiredLayerWeightSuffixes{
            ".self_attn.q_proj.weight",
            ".self_attn.k_proj.weight",
            ".self_attn.v_proj.weight",
            ".self_attn.o_proj.weight",
            ".mlp.gate_proj.weight",
            ".mlp.up_proj.weight",
            ".mlp.down_proj.weight",
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
    };

    for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        for (const std::string_view suffix: kRequiredLayerWeightSuffixes) {
            AM_RETURN_IF_ERROR(RequireLayerWeight(weights, layer, suffix));
        }
    }

    return Status::Ok();
}

}// namespace aethermind
