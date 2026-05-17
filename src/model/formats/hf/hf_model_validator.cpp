#include "aethermind/model/formats/hf/hf_model_validator.h"
#include "aethermind/utils/overflow_check.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
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

std::string ShapeToString(const std::vector<int64_t>& shape) {
    std::string result = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) result += ", ";
        result += std::to_string(shape[i]);
    }
    result += "]";
    return result;
}

Status ValidateWeightViewIntegrity(const RawWeightView& view, std::string_view weight_name) {
    if (!view.IsValid()) {
        return Status::InvalidArgument(std::string("Weight '") + std::string(weight_name) +
                                       "' has an invalid RawWeightView (null storage, invalid dtype, "
                                       "or null data with non-zero byte size)");
    }

    if (view.shape.empty()) {
        return Status::InvalidArgument(std::string("Weight '") + std::string(weight_name) +
                                       "' has empty shape");
    }

    for (int64_t dim: view.shape) {
        if (dim <= 0) {
            return Status::InvalidArgument(
                    std::string("Weight '") + std::string(weight_name) +
                    "' has non-positive shape dimension " + std::to_string(dim));
        }
    }

    uint64_t numel = 0;
    if (SafeMultiplyU64(view.shape, &numel)) {
        return Status::InvalidArgument(std::string("Weight '") + std::string(weight_name) +
                                       "' shape element count overflows uint64_t");
    }

    const auto itemsize = static_cast<size_t>(view.dtype.nbytes());
    if (itemsize == 0) {
        return Status::InvalidArgument(std::string("Weight '") + std::string(weight_name) +
                                       "' has dtype with zero byte size");
    }

    size_t expected_bytes = 0;
    if (CheckOverflowMul(numel, itemsize, &expected_bytes)) {
        return Status::InvalidArgument(std::string("Weight '") + std::string(weight_name) +
                                       "' byte size overflows size_t");
    }

    if (view.bytes != expected_bytes) {
        return Status::InvalidArgument(
                std::string("Weight '") + std::string(weight_name) +
                "' byte size mismatch: shape " + ShapeToString(view.shape) +
                " × " + std::to_string(itemsize) + " bytes/elem" +
                " = " + std::to_string(expected_bytes) +
                " but view reports " + std::to_string(view.bytes));
    }

    return Status::Ok();
}

Status ValidateWeight(const RawWeightTable& weights, std::string_view weight_name) {
    const auto it = weights.find(std::string(weight_name));
    if (it == weights.end()) {
        return Status::InvalidArgument(std::string("Model weight set is missing required weight '") +
                                       std::string(weight_name) + "'");
    }
    return ValidateWeightViewIntegrity(it->second, weight_name);
}

Status ValidateLayerWeight(const RawWeightTable& weights,
                           int64_t layer_index,
                           std::string_view suffix) {
    const std::string name = "model.layers." + std::to_string(layer_index) + std::string(suffix);
    return ValidateWeight(weights, name);
}

}// namespace

Status HfModelValidator::ValidateConfig(const HfModelConfig& config, const ModelValidationOptions&) {
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

Status HfModelValidator::ValidateWeightSet(const HfModelConfig& config,
                                           const RawWeightTable& weights,
                                           const ModelValidationOptions&) {
    AM_RETURN_IF_ERROR(RequirePositive(config.num_hidden_layers, "num_hidden_layers"));
    AM_RETURN_IF_ERROR(ValidateWeight(weights, "model.embed_tokens.weight"));
    AM_RETURN_IF_ERROR(ValidateWeight(weights, "model.norm.weight"));

    for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
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

        for (const std::string_view suffix: kRequiredLayerWeightSuffixes) {
            AM_RETURN_IF_ERROR(ValidateLayerWeight(weights, layer, suffix));
        }
    }

    return Status::Ok();
}

Status HfModelValidator::ValidateResolvedModel(const HfModelConfig&,
                                               const ModelWeightIndex&,
                                               const ModelValidationOptions&) {
    return Status(StatusCode::kUnimplemented,
                  "HfModelValidator::ValidateResolvedModel is not implemented yet");
}

}// namespace aethermind
