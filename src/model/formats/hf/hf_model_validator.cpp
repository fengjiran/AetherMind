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

Status RequirePositive(int64_t value, std::string_view field_name) {
    if (value <= 0) {
        return Status::InvalidArgument(std::string("Model config field '") +
                                       std::string(field_name) +
                                       "' must be positive, got " + std::to_string(value));
    }
    return Status::Ok();
}

bool IsUnknownDType(const DataType& dtype) {
    return dtype == DataType{};
}

bool IsSupportedDenseWeightDTypeHint(const DataType& dtype) {
    return IsUnknownDType(dtype) || dtype.IsFloat32() || dtype.IsFloat16() || dtype.IsBFloat16();
}

bool HasUnsupportedNamedDTypeHint(const HfModelConfig& config) {
    return !config.weight_dtype_hint_name.empty() &&
           config.weight_dtype_hint_name != "auto" &&
           IsUnknownDType(config.weight_dtype_hint);
}

bool IsSupportedRopeScalingType(std::string_view type) {
    const auto is = [&](std::string_view value) noexcept {
        return type == value;
    };

    return is("linear") || is("dynamic") || is("yarn") || is("llama3") || is("longrope");
}

bool IsSupportedActivation(std::string_view act) {
    constexpr std::string_view kSupported[] = {
            "silu",
            "gelu",
            "relu",
    };

    return std::ranges::any_of(kSupported, [act](std::string_view s) { return act == s; });
}

bool HasRopeScaling(const HfRopeConfig& rope) {
    return rope.scaling_factor.has_value() || !rope.scaling_type.empty();
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

Status HfModelValidator::ValidateConfig(const HfModelConfig& config, const ModelValidationOptions& options) {
    if (config.model_type.empty()) {
        return Status::InvalidArgument("Model config field 'model_type' must be provided");
    }

    if (config.model_type != "llama") {
        return Status::InvalidArgument("Only model_type=llama is supported in AetherMind Phase 1");
    }

    AM_RETURN_IF_ERROR(RequirePositive(config.hidden_size, "hidden_size"));
    AM_RETURN_IF_ERROR(RequirePositive(config.intermediate_size, "intermediate_size"));
    AM_RETURN_IF_ERROR(RequirePositive(config.num_hidden_layers, "num_hidden_layers"));
    AM_RETURN_IF_ERROR(RequirePositive(config.num_attention_heads, "num_attention_heads"));
    AM_RETURN_IF_ERROR(RequirePositive(config.num_key_value_heads, "num_key_value_heads"));
    AM_RETURN_IF_ERROR(RequirePositive(config.vocab_size, "vocab_size"));
    AM_RETURN_IF_ERROR(RequirePositive(config.max_position_embeddings, "max_position_embeddings"));

    if (config.rms_norm_eps <= 0.0) {
        return Status::InvalidArgument("Model config field 'rms_norm_eps' must be positive");
    }

    if (config.rope.theta <= 0.0) {
        return Status::InvalidArgument("Model config field 'rope.theta' must be positive");
    }

    if (config.hidden_size % config.num_attention_heads != 0) {
        return Status::InvalidArgument("Model config hidden_size must be divisible by num_attention_heads");
    }

    const int64_t inferred_head_dim = config.hidden_size / config.num_attention_heads;
    if (config.head_dim != 0 && config.head_dim != inferred_head_dim) {
        return Status::InvalidArgument("Model config head_dim must match hidden_size / num_attention_heads");
    }

    if (config.num_attention_heads % config.num_key_value_heads != 0) {
        return Status::InvalidArgument("Model config num_attention_heads must be divisible by num_key_value_heads");
    }

    if (config.num_key_value_heads > config.num_attention_heads) {
        return Status::InvalidArgument("Model config num_key_value_heads must not exceed num_attention_heads");
    }

    if (config.intermediate_size < config.hidden_size) {
        return Status::InvalidArgument("Model config intermediate_size must be greater than or equal to hidden_size");
    }

    if (!IsSupportedActivation(config.hidden_act)) {
        return Status::InvalidArgument(
                "Model config field 'hidden_act' must be one of: silu, gelu, relu");
    }

    if (!options.allow_bias && (config.attention_bias || config.mlp_bias)) {
        return Status::InvalidArgument("Bias linear is not supported in AetherMind Phase 1");
    }

    const bool has_rope_scaling = HasRopeScaling(config.rope);
    if (!options.allow_rope_scaling && has_rope_scaling) {
        return Status::InvalidArgument("RoPE scaling is not supported in AetherMind Phase 1");
    }

    if (options.allow_rope_scaling && has_rope_scaling) {
        if (!config.rope.scaling_factor.has_value()) {
            return Status::InvalidArgument("Model config field 'rope.scaling_factor' must be provided when RoPE scaling is configured");
        }

        if (*config.rope.scaling_factor <= 0.0) {
            return Status::InvalidArgument("Model config field 'rope.scaling_factor' must be positive");
        }

        if (!IsSupportedRopeScalingType(config.rope.scaling_type)) {
            return Status::InvalidArgument("Unsupported RoPE scaling type in model config");
        }
    }

    if (HasUnsupportedNamedDTypeHint(config) || !IsSupportedDenseWeightDTypeHint(config.weight_dtype_hint)) {
        return Status::InvalidArgument("Only dense float torch_dtype hints are supported in AetherMind Phase 1");
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
