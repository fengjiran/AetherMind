#include "aethermind/model/formats/hf/hf_model_validator.h"
#include "aethermind/utils/overflow_check.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <ranges>
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
    constexpr std::string_view kSupported[] = {
            "linear",
            "dynamic",
            "yarn",
            "llama3",
            "longrope",
    };

    return std::ranges::any_of(kSupported, [type](std::string_view s) { return type == s; });
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

    if (!view.is_contiguous) {
        return Status::InvalidArgument(std::string("Weight '") + std::string(weight_name) +
                                       "' is non-contiguous; non-contiguous weights are not supported");
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

    if (numel == 0) {
        return Status::InvalidArgument(std::string("Weight '") + std::string(weight_name) +
                                       "' is an empty tensor");
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

    if (view.bytes == 0) {
        return Status::InvalidArgument(std::string("Weight '") + std::string(weight_name) +
                                       "' is an empty tensor");
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

constexpr std::string_view kInputNormSuffix = ".input_layernorm.weight";
constexpr std::string_view kPostAttnNormSuffix = ".post_attention_layernorm.weight";

bool IsLayerNormSuffix(std::string_view suffix) {
    return suffix == kInputNormSuffix || suffix == kPostAttnNormSuffix;
}

bool ContainsAnyOf(std::string_view name, std::string_view pattern) {
    return name.find(pattern) != std::string_view::npos;
}

bool EndsWithComponent(std::string_view name, std::string_view component) {
    if (name.size() < component.size()) return false;
    if (!name.ends_with(component)) return false;
    if (name.size() == component.size()) return true;
    return name[name.size() - component.size() - 1] == '.';
}

Status DetectUnsupportedTensors(const RawWeightTable& weights,
                                const ModelValidationOptions& options) {
    // MoE tensors are always rejected regardless of options.
    constexpr std::string_view kMoePatterns[] = {
            "experts",
            "router",
            "moe",
            "shared_experts",
    };

    constexpr std::string_view kBiasPatterns[] = {
            ".q_proj.bias",
            ".k_proj.bias",
            ".v_proj.bias",
            ".o_proj.bias",
            ".gate_proj.bias",
            ".up_proj.bias",
            ".down_proj.bias",
            "lm_head.bias",
    };

    constexpr std::string_view kLoraPatterns[] = {
            "lora_A",
            "lora_B",
            "adapter",
            "peft",
    };

    constexpr std::string_view kQuantizedPatterns[] = {
            "qweight",
            "qzeros",
            "scales",
            "g_idx",
            "bits",
            "group_size",
    };

    for (const auto& name: (weights | std::views::keys)) {
        for (const auto& pattern: kMoePatterns) {
            if (ContainsAnyOf(name, pattern)) {
                return Status::InvalidArgument(
                        "Unsupported tensor found: tensor=" + name +
                        ", reason=MoE weights are not supported in AetherMind Phase 1");
            }
        }

        if (!options.allow_bias) {
            for (const auto& pattern: kBiasPatterns) {
                if (ContainsAnyOf(name, pattern)) {
                    return Status::InvalidArgument(
                            "Unsupported tensor found: tensor=" + name +
                            ", reason=Bias weights are not supported in AetherMind Phase 1");
                }
            }
        }

        if (!options.allow_lora_or_adapter) {
            for (const auto& pattern: kLoraPatterns) {
                if (ContainsAnyOf(name, pattern)) {
                    return Status::InvalidArgument(
                            "Unsupported tensor found: tensor=" + name +
                            ", reason=LoRA/adapter weights are not supported in AetherMind Phase 1");
                }
            }
        }

        if (!options.allow_quantized_tensors) {
            for (const auto& pattern: kQuantizedPatterns) {
                if (EndsWithComponent(name, pattern)) {
                    return Status::InvalidArgument(
                            "Unsupported tensor found: tensor=" + name +
                            ", reason=Quantized weights are not supported in AetherMind Phase 1");
                }
            }
        }
    }

    return Status::Ok();
}

Status ValidateLmHeadRequirement(const HfModelConfig& config,
                                 const RawWeightTable& weights,
                                 const ModelValidationOptions& options) {
    const bool has_lm_head = weights.contains("lm_head.weight");

    if (!has_lm_head && !config.tie_word_embeddings) {
        return Status::InvalidArgument(
                "Required weight 'lm_head.weight' is missing and "
                "tie_word_embeddings is false");
    }

    if (!has_lm_head && options.require_lm_head_when_tied) {
        return Status::InvalidArgument(
                "Required weight 'lm_head.weight' is missing; "
                "require_lm_head_when_tied is enabled");
    }

    return Status::Ok();
}

bool IsLayerWeightName(std::string_view key) {
    constexpr std::string_view kPrefix = "model.layers.";
    return key.size() > kPrefix.size() && key.substr(0, kPrefix.size()) == kPrefix;
}

StatusOr<int64_t> ExtractLayerIndex(std::string_view key) {
    constexpr std::string_view kPrefix = "model.layers.";
    const auto rest = key.substr(kPrefix.size());
    const auto dot = rest.find('.');
    if (dot == std::string_view::npos) {
        return Status::InvalidArgument("Invalid layer weight name '" + std::string(key) + "': missing suffix after layer index");
    }

    const auto digits = rest.substr(0, dot);
    if (digits.empty()) {
        return Status::InvalidArgument("Invalid layer weight name '" + std::string(key) + "': missing layer index");
    }

    int64_t index = 0;
    const auto* begin = digits.data();
    const auto* end = begin + digits.size();
    const auto [ptr, error] = std::from_chars(begin, end, index);
    if (error == std::errc::result_out_of_range) {
        return Status::InvalidArgument("Invalid layer weight name '" + std::string(key) + "': layer index is out of range");
    }

    if (error != std::errc{} || ptr != end) {
        return Status::InvalidArgument("Invalid layer weight name '" + std::string(key) + "': layer index is not numeric");
    }
    return index;
}

Status ValidateLayerIndexCompleteness(const RawWeightTable& weights, int64_t num_hidden_layers) {
    for (const auto& name: (weights | std::views::keys)) {
        if (!IsLayerWeightName(name)) {
            continue;
        }

        const auto index = ExtractLayerIndex(name);
        if (!index.ok()) {
            return index.status();
        }

        if (*index >= num_hidden_layers) {
            return Status::InvalidArgument(
                    "Weight '" + name + "' has layer index " + std::to_string(*index) +
                    " but num_hidden_layers is " + std::to_string(num_hidden_layers));
        }
    }
    return Status::Ok();
}

std::string_view ExtractLayerSuffix(std::string_view key) {
    constexpr std::string_view kPrefix = "model.layers.";
    const auto rest = key.substr(kPrefix.size());
    const auto dot = rest.find('.');
    return rest.substr(dot);
}

bool IsKnownLayerWeight(const HfModelConfig& config, std::string_view name) {
    if (!IsLayerWeightName(name)) {
        return false;
    }

    if (const auto index = ExtractLayerIndex(name);
        !index.ok() || *index >= config.num_hidden_layers) {
        return false;
    }

    const auto suffix = ExtractLayerSuffix(name);
    return std::ranges::any_of(kRequiredLayerWeightSuffixes,
                               [suffix](std::string_view required) { return suffix == required; });
}

bool IsKnownIgnorableTensor(const HfModelConfig& config, std::string_view name) {
    constexpr std::string_view kGlobalRotaryInvFreq = "model.rotary_emb.inv_freq";
    constexpr std::string_view kLayerRotaryInvFreq = ".self_attn.rotary_emb.inv_freq";
    if (name == kGlobalRotaryInvFreq) {
        return true;
    }

    if (!IsLayerWeightName(name)) {
        return false;
    }

    if (const auto index = ExtractLayerIndex(name);
        !index.ok() || *index >= config.num_hidden_layers) {
        return false;
    }
    return ExtractLayerSuffix(name) == kLayerRotaryInvFreq;
}

bool IsKnownWeightTensor(const HfModelConfig& config, std::string_view name) {
    constexpr std::string_view kEmbedTokens = "model.embed_tokens.weight";
    constexpr std::string_view kFinalNorm = "model.norm.weight";
    constexpr std::string_view kLmHead = "lm_head.weight";
    return name == kEmbedTokens || name == kFinalNorm ||
           name == kLmHead || IsKnownLayerWeight(config, name);
}

Status ValidateUnknownTensorPolicy(const HfModelConfig& config,
                                   const RawWeightTable& weights,
                                   const ModelValidationOptions& options) {
    for (const auto& name: (weights | std::views::keys)) {
        if (IsKnownWeightTensor(config, name)) {
            continue;
        }

        if (!options.strict_tensor_names && options.allow_unknown_tensors && IsKnownIgnorableTensor(config, name)) {
            continue;
        }

        if (options.strict_tensor_names || !options.allow_unknown_tensors) {
            return Status::InvalidArgument("Unexpected tensor found: tensor=" + name);
        }
    }
    return Status::Ok();
}

bool IsSupportedWeightDType(const DataType& dtype) {
    return dtype.IsFloat32() || dtype.IsFloat16() || dtype.IsBFloat16();
}

Status ValidateWeightDType(const RawWeightView& view, std::string_view weight_name) {
    if (!IsSupportedWeightDType(view.dtype)) {
        return Status::InvalidArgument(
                "Invalid tensor dtype: tensor=" + std::string(weight_name) +
                ", expected one of [F32, F16, BF16], actual=" + std::string(DataTypeToString(view.dtype)));
    }
    return Status::Ok();
}

Status ExpectRank(const RawWeightView& view, int64_t expected_rank, std::string_view weight_name) {
    if (static_cast<int64_t>(view.shape.size()) != expected_rank) {
        return Status::InvalidArgument("Invalid tensor rank: tensor=" + std::string(weight_name) +
                                       ", expected=" + std::to_string(expected_rank) +
                                       ", actual=" + std::to_string(view.shape.size()));
    }
    return Status::Ok();
}

Status ExpectShape(const RawWeightView& view,
                   std::initializer_list<int64_t> expected_shape,
                   std::string_view weight_name) {
    if (const auto actual_size = view.shape.size(); actual_size != expected_shape.size()) {
        return Status::InvalidArgument("Invalid tensor shape: tensor=" + std::string(weight_name) +
                                       ", expected=" + ShapeToString(std::vector<int64_t>(expected_shape)) +
                                       ", actual=" + ShapeToString(view.shape));
    }

    size_t i = 0;
    for (const int64_t expected_dim: expected_shape) {
        if (view.shape[i] != expected_dim) {
            return Status::InvalidArgument("Invalid tensor shape: tensor=" + std::string(weight_name) +
                                           ", expected=" + ShapeToString(std::vector<int64_t>(expected_shape)) +
                                           ", actual=" + ShapeToString(view.shape));
        }
        ++i;
    }

    return Status::Ok();
}

Status ValidateUniformLinearDType(std::string_view weight_name,
                                  const RawWeightView& view,
                                  bool* has_linear_dtype,
                                  DataType* linear_dtype) {
    if (!*has_linear_dtype) {
        *linear_dtype = view.dtype;
        *has_linear_dtype = true;
        return Status::Ok();
    }

    if (view.dtype != *linear_dtype) {
        return Status::InvalidArgument(
                "Mixed linear tensor dtype: tensor=" + std::string(weight_name) +
                ", expected=" + std::string(DataTypeToString(*linear_dtype)) +
                ", actual=" + std::string(DataTypeToString(view.dtype)));
    }
    return Status::Ok();
}

Status ValidateResolvedModelDTypes(const HfModelConfig& config,
                                   const ResolvedModelWeights& resolved,
                                   const ModelValidationOptions& options) {
    bool has_linear_dtype = false;
    DataType linear_dtype{};

    const auto validate_linear = [&](std::string_view weight_name, const RawWeightView& view) -> Status {
        AM_RETURN_IF_ERROR(ValidateWeightDType(view, weight_name));
        if (options.require_uniform_linear_dtype) {
            AM_RETURN_IF_ERROR(ValidateUniformLinearDType(weight_name, view, &has_linear_dtype, &linear_dtype));
        }
        return Status::Ok();
    };

    const auto validate_norm = [](std::string_view weight_name, const RawWeightView& view) -> Status {
        return ValidateWeightDType(view, weight_name);
    };

    AM_RETURN_IF_ERROR(validate_linear("model.embed_tokens.weight", resolved.embed_tokens));
    AM_RETURN_IF_ERROR(validate_norm("model.norm.weight", resolved.final_norm));

    if (resolved.lm_head.has_value()) {
        AM_RETURN_IF_ERROR(ValidateWeightDType(*resolved.lm_head, "lm_head.weight"));
        if (config.tie_word_embeddings && resolved.lm_head->dtype != resolved.embed_tokens.dtype) {
            return Status::InvalidArgument(
                    "Tied embedding dtype mismatch: tensor=lm_head.weight, expected=" +
                    std::string(DataTypeToString(resolved.embed_tokens.dtype)) +
                    ", actual=" + std::string(DataTypeToString(resolved.lm_head->dtype)));
        }
        if (options.require_uniform_linear_dtype) {
            AM_RETURN_IF_ERROR(ValidateUniformLinearDType("lm_head.weight", *resolved.lm_head, &has_linear_dtype, &linear_dtype));
        }
    }

    for (size_t i = 0; i < resolved.layers.size(); ++i) {
        const auto& layer = resolved.layers[i];
        const std::string prefix = "model.layers." + std::to_string(i);

        AM_RETURN_IF_ERROR(validate_norm(prefix + ".input_layernorm.weight", layer.norm.input_rmsnorm));
        AM_RETURN_IF_ERROR(validate_norm(prefix + ".post_attention_layernorm.weight", layer.norm.post_attn_rmsnorm));

        AM_RETURN_IF_ERROR(validate_linear(prefix + ".self_attn.q_proj.weight", layer.attn.q_proj));
        AM_RETURN_IF_ERROR(validate_linear(prefix + ".self_attn.k_proj.weight", layer.attn.k_proj));
        AM_RETURN_IF_ERROR(validate_linear(prefix + ".self_attn.v_proj.weight", layer.attn.v_proj));
        AM_RETURN_IF_ERROR(validate_linear(prefix + ".self_attn.o_proj.weight", layer.attn.o_proj));

        AM_RETURN_IF_ERROR(validate_linear(prefix + ".mlp.gate_proj.weight", layer.mlp.gate_proj));
        AM_RETURN_IF_ERROR(validate_linear(prefix + ".mlp.up_proj.weight", layer.mlp.up_proj));
        AM_RETURN_IF_ERROR(validate_linear(prefix + ".mlp.down_proj.weight", layer.mlp.down_proj));
    }

    return Status::Ok();
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

    if (const int64_t inferred_head_dim = config.hidden_size / config.num_attention_heads;
        config.head_dim != 0 && config.head_dim != inferred_head_dim) {
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
        if (config.rope.scaling_type.empty()) {
            return Status::InvalidArgument(
                    "Model config field 'rope.scaling_type' must be provided when RoPE scaling is configured");
        }

        if (!IsSupportedRopeScalingType(config.rope.scaling_type)) {
            return Status::InvalidArgument("Unsupported RoPE scaling type in model config");
        }

        if (!config.rope.scaling_factor.has_value()) {
            return Status::InvalidArgument(
                    "Model config field 'rope.scaling_factor' must be provided when RoPE scaling is configured");
        }

        if (*config.rope.scaling_factor <= 0.0) {
            return Status::InvalidArgument("Model config field 'rope.scaling_factor' must be positive");
        }
    }

    if (HasUnsupportedNamedDTypeHint(config) || !IsSupportedDenseWeightDTypeHint(config.weight_dtype_hint)) {
        return Status::InvalidArgument("Only dense float torch_dtype hints are supported in AetherMind Phase 1");
    }

    return Status::Ok();
}

Status HfModelValidator::ValidateWeightSet(const HfModelConfig& config,
                                           const RawWeightTable& weights,
                                           const ModelValidationOptions& options) {
    AM_RETURN_IF_ERROR(RequirePositive(config.num_hidden_layers, "num_hidden_layers"));
    AM_RETURN_IF_ERROR(DetectUnsupportedTensors(weights, options));
    AM_RETURN_IF_ERROR(ValidateLayerIndexCompleteness(weights, config.num_hidden_layers));
    AM_RETURN_IF_ERROR(ValidateUnknownTensorPolicy(config, weights, options));

    bool has_linear_dtype = false;
    DataType linear_dtype{};

    AM_RETURN_IF_ERROR(ValidateWeight(weights, "model.embed_tokens.weight"));
    AM_RETURN_IF_ERROR(ValidateWeightDType(weights.at("model.embed_tokens.weight"), "model.embed_tokens.weight"));
    AM_RETURN_IF_ERROR(ExpectRank(weights.at("model.embed_tokens.weight"), 2, "model.embed_tokens.weight"));
    if (options.require_uniform_linear_dtype) {
        AM_RETURN_IF_ERROR(ValidateUniformLinearDType("model.embed_tokens.weight",
                                                      weights.at("model.embed_tokens.weight"),
                                                      &has_linear_dtype, &linear_dtype));
    }

    AM_RETURN_IF_ERROR(ValidateWeight(weights, "model.norm.weight"));
    AM_RETURN_IF_ERROR(ValidateWeightDType(weights.at("model.norm.weight"), "model.norm.weight"));
    AM_RETURN_IF_ERROR(ExpectRank(weights.at("model.norm.weight"), 1, "model.norm.weight"));

    for (int64_t i = 0; i < config.num_hidden_layers; ++i) {
        for (const std::string_view suffix: kRequiredLayerWeightSuffixes) {
            const std::string name = "model.layers." + std::to_string(i) + std::string(suffix);
            AM_RETURN_IF_ERROR(ValidateWeight(weights, name));
            const auto& view = weights.at(name);
            AM_RETURN_IF_ERROR(ValidateWeightDType(view, name));

            const bool is_norm = IsLayerNormSuffix(suffix);
            AM_RETURN_IF_ERROR(ExpectRank(view, is_norm ? 1 : 2, name));
            if (options.require_uniform_linear_dtype && !is_norm) {
                AM_RETURN_IF_ERROR(ValidateUniformLinearDType(name, view, &has_linear_dtype, &linear_dtype));
            }
        }
    }

    if (const auto lm_head_it = weights.find("lm_head.weight"); lm_head_it != weights.end()) {
        AM_RETURN_IF_ERROR(ValidateWeightViewIntegrity(lm_head_it->second, "lm_head.weight"));
        AM_RETURN_IF_ERROR(ValidateWeightDType(lm_head_it->second, "lm_head.weight"));
        AM_RETURN_IF_ERROR(ExpectRank(lm_head_it->second, 2, "lm_head.weight"));
        if (options.require_uniform_linear_dtype) {
            AM_RETURN_IF_ERROR(ValidateUniformLinearDType("lm_head.weight", lm_head_it->second, &has_linear_dtype, &linear_dtype));
        }
    }

    AM_RETURN_IF_ERROR(ValidateLmHeadRequirement(config, weights, options));

    return Status::Ok();
}

Status HfModelValidator::ValidateResolvedModel(const HfModelConfig& config,
                                               const ResolvedModelWeights& resolved,
                                               const ModelValidationOptions& options) {
    AM_RETURN_IF_ERROR(RequirePositive(config.num_attention_heads, "num_attention_heads"));
    AM_RETURN_IF_ERROR(RequirePositive(config.num_hidden_layers, "num_hidden_layers"));

    const int64_t hidden = config.hidden_size;
    const int64_t vocab = config.vocab_size;
    const int64_t intermediate = config.intermediate_size;
    const int64_t head_dim = config.hidden_size / config.num_attention_heads;
    const int64_t kv_hidden = config.num_key_value_heads * head_dim;

    if (resolved.layers.size() != static_cast<size_t>(config.num_hidden_layers)) {
        return Status::InvalidArgument("Resolved model layer count mismatch: expected=" +
                                       std::to_string(config.num_hidden_layers) +
                                       ", actual=" + std::to_string(resolved.layers.size()));
    }

    AM_RETURN_IF_ERROR(ExpectShape(resolved.embed_tokens,
                                   {vocab, hidden},
                                   "model.embed_tokens.weight"));
    AM_RETURN_IF_ERROR(ExpectShape(resolved.final_norm,
                                   {hidden},
                                   "model.norm.weight"));

    if (resolved.lm_head.has_value()) {
        AM_RETURN_IF_ERROR(ExpectShape(*resolved.lm_head,
                                       {vocab, hidden},
                                       "lm_head.weight"));
    } else if (!config.tie_word_embeddings) {
        return Status::InvalidArgument(
                "Required weight 'lm_head.weight' is missing and tie_word_embeddings is false");
    }

    for (size_t i = 0; i < resolved.layers.size(); ++i) {
        const auto& layer = resolved.layers[i];
        const std::string prefix = "model.layers." + std::to_string(i);

        AM_RETURN_IF_ERROR(ExpectShape(layer.norm.input_rmsnorm,
                                       {hidden},
                                       prefix + ".input_layernorm.weight"));
        AM_RETURN_IF_ERROR(ExpectShape(layer.norm.post_attn_rmsnorm,
                                       {hidden},
                                       prefix + ".post_attention_layernorm.weight"));
        AM_RETURN_IF_ERROR(ExpectShape(layer.attn.q_proj,
                                       {hidden, hidden},
                                       prefix + ".self_attn.q_proj.weight"));
        AM_RETURN_IF_ERROR(ExpectShape(layer.attn.k_proj,
                                       {kv_hidden, hidden},
                                       prefix + ".self_attn.k_proj.weight"));
        AM_RETURN_IF_ERROR(ExpectShape(layer.attn.v_proj,
                                       {kv_hidden, hidden},
                                       prefix + ".self_attn.v_proj.weight"));
        AM_RETURN_IF_ERROR(ExpectShape(layer.attn.o_proj,
                                       {hidden, hidden},
                                       prefix + ".self_attn.o_proj.weight"));
        AM_RETURN_IF_ERROR(ExpectShape(layer.mlp.gate_proj,
                                       {intermediate, hidden},
                                       prefix + ".mlp.gate_proj.weight"));
        AM_RETURN_IF_ERROR(ExpectShape(layer.mlp.up_proj,
                                       {intermediate, hidden},
                                       prefix + ".mlp.up_proj.weight"));
        AM_RETURN_IF_ERROR(ExpectShape(layer.mlp.down_proj,
                                       {hidden, intermediate},
                                       prefix + ".mlp.down_proj.weight"));
    }

    AM_RETURN_IF_ERROR(ValidateResolvedModelDTypes(config, resolved, options));

    return Status::Ok();
}

}// namespace aethermind
