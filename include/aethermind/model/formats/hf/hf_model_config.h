#ifndef AETHERMIND_MODEL_FORMATS_HF_HF_MODEL_CONFIG_H
#define AETHERMIND_MODEL_FORMATS_HF_HF_MODEL_CONFIG_H

/// @file hf_model_config.h
/// @brief Parsed representation of a HuggingFace model's config.json.
///
/// Field names mirror config.json keys so parsing is a direct projection.
/// Values that config.json may omit keep zero-value defaults; consumers
/// derive or validate them (see ModelGraphBuilder and HfModelValidator).
#include "aethermind/dtypes/data_type.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace aethermind {

// NOLINTBEGIN(readability-identifier-naming)
// Names below mirror config.json keys and HF strings 1:1, so the naming lint
// is suppressed for the whole block instead of per name.
/// @brief RoPE algorithm spellings accepted from HF config.json.
///
/// The underlying type is fixed to uint8_t to keep the config struct compact
/// and its serialization stable.
enum class HfRoPEAlgorithm : uint8_t {
    kStandard = 0, // Empty or "default" HF rope_type.
    kLinear = 1,
    kDynamicNtk = 2,
    kYarn = 3,
    kLlama3 = 4,
    kLongRope = 5,
    kSu = 6,       // Legacy Phi-3 spelling normalized to LongRoPE.
    kUnknown = 255 // Parsed but not recognized by this engine.
};

/// @brief Maps an HF config.json rope_type string to its format-layer enum.
///
/// @param type_str Value of the legacy `rope_scaling.type` or current
///                 `rope_parameters.rope_type` key.
/// @return Matching enum value; kStandard for empty or "default", kUnknown for
/// unrecognized strings.
/// @note Inverse of ToString except that kUnknown round-trips to "unknown",
/// which is not a valid HF input string.
inline HfRoPEAlgorithm ParseHfRoPEAlgorithm(std::string_view type_str) noexcept {
    const auto is = [type_str](std::string_view value) noexcept {
        return type_str == value;
    };

    if (type_str.empty() || is("default")) {
        return HfRoPEAlgorithm::kStandard;
    }

    if (is("linear")) {
        return HfRoPEAlgorithm::kLinear;
    }

    if (is("dynamic") || is("dynamic_ntk")) {
        return HfRoPEAlgorithm::kDynamicNtk;
    }

    if (is("yarn")) {
        return HfRoPEAlgorithm::kYarn;
    }

    if (is("llama3")) {
        return HfRoPEAlgorithm::kLlama3;
    }

    if (is("longrope")) {
        return HfRoPEAlgorithm::kLongRope;
    }

    if (is("su")) {
        return HfRoPEAlgorithm::kSu;
    }
    return HfRoPEAlgorithm::kUnknown;
}

/// @brief Maps an HF algorithm enum value back to its config.json string.
///
/// @param algorithm Enum value to convert.
/// @return HF string for the value; "unknown" for kUnknown.
inline std::string_view ToString(HfRoPEAlgorithm algorithm) noexcept {
    switch (algorithm) {
        case HfRoPEAlgorithm::kStandard:
            return "default";
        case HfRoPEAlgorithm::kLinear:
            return "linear";
        case HfRoPEAlgorithm::kDynamicNtk:
            return "dynamic";
        case HfRoPEAlgorithm::kYarn:
            return "yarn";
        case HfRoPEAlgorithm::kLlama3:
            return "llama3";
        case HfRoPEAlgorithm::kLongRope:
            return "longrope";
        case HfRoPEAlgorithm::kSu:
            return "su";
        case HfRoPEAlgorithm::kUnknown:
            return "unknown";
    }
    return "unknown";
}

/// @brief RoPE configuration for the model.
///
/// Raw HuggingFace RoPE fields. ModelGraphBuilder is responsible for turning
/// these format-specific optional fields into one typed semantic alternative.
struct HfRopeConfig {
    double theta = 10000.0; // Standard RoPE base frequency.
    std::optional<double> factor{};
    std::optional<int64_t> original_context_length{};
    std::optional<double> beta_fast{};
    std::optional<double> beta_slow{};
    std::optional<double> attention_factor{};
    std::optional<double> low_frequency_factor{};
    std::optional<double> high_frequency_factor{};
    std::optional<double> mscale{};
    std::optional<double> mscale_all_dim{};
    std::optional<bool> truncate_correction_range{};
    std::optional<double> partial_rotary_factor{};
    std::optional<int64_t> rotary_dim{};
    std::vector<double> short_factors{};
    std::vector<double> long_factors{};
    HfRoPEAlgorithm algorithm = HfRoPEAlgorithm::kStandard;
};

/// @brief Parsed HuggingFace config.json for a model directory.
///
/// Zero-value defaults mean "absent in config.json"; consumers must derive
/// or reject them (for example, head_dim derives as
/// hidden_size / num_attention_heads).
struct HfModelConfig {
    std::string model_type{};
    std::vector<std::string> architectures{};

    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t vocab_size = 0;
    int64_t max_position_embeddings = 0;
    int64_t head_dim = 0; // 0 = derive from hidden_size / num_attention_heads.

    double rms_norm_eps = 0.0;
    std::string hidden_act = "silu";

    bool tie_word_embeddings = false;
    bool attention_bias = false;
    bool mlp_bias = false;

    std::string weight_dtype_hint_name{}; // Raw config.json "torch_dtype" string.
    DataType weight_dtype_hint{};         // Parsed hint; undefined → Float32 fallback.
    HfRopeConfig rope{};
};
// NOLINTEND(readability-identifier-naming)

} // namespace aethermind

#endif // AETHERMIND_MODEL_FORMATS_HF_HF_MODEL_CONFIG_H
