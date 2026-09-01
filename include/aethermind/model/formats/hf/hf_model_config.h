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
/// @brief RoPE scaling strategies defined in HF config.json.
///
/// The underlying type is fixed to uint8_t to keep the config struct compact
/// and its serialization stable.
enum class HfRopeScalingType : uint8_t {
    kNone = 0,     // No rope_scaling entry; standard RoPE applies.
    kLinear,       // HF string: "linear"
    kDynamicNtk,   // HF string: "dynamic"; Dynamic NTK scaling
    kYarn,         // HF string: "yarn"
    kLlama3,       // HF string: "llama3"; LLaMA 3.x piecewise scaling
    kLongRope,     // HF string: "longrope"
    kSu,           // HF string: "su"; legacy/extended HF scaling type
    kUnknown = 255 // Parsed but not recognized by this engine.
};

/// @brief Maps an HF config.json rope_scaling.type string to its enum value.
///
/// @param type_str Value of the "rope_scaling.type" key.
/// @return Matching enum value; kNone for empty or "default", kUnknown for
/// unrecognized strings.
/// @note Inverse of ToString except that kUnknown round-trips to "unknown",
/// which is not a valid HF input string.
inline HfRopeScalingType ParseRopeScalingType(std::string_view type_str) noexcept {
    const auto is = [type_str](std::string_view value) noexcept {
        return type_str == value;
    };

    if (type_str.empty() || is("default")) {
        return HfRopeScalingType::kNone;
    }

    if (is("linear")) {
        return HfRopeScalingType::kLinear;
    }

    if (is("dynamic") || is("dynamic_ntk")) {
        return HfRopeScalingType::kDynamicNtk;
    }

    if (is("yarn")) {
        return HfRopeScalingType::kYarn;
    }

    if (is("llama3")) {
        return HfRopeScalingType::kLlama3;
    }

    if (is("longrope")) {
        return HfRopeScalingType::kLongRope;
    }

    if (is("su")) {
        return HfRopeScalingType::kSu;
    }
    return HfRopeScalingType::kUnknown;
}

/// @brief Maps a scaling enum value back to its HF config.json string.
///
/// @param scaling_type Enum value to convert.
/// @return HF string for the value; "unknown" for kUnknown.
inline std::string_view ToString(HfRopeScalingType scaling_type) noexcept {
    switch (scaling_type) {
        case HfRopeScalingType::kNone:
            return "default";
        case HfRopeScalingType::kLinear:
            return "linear";
        case HfRopeScalingType::kDynamicNtk:
            return "dynamic";
        case HfRopeScalingType::kYarn:
            return "yarn";
        case HfRopeScalingType::kLlama3:
            return "llama3";
        case HfRopeScalingType::kLongRope:
            return "longrope";
        case HfRopeScalingType::kSu:
            return "su";
        case HfRopeScalingType::kUnknown:
            return "unknown";
    }
    return "unknown";
}

/// @brief RoPE configuration for the model.
///
/// theta and scaling_type are unconditional; scaling_factor is present only
/// when config.json provides it.
struct HfRopeConfig {
    double theta = 10000.0; // Standard RoPE base frequency.
    std::optional<double> scaling_factor{};
    HfRopeScalingType scaling_type = HfRopeScalingType::kNone;
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
