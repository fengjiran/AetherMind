#ifndef AETHERMIND_MODEL_FORMATS_HF_HF_MODEL_CONFIG_H
#define AETHERMIND_MODEL_FORMATS_HF_HF_MODEL_CONFIG_H

#include "data_type.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace aethermind {

// NOLINTBEGIN(readability-identifier-naming)
// HuggingFace config.json 中定义的 RoPE 缩放策略。
// 固定底层类型，便于 Config 结构体紧凑存储与稳定序列化。
enum class HfRopeScalingType : uint8_t {
    kNone = 0,    // 未配置 rope_scaling；使用标准 RoPE
    kLinear,      // HF string: "linear"
    kDynamicNtk,  // HF string: "dynamic"; Dynamic NTK scaling
    kYarn,        // HF string: "yarn"
    kLlama3,      // HF string: "llama3"; LLaMA 3.x 分段缩放
    kLongRope,    // HF string: "longrope"
    kSu,          // HF string: "su"; legacy/extended HF scaling type
    kUnknown = 255// HF 字符串已解析但当前引擎尚未识别
};

inline HfRopeScalingType ParseRopeScalingType(std::string_view type_str) noexcept {
    const auto is = [type_str](std::string_view value) noexcept {
        return type_str.compare(value) == 0;
    };

    if (type_str.empty() || is("default")) return HfRopeScalingType::kNone;
    if (is("linear")) return HfRopeScalingType::kLinear;
    if (is("dynamic") || is("dynamic_ntk")) return HfRopeScalingType::kDynamicNtk;
    if (is("yarn")) return HfRopeScalingType::kYarn;
    if (is("llama3")) return HfRopeScalingType::kLlama3;
    if (is("longrope")) return HfRopeScalingType::kLongRope;
    if (is("su")) return HfRopeScalingType::kSu;
    return HfRopeScalingType::kUnknown;
}

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

struct HfRopeConfig {
    double theta = 10000.0;
    std::optional<double> scaling_factor{};
    HfRopeScalingType scaling_type = HfRopeScalingType::kNone;
};

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
    int64_t head_dim = 0;

    double rms_norm_eps = 0.0;
    std::string hidden_act = "silu";

    bool tie_word_embeddings = false;
    bool attention_bias = false;
    bool mlp_bias = false;

    std::string weight_dtype_hint_name{};
    DataType weight_dtype_hint{};
    HfRopeConfig rope{};
};
// NOLINTEND(readability-identifier-naming)

}// namespace aethermind

#endif// AETHERMIND_MODEL_FORMATS_HF_HF_MODEL_CONFIG_H
