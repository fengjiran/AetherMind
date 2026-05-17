#ifndef AETHERMIND_MODEL_FORMATS_HF_HF_MODEL_CONFIG_H
#define AETHERMIND_MODEL_FORMATS_HF_HF_MODEL_CONFIG_H

#include "data_type.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace aethermind {

// NOLINTBEGIN(readability-identifier-naming)
struct HfRopeConfig {
    double theta = 10000.0;
    std::optional<double> scaling_factor{};
    std::string scaling_type{};
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
