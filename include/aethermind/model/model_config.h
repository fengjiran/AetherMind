#ifndef AETHERMIND_MODEL_MODEL_CONFIG_H
#define AETHERMIND_MODEL_MODEL_CONFIG_H

#include <cstdint>
#include <string>
#include <vector>

namespace aethermind {

// NOLINTBEGIN(readability-identifier-naming)
struct ModelConfig {
    std::string model_type{};
    std::vector<std::string> architectures{};

    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t vocab_size = 0;

    double rms_norm_eps = 0.0;
    bool tie_word_embeddings = false;
};
// NOLINTEND(readability-identifier-naming)

}// namespace aethermind

#endif// AETHERMIND_MODEL_MODEL_CONFIG_H
