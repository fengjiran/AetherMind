#ifndef AETHERMIND_MODEL_GRAPH_OP_PARAMS_H
#define AETHERMIND_MODEL_GRAPH_OP_PARAMS_H

#include "aethermind/model/formats/hf/hf_model_config.h"

#include <cstdint>
#include <optional>
#include <variant>

namespace aethermind {

struct EmbeddingParams {};

struct RmsNormParams {
    float eps = 1.0e-5f;
};

struct LinearParams {};

struct RoPEParams {
    int64_t head_dim = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t max_position_embeddings = 0;
    double theta = 10000.0;
    std::optional<double> scaling_factor{};
    HfRopeScalingType scaling_type = HfRopeScalingType::kNone;
};

struct MatMulParams {
    bool transpose_rhs = false;
};

struct SoftmaxParams {
    int64_t axis = -1;
};

struct AddParams {};

struct SiluMulParams {};

struct KVCacheUpdateParams {};

struct AttentionParams {
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
};

struct ArgmaxParams {
    int64_t axis = -1;
};

using OpParams = std::variant<std::monostate,
                              EmbeddingParams,
                              RmsNormParams,
                              LinearParams,
                              RoPEParams,
                              MatMulParams,
                              SoftmaxParams,
                              AddParams,
                              SiluMulParams,
                              KVCacheUpdateParams,
                              AttentionParams,
                              ArgmaxParams>;

}// namespace aethermind

#endif
