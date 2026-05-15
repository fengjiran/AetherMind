#ifndef AETHERMIND_MODEL_MODEL_WEIGHT_INDEX_H
#define AETHERMIND_MODEL_MODEL_WEIGHT_INDEX_H

#include "aethermind/model/raw_weight.h"

#include <optional>
#include <vector>

namespace aethermind {

struct AttentionRawWeights {
    RawWeightView q_proj{};
    RawWeightView k_proj{};
    RawWeightView v_proj{};
    RawWeightView o_proj{};
};

struct FfnRawWeights {
    RawWeightView gate_proj{};
    RawWeightView up_proj{};
    RawWeightView down_proj{};
};

struct NormRawWeights {
    RawWeightView input_rmsnorm{};
    RawWeightView post_attn_rmsnorm{};
};

struct DecoderLayerRawWeights {
    NormRawWeights norm{};
    AttentionRawWeights attn{};
    FfnRawWeights ffn{};
};

struct ModelWeightIndex {
    RawWeightView embed_tokens{};
    RawWeightView final_norm{};
    std::optional<RawWeightView> lm_head{};
    std::vector<DecoderLayerRawWeights> layers{};

    AM_NODISCARD size_t NumLayers() const noexcept;
};

}// namespace aethermind

#endif
