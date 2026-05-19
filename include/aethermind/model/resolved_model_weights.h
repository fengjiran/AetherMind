#ifndef AETHERMIND_MODEL_RESOLVED_MODEL_WEIGHTS_H
#define AETHERMIND_MODEL_RESOLVED_MODEL_WEIGHTS_H

#include "aethermind/model/raw_weight.h"

#include <optional>
#include <vector>

namespace aethermind {

struct AttnRawWeights {
    RawWeightView q_proj{};
    RawWeightView k_proj{};
    RawWeightView v_proj{};
    RawWeightView o_proj{};
};

struct MLPRawWeights {
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
    AttnRawWeights attn{};
    MLPRawWeights mlp{};
};

struct ResolvedModelWeights {
    RawWeightView embed_tokens{};
    RawWeightView final_norm{};
    std::optional<RawWeightView> lm_head{};
    std::vector<DecoderLayerRawWeights> layers{};

    AM_NODISCARD size_t NumLayers() const noexcept;
};

}// namespace aethermind

#endif
