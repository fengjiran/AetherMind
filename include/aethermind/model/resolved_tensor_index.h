#ifndef AETHERMIND_MODEL_RESOLVED_TENSOR_INDEX_H
#define AETHERMIND_MODEL_RESOLVED_TENSOR_INDEX_H

#include "data_type.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace aethermind {

struct RawTensorBacking {
    virtual ~RawTensorBacking() = default;
};

struct RawTensorView {
    const std::byte* data = nullptr;
    size_t bytes = 0;
    DataType dtype{};
    std::vector<int64_t> shape{};
    std::shared_ptr<const RawTensorBacking> backing{};

    AM_NODISCARD bool IsValid() const noexcept {
        return backing != nullptr && dtype.bits() > 0 && (data != nullptr || bytes == 0);
    }

    AM_NODISCARD bool IsAligned(size_t alignment) const noexcept {
        return alignment != 0 && data != nullptr &&
               reinterpret_cast<std::uintptr_t>(data) % alignment == 0;
    }
};

using RawTensorTable = std::unordered_map<std::string, RawTensorView>;

struct ResolvedAttentionRawWeights {
    RawTensorView q_proj{};
    RawTensorView k_proj{};
    RawTensorView v_proj{};
    RawTensorView o_proj{};
};

struct ResolvedFfnRawWeights {
    RawTensorView gate_proj{};
    RawTensorView up_proj{};
    RawTensorView down_proj{};
};

struct ResolvedNormRawWeights {
    RawTensorView input_rmsnorm{};
    RawTensorView post_attn_rmsnorm{};
};

struct ResolvedDecoderLayerRaw {
    ResolvedNormRawWeights norm{};
    ResolvedAttentionRawWeights attn{};
    ResolvedFfnRawWeights ffn{};
};

struct ResolvedTensorIndex {
    RawTensorView embed_tokens{};
    RawTensorView final_norm{};
    std::optional<RawTensorView> lm_head{};
    std::vector<ResolvedDecoderLayerRaw> layers{};

    AM_NODISCARD size_t NumLayers() const noexcept;
};

}// namespace aethermind

#endif
