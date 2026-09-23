#ifndef AETHERMIND_MODEL_ROPE_PARAMS_BUILDER_INTERNAL_H
#define AETHERMIND_MODEL_ROPE_PARAMS_BUILDER_INTERNAL_H

/// @file rope_params_builder.h
/// @brief Family-independent HF RoPE normalization shared by per-family graph
/// builders. Internal to the model module.

#include "aethermind/base/status.h"
#include "aethermind/model/formats/hf/hf_model_config.h"
#include "aethermind/operators/op_params.h"

#include <cstdint>

namespace aethermind::detail {

/// @brief Normalizes raw HF RoPE fields into one typed semantic alternative.
///
/// This is the single HF-to-semantic conversion boundary for RoPE: optional HF
/// fields and aliases are normalized into one complete RoPEAlgorithmParams
/// variant, and unsupported HF variants are rejected without producing a
/// partial graph. Family-independent — Qwen and MoE builders reuse it as-is.
StatusOr<RoPEParams> MakeRoPEParams(const HfModelConfig& config,
                                    int64_t head_dim);

} // namespace aethermind::detail

#endif // AETHERMIND_MODEL_ROPE_PARAMS_BUILDER_INTERNAL_H