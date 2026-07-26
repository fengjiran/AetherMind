#include "aethermind/operators/operator_inference.h"

namespace aethermind::detail {

StatusOr<InferenceResult> InferRoPE(const OpParams& params, std::span<const TensorSpec> inputs) {
    const auto* rope_params = std::get_if<RoPEParams>(&params);
    if (rope_params == nullptr) {
        return Status::InvalidArgument("RoPE node requires RoPEParams");
    }

    if (rope_params->head_dim <= 0 || rope_params->num_attention_heads <= 0 ||
        rope_params->num_key_value_heads <= 0 || rope_params->max_position_embeddings <= 0) {
        return Status::InvalidArgument("RoPEParams dimensions must be positive");
    }

    if (!std::isfinite(rope_params->theta) || rope_params->theta <= 0.0) {
        return Status::InvalidArgument("RoPEParams theta must be finite and positive");
    }
    AM_RETURN_IF_ERROR(ValidateInferenceInputCount(OpType::kRoPE, inputs));

    const auto& q_spec = inputs[0];
    const auto& k_spec = inputs[1];
    const auto& pos_spec = inputs[2];

    if (q_spec.dtype != DataType::Float32() || k_spec.dtype != DataType::Float32()) {
        return Status::InvalidArgument("RoPE only supports float32 q/k inputs in Phase 1");
    }

    if (pos_spec.dtype != DataType::Int(64)) {
        return Status::InvalidArgument("RoPE position_ids must be int64");
    }

    // GQA-compatible: q and k may have different last dimensions
    // (q_last = num_attention_heads * head_dim, k_last = num_key_value_heads * head_dim).
    // Only require rank equality and batch-dimension consistency.
    if (q_spec.shape.IsRanked() && k_spec.shape.IsRanked()) {
        if (q_spec.shape.rank() != k_spec.shape.rank()) {
            return Status::InvalidArgument("RoPE q and k must have the same rank");
        }

        if (const auto rank = q_spec.shape.rank().value(); rank >= 1) {
            for (size_t i = 0; i < rank - 1; ++i) {
                if (q_spec.shape[i] != k_spec.shape[i]) {
                    return Status::InvalidArgument(
                            "RoPE q and k batch dimensions must be identical");
                }
            }
        }
    }

    return InferenceResult{
            .outputs = {q_spec, k_spec},
    };
}

}// namespace aethermind::detail
