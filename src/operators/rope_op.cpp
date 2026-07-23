#include "aethermind/dtypes/data_type.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/shape_inference/tensor_spec.h"

#include <span>
#include <string>

namespace aethermind::detail {

StatusOr<InferenceResult> InferRoPE(const OpParams& /*params*/, std::span<const TensorSpec> inputs) {
    if (inputs.size() != 3) {
        return Status::InvalidArgument(
                "RoPE expects exactly 3 inputs (q, k, position_ids), got " +
                std::to_string(inputs.size()));
    }

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
