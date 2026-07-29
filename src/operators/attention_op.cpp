#include "aethermind/operators/operator_inference.h"

namespace aethermind::detail {

StatusOr<InferenceResult> InferAttention(const OpParams& params, std::span<const TensorSpec> inputs) {
    if (!std::holds_alternative<AttentionParams>(params)) {
        return Status::InvalidArgument("Attention node requires AttentionParams");
    }
    AM_RETURN_IF_ERROR(ValidateInferenceInputCount(OpType::kAttention, inputs));

    const auto& q_spec = inputs[0];

    if (q_spec.dtype != DataType::Float32()) {
        return Status::InvalidArgument("Attention only supports float32 q input in Phase 1");
    }

    return InferenceResult{
            .outputs = {q_spec},
    };
}

}// namespace aethermind::detail
