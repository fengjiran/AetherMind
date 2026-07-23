#include "aethermind/dtypes/data_type.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/shape_inference/tensor_spec.h"

#include <span>
#include <string>

namespace aethermind::detail {

StatusOr<InferenceResult> InferAttention(const OpParams& /*params*/, std::span<const TensorSpec> inputs) {
    if (inputs.size() != 3) {
        return Status::InvalidArgument(
                "Attention expects exactly 3 inputs (q, kCache, vCache), got " + std::to_string(inputs.size()));
    }

    const auto& q_spec = inputs[0];

    if (q_spec.dtype != DataType::Float32()) {
        return Status::InvalidArgument("Attention only supports float32 q input in Phase 1");
    }

    return InferenceResult{
            .outputs = {q_spec},
    };
}

}// namespace aethermind::detail
