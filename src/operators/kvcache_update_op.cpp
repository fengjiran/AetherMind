#include "aethermind/dtypes/data_type.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/shape_inference/tensor_spec.h"

#include <span>
#include <string>

namespace aethermind::detail {

StatusOr<InferenceResult> InferKVCacheUpdate(const OpParams& /*params*/, std::span<const TensorSpec> inputs) {
    if (inputs.size() != 4) {
        return Status::InvalidArgument(
                "KVCacheUpdate expects exactly 4 inputs (k, v, kCacheIn, vCacheIn), got " + std::to_string(inputs.size()));
    }

    const auto& k_spec = inputs[0];
    const auto& v_spec = inputs[1];
    const auto& k_cache_in_spec = inputs[2];
    const auto& v_cache_in_spec = inputs[3];

    if (k_spec.dtype != DataType::Float32() || v_spec.dtype != DataType::Float32()) {
        return Status::InvalidArgument("KVCacheUpdate only supports float32 k/v inputs in Phase 1");
    }

    return InferenceResult{
            .outputs = {k_cache_in_spec, v_cache_in_spec},
    };
}

}// namespace aethermind::detail
