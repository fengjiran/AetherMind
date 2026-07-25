#include "aethermind/dtypes/data_type.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/shape_inference/tensor_spec.h"

#include <span>
#include <string>

namespace aethermind::detail {

StatusOr<InferenceResult> InferKVCacheUpdate(const OpParams& params, std::span<const TensorSpec> inputs) {
    if (!std::holds_alternative<KVCacheUpdateParams>(params)) {
        return Status::InvalidArgument("KVCacheUpdate node requires KVCacheUpdateParams");
    }
    AM_RETURN_IF_ERROR(ValidateInferenceInputCount(OpType::kKVCacheUpdate, inputs));

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
