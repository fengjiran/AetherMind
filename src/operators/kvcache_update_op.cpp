#include "aethermind/operators/kvcache_update_op.h"
#include "aethermind/operators/operator_inference.h"

namespace aethermind::detail {

StatusOr<InferenceResult> InferKVCacheUpdate(const OpParams& params,
                                             std::span<const TensorSpec> inputs) {
    if (!std::holds_alternative<KVCacheUpdateParams>(params)) {
        return Status::InvalidArgument(
                "KVCacheUpdate node requires KVCacheUpdateParams");
    }
    AM_RETURN_IF_ERROR(ValidateInferenceInputCount(OpType::kKVCacheUpdate, inputs));

    const auto& k_spec = inputs[0];
    const auto& v_spec = inputs[1];
    const auto& k_cache_in_spec = inputs[2];
    const auto& v_cache_in_spec = inputs[3];

    if (!IsKVCacheUpdateSupportedDType(k_spec.dtype)) {
        return Status::InvalidArgument(
                MakeKVCacheUpdateUnsupportedDTypeMessage("KVCacheUpdate k"));
    }

    if (!IsKVCacheUpdateSupportedDType(v_spec.dtype)) {
        return Status::InvalidArgument(
                MakeKVCacheUpdateUnsupportedDTypeMessage("KVCacheUpdate v"));
    }

    return InferenceResult{
            .outputs = {k_cache_in_spec, v_cache_in_spec},
    };
}

}// namespace aethermind::detail
