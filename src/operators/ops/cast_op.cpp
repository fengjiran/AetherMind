#include "aethermind/operators/ops/cast_op.h"
#include "aethermind/operators/operator_inference.h"

namespace aethermind::detail {

StatusOr<InferenceResult> InferCast(const OpParams& params,
                                    std::span<const TensorSpec> inputs) {
    if (!std::holds_alternative<CastParams>(params)) {
        return Status::InvalidArgument("Cast node requires CastParams");
    }

    AM_RETURN_IF_ERROR(ValidateInferenceInputCount(OpType::kCast, inputs));

    const auto& cast_params = std::get<CastParams>(params);
    const TensorSpec& input = inputs[0];
    if (!IsCastSupportedDType(input.dtype)) {
        return Status::InvalidArgument(
                MakeCastUnsupportedDTypeMessage("Cast input"));
    }

    if (!IsCastSupportedDType(cast_params.target_dtype)) {
        return Status::InvalidArgument(
                MakeCastUnsupportedDTypeMessage("Cast target"));
    }

    // A same-dtype Cast is not expressible: the graph transition must be
    // explicit, and emitting a no-op node would only complicate folding.
    if (cast_params.target_dtype == input.dtype) {
        return Status::InvalidArgument(
                "Cast target dtype must differ from the input dtype");
    }

    InferenceResult result;
    result.outputs.emplace_back(cast_params.target_dtype, input.shape);
    return result;
}

} // namespace aethermind::detail