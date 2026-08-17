#include "aethermind/operators/ops/silu_op.h"
#include "aethermind/operators/operator_inference.h"

namespace aethermind {

namespace detail {

StatusOr<InferenceResult> InferSilu(const OpParams& params,
                                    std::span<const TensorSpec> inputs) {
    if (!std::holds_alternative<SiluParams>(params)) {
        return Status::InvalidArgument("Silu node requires SiluParams");
    }

    AM_RETURN_IF_ERROR(ValidateInferenceInputCount(OpType::kSilu, inputs));

    if (!IsSiluSupportedDType(inputs[0].dtype)) {
        return Status::InvalidArgument(MakeSiluUnsupportedDTypeMessage("Silu input"));
    }

    InferenceResult result;
    result.outputs.push_back(inputs[0]);
    return result;
}

}// namespace detail

}// namespace aethermind
