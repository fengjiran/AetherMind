#include "aethermind/operators/softmax_op.h"
#include "aethermind/operators/operator_inference.h"

namespace aethermind::detail {

StatusOr<InferenceResult> InferSoftmax(const OpParams& params,
                                       std::span<const TensorSpec> inputs) {
    if (!std::holds_alternative<SoftmaxParams>(params)) {
        return Status::InvalidArgument("Softmax node requires SoftmaxParams");
    }
    AM_RETURN_IF_ERROR(ValidateInferenceInputCount(OpType::kSoftmax, inputs));

    if (!IsSoftmaxSupportedDType(inputs[0].dtype)) {
        return Status::InvalidArgument(
                MakeSoftmaxUnsupportedDTypeMessage("Softmax input"));
    }

    InferenceResult result;
    result.outputs.push_back(inputs[0]);
    return result;
}

}// namespace aethermind::detail
