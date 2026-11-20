#include "aethermind/operators/ops/softmax_op.h"
#include "aethermind/operators/operator_inference.h"

namespace aethermind::detail {

StatusOr<InferenceResult> InferSoftmax(const OpParams& params,
                                       std::span<const TensorSpec> inputs) {
    const auto* softmax = std::get_if<SoftmaxParams>(&params);
    if (softmax == nullptr) {
        return Status::InvalidArgument("Softmax node requires SoftmaxParams");
    }
    AM_RETURN_IF_ERROR(ValidateInferenceInputCount(OpType::kSoftmax, inputs));

    if (!IsSoftmaxSupportedDType(inputs[0].dtype)) {
        return Status::InvalidArgument(
                MakeSoftmaxUnsupportedDTypeMessage("Softmax input"));
    }

    if (const auto rank = inputs[0].shape.rank()) {
        const int64_t axis = softmax->axis < 0 ? softmax->axis + static_cast<int64_t>(*rank)
                                               : softmax->axis;
        if (axis < 0 || static_cast<size_t>(axis) >= *rank) {
            return Status::InvalidArgument("Softmax axis is out of range");
        }
    }

    InferenceResult result;
    result.outputs.push_back(inputs[0]);
    return result;
}

} // namespace aethermind::detail
