#include "aethermind/operators/ops/argmax_op.h"
#include "aethermind/operators/operator_inference.h"

namespace aethermind::detail {

StatusOr<InferenceResult> InferArgmax(const OpParams& params,
                                      std::span<const TensorSpec> inputs) {
    const auto* argmax_params = std::get_if<ArgmaxParams>(&params);
    if (argmax_params == nullptr) {
        return Status::InvalidArgument("Argmax node requires ArgmaxParams");
    }
    AM_RETURN_IF_ERROR(ValidateInferenceInputCount(OpType::kArgmax, inputs));
    const auto& input_spec = inputs[0];

    if (!IsArgmaxSupportedDType(input_spec.dtype)) {
        return Status::InvalidArgument(
                MakeArgmaxUnsupportedDTypeMessage("Argmax input"));
    }

    InferenceResult result;
    if (!input_spec.shape.IsRanked()) {
        result.outputs.emplace_back(DataType::Int(64), SymbolicShape(std::nullopt));
        return result;
    }

    const auto rank = *input_spec.shape.rank();
    int64_t axis = argmax_params->axis;
    if (axis < 0) {
        axis += static_cast<int64_t>(rank);
    }

    if (axis < 0 || static_cast<size_t>(axis) >= rank) {
        return Status::InvalidArgument("Argmax axis is out of range");
    }

    std::vector<ShapeSymbol> output_dims;
    for (size_t i = 0; i < rank; ++i) {
        if (static_cast<int64_t>(i) != axis) {
            output_dims.push_back(input_spec.shape[i]);
        }
    }
    result.outputs.emplace_back(DataType::Int(64), SymbolicShape(output_dims));

    return result;
}

} // namespace aethermind::detail
