#include "aethermind/dtypes/data_type.h"
#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/shape_inference/shape_symbol.h"
#include "aethermind/shape_inference/tensor_spec.h"

#include <span>
#include <string>

namespace aethermind::detail {

StatusOr<InferenceResult> InferArgmax(const OpParams& params,
                                      std::span<const TensorSpec> inputs) {
    const auto* typed = std::get_if<ArgmaxParams>(&params);
    if (typed == nullptr) {
        return Status::InvalidArgument("Argmax node requires ArgmaxParams");
    }
    if (inputs.size() != 1) {
        return Status::InvalidArgument("Argmax requires exactly 1 input");
    }
    if (inputs[0].dtype != DataType::Float32()) {
        return Status::InvalidArgument("Argmax input must be float32");
    }
    InferenceResult result;
    if (!inputs[0].shape.IsRanked()) {
        result.outputs.push_back({DataType::Int(64), SymbolicShape(std::nullopt)});
        return result;
    }
    const auto rank = inputs[0].shape.rank().value();
    int64_t axis = typed->axis;
    if (axis < 0) {
        axis += static_cast<int64_t>(rank);
    }
    if (axis < 0 || static_cast<size_t>(axis) >= rank) {
        return Status::InvalidArgument("Argmax axis is out of range");
    }
    std::vector<ShapeSymbol> output_dims;
    for (size_t i = 0; i < rank; ++i) {
        if (static_cast<int64_t>(i) != axis) {
            output_dims.push_back(inputs[0].shape[i]);
        }
    }
    result.outputs.push_back({DataType::Int(64), SymbolicShape(output_dims)});
    return result;
}

}// namespace aethermind::detail
