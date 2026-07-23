#include "aethermind/dtypes/data_type.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/shape_inference/tensor_spec.h"

#include <span>
#include <string>

namespace aethermind::detail {

StatusOr<InferenceResult> InferSoftmax(const OpParams& /*params*/,
                                       std::span<const TensorSpec> inputs) {
    if (inputs.size() != 1) {
        return Status::InvalidArgument("Softmax requires exactly 1 input");
    }
    if (inputs[0].dtype != DataType::Float32()) {
        return Status::InvalidArgument("Softmax input must be float32");
    }
    InferenceResult result;
    result.outputs.push_back(inputs[0]);
    return result;
}

}// namespace aethermind::detail
