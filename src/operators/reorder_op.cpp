#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator_inference.h"

namespace aethermind::detail {

// Reorder semantic inference.
//
// Validation order (parameter-before-input precedence):
//   1. require ReorderParams variant
//   2. require exactly one input (via schema arity)
//   3. copy inputs[0] directly as the sole output
//
// Inference algorithm:
//   - The output TensorSpec is an exact copy of the input: same dtype,
//     same SymbolicShape (with identical ShapeSymbol identities), same
//     unranked state if applicable.
//   - No constraints are emitted: logical identity is trivially provable.
//   - runtime_checks is always empty.
//   - No rank/dimension/dtype/stride/layout inspection is performed.
StatusOr<InferenceResult> InferReorder(const OpParams& params,
                                       std::span<const TensorSpec> inputs) {
    const auto* reorder_params = std::get_if<ReorderParams>(&params);
    if (reorder_params == nullptr) {
        return Status::InvalidArgument("Reorder node requires ReorderParams");
    }

    AM_RETURN_IF_ERROR(ValidateInferenceInputCount(OpType::kReorder, inputs));

    InferenceResult result;
    result.outputs.push_back(inputs[0]);
    return result;
}

}// namespace aethermind::detail
