#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator_inference.h"

namespace aethermind::detail {

// Reorder changes physical storage intent only, so semantic inference forwards
// the complete TensorSpec without inspecting dtype, rank, or dimensions.
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

} // namespace aethermind::detail
