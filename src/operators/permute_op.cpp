#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator_inference.h"

#include <algorithm>
#include <vector>

namespace aethermind::detail {

StatusOr<InferenceResult> InferPermute(const OpParams& params,
                                       std::span<const TensorSpec> inputs) {
    const auto* permute_params = std::get_if<PermuteParams>(&params);
    if (permute_params == nullptr) {
        return Status::InvalidArgument("Permute node requires PermuteParams");
    }

    // Check parameter-only invariants first so malformed parameters retain
    // precedence over input-structure errors.
    {
        std::vector<uint32_t> sorted = permute_params->permutation;
        std::ranges::sort(sorted);
        if (std::ranges::adjacent_find(sorted) != sorted.end()) {
            return Status::InvalidArgument(
                    "Permute permutation must be a bijection: duplicate axis");
        }
    }

    AM_RETURN_IF_ERROR(ValidateInferenceInputCount(OpType::kPermute, inputs));

    const TensorSpec& input = inputs[0];
    if (!input.shape.IsRanked()) {
        return Status::InvalidArgument("Permute input must be ranked");
    }
    const size_t input_rank = *input.shape.rank();

    if (permute_params->permutation.size() != input_rank) {
        return Status::InvalidArgument(
                "Permute permutation length must match input rank");
    }

    for (const uint32_t axis: permute_params->permutation) {
        if (static_cast<size_t>(axis) >= input_rank) {
            return Status::InvalidArgument(
                    "Permute permutation axis out of range");
        }
    }

    // Copy the original symbols so downstream equality reasoning preserves
    // symbolic identity, not only static values.
    std::vector<ShapeSymbol> output_dims;
    output_dims.reserve(permute_params->permutation.size());
    for (const uint32_t axis: permute_params->permutation) {
        output_dims.push_back(input.shape[axis]);
    }
    SymbolicShape output_shape(std::move(output_dims));

    InferenceResult result;
    result.outputs.emplace_back(input.dtype, output_shape);
    return result;
}

}// namespace aethermind::detail
