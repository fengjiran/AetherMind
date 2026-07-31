#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator_inference.h"

#include <algorithm>
#include <vector>

namespace aethermind::detail {

// Permute semantic inference.
//
// Validation order (per plan, parameter-before-input precedence):
//   1. require PermuteParams variant
//   2. parameter-only invariant: permutation is a bijection (no duplicate
//      axes). Reported before any input check so a malformed PermuteParams
//      is surfaced even when arity is also wrong.
//   3. require exactly one input
//   4. require ranked input
//   5. permutation length == input rank
//   6. every permutation[k] < input rank
//
// Inference algorithm:
//   - Build output dims by copying input.shape[permutation[j]] for each
//     output axis j. ShapeSymbol identity is preserved (not just static
//     value) so downstream constraint solving can equate them.
//   - Empty permutation (rank zero) is identity over a scalar.
//   - No constraints are emitted: bijection guarantees volume equality
//     automatically and semantics are statically decidable.
//   - runtime_checks is always empty.
StatusOr<InferenceResult> InferPermute(const OpParams& params,
                                       std::span<const TensorSpec> inputs) {
    const auto* permute_params = std::get_if<PermuteParams>(&params);
    if (permute_params == nullptr) {
        return Status::InvalidArgument("Permute node requires PermuteParams");
    }

    // Parameter-only invariant: permutation must be a bijection over its
    // own index set (no duplicate axes). Validated before any input check
    // so a malformed PermuteParams is reported even when arity is wrong.
    // Length and axis-range checks against input rank are deferred to
    // after the arity check (they depend on input shape).
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

    // Validate axis range before constructing output.
    for (const uint32_t axis: permute_params->permutation) {
        if (static_cast<size_t>(axis) >= input_rank) {
            return Status::InvalidArgument(
                    "Permute permutation axis out of range");
        }
    }

    // Build output dims by permuting input dims. ShapeSymbol identity is
    // preserved (copy the same symbol), not just the static value.
    std::vector<ShapeSymbol> output_dims;
    output_dims.reserve(permute_params->permutation.size());
    for (const uint32_t axis: permute_params->permutation) {
        output_dims.push_back(input.shape[axis]);
    }
    SymbolicShape output_shape(std::move(output_dims));

    InferenceResult result;
    // Output dtype follows input dtype; no dtype restriction.
    result.outputs.emplace_back(input.dtype, output_shape);
    // No constraints: bijection guarantees volume equality automatically;
    // semantics are statically decidable, so no deferred runtime checks.
    return result;
}

}// namespace aethermind::detail
