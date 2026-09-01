#include "aethermind/operators/ops/embedding_op.h"
#include "aethermind/operators/operator_inference.h"

namespace aethermind {
namespace detail {

// Model-weight dimensions must be positive, while zero token dimensions are
// valid and produce empty outputs. Symbolic weight bounds are deferred.
StatusOr<InferenceResult> InferEmbedding(const OpParams& params,
                                         std::span<const TensorSpec> inputs) {
    if (!std::holds_alternative<EmbeddingParams>(params)) {
        return Status::InvalidArgument("Embedding node requires EmbeddingParams");
    }

    AM_RETURN_IF_ERROR(ValidateInferenceInputCount(OpType::kEmbedding, inputs));

    const TensorSpec& input_ids_spec = inputs[0];
    const TensorSpec& weight_spec = inputs[1];
    const auto& input_ids_shape = input_ids_spec.shape;
    const auto& weight_shape = weight_spec.shape;

    if (!IsSupportedTokenIdDType(input_ids_spec.dtype)) {
        return Status::InvalidArgument(
                "Embedding token_ids must be int32, int64, or uint32");
    }

    if (!IsEmbeddingSupportedWeightDType(weight_spec.dtype)) {
        return Status::InvalidArgument(
                MakeEmbeddingUnsupportedWeightDTypeMessage("Embedding"));
    }

    const auto input_rank_opt = input_ids_shape.rank();
    if (!input_rank_opt.has_value() || *input_rank_opt < 1) {
        return Status::InvalidArgument("Embedding input must have rank >= 1");
    }
    const size_t input_rank = *input_rank_opt;

    if (!HasRank(weight_shape, 2)) {
        return Status::InvalidArgument("Embedding weight must be rank 2");
    }

    InferenceResult res;
    // Weight dimensions (vocab_size, hidden_size) must be positive. Static
    // violations are rejected here; symbolic dims emit a runtime constraint.
    for (size_t i = 0; i < *weight_shape.rank(); ++i) {
        if (const ShapeSymbol& dim = weight_shape[i]; dim.IsStatic()) {
            if (dim.GetStaticValue() <= 0) {
                return Status::InvalidArgument(
                        "Embedding weight dimension must be positive when statically known.");
            }
        } else {
            res.runtime_checks.emplace_back(
                    DimPositiveConstraint{
                            {.tensor_port = {.direction = TensorPortType::kInput,
                                             .tensor_idx = 1},
                             .dim_index = i}},
                    "Embedding weight dimension must be positive");
        }
    }

    // Output shape preserves all input_ids axes and appends the hidden axis,
    // matching PyTorch nn.Embedding / NumPy fancy indexing semantics.
    std::vector<ShapeSymbol> output_shape;
    output_shape.reserve(input_rank + 1);
    for (size_t i = 0; i < input_rank; ++i) {
        output_shape.push_back(input_ids_shape[i]);
    }
    output_shape.push_back(weight_shape[1]);

    res.outputs.emplace_back(weight_spec.dtype,
                             SymbolicShape(std::move(output_shape)));
    return res;
}

} // namespace detail

} // namespace aethermind
