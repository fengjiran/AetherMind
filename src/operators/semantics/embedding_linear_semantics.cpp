#include "operator_semantics_internal.h"

#include "aethermind/dtypes/data_type.h"
#include "aethermind/shape_inference/broadcast.h"
#include "aethermind/shape_inference/shape_symbol.h"
#include "aethermind/shape_inference/tensor_spec.h"

namespace aethermind {
namespace detail {
namespace {

bool IsSupportedTokenIdDType(const DataType& dtype) {
    return dtype == DataType::Int(32) || dtype == DataType::Int(64) || dtype == DataType::UInt(32);
}

}// namespace

StatusOr<InferenceResult> InferEmbedding(const OpParams& /*params*/,
                                           std::span<const TensorSpec> inputs) {
    if (inputs.size() != 2) {
        return Status::InvalidArgument("Embedding requires exactly 2 inputs");
    }
    const TensorSpec& token_spec = inputs[0];
    const TensorSpec& weight_spec = inputs[1];
    if (!IsSupportedTokenIdDType(token_spec.dtype)) {
        return Status::InvalidArgument("Embedding token_ids must be int32, int64, or uint32");
    }
    if (weight_spec.dtype != DataType::Float32()) {
        return Status::InvalidArgument("Embedding weight must be float32");
    }
    if (!HasRank(token_spec.shape, 1)) {
        return Status::InvalidArgument("Embedding token_ids must be rank 1");
    }
    if (!HasRank(weight_spec.shape, 2)) {
        return Status::InvalidArgument("Embedding weight must be rank 2");
    }
    const auto& tokens_shape = token_spec.shape;
    const auto& weight_shape = weight_spec.shape;
    if (tokens_shape[0].IsStatic() && tokens_shape[0].GetStaticValue() <= 0) {
        return Status::InvalidArgument("Embedding token_ids dimension must be positive");
    }
    if (weight_shape[0].IsStatic() && weight_shape[0].GetStaticValue() <= 0) {
        return Status::InvalidArgument("Embedding weight dimension 0 must be positive");
    }
    if (weight_shape[1].IsStatic() && weight_shape[1].GetStaticValue() <= 0) {
        return Status::InvalidArgument("Embedding weight dimension 1 must be positive");
    }
    InferenceResult result;
    result.outputs.push_back({weight_spec.dtype, SymbolicShape({tokens_shape[0], weight_shape[1]})});
    return result;
}

StatusOr<InferenceResult> InferLinear(const OpParams& /*params*/,
                                        std::span<const TensorSpec> inputs) {
    if (inputs.size() != 2) {
        return Status::InvalidArgument("Linear requires exactly 2 inputs");
    }
    const TensorSpec& input_spec = inputs[0];
    const TensorSpec& weight_spec = inputs[1];
    if (input_spec.dtype != DataType::Float32()) {
        return Status::InvalidArgument("Linear input must be float32");
    }
    if (weight_spec.dtype != DataType::Float32()) {
        return Status::InvalidArgument("Linear weight must be float32");
    }
    if (!input_spec.shape.IsRanked()) {
        return Status::InvalidArgument("Linear input must be ranked");
    }
    if (!HasRank(weight_spec.shape, 2)) {
        return Status::InvalidArgument("Linear weight must be rank 2");
    }
    const auto& input_shape = input_spec.shape;
    const auto& weight_shape = weight_spec.shape;
    const auto input_rank = input_shape.rank().value();
    if (input_rank != 1 && input_rank != 2) {
        return Status::InvalidArgument("Linear input must be rank 1 or 2");
    }
    const ShapeSymbol& in_features = input_shape[input_rank - 1];
    const ShapeSymbol& weight_in = weight_shape[1];
    if (in_features.IsStatic() && in_features.GetStaticValue() <= 0) {
        return Status::InvalidArgument("Linear input last dimension must be positive");
    }
    if (weight_shape[0].IsStatic() && weight_shape[0].GetStaticValue() <= 0) {
        return Status::InvalidArgument("Linear weight output dimension must be positive");
    }
    if (weight_in.IsStatic() && weight_in.GetStaticValue() <= 0) {
        return Status::InvalidArgument("Linear weight input dimension must be positive");
    }
    if (in_features.IsStatic() && weight_in.IsStatic() &&
        in_features.GetStaticValue() != weight_in.GetStaticValue()) {
        return Status::InvalidArgument("Linear input last dimension must match weight input dimension");
    }
    auto unified = UnifyShapeSymbol(in_features, weight_in);
    if (!unified.ok()) {
        return unified.status();
    }
    InferenceResult result;
    if (input_rank == 1) {
        result.outputs.push_back({input_spec.dtype, SymbolicShape({weight_shape[0]})});
    } else {
        result.outputs.push_back({input_spec.dtype, SymbolicShape({input_shape[0], weight_shape[0]})});
    }
    if (in_features != weight_in) {
        result.runtime_checks.push_back(ShapeConstraint{
                DimEqualConstraint{{{TensorPortType::kInput, 0}, static_cast<size_t>(input_rank - 1)},
                                   {{TensorPortType::kInput, 1}, 1}},
                "Linear input last dimension must match weight input dimension"});
    }
    return result;
}

namespace {

struct RhsAxes {
    size_t inner;
    size_t outer;
};

RhsAxes ResolveRhsAxes(const MatMulParams& params, size_t rhs_rank) {
    if (params.transpose_rhs) {
        return {rhs_rank - 1, rhs_rank - 2};
    }
    return {rhs_rank - 2, rhs_rank - 1};
}

SymbolicShape MakeBatchShape(const SymbolicShape& shape, size_t rank) {
    std::vector<ShapeSymbol> batch_dims;
    for (size_t i = 0; i < rank - 2; ++i) {
        batch_dims.push_back(shape[i]);
    }
    return SymbolicShape(batch_dims);
}

}// namespace

StatusOr<InferenceResult> InferMatMul(const OpParams& params,
                                        std::span<const TensorSpec> inputs) {
    if (inputs.size() != 2) {
        return Status::InvalidArgument("MatMul requires exactly 2 inputs");
    }
    const auto* typed = std::get_if<MatMulParams>(&params);
    if (typed == nullptr) {
        return Status::InvalidArgument("MatMul requires MatMulParams");
    }
    const TensorSpec& lhs_spec = inputs[0];
    const TensorSpec& rhs_spec = inputs[1];
    if (lhs_spec.dtype != DataType::Float32()) {
        return Status::InvalidArgument("MatMul lhs must be float32");
    }
    if (rhs_spec.dtype != DataType::Float32()) {
        return Status::InvalidArgument("MatMul rhs must be float32");
    }
    if (!lhs_spec.shape.IsRanked()) {
        return Status::InvalidArgument("MatMul lhs must be ranked");
    }
    if (!rhs_spec.shape.IsRanked()) {
        return Status::InvalidArgument("MatMul rhs must be ranked");
    }
    const auto lhs_rank = lhs_spec.shape.rank().value();
    const auto rhs_rank = rhs_spec.shape.rank().value();
    if (lhs_rank < 2) {
        return Status::InvalidArgument("MatMul lhs must have rank >= 2");
    }
    if (rhs_rank < 2) {
        return Status::InvalidArgument("MatMul rhs must have rank >= 2");
    }
    const RhsAxes rhs_axes = ResolveRhsAxes(*typed, rhs_rank);
    const ShapeSymbol& lhs_inner = lhs_spec.shape[lhs_rank - 1];
    const ShapeSymbol& rhs_inner = rhs_spec.shape[rhs_axes.inner];
    if (lhs_inner.IsStatic() && lhs_inner.GetStaticValue() <= 0) {
        return Status::InvalidArgument("MatMul lhs inner dimension must be positive");
    }
    if (rhs_inner.IsStatic() && rhs_inner.GetStaticValue() <= 0) {
        return Status::InvalidArgument("MatMul rhs inner dimension must be positive");
    }
    if (lhs_inner.IsStatic() && rhs_inner.IsStatic() &&
        lhs_inner.GetStaticValue() != rhs_inner.GetStaticValue()) {
        return Status::InvalidArgument("MatMul inner dimensions must be equal");
    }
    const ShapeSymbol& lhs_outer = lhs_spec.shape[lhs_rank - 2];
    const ShapeSymbol& rhs_outer = rhs_spec.shape[rhs_axes.outer];
    if (lhs_outer.IsStatic() && lhs_outer.GetStaticValue() <= 0) {
        return Status::InvalidArgument("MatMul lhs outer dimension must be positive");
    }
    if (rhs_outer.IsStatic() && rhs_outer.GetStaticValue() <= 0) {
        return Status::InvalidArgument("MatMul rhs outer dimension must be positive");
    }
    auto lhs_batch = MakeBatchShape(lhs_spec.shape, lhs_rank);
    auto rhs_batch = MakeBatchShape(rhs_spec.shape, rhs_rank);
    auto broadcast_result = InferBroadcastShape(lhs_batch, rhs_batch);
    if (!broadcast_result.ok()) {
        return broadcast_result.status();
    }
    std::vector<ShapeSymbol> output_dims = broadcast_result->output_shape.shape().value();
    output_dims.push_back(lhs_outer);
    output_dims.push_back(rhs_outer);
    InferenceResult result;
    result.outputs.push_back({lhs_spec.dtype, SymbolicShape(output_dims)});
    if (!lhs_inner.IsStatic() || !rhs_inner.IsStatic()) {
        if (lhs_inner != rhs_inner) {
            result.runtime_checks.push_back(ShapeConstraint{
                    DimEqualConstraint{{{TensorPortType::kInput, 0}, lhs_rank - 1},
                                       {{TensorPortType::kInput, 1}, rhs_axes.inner}},
                    "MatMul inner dimensions must be equal"});
        }
    }
    for (const auto& deferred: broadcast_result->deferred_axes) {
        result.runtime_checks.push_back(ShapeConstraint{
                DimBroadcastableConstraint{{{TensorPortType::kInput, 0}, deferred.lhs_axis},
                                           {{TensorPortType::kInput, 1}, deferred.rhs_axis}},
                "MatMul batch dimensions must be broadcastable"});
    }
    return result;
}

}// namespace detail
}// namespace aethermind
