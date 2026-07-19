#include "operator_semantics_internal.h"

#include "aethermind/dtypes/data_type.h"
#include "aethermind/operators/add_op.h"
#include "aethermind/shape_inference/broadcast.h"
#include "aethermind/shape_inference/tensor_spec.h"

namespace aethermind {
namespace detail {
namespace {

StatusOr<InferenceResult> AnalyzeBroadcastBinary(const OpParams& /*params*/,
                                                 std::span<const TensorSpec> inputs,
                                                 const char* op_name) {
    if (inputs.size() != 2) {
        return Status::InvalidArgument(
                std::string(op_name) + " requires exactly 2 inputs");
    }
    const TensorSpec& lhs_spec = inputs[0];
    const TensorSpec& rhs_spec = inputs[1];
    if (lhs_spec.dtype != DataType::Float32()) {
        return Status::InvalidArgument(
                std::string(op_name) + " lhs must be float32");
    }
    if (rhs_spec.dtype != DataType::Float32()) {
        return Status::InvalidArgument(
                std::string(op_name) + " rhs must be float32");
    }
    auto broadcast_result = InferBroadcastShape(lhs_spec.shape, rhs_spec.shape);
    if (!broadcast_result.ok()) {
        return broadcast_result.status();
    }
    InferenceResult result;
    result.outputs.push_back({lhs_spec.dtype, broadcast_result->output_shape});
    for (const auto& deferred: broadcast_result->deferred_axes) {
        result.runtime_checks.push_back(ShapeConstraint{
                DimBroadcastableConstraint{{{TensorPortType::kInput, 0}, deferred.lhs_axis},
                                           {{TensorPortType::kInput, 1}, deferred.rhs_axis}},
                std::string(op_name) + " dimensions must be broadcastable"});
    }
    return result;
}

}// namespace

StatusOr<InferenceResult> AnalyzeAdd(const OpParams& /*params*/,
                                     std::span<const TensorSpec> inputs) {
    if (inputs.size() != 2) {
        return Status::InvalidArgument("Add requires exactly 2 inputs");
    }
    const TensorSpec& lhs_spec = inputs[0];
    const TensorSpec& rhs_spec = inputs[1];
    if (lhs_spec.dtype != rhs_spec.dtype) {
        return Status::InvalidArgument("Add inputs must have the same dtype");
    }
    if (!IsAddSupportedDType(lhs_spec.dtype)) {
        return Status::InvalidArgument(MakeAddUnsupportedDTypeMessage("Add"));
    }
    auto broadcast_result = InferBroadcastShape(lhs_spec.shape, rhs_spec.shape);
    if (!broadcast_result.ok()) {
        return broadcast_result.status();
    }
    InferenceResult result;
    result.outputs.push_back({lhs_spec.dtype, broadcast_result->output_shape});
    for (const auto& deferred: broadcast_result->deferred_axes) {
        result.runtime_checks.push_back(ShapeConstraint{
                DimBroadcastableConstraint{{{TensorPortType::kInput, 0}, deferred.lhs_axis},
                                           {{TensorPortType::kInput, 1}, deferred.rhs_axis}},
                "Add dimensions must be broadcastable"});
    }
    return result;
}

StatusOr<InferenceResult> AnalyzeSilu(const OpParams& /*params*/,
                                      std::span<const TensorSpec> inputs) {
    if (inputs.size() != 1) {
        return Status::InvalidArgument("Silu requires exactly 1 input");
    }
    if (inputs[0].dtype != DataType::Float32()) {
        return Status::InvalidArgument("Silu input must be float32");
    }
    InferenceResult result;
    result.outputs.push_back(inputs[0]);
    return result;
}

StatusOr<InferenceResult> AnalyzeSiluMul(const OpParams& /*params*/,
                                         std::span<const TensorSpec> inputs) {
    return AnalyzeBroadcastBinary(/*params=*/{}, inputs, "SiluMul");
}

StatusOr<InferenceResult> AnalyzeElementwiseMul(const OpParams& /*params*/,
                                                std::span<const TensorSpec> inputs) {
    return AnalyzeBroadcastBinary(/*params=*/{}, inputs, "ElementwiseMul");
}

StatusOr<InferenceResult> AnalyzeSoftmax(const OpParams& /*params*/,
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

StatusOr<InferenceResult> AnalyzeArgmax(const OpParams& params,
                                        std::span<const TensorSpec> inputs) {
    if (inputs.size() != 1) {
        return Status::InvalidArgument("Argmax requires exactly 1 input");
    }
    const auto* typed = std::get_if<ArgmaxParams>(&params);
    if (typed == nullptr) {
        return Status::InvalidArgument("Argmax requires ArgmaxParams");
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

}// namespace detail
}// namespace aethermind
