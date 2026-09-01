#include "aethermind/operators/operator_inference.h"
#include "aethermind/operators/ops/linear_op.h"
#include "aethermind/shape_inference/shape_constraint.h"

namespace aethermind {

namespace detail {

StatusOr<InferenceResult> InferGateUpLinear(const OpParams& params,
                                            std::span<const TensorSpec> inputs) {
    if (!std::holds_alternative<GateUpLinearParams>(params)) {
        return Status::InvalidArgument(
                "GateUpLinear node requires GateUpLinearParams");
    }

    const auto& gate_up_params = std::get<GateUpLinearParams>(params);
    if (gate_up_params.has_bias) {
        return Status::InvalidArgument(
                "GateUpLinear bias inputs are not supported yet");
    }

    AM_RETURN_IF_ERROR(ValidateInferenceInputCount(OpType::kGateUpLinear, inputs));

    const TensorSpec& input_spec = inputs[0];
    const TensorSpec& weight_spec = inputs[1];
    const auto& input_shape = input_spec.shape;
    const auto& weight_shape = weight_spec.shape;
    const auto input_rank = input_shape.rank();
    if (!input_rank.has_value() || *input_rank < 1U) {
        return Status::InvalidArgument(
                "GateUpLinear input must have rank >= 1");
    }

    if (!HasRank(weight_shape, 2U)) {
        return Status::InvalidArgument(
                "GateUpLinear gate_up_weight must be rank 2");
    }

    if (gate_up_params.gate_out_features < 0 || gate_up_params.up_out_features < 0) {
        return Status::InvalidArgument(
                "GateUpLinear gate/up out_features must be non-negative");
    }

    if (gate_up_params.gate_out_features >
        std::numeric_limits<int64_t>::max() - gate_up_params.up_out_features) {
        return Status::InvalidArgument(
                "GateUpLinear gate/up out_features overflow int64");
    }

    if (!IsLinearSupportedActivationDType(input_spec.dtype)) {
        return Status::InvalidArgument(
                MakeLinearUnsupportedActivationDTypeMessage("GateUpLinear"));
    }

    if (!IsLinearSupportedWeightDType(weight_spec.dtype)) {
        return Status::InvalidArgument(
                MakeLinearUnsupportedWeightDTypeMessage("GateUpLinear"));
    }

    const int64_t expected_rows = gate_up_params.gate_out_features + gate_up_params.up_out_features;
    const ShapeSymbol& packed_rows = weight_shape[0];
    if (packed_rows.IsStatic() && packed_rows.GetStaticValue() != expected_rows) {
        return Status::InvalidArgument(
                "GateUpLinear gate_up_weight rows must equal gate_out + up_out features");
    }

    const ShapeSymbol& in_features = input_shape[*input_rank - 1U];
    const ShapeSymbol& weight_in = weight_shape[1];
    InferenceResult result;
    if (!AreProvablyEqual(in_features, weight_in)) {
        if (in_features.IsStatic() && weight_in.IsStatic()) {
            return Status::InvalidArgument(
                    "GateUpLinear gate_up_weight length must equal input last dimension");
        }

        result.runtime_checks.emplace_back(
                DimEqualConstraint{
                        .lhs = {.tensor_port = {.direction = TensorPortType::kInput,
                                                .tensor_idx = 0},
                                .dim_index = *input_rank - 1U},
                        .rhs = {.tensor_port = {.direction = TensorPortType::kInput,
                                                .tensor_idx = 1},
                                .dim_index = 1}},
                "GateUpLinear input last dimension must match gate_up_weight input dimension");
    }

    std::vector<ShapeSymbol> base_shape;
    base_shape.reserve(*input_rank - 1U);
    for (size_t i = 0; i < *input_rank - 1U; ++i) {
        base_shape.push_back(input_shape[i]);
    }

    const auto make_output = [&](int64_t out_features) {
        std::vector<ShapeSymbol> shape = base_shape;
        shape.push_back(ShapeSymbol::CreateFromValue(out_features));
        return TensorSpec(input_spec.dtype, SymbolicShape(std::move(shape)));
    };

    result.outputs.push_back(make_output(gate_up_params.gate_out_features));
    result.outputs.push_back(make_output(gate_up_params.up_out_features));
    return result;
}

} // namespace detail

} // namespace aethermind
