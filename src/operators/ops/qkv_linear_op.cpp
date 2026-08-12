#include "aethermind/operators/ops/qkv_linear_op.h"
#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/operators/operator_registry.h"
#include "aethermind/operators/ops/linear_op.h"
#include "aethermind/shape_inference/shape_constraint.h"

namespace aethermind {

Status QkvLinearOp::Prepare(OperatorContext& ctx) {
    if (ctx.backend == nullptr) {
        return Status::InvalidArgument(
                "QkvLinear Prepare requires OperatorContext.backend");
    }

    const auto resolved = ctx.backend->ResolveKernelInfo(
            OpType::kQkvLinear,
            ctx.selector);

    if (!resolved.ok()) {
        return resolved.status();
    }

    resolved_kernel_ = resolved.value();
    if (resolved_kernel_.fn == nullptr) {
        return Status::Internal(
                "QkvLinear Prepare resolved a kernel with null fn");
    }
    return Status::Ok();
}

Status QkvLinearOp::Run(KernelContext& ctx,
                        const RuntimeBindingContext& bindings,
                        size_t step_index) const noexcept {
    if (resolved_kernel_.fn == nullptr) {
        return Status::FailedPrecondition(
                "QkvLinear Run called before Prepare");
    }

    const auto binding = bindings.GetStepTensorBinding(step_index);
    if (!binding.ok()) {
        return binding.status();
    }

    const auto* b = binding.value();
    if (b->inputs.size() != 2) {
        return Status::InvalidArgument(
                "QkvLinear requires 2 input tensor bindings, got " +
                std::to_string(b->inputs.size()));
    }

    if (b->outputs.size() != 3) {
        return Status::InvalidArgument(
                "QkvLinear requires 3 output tensor bindings, got " +
                std::to_string(b->outputs.size()));
    }

    return InvokeResolvedKernel(ctx, b->inputs, b->outputs);
}

AM_REGISTER_OPERATOR(OpType::kQkvLinear, QkvLinearOp)


namespace detail {

StatusOr<InferenceResult> InferQkvLinear(const OpParams& params,
                                         std::span<const TensorSpec> inputs) {
    if (!std::holds_alternative<QkvLinearParams>(params)) {
        return Status::InvalidArgument(
                "QkvLinear node requires QkvLinearParams");
    }

    const auto& qkv_params = std::get<QkvLinearParams>(params);
    // The two-port schema (input, qkv_weight) cannot express the Q/K/V bias
    // inputs declared by QkvLinearParams::has_bias; reject until the schema
    // gains optional bias ports.
    if (qkv_params.has_bias) {
        return Status::InvalidArgument(
                "QkvLinear bias inputs are not supported yet");
    }

    AM_RETURN_IF_ERROR(ValidateInferenceInputCount(OpType::kQkvLinear, inputs));

    const TensorSpec& input_spec = inputs[0];
    const TensorSpec& weight_spec = inputs[1];
    const auto& input_shape = input_spec.shape;
    const auto& weight_shape = weight_spec.shape;
    const auto input_rank = input_shape.rank();
    if (!input_rank.has_value() || *input_rank < 1) {
        return Status::InvalidArgument("QkvLinear input must have rank >= 1");
    }

    if (!HasRank(weight_shape, 2)) {
        return Status::InvalidArgument("QkvLinear qkv_weight must be rank 2");
    }

    if (qkv_params.q_out_features < 0 || qkv_params.k_out_features < 0 ||
        qkv_params.v_out_features < 0) {
        return Status::InvalidArgument(
                "QkvLinear q/k/v out_features must be non-negative");
    }

    if (!IsLinearSupportedActivationDType(input_spec.dtype)) {
        return Status::InvalidArgument(
                MakeLinearUnsupportedActivationDTypeMessage("QkvLinear"));
    }

    if (!IsLinearSupportedWeightDType(weight_spec.dtype)) {
        return Status::InvalidArgument(
                MakeLinearUnsupportedWeightDTypeMessage("QkvLinear"));
    }

    // Params are compile-time constants, but the shape-constraint vocabulary
    // only compares tensor dimensions, so a symbolic packed row count is
    // deferred to runtime validation instead of emitting a deferred check.
    const ShapeSymbol& packed_rows = weight_shape[0];
    const int64_t expected_rows = qkv_params.q_out_features + qkv_params.k_out_features +
                                  qkv_params.v_out_features;
    if (packed_rows.IsStatic() && packed_rows.GetStaticValue() != expected_rows) {
        return Status::InvalidArgument(
                "QkvLinear qkv_weight rows must equal q_out + k_out + v_out features");
    }

    // Reject proven incompatibility now; preserve symbolic compatibility as a
    // runtime equality check.
    const ShapeSymbol& in_features = input_shape[*input_rank - 1];
    const ShapeSymbol& weight_in = weight_shape[1];

    InferenceResult res;
    if (!AreProvablyEqual(in_features, weight_in)) {
        if (in_features.IsStatic() && weight_in.IsStatic()) {
            return Status::InvalidArgument(
                    "QkvLinear qkv_weight length must equal input last dimension");
        }

        res.runtime_checks.emplace_back(
                DimEqualConstraint{
                        .lhs = {.tensor_port = {.direction = TensorPortType::kInput,
                                                .tensor_idx = 0},
                                .dim_index = *input_rank - 1},
                        .rhs = {.tensor_port = {.direction = TensorPortType::kInput,
                                                .tensor_idx = 1},
                                .dim_index = 1}},
                "QkvLinear input last dimension must match qkv_weight input dimension");
    }

    std::vector<ShapeSymbol> base_shape;
    base_shape.reserve(*input_rank - 1);
    for (size_t i = 0; i < *input_rank - 1; ++i) {
        base_shape.push_back(input_shape[i]);
    }

    const auto make_output = [&](int64_t out_features) {
        std::vector<ShapeSymbol> shape = base_shape;
        shape.push_back(ShapeSymbol::CreateFromValue(out_features));
        return TensorSpec(input_spec.dtype, SymbolicShape(std::move(shape)));
    };

    res.outputs.push_back(make_output(qkv_params.q_out_features));
    res.outputs.push_back(make_output(qkv_params.k_out_features));
    res.outputs.push_back(make_output(qkv_params.v_out_features));

    return res;
}

}// namespace detail

}// namespace aethermind
