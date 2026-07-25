#include "aethermind/operators/rmsnorm_op.h"
#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/operators/operator_registry.h"

namespace aethermind {
Status RmsNormOp::Prepare(OperatorContext& ctx) {
    if (ctx.backend == nullptr) {
        return Status::InvalidArgument("RmsNorm Prepare requires OperatorContext.backend");
    }

    const auto resolved = ctx.backend->ResolveKernelInfo(
            OpType::kRmsNorm,
            ctx.selector);

    if (!resolved.ok()) {
        return resolved.status();
    }

    resolved_kernel_ = resolved.value();
    if (resolved_kernel_.fn == nullptr) {
        return Status::Internal("RmsNorm Prepare resolved a kernel with null fn");
    }
    const auto eps_bytes = std::as_bytes(std::span{&params_.eps, size_t{1}});
    resolved_kernel_.attrs.assign(eps_bytes.begin(), eps_bytes.end());
    return Status::Ok();
}

Status RmsNormOp::Run(KernelContext& ctx,
                      const RuntimeBindingContext& bindings,
                      size_t step_index) const noexcept {
    if (resolved_kernel_.fn == nullptr) {
        return Status::FailedPrecondition("RmsNorm Run called before Prepare");
    }

    const auto binding = bindings.GetStepTensorBinding(step_index);
    if (!binding.ok()) {
        return binding.status();
    }

    const auto* b = binding.value();
    if (b->inputs.size() != 2) {
        return Status::InvalidArgument(
                "RmsNorm requires 2 input tensor bindings, got " +
                std::to_string(b->inputs.size()));
    }

    if (b->outputs.size() != 1) {
        return Status::InvalidArgument(
                "RmsNorm requires 1 output tensor binding, got " +
                std::to_string(b->outputs.size()));
    }

    return InvokeResolvedKernel(ctx, b->inputs, b->outputs);
}

AM_REGISTER_OPERATOR(OpType::kRmsNorm, RmsNormOp)


namespace detail {

// Shape inference and constraint analysis for the RmsNorm operator.
//
// Steps:
//   1. Validate OpParams variant type and eps scalar.
//   2. Validate input count via ValidateInferenceInputCount (exactly 2:
//      input and weight, checked against the operator schema).
//   3. Validate input rank (>=1). Leading batch/sequence dims may be
//      zero; only the last (hidden) dim must be positive (zero-length
//      reduction is undefined).
//   4. Validate weight rank (=1) and positive length.
//   5. Reconcile hidden_size (input last dim) with weight_len: fail-fast on
//      static hard conflict; otherwise emit a DimEqualConstraint deferred to
//      runtime and verified by the Executor.
//   6. Validate input/weight dtypes against the supported dtype set.  Mixed
//      input/weight dtypes are allowed; the kernel converts the weight to the
//      input dtype at runtime (HuggingFace LlamaRMSNorm convention).
//   7. Build InferenceResult: the output spec mirrors the input spec
//      (RmsNorm is shape-preserving), plus any deferred runtime checks.
StatusOr<InferenceResult> InferRmsNorm(const OpParams& params,
                                       std::span<const TensorSpec> inputs) {
    const auto* typed = std::get_if<RmsNormParams>(&params);
    if (typed == nullptr) {
        return Status::InvalidArgument("RmsNorm node requires RmsNormParams");
    }

    if (!std::isfinite(typed->eps) || typed->eps <= 0.0F) {
        return Status::InvalidArgument("RmsNormParams eps must be finite and positive");
    }

    AM_RETURN_IF_ERROR(ValidateInferenceInputCount(OpType::kRmsNorm, inputs));

    const auto& input_spec = inputs[0];
    const auto& weight_spec = inputs[1];

    // Mixed input/weight dtypes are allowed; the kernel converts the weight to
    // the input dtype at runtime (HuggingFace LlamaRMSNorm convention).
    if (!IsRmsNormSupportedDType(input_spec.dtype)) {
        return Status::InvalidArgument(
                MakeRmsNormUnsupportedDTypeMessage("RmsNorm input"));
    }

    if (!IsRmsNormSupportedDType(weight_spec.dtype)) {
        return Status::InvalidArgument(
                MakeRmsNormUnsupportedDTypeMessage("RmsNorm weight"));
    }

    const auto input_rank = input_spec.shape.rank();
    if (!input_rank.has_value() || *input_rank < 1) {
        return Status::InvalidArgument("RmsNorm input must have rank >= 1");
    }

    // Only the last (hidden) dimension must be positive: zero-length reduction
    // has no valid definition. Leading batch/sequence dimensions may be zero.
    const size_t rank = *input_rank;
    InferenceResult res;
    const ShapeSymbol& hidden_size = input_spec.shape[rank - 1];
    if (hidden_size.IsStatic()) {
        if (hidden_size.GetStaticValue() <= 0) {
            return Status::InvalidArgument(
                    "RmsNorm input last dimension must be positive when statically known.");
        }
    } else {
        res.runtime_checks.emplace_back(
                DimPositiveConstraint{.dim = {.tensor_port = {.direction = TensorPortType::kInput,
                                                              .tensor_idx = 0},
                                              .dim_index = rank - 1}},
                "RmsNorm input last dimension must be positive");
    }

    if (!HasRank(weight_spec.shape, 1)) {
        return Status::InvalidArgument("RmsNorm weight must be rank-1");
    }

    const ShapeSymbol& weight_len = weight_spec.shape[0];

    // Weight length positivity is enforced transitively: hidden_size > 0
    // (checked above) + hidden_size == weight_len (DimEqualConstraint below)
    // ⇒ weight_len > 0. No separate DimPositiveConstraint needed.
    //
    // Static mismatch is unrecoverable; dynamic/symbolic mismatches are deferred
    // to the Executor via a DimEqualConstraint.
    if (!AreProvablyEqual(hidden_size, weight_len)) {
        if (hidden_size.IsStatic() && weight_len.IsStatic()) {
            return Status::InvalidArgument(
                    "RmsNorm weight length must equal input last dimension");
        }

        res.runtime_checks.emplace_back(
                DimEqualConstraint{
                        .lhs = {.tensor_port = {.direction = TensorPortType::kInput,
                                                .tensor_idx = 0},
                                .dim_index = rank - 1},
                        .rhs = {.tensor_port = {.direction = TensorPortType::kInput,
                                                .tensor_idx = 1},
                                .dim_index = 0}},
                "RmsNorm hidden dimension must match weight length");
    }

    res.outputs.emplace_back(input_spec);
    return res;
}

}// namespace detail

}// namespace aethermind
