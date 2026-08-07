#include "aethermind/operators/elementwise_mul_op.h"
#include "aethermind/operators/op_params.h"
#include "aethermind/utils/overflow_check.h"
#include "const_eval_internal.h"

namespace aethermind {
namespace {

// Integer overflow is checked and reported as Status::Overflow before any
// wrapping can occur. Float paths are unchecked (inf is valid in folded graphs).
template<typename T>
Status MultiplyScalar(T lhs, T rhs, T& out) {
    if constexpr (std::is_integral_v<T>) {
        if (CheckOverflowMul(lhs, rhs, &out)) {
            return Status::Overflow("Mul constant evaluator integer overflow");
        }
    } else {
        out = lhs * rhs;
    }
    return Status::Ok();
}

struct MulScalarOp {
    template<typename T>
    static Status Apply(T lhs, T rhs, T& out) {
        return MultiplyScalar(lhs, rhs, out);
    }
};

// TU-local evaluator — registered via GetMulConstEvaluator() accessor.
class ElementwiseMulConstEvaluator final : public ConstEvaluator {
public:
    // Validates shapes and dtype match; requires identical shapes
    // (no broadcast support for element-wise Mul).
    // Produces a contiguous-output plan (element-wise op is always dense)
    // and an elementwise cost estimate for the pass's budget enforcement.
    StatusOr<ConstEvalPlan> Plan(std::span<const GraphValueDesc> inputs,
                                 std::span<const GraphValueDesc> outputs,
                                 const OpParams& params,
                                 AM_MAYBE_UNUSED const ConstEvalPolicy& policy) const override {
        if (inputs.size() != 2U || outputs.size() != 1U ||
            !std::holds_alternative<ElementwiseMulParams>(params)) {
            return Status::Unimplemented(
                    "ElementwiseMul constant evaluator requires two inputs and one output");
        }

        const TensorSpec& lhs = inputs[0].spec;
        const TensorSpec& rhs = inputs[1].spec;
        const TensorSpec& output = outputs[0].spec;
        if (!IsElementwiseMulSupportedDType(lhs.dtype) || rhs.dtype != lhs.dtype ||
            output.dtype != lhs.dtype) {
            return Status::Unimplemented(
                    MakeElementwiseMulUnsupportedDTypeMessage(
                            "ElementwiseMul constant evaluator"));
        }

        auto lhs_shape = ExtractStaticShape(lhs);
        AM_RETURN_IF_ERROR(lhs_shape.status());
        auto rhs_shape = ExtractStaticShape(rhs);
        AM_RETURN_IF_ERROR(rhs_shape.status());
        auto shape = ExtractStaticShape(output);
        AM_RETURN_IF_ERROR(shape.status());

        if (*lhs_shape != *shape || *rhs_shape != *shape) {
            return Status::Unimplemented(
                    "ElementwiseMul constant evaluator requires identical "
                    "static shapes for lhs, rhs, and output");
        }

        auto numel = CountElements(*shape);
        AM_RETURN_IF_ERROR(numel.status());
        auto cost = EstimateCost(output, *numel, detail::kMulOpsPerElement);
        AM_RETURN_IF_ERROR(cost.status());

        auto strides = MakeContiguousStrides(*shape);
        AM_RETURN_IF_ERROR(strides.status());

        ConstEvalPlan plan;
        plan.cost = *cost;
        plan.outputs.push_back({
                .spec = output,
                .quantization = outputs[0].quantization,
                .strides = std::move(*strides),
                .nbytes = cost->output_bytes,
                .name = "folded_" + outputs[0].name,
        });
        return plan;
    }

    // Flat fast path when both inputs are contiguous; strided kernel otherwise.
    Status Evaluate(std::span<const TensorView> inputs,
                    std::span<MutableTensorView> outputs,
                    const OpParams& params) const override {
        if (inputs.size() != 2U || outputs.size() != 1U ||
            !std::holds_alternative<ElementwiseMulParams>(params)) {
            return Status::InvalidArgument(
                    "ElementwiseMul constant evaluator received invalid view arity");
        }

        const DataType dtype = inputs[0].dtype();
        if (!IsElementwiseMulSupportedDType(dtype) || inputs[1].dtype() != dtype ||
            outputs[0].dtype() != dtype) {
            return Status::InvalidArgument(
                    MakeElementwiseMulUnsupportedDTypeMessage(
                            "ElementwiseMul constant evaluator"));
        }

        if (inputs[0].shape() != outputs[0].shape() ||
            inputs[1].shape() != outputs[0].shape()) {
            return Status::InvalidArgument(
                    "ElementwiseMul constant evaluator received mismatched shapes");
        }

        if (!outputs[0].is_contiguous()) {
            return Status::InvalidArgument(
                    "ElementwiseMul constant evaluator requires contiguous output tensor");
        }

        if (inputs[0].is_contiguous() && inputs[1].is_contiguous()) {
            return detail::EvaluateBinaryFlatByDType<MulScalarOp>(
                    dtype, inputs, outputs, outputs[0].numel());
        }

        return detail::EvaluateBinaryStridedByDType<MulScalarOp>(
                dtype, inputs, outputs, inputs[0].strides(), inputs[1].strides());
    }
};

}// namespace

const ConstEvaluator& detail::GetMulConstEvaluator() noexcept {
    static const ElementwiseMulConstEvaluator kEvaluator;
    return kEvaluator;
}

}// namespace aethermind
