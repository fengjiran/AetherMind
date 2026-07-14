#include "aethermind/model/graph/op_params.h"
#include "aethermind/utils/overflow_check.h"
#include "const_eval_internal.h"

#include <type_traits>

namespace aethermind {
namespace {

bool IsFoldableMulDType(const DataType& dtype) {
    return dtype == DataType::Float32() ||
           dtype == DataType::Double() ||
           dtype == DataType::BFloat(16) ||
           dtype == DataType::Int(32) ||
           dtype == DataType::Int(64);
}

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

class ElementwiseMulConstEvaluator final : public ConstEvaluator {
public:
    AM_NODISCARD StatusOr<ConstEvalPlan> Plan(std::span<const NodeOutputDesc> inputs,
                                              std::span<const NodeOutputDesc> outputs,
                                              const OpParams& params,
                                              const ConstEvalPolicy& policy) const override {
        if (inputs.size() != 2U || outputs.size() != 1U ||
            !std::holds_alternative<ElementwiseMulParams>(params)) {
            return Status::Unimplemented(
                    "ElementwiseMul constant evaluator requires two inputs and one output");
        }

        const TensorSpec& lhs = inputs[0].spec;
        const TensorSpec& rhs = inputs[1].spec;
        const TensorSpec& output = outputs[0].spec;
        if (!IsFoldableMulDType(lhs.dtype) || rhs.dtype != lhs.dtype || output.dtype != lhs.dtype) {
            return Status::Unimplemented(
                    "ElementwiseMul constant evaluator only supports float32, float64, bfloat16, "
                    "int32, and int64 tensors");
        }

        auto lhs_shape = ExtractStaticShape(lhs);
        AM_RETURN_IF_ERROR(lhs_shape.status());
        auto rhs_shape = ExtractStaticShape(rhs);
        AM_RETURN_IF_ERROR(rhs_shape.status());
        auto shape = ExtractStaticShape(output);
        AM_RETURN_IF_ERROR(shape.status());
        if (lhs_shape->empty() || rhs_shape->empty() || shape->empty()) {
            return Status::Unimplemented(
                    "ElementwiseMul constant evaluator requires non-scalar tensor shapes");
        }

        if (*lhs_shape != *shape || *rhs_shape != *shape) {
            return Status::Unimplemented(
                    "ElementwiseMul constant evaluator requires identical static shapes for lhs, rhs, and output");
        }

        auto numel = CountElements(*shape);
        AM_RETURN_IF_ERROR(numel.status());
        if (static_cast<size_t>(*numel) > policy.max_compute_elements) {
            return Status::Unimplemented(
                    "ElementwiseMul constant evaluator compute budget exceeded");
        }

        auto nbytes = CountBytes(output);
        AM_RETURN_IF_ERROR(nbytes.status());
        if (*nbytes > policy.max_output_bytes) {
            return Status::Unimplemented(
                    "ElementwiseMul constant evaluator output byte budget exceeded");
        }

        auto strides = MakeContiguousStrides(*shape);
        AM_RETURN_IF_ERROR(strides.status());

        ConstEvalPlan plan;
        plan.outputs.push_back({
                .spec = output,
                .quantization = outputs[0].quantization,
                .strides = std::move(*strides),
                .nbytes = *nbytes,
                .debug_name = "folded_" + outputs[0].debug_name,
        });
        return plan;
    }

    AM_NODISCARD Status Evaluate(std::span<const TensorView> inputs,
                                 std::span<MutableTensorView> outputs,
                                 const OpParams& params) const override {
        if (inputs.size() != 2U || outputs.size() != 1U ||
            !std::holds_alternative<ElementwiseMulParams>(params)) {
            return Status::InvalidArgument(
                    "ElementwiseMul constant evaluator received invalid view arity");
        }

        const DataType dtype = inputs[0].dtype();
        if (!IsFoldableMulDType(dtype) || inputs[1].dtype() != dtype ||
            outputs[0].dtype() != dtype) {
            return Status::InvalidArgument(
                    "ElementwiseMul constant evaluator received unsupported dtype");
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
            return detail::EvaluateBinaryByDType<MulScalarOp>(dtype, inputs, outputs, outputs[0].numel());
        }

        return detail::EvaluateBinaryStridedByDType<MulScalarOp>(dtype, inputs, outputs,
                                                                 inputs[0].strides(), inputs[1].strides());
    }
};

}// namespace

const ConstEvaluator& detail::GetMulConstEvaluator() noexcept {
    static const ElementwiseMulConstEvaluator kEvaluator;
    return kEvaluator;
}

}// namespace aethermind
