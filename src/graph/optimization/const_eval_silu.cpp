#include "aethermind/dtypes/float8_e4m3fn.h"
#include "aethermind/dtypes/float8_e5m2.h"
#include "aethermind/operators/op_params.h"
#include "aethermind/operators/silu_op.h"
#include "const_eval_internal.h"

#include <cmath>

namespace aethermind {
namespace {

// Stable SiLU: sign-dependent formula prevents overflow at large |x|.
// Half, bfloat16 and fp8 inputs are promoted to float for intermediate
// computation and rounded once on output to preserve accuracy in the
// wider type (matching the arithmetic model of their dtype headers).
struct SiluScalarOp {
    template<typename T>
    static Status Apply(T input, T& output) {
        const auto x = static_cast<float>(input);
        float result;
        if (x >= 0.0F) {
            result = x / (1.0F + std::exp(-x));
        } else {
            const float exp_x = std::exp(x);
            result = x * exp_x / (1.0F + exp_x);
        }
        output = static_cast<T>(result);
        return Status::Ok();
    }
};

Status EvaluateSiluFlatByDType(const DataType& dtype,
                               std::span<const TensorView> inputs,
                               std::span<MutableTensorView> outputs,
                               int64_t numel) {
    if (dtype == DataType::Float32()) {
        return detail::EvaluateUnaryFlatTyped<SiluScalarOp, float>(inputs, outputs, numel);
    }

    if (dtype == DataType::Float(16)) {
        return detail::EvaluateUnaryFlatTyped<SiluScalarOp, Half>(inputs, outputs, numel);
    }

    if (dtype == DataType::BFloat(16)) {
        return detail::EvaluateUnaryFlatTyped<SiluScalarOp, BFloat16>(inputs, outputs, numel);
    }

    if (dtype == DataType::Float8E4M3FN()) {
        return detail::EvaluateUnaryFlatTyped<SiluScalarOp, Float8_e4m3fn>(inputs, outputs, numel);
    }

    if (dtype == DataType::Float8E5M2()) {
        return detail::EvaluateUnaryFlatTyped<SiluScalarOp, Float8_e5m2>(inputs, outputs, numel);
    }
    return Status::InvalidArgument("Silu constant evaluator received unsupported dtype");
}

Status EvaluateSiluStridedByDType(const DataType& dtype,
                                  std::span<const TensorView> inputs,
                                  std::span<MutableTensorView> outputs,
                                  std::span<const int64_t> input_strides) {
    if (dtype == DataType::Float32()) {
        return detail::EvaluateUnaryStridedKernel<SiluScalarOp, float>(
                inputs, outputs, input_strides);
    }

    if (dtype == DataType::Float(16)) {
        return detail::EvaluateUnaryStridedKernel<SiluScalarOp, Half>(
                inputs, outputs, input_strides);
    }

    if (dtype == DataType::BFloat(16)) {
        return detail::EvaluateUnaryStridedKernel<SiluScalarOp, BFloat16>(
                inputs, outputs, input_strides);
    }

    if (dtype == DataType::Float8E4M3FN()) {
        return detail::EvaluateUnaryStridedKernel<SiluScalarOp, Float8_e4m3fn>(
                inputs, outputs, input_strides);
    }

    if (dtype == DataType::Float8E5M2()) {
        return detail::EvaluateUnaryStridedKernel<SiluScalarOp, Float8_e5m2>(
                inputs, outputs, input_strides);
    }
    return Status::InvalidArgument("Silu constant evaluator received unsupported dtype");
}

// TU-local evaluator — registered via GetSiluConstEvaluator() accessor.
// The registry holds a function pointer, not the concrete type.
class SiluConstEvaluator final : public ConstEvaluator {
public:
    // Validates shapes and dtype; produces a contiguous-output plan and cost estimate.
    // SiLU is element-wise so the output is always dense and contiguous.
    StatusOr<ConstEvalPlan> Plan(std::span<const GraphValueDesc> inputs,
                                 std::span<const GraphValueDesc> outputs,
                                 const OpParams& params,
                                 AM_MAYBE_UNUSED const ConstEvalPolicy& policy) const override {
        if (inputs.size() != 1U || outputs.size() != 1U ||
            !std::holds_alternative<SiluParams>(params)) {
            return Status::Unimplemented(
                    "Silu constant evaluator requires one input and one output");
        }

        const TensorSpec& input = inputs[0].spec;
        const TensorSpec& output = outputs[0].spec;
        if (!IsSiluSupportedDType(input.dtype) || output.dtype != input.dtype) {
            return Status::Unimplemented(
                    MakeSiluUnsupportedDTypeMessage("Silu constant evaluator"));
        }

        auto input_shape = ExtractStaticShape(input);
        AM_RETURN_IF_ERROR(input_shape.status());
        auto output_shape = ExtractStaticShape(output);
        AM_RETURN_IF_ERROR(output_shape.status());

        if (*input_shape != *output_shape) {
            return Status::Unimplemented(
                    "Silu constant evaluator requires identical "
                    "static shapes for input and output");
        }

        auto numel = CountElements(*output_shape);
        AM_RETURN_IF_ERROR(numel.status());
        auto cost = EstimateCost(output, *numel, detail::kSiluOpsPerElement);
        AM_RETURN_IF_ERROR(cost.status());

        auto output_strides = MakeContiguousStrides(*output_shape);
        AM_RETURN_IF_ERROR(output_strides.status());

        ConstEvalPlan plan;
        plan.cost = *cost;
        plan.outputs.push_back({
                .spec = output,
                .quantization = outputs[0].quantization,
                .strides = std::move(*output_strides),
                .nbytes = cost->output_bytes,
                .name = "folded_" + outputs[0].name,
        });
        return plan;
    }

    // Flat fast path for contiguous input; strided kernel for non-contiguous.
    // The flat path avoids stride indirection and is measurably faster.
    Status Evaluate(std::span<const TensorView> inputs,
                    std::span<MutableTensorView> outputs,
                    const OpParams& params) const override {
        if (inputs.size() != 1U || outputs.size() != 1U ||
            !std::holds_alternative<SiluParams>(params)) {
            return Status::InvalidArgument(
                    "Silu constant evaluator received invalid view arity");
        }

        const DataType dtype = inputs[0].dtype();
        if (!IsSiluSupportedDType(dtype) || outputs[0].dtype() != dtype) {
            return Status::InvalidArgument(
                    MakeSiluUnsupportedDTypeMessage("Silu constant evaluator"));
        }

        if (inputs[0].shape() != outputs[0].shape()) {
            return Status::InvalidArgument(
                    "Silu constant evaluator received mismatched shapes");
        }

        if (!outputs[0].is_contiguous()) {
            return Status::InvalidArgument(
                    "Silu constant evaluator requires contiguous output tensor");
        }

        if (inputs[0].is_contiguous()) {
            return EvaluateSiluFlatByDType(dtype, inputs, outputs, outputs[0].numel());
        }

        return EvaluateSiluStridedByDType(dtype, inputs, outputs, inputs[0].strides());
    }
};

}// namespace

const ConstEvaluator& detail::GetSiluConstEvaluator() noexcept {
    static const SiluConstEvaluator kEvaluator;
    return kEvaluator;
}

}// namespace aethermind
