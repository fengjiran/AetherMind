#include "aethermind/model/graph/op_params.h"
#include "const_eval_internal.h"

#include <cmath>
#include <type_traits>

namespace aethermind {
namespace {

bool IsFoldableSiluDType(const DataType& dtype) {
    return dtype == DataType::Float32() ||
           dtype == DataType::Double() ||
           dtype == DataType::BFloat(16);
}

// Stable SiLU: sign-dependent formula prevents overflow at large |x|.
// BFloat16 is promoted to float for intermediate computation and
// rounded once on output to preserve accuracy in the wider type.
struct SiluScalarOp {
    template<typename T>
    static Status Apply(T input, T& output) {
        using ComputeType = std::conditional_t<std::is_same_v<T, BFloat16>, float, T>;
        const auto x = static_cast<ComputeType>(input);
        ComputeType result;
        if (x >= ComputeType{0}) {
            result = x / (ComputeType{1} + std::exp(-x));
        } else {
            const ComputeType exp_x = std::exp(x);
            result = x * exp_x / (ComputeType{1} + exp_x);
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

    if (dtype == DataType::Double()) {
        return detail::EvaluateUnaryFlatTyped<SiluScalarOp, double>(inputs, outputs, numel);
    }

    if (dtype == DataType::BFloat(16)) {
        return detail::EvaluateUnaryFlatTyped<SiluScalarOp, BFloat16>(inputs, outputs, numel);
    }
    return Status::InvalidArgument("Silu constant evaluator received unsupported dtype");
}

Status EvaluateSiluStridedByDType(const DataType& dtype,
                                  std::span<const TensorView> inputs,
                                  std::span<MutableTensorView> outputs,
                                  std::span<const int64_t> input_strides) {
    if (dtype == DataType::Float32()) {
        return detail::EvaluateUnaryStridedKernel<SiluScalarOp, float>(inputs, outputs, input_strides);
    }

    if (dtype == DataType::Double()) {
        return detail::EvaluateUnaryStridedKernel<SiluScalarOp, double>(inputs, outputs, input_strides);
    }

    if (dtype == DataType::BFloat(16)) {
        return detail::EvaluateUnaryStridedKernel<SiluScalarOp, BFloat16>(inputs, outputs, input_strides);
    }
    return Status::InvalidArgument("Silu constant evaluator received unsupported dtype");
}

// TU-local evaluator — registered via GetSiluConstEvaluator() accessor.
// The registry holds a function pointer, not the concrete type.
class SiluConstEvaluator final : public ConstEvaluator {
public:
    // Validates shapes, dtype, and budgets; produces a contiguous-output plan.
    // SiLU is element-wise so the output is always dense and contiguous.
    AM_NODISCARD StatusOr<ConstEvalPlan> Plan(std::span<const NodeOutputDesc> inputs,
                                              std::span<const NodeOutputDesc> outputs,
                                              const OpParams& params,
                                              const ConstEvalPolicy& policy) const override {
        if (inputs.size() != 1U || outputs.size() != 1U || !std::holds_alternative<SiluParams>(params)) {
            return Status::Unimplemented(
                    "Silu constant evaluator requires one input and one output");
        }

        const TensorSpec& input = inputs[0].spec;
        const TensorSpec& output = outputs[0].spec;
        if (!IsFoldableSiluDType(input.dtype) || output.dtype != input.dtype) {
            return Status::Unimplemented(
                    "Silu constant evaluator only supports float32, float64, and bfloat16 tensors");
        }

        auto input_shape = ExtractStaticShape(input);
        AM_RETURN_IF_ERROR(input_shape.status());
        auto output_shape = ExtractStaticShape(output);
        AM_RETURN_IF_ERROR(output_shape.status());

        if (*input_shape != *output_shape) {
            return Status::Unimplemented(
                    "Silu constant evaluator requires identical static shapes for input and output");
        }

        auto numel = CountElements(*output_shape);
        AM_RETURN_IF_ERROR(numel.status());
        if (static_cast<size_t>(*numel) > policy.max_compute_elements) {
            return Status::Unimplemented(
                    "Silu constant evaluator compute budget exceeded");
        }

        auto nbytes = CountBytes(output);
        AM_RETURN_IF_ERROR(nbytes.status());
        if (*nbytes > policy.max_output_bytes) {
            return Status::Unimplemented(
                    "Silu constant evaluator output byte budget exceeded");
        }

        auto output_strides = MakeContiguousStrides(*output_shape);
        AM_RETURN_IF_ERROR(output_strides.status());

        ConstEvalPlan plan;
        plan.outputs.push_back({
                .spec = output,
                .quantization = outputs[0].quantization,
                .strides = std::move(*output_strides),
                .nbytes = *nbytes,
                .debug_name = "folded_" + outputs[0].debug_name,
        });
        return plan;
    }

    // Flat fast path for contiguous input; strided kernel for non-contiguous.
    // The flat path avoids stride indirection and is measurably faster.
    AM_NODISCARD Status Evaluate(std::span<const TensorView> inputs,
                                 std::span<MutableTensorView> outputs,
                                 const OpParams& params) const override {
        if (inputs.size() != 1U || outputs.size() != 1U ||
            !std::holds_alternative<SiluParams>(params)) {
            return Status::InvalidArgument(
                    "Silu constant evaluator received invalid view arity");
        }

        const DataType dtype = inputs[0].dtype();
        if (!IsFoldableSiluDType(dtype) || outputs[0].dtype() != dtype) {
            return Status::InvalidArgument(
                    "Silu constant evaluator received unsupported dtype");
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
