#include "aethermind/operators/op_params.h"
#include "aethermind/operators/ops/cast_op.h"
#include "const_eval_internal.h"

namespace aethermind {
namespace {

// Dispatches the six supported (source, target) dtype pairs. The conversion
// routes through float so Half/BFloat16 narrowing reuses their frozen bit
// contracts (RNE, gradual underflow, quiet NaN, signed zero).
Status EvaluateCastTyped(const DataType& input_dtype,
                         const DataType& output_dtype,
                         std::span<const TensorView> inputs,
                         std::span<MutableTensorView> outputs,
                         std::span<const int64_t> input_strides,
                         bool flat) {
    const int64_t numel = outputs[0].numel();
    if (input_dtype == DataType::Float32()) {
        if (output_dtype == DataType::Float(16)) {
            return flat ? detail::EvaluateCastFlatTyped<float, Half>(inputs, outputs, numel)
                        : detail::EvaluateCastStridedKernel<float, Half>(
                                  inputs, outputs, input_strides);
        }

        if (output_dtype == DataType::BFloat(16)) {
            return flat ? detail::EvaluateCastFlatTyped<float, BFloat16>(inputs, outputs, numel)
                        : detail::EvaluateCastStridedKernel<float, BFloat16>(
                                  inputs, outputs, input_strides);
        }
    } else if (input_dtype == DataType::Float(16)) {
        if (output_dtype == DataType::Float32()) {
            return flat ? detail::EvaluateCastFlatTyped<Half, float>(inputs, outputs, numel)
                        : detail::EvaluateCastStridedKernel<Half, float>(
                                  inputs, outputs, input_strides);
        }

        if (output_dtype == DataType::BFloat(16)) {
            return flat ? detail::EvaluateCastFlatTyped<Half, BFloat16>(inputs, outputs, numel)
                        : detail::EvaluateCastStridedKernel<Half, BFloat16>(
                                  inputs, outputs, input_strides);
        }
    } else if (input_dtype == DataType::BFloat(16)) {
        if (output_dtype == DataType::Float32()) {
            return flat ? detail::EvaluateCastFlatTyped<BFloat16, float>(inputs, outputs, numel)
                        : detail::EvaluateCastStridedKernel<BFloat16, float>(
                                  inputs, outputs, input_strides);
        }

        if (output_dtype == DataType::Float(16)) {
            return flat ? detail::EvaluateCastFlatTyped<BFloat16, Half>(inputs, outputs, numel)
                        : detail::EvaluateCastStridedKernel<BFloat16, Half>(
                                  inputs, outputs, input_strides);
        }
    }
    return Status::InvalidArgument(
            "Cast constant evaluator received an unsupported dtype pair");
}

// TU-local evaluator — registered via GetCastConstEvaluator() accessor.
class CastConstEvaluator final : public ConstEvaluator {
public:
    // Cast is element-wise, so the output is always dense and contiguous.
    StatusOr<ConstEvalPlan> Plan(std::span<const GraphValueDesc> inputs,
                                 std::span<const GraphValueDesc> outputs,
                                 const OpParams& params,
                                 AM_MAYBE_UNUSED const ConstEvalPolicy& policy) const override {
        if (inputs.size() != 1U || outputs.size() != 1U ||
            !std::holds_alternative<CastParams>(params)) {
            return Status::Unimplemented(
                    "Cast constant evaluator requires one input and one output");
        }

        const auto& cast_params = std::get<CastParams>(params);
        const TensorSpec& input = inputs[0].spec;
        const TensorSpec& output = outputs[0].spec;
        if (!IsCastSupportedDType(input.dtype) || !IsCastSupportedDType(output.dtype) ||
            output.dtype != cast_params.target_dtype || output.dtype == input.dtype) {
            return Status::Unimplemented(
                    MakeCastUnsupportedDTypeMessage("Cast constant evaluator"));
        }

        auto input_shape = ExtractStaticShape(input);
        AM_RETURN_IF_ERROR(input_shape.status());
        auto output_shape = ExtractStaticShape(output);
        AM_RETURN_IF_ERROR(output_shape.status());
        if (*input_shape != *output_shape) {
            return Status::Unimplemented(
                    "Cast constant evaluator requires identical "
                    "static shapes for input and output");
        }

        auto numel = CountElements(*output_shape);
        AM_RETURN_IF_ERROR(numel.status());
        auto cost = EstimateCost(output, *numel, detail::kCastOpsPerElement);
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

    Status Evaluate(std::span<const TensorView> inputs,
                    std::span<MutableTensorView> outputs,
                    const OpParams& params) const override {
        if (inputs.size() != 1U || outputs.size() != 1U ||
            !std::holds_alternative<CastParams>(params)) {
            return Status::InvalidArgument(
                    "Cast constant evaluator received invalid view arity");
        }

        const DataType input_dtype = inputs[0].dtype();
        const DataType output_dtype = outputs[0].dtype();
        if (!IsCastSupportedDType(input_dtype) || !IsCastSupportedDType(output_dtype) ||
            output_dtype != std::get<CastParams>(params).target_dtype ||
            output_dtype == input_dtype) {
            return Status::InvalidArgument(
                    MakeCastUnsupportedDTypeMessage("Cast constant evaluator"));
        }

        if (inputs[0].shape() != outputs[0].shape()) {
            return Status::InvalidArgument(
                    "Cast constant evaluator received mismatched shapes");
        }

        if (!outputs[0].is_contiguous()) {
            return Status::InvalidArgument(
                    "Cast constant evaluator requires contiguous output tensor");
        }

        return EvaluateCastTyped(input_dtype, output_dtype, inputs, outputs,
                                 inputs[0].strides(), inputs[0].is_contiguous());
    }
};

} // namespace

const ConstEvaluator& detail::GetCastConstEvaluator() noexcept {
    static const CastConstEvaluator kEvaluator;
    return kEvaluator;
}

} // namespace aethermind
