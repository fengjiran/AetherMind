#include "aethermind/dtypes/float8_e4m3fn.h"
#include "aethermind/dtypes/float8_e5m2.h"
#include "aethermind/operators/op_params.h"
#include "aethermind/operators/silu_mul_op.h"
#include "const_eval_internal.h"

#include <cmath>

namespace aethermind {
namespace {

// Computes output = silu(gate) * up where silu(x) = x / (1 + exp(-x)).
// The branch on g >= 0 mirrors SiluScalarOp to avoid exp overflow for
// large negative inputs. Half, bfloat16 and fp8 inputs are promoted to
// float for intermediate computation and rounded once on output to
// preserve accuracy in the wider type (matching the arithmetic model
// of their dtype headers).
struct SiluMulScalarOp {
    template<typename T>
    static Status Apply(T gate, T up, T& out) {
        const auto g = static_cast<float>(gate);
        const auto u = static_cast<float>(up);
        float silu;
        if (g >= 0.0F) {
            silu = g / (1.0F + std::exp(-g));
        } else {
            const float exp_g = std::exp(g);
            silu = g * exp_g / (1.0F + exp_g);
        }
        out = static_cast<T>(silu * u);
        return Status::Ok();
    }
};

Status EvaluateSiluMulFlatByDType(const DataType& dtype,
                                  std::span<const TensorView> inputs,
                                  std::span<MutableTensorView> outputs,
                                  int64_t numel) {
    if (dtype == DataType::Float32()) {
        return detail::EvaluateBinaryFlatTyped<SiluMulScalarOp, float>(inputs, outputs, numel);
    }

    if (dtype == DataType::Float(16)) {
        return detail::EvaluateBinaryFlatTyped<SiluMulScalarOp, Half>(inputs, outputs, numel);
    }

    if (dtype == DataType::BFloat(16)) {
        return detail::EvaluateBinaryFlatTyped<SiluMulScalarOp, BFloat16>(inputs, outputs, numel);
    }

    if (dtype == DataType::Float8E4M3FN()) {
        return detail::EvaluateBinaryFlatTyped<SiluMulScalarOp, Float8_e4m3fn>(inputs, outputs, numel);
    }

    if (dtype == DataType::Float8E5M2()) {
        return detail::EvaluateBinaryFlatTyped<SiluMulScalarOp, Float8_e5m2>(inputs, outputs, numel);
    }
    return Status::InvalidArgument("SiluMul constant evaluator received unsupported dtype");
}

Status EvaluateSiluMulStridedByDType(const DataType& dtype,
                                     std::span<const TensorView> inputs,
                                     std::span<MutableTensorView> outputs,
                                     std::span<const int64_t> gate_strides,
                                     std::span<const int64_t> up_strides) {
    if (dtype == DataType::Float32()) {
        return detail::EvaluateBinaryStridedKernel<SiluMulScalarOp, float>(
                inputs, outputs, gate_strides, up_strides);
    }

    if (dtype == DataType::Float(16)) {
        return detail::EvaluateBinaryStridedKernel<SiluMulScalarOp, Half>(
                inputs, outputs, gate_strides, up_strides);
    }

    if (dtype == DataType::BFloat(16)) {
        return detail::EvaluateBinaryStridedKernel<SiluMulScalarOp, BFloat16>(
                inputs, outputs, gate_strides, up_strides);
    }

    if (dtype == DataType::Float8E4M3FN()) {
        return detail::EvaluateBinaryStridedKernel<SiluMulScalarOp, Float8_e4m3fn>(
                inputs, outputs, gate_strides, up_strides);
    }

    if (dtype == DataType::Float8E5M2()) {
        return detail::EvaluateBinaryStridedKernel<SiluMulScalarOp, Float8_e5m2>(
                inputs, outputs, gate_strides, up_strides);
    }
    return Status::InvalidArgument("SiluMul constant evaluator received unsupported dtype");
}

// TU-local evaluator — registered via GetSiluMulConstEvaluator() accessor.
class SiluMulConstEvaluator final : public ConstEvaluator {
public:
    // Validates shapes and dtype match across gate/up/output.
    // Produces a contiguous-output plan and cost estimate.
    StatusOr<ConstEvalPlan> Plan(std::span<const GraphValueDesc> inputs,
                                 std::span<const GraphValueDesc> outputs,
                                 const OpParams& params,
                                 AM_MAYBE_UNUSED const ConstEvalPolicy& policy) const override {
        if (inputs.size() != 2U || outputs.size() != 1U ||
            !std::holds_alternative<SiluMulParams>(params)) {
            return Status::Unimplemented(
                    "SiluMul constant evaluator requires two inputs and one output");
        }

        const TensorSpec& gate = inputs[0].spec;
        const TensorSpec& up = inputs[1].spec;
        const TensorSpec& output = outputs[0].spec;
        if (!IsSiluMulSupportedDType(gate.dtype) || up.dtype != gate.dtype ||
            output.dtype != gate.dtype) {
            return Status::Unimplemented(
                    MakeSiluMulUnsupportedDTypeMessage("SiluMul constant evaluator"));
        }

        auto gate_shape = ExtractStaticShape(gate);
        AM_RETURN_IF_ERROR(gate_shape.status());
        auto up_shape = ExtractStaticShape(up);
        AM_RETURN_IF_ERROR(up_shape.status());
        auto output_shape = ExtractStaticShape(output);
        AM_RETURN_IF_ERROR(output_shape.status());

        if (*gate_shape != *output_shape || *up_shape != *output_shape) {
            return Status::Unimplemented(
                    "SiluMul constant evaluator requires identical static shapes for gate, up, and output");
        }

        auto numel = CountElements(*output_shape);
        AM_RETURN_IF_ERROR(numel.status());
        auto cost = EstimateCost(output, *numel, detail::kSiluMulOpsPerElement);
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

    // Flat fast path when both inputs are contiguous; strided kernel otherwise.
    Status Evaluate(std::span<const TensorView> inputs,
                    std::span<MutableTensorView> outputs,
                    const OpParams& params) const override {
        if (inputs.size() != 2U || outputs.size() != 1U || !std::holds_alternative<SiluMulParams>(params)) {
            return Status::InvalidArgument(
                    "SiluMul constant evaluator received invalid view arity");
        }

        const auto& gate = inputs[0];
        const auto& up = inputs[1];
        const auto& out = outputs[0];

        const DataType dtype = gate.dtype();
        if (!IsSiluMulSupportedDType(dtype) || up.dtype() != dtype || out.dtype() != dtype) {
            return Status::InvalidArgument(
                    MakeSiluMulUnsupportedDTypeMessage("SiluMul constant evaluator"));
        }

        if (gate.shape() != out.shape() || up.shape() != out.shape()) {
            return Status::InvalidArgument(
                    "SiluMul constant evaluator received mismatched shapes");
        }

        if (!out.is_contiguous()) {
            return Status::InvalidArgument(
                    "SiluMul constant evaluator requires contiguous output tensor");
        }

        if (gate.is_contiguous() && up.is_contiguous()) {
            return EvaluateSiluMulFlatByDType(dtype, inputs, outputs, out.numel());
        }

        return EvaluateSiluMulStridedByDType(
                dtype, inputs, outputs, gate.strides(), up.strides());
    }
};

}// namespace

const ConstEvaluator& detail::GetSiluMulConstEvaluator() noexcept {
    static const SiluMulConstEvaluator kEvaluator;
    return kEvaluator;
}

}// namespace aethermind
