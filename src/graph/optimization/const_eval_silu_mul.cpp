#include "aethermind/operators/op_params.h"
#include "const_eval_internal.h"

#include <cmath>
#include <type_traits>

namespace aethermind {
namespace {

bool IsFoldableSiluMulDType(const DataType& dtype) {
    return dtype == DataType::Float32() ||
           dtype == DataType::Double() ||
           dtype == DataType::BFloat(16);
}

// Computes output = silu(gate) * up where silu(x) = x / (1 + exp(-x)).
// The branch on x >= 0 mirrors SiluScalarOp to avoid exp overflow for
// large negative inputs.
struct SiluMulScalarOp {
    template<typename T>
    static Status Apply(T gate, T up, T& out) {
        using ComputeType = std::conditional_t<std::is_same_v<T, BFloat16>, float, T>;
        const auto x = static_cast<ComputeType>(gate);
        ComputeType silu;
        if (x >= ComputeType{0}) {
            silu = x / (ComputeType{1} + std::exp(-x));
        } else {
            const ComputeType exp_x = std::exp(x);
            silu = x * exp_x / (ComputeType{1} + exp_x);
        }
        out = static_cast<T>(silu * static_cast<ComputeType>(up));
        return Status::Ok();
    }
};

// TU-local evaluator — registered via GetSiluMulConstEvaluator() accessor.
class SiluMulConstEvaluator final : public ConstEvaluator {
public:
    // Validates shapes and dtype match across gate/up/output.
    // Produces a contiguous-output plan and cost estimate.
    AM_NODISCARD StatusOr<ConstEvalPlan> Plan(std::span<const GraphValueDesc> inputs,
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
        if (!IsFoldableSiluMulDType(gate.dtype) || up.dtype != gate.dtype ||
            output.dtype != gate.dtype) {
            return Status::Unimplemented(
                    "SiluMul constant evaluator only supports float32, float64, and bfloat16 tensors");
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
                .debug_name = "folded_" + outputs[0].name,
        });
        return plan;
    }

    // Flat fast path when both inputs are contiguous; strided kernel otherwise.
    AM_NODISCARD Status Evaluate(std::span<const TensorView> inputs,
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
        if (!IsFoldableSiluMulDType(dtype) || up.dtype() != dtype || out.dtype() != dtype) {
            return Status::InvalidArgument(
                    "SiluMul constant evaluator received unsupported dtype");
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
            return detail::EvaluateBinaryFlatByDType<SiluMulScalarOp>(
                    dtype, inputs, outputs, out.numel());
        }

        return detail::EvaluateBinaryStridedByDType<SiluMulScalarOp>(
                dtype, inputs, outputs, gate.strides(), up.strides());
    }
};

}// namespace

const ConstEvaluator& detail::GetSiluMulConstEvaluator() noexcept {
    static const SiluMulConstEvaluator kEvaluator;
    return kEvaluator;
}

}// namespace aethermind
