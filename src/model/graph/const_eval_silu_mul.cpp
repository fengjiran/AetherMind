#include "aethermind/model/graph/op_params.h"
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

class SiluMulConstEvaluator final : public ConstEvaluator {
public:
    AM_NODISCARD StatusOr<ConstEvalPlan> Plan(std::span<const NodeOutputDesc> inputs,
                                              std::span<const NodeOutputDesc> outputs,
                                              const OpParams& params,
                                              const ConstEvalPolicy& policy) const override {
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
        auto shape = ExtractStaticShape(output);
        AM_RETURN_IF_ERROR(shape.status());
        if (gate_shape->empty() || up_shape->empty() || shape->empty()) {
            return Status::Unimplemented(
                    "SiluMul constant evaluator requires non-scalar tensor shapes");
        }

        if (*gate_shape != *shape || *up_shape != *shape) {
            return Status::Unimplemented(
                    "SiluMul constant evaluator requires identical static shapes for gate, up, and output");
        }

        auto numel = CountElements(*shape);
        AM_RETURN_IF_ERROR(numel.status());
        if (static_cast<size_t>(*numel) > policy.max_compute_elements) {
            return Status::Unimplemented(
                    "SiluMul constant evaluator compute budget exceeded");
        }

        auto nbytes = CountBytes(output);
        AM_RETURN_IF_ERROR(nbytes.status());
        if (*nbytes > policy.max_output_bytes) {
            return Status::Unimplemented(
                    "SiluMul constant evaluator output byte budget exceeded");
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
            !std::holds_alternative<SiluMulParams>(params)) {
            return Status::InvalidArgument(
                    "SiluMul constant evaluator received invalid view arity");
        }

        const DataType dtype = inputs[0].dtype();
        if (!IsFoldableSiluMulDType(dtype) || inputs[1].dtype() != dtype ||
            outputs[0].dtype() != dtype) {
            return Status::InvalidArgument(
                    "SiluMul constant evaluator received unsupported dtype");
        }

        if (inputs[0].shape() != outputs[0].shape() ||
            inputs[1].shape() != outputs[0].shape()) {
            return Status::InvalidArgument(
                    "SiluMul constant evaluator received mismatched shapes");
        }

        if (!outputs[0].is_contiguous()) {
            return Status::InvalidArgument(
                    "SiluMul constant evaluator requires contiguous output tensor");
        }

        if (inputs[0].is_contiguous() && inputs[1].is_contiguous()) {
            return detail::EvaluateBinaryFlatByDType<SiluMulScalarOp>(dtype, inputs, outputs, outputs[0].numel());
        }

        return detail::EvaluateBinaryStridedByDType<SiluMulScalarOp>(dtype, inputs, outputs,
                                                                     inputs[0].strides(), inputs[1].strides());
    }
};

}// namespace

const ConstEvaluator& detail::GetSiluMulConstEvaluator() noexcept {
    static const SiluMulConstEvaluator kEvaluator;
    return kEvaluator;
}

}// namespace aethermind
