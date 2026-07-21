#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/add_op.h"
#include "aethermind/shape_inference/broadcast.h"
#include "aethermind/utils/overflow_check.h"
#include "const_eval_internal.h"

#include <algorithm>
#include <type_traits>

namespace aethermind {
namespace {

// Integer overflow is checked and reported as Status::Overflow before any
// wrapping can occur. Float paths are unchecked (inf is valid in folded graphs).
template<typename T>
Status AddScalar(T lhs, T rhs, T& out) {
    if constexpr (std::is_integral_v<T>) {
        if (CheckOverflowAdd(lhs, rhs, &out)) {
            return Status::Overflow("Add constant evaluator integer overflow");
        }
    } else {
        out = lhs + rhs;
    }
    return Status::Ok();
}

struct AddScalarOp {
    template<typename T>
    static Status Apply(T lhs, T rhs, T& out) {
        return AddScalar(lhs, rhs, out);
    }
};

// TU-local evaluator — registered via GetAddConstEvaluator() accessor.
class AddConstEvaluator final : public ConstEvaluator {
public:
    // Validates shapes with broadcast compatibility, dtype match, and budgets.
    // Produces a contiguous-output plan (broadcast result is always dense).
    AM_NODISCARD StatusOr<ConstEvalPlan> Plan(std::span<const GraphValueDesc> inputs,
                                              std::span<const GraphValueDesc> outputs,
                                              const OpParams& params,
                                              const ConstEvalPolicy& policy) const override {
        if (inputs.size() != 2U || outputs.size() != 1U ||
            !std::holds_alternative<AddParams>(params)) {
            return Status::Unimplemented(
                    "Add constant evaluator requires two inputs and one output");
        }

        const TensorSpec& lhs = inputs[0].spec;
        const TensorSpec& rhs = inputs[1].spec;
        const TensorSpec& output = outputs[0].spec;
        if (!IsAddSupportedDType(lhs.dtype) || rhs.dtype != lhs.dtype || output.dtype != lhs.dtype) {
            return Status::Unimplemented(
                    MakeAddUnsupportedDTypeMessage("Add constant evaluator"));
        }

        auto lhs_shape = ExtractStaticShape(lhs);
        AM_RETURN_IF_ERROR(lhs_shape.status());
        auto rhs_shape = ExtractStaticShape(rhs);
        AM_RETURN_IF_ERROR(rhs_shape.status());
        auto shape = ExtractStaticShape(output);
        AM_RETURN_IF_ERROR(shape.status());

        if (auto broadcast_shape = BroadcastShapes(*lhs_shape, *rhs_shape);
            !broadcast_shape.ok() || *broadcast_shape != *shape) {
            return Status::Unimplemented(
                    "Add constant evaluator requires broadcast-compatible static shapes matching output");
        }

        auto numel = CountElements(*shape);
        AM_RETURN_IF_ERROR(numel.status());
        if (static_cast<size_t>(*numel) > policy.max_compute_elements) {
            return Status::Unimplemented(
                    "Add constant evaluator compute budget exceeded");
        }

        auto nbytes = CountBytes(output);
        AM_RETURN_IF_ERROR(nbytes.status());
        if (*nbytes > policy.max_output_bytes) {
            return Status::Unimplemented(
                    "Add constant evaluator output byte budget exceeded");
        }

        auto strides = MakeContiguousStrides(*shape);
        AM_RETURN_IF_ERROR(strides.status());

        ConstEvalPlan plan;
        plan.outputs.push_back({
                .spec = output,
                .quantization = outputs[0].quantization,
                .strides = std::move(*strides),
                .nbytes = *nbytes,
                .debug_name = "folded_" + outputs[0].name,
        });
        return plan;
    }

    // Fast path when both inputs match output shape and are contiguous;
    // otherwise uses broadcast-aware strided kernel.
    AM_NODISCARD Status Evaluate(std::span<const TensorView> inputs,
                                 std::span<MutableTensorView> outputs,
                                 const OpParams& params) const override {
        if (inputs.size() != 2U || outputs.size() != 1U ||
            !std::holds_alternative<AddParams>(params)) {
            return Status::InvalidArgument(
                    "Add constant evaluator received invalid view arity");
        }

        const auto& lhs = inputs[0];
        const auto& rhs = inputs[1];
        const auto& out = outputs[0];

        const DataType dtype = lhs.dtype();
        if (!IsAddSupportedDType(dtype) || rhs.dtype() != dtype || out.dtype() != dtype) {
            return Status::InvalidArgument(
                    MakeAddUnsupportedDTypeMessage("Add constant evaluator"));
        }

        if (auto broadcast_shape = BroadcastShapes(lhs.shape(), rhs.shape());
            !broadcast_shape.ok() || !std::ranges::equal(*broadcast_shape, out.shape())) {
            return Status::InvalidArgument(
                    "Add constant evaluator received mismatched shapes");
        }

        if (!out.is_contiguous()) {
            return Status::InvalidArgument(
                    "Add constant evaluator requires contiguous output tensor");
        }

        if (lhs.shape() == out.shape() && rhs.shape() == out.shape() && lhs.is_contiguous() && rhs.is_contiguous()) {
            return detail::EvaluateBinaryFlatByDType<AddScalarOp>(dtype, inputs, outputs, out.numel());
        }

        auto lhs_strides = BroadcastInputStrides(lhs.shape(), lhs.strides(),
                                                 out.shape());
        AM_RETURN_IF_ERROR(lhs_strides.status());
        auto rhs_strides = BroadcastInputStrides(rhs.shape(), rhs.strides(),
                                                 out.shape());
        AM_RETURN_IF_ERROR(rhs_strides.status());
        return detail::EvaluateBinaryStridedByDType<AddScalarOp>(dtype, inputs, outputs, *lhs_strides, *rhs_strides);
    }
};

}// namespace

const ConstEvaluator& detail::GetAddConstEvaluator() noexcept {
    static const AddConstEvaluator kEvaluator;
    return kEvaluator;
}

}// namespace aethermind
