#include "aethermind/model/graph/const_evaluator.h"
#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/op_type.h"
#include "aethermind/utils/overflow_check.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <type_traits>
#include <variant>

namespace aethermind {

// ── Shape/stride helpers (shared by const evaluator and folding pass) ──

StatusOr<std::vector<int64_t>> ExtractStaticShape(const TensorSpec& spec) {
    if (!spec.shape.IsStatic()) {
        return Status::Unimplemented("requires static tensor shape");
    }

    const auto rank = spec.shape.rank();
    if (!rank.has_value()) {
        return Status::Unimplemented("requires ranked tensor shape");
    }

    std::vector<int64_t> shape(*rank);
    for (size_t i = 0; i < *rank; ++i) {
        shape[i] = spec.shape[i].GetStaticValue();
    }
    return shape;
}

StatusOr<int64_t> CountElements(std::span<const int64_t> shape) {
    int64_t numel = 1;
    for (const int64_t dim: shape) {
        if (dim < 0) {
            return Status::InvalidArgument(
                    "static tensor dimensions must be non-negative");
        }

        if (dim != 0 && numel > std::numeric_limits<int64_t>::max() / dim) {
            return Status::ResourceExhausted(
                    "tensor element count overflow");
        }
        numel *= dim;
    }
    return numel;
}

StatusOr<size_t> CountBytes(const TensorSpec& spec) {
    auto shape = ExtractStaticShape(spec);
    AM_RETURN_IF_ERROR(shape.status());

    auto numel = CountElements(*shape);
    AM_RETURN_IF_ERROR(numel.status());

    const auto element_bytes = static_cast<size_t>(spec.dtype.nbytes());
    const auto element_count = static_cast<size_t>(*numel);
    if (element_bytes != 0U && element_count > std::numeric_limits<size_t>::max() / element_bytes) {
        return Status::ResourceExhausted("tensor byte size overflow");
    }
    return element_count * element_bytes;
}

bool SameStaticShape(const TensorSpec& lhs, const TensorSpec& rhs) {
    if (!lhs.shape.IsStatic() || !rhs.shape.IsStatic()) {
        return false;
    }
    return lhs.shape == rhs.shape;
}

std::vector<int64_t> MakeContiguousStrides(std::span<const int64_t> shape) {
    std::vector<int64_t> strides(shape.size());
    if (shape.empty()) {
        return strides;
    }

    strides.back() = 1;
    for (size_t i = shape.size() - 1U; i > 0U; --i) {
        strides[i - 1U] = strides[i] * shape[i];
    }
    return strides;
}

namespace {

bool IsFoldableAddDType(const DataType& dtype) {
    return dtype == DataType::Float32() || dtype == DataType::Double() ||
           dtype == DataType::Int(32) || dtype == DataType::Int(64);
}

template<typename T>
Status AddScalar(T lhs, T rhs, T& out) {
    if constexpr (std::is_integral_v<T>) {
        if (CheckOverflowAdd(lhs, rhs, &out)) {
            return Status::Unimplemented("Add constant evaluator integer overflow");
        }
    } else {
        out = lhs + rhs;
    }
    return Status::Ok();
}

template<typename T>
Status EvaluateAddTyped(std::span<const TensorView> inputs,
                        std::span<MutableTensorView> outputs,
                        int64_t numel) {
    const auto* lhs = inputs[0].data<T>();
    const auto* rhs = inputs[1].data<T>();
    auto* out = outputs[0].data<T>();
    for (int64_t i = 0; i < numel; ++i) {
        AM_RETURN_IF_ERROR(AddScalar(lhs[i], rhs[i], out[i]));
    }
    return Status::Ok();
}

Status EvaluateAddByDType(const DataType& dtype,
                          std::span<const TensorView> inputs,
                          std::span<MutableTensorView> outputs,
                          int64_t numel) {
    if (dtype == DataType::Float32()) {
        return EvaluateAddTyped<float>(inputs, outputs, numel);
    }

    if (dtype == DataType::Double()) {
        return EvaluateAddTyped<double>(inputs, outputs, numel);
    }

    if (dtype == DataType::Int(32)) {
        return EvaluateAddTyped<int32_t>(inputs, outputs, numel);
    }

    if (dtype == DataType::Int(64)) {
        return EvaluateAddTyped<int64_t>(inputs, outputs, numel);
    }
    return Status::InvalidArgument("Add constant evaluator received unsupported dtype");
}

class AddConstEvaluator final : public ConstEvaluator {
public:
    AM_NODISCARD StatusOr<ConstEvalPlan> Plan(std::span<const NodeOutputDesc> inputs,
                                              std::span<const NodeOutputDesc> outputs,
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
        if (!IsFoldableAddDType(lhs.dtype) || rhs.dtype != lhs.dtype || output.dtype != lhs.dtype) {
            return Status::Unimplemented(
                    "Add constant evaluator only supports float32, float64, int32, and int64 tensors");
        }

        if (!SameStaticShape(lhs, rhs) || !SameStaticShape(lhs, output)) {
            return Status::Unimplemented(
                    "Add constant evaluator requires matching static shapes");
        }

        auto shape = ExtractStaticShape(output);
        AM_RETURN_IF_ERROR(shape.status());
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

        ConstEvalPlan plan;
        plan.outputs.push_back({
                .spec = output,
                .quantization = outputs[0].quantization,
                .strides = MakeContiguousStrides(*shape),
                .nbytes = *nbytes,
                .debug_name = "folded_" + outputs[0].debug_name,
        });
        return plan;
    }

    AM_NODISCARD Status Evaluate(std::span<const TensorView> inputs,
                                 std::span<MutableTensorView> outputs,
                                 const OpParams& params) const override {
        if (inputs.size() != 2U || outputs.size() != 1U ||
            !std::holds_alternative<AddParams>(params)) {
            return Status::InvalidArgument(
                    "Add constant evaluator received invalid view arity");
        }

        const DataType dtype = inputs[0].dtype();
        if (!IsFoldableAddDType(dtype) || inputs[1].dtype() != dtype || outputs[0].dtype() != dtype) {
            return Status::InvalidArgument(
                    "Add constant evaluator received unsupported dtype");
        }

        if (inputs[0].shape() != inputs[1].shape() ||
            inputs[0].shape() != outputs[0].shape()) {
            return Status::InvalidArgument(
                    "Add constant evaluator received mismatched shapes");
        }

        // This evaluator only supports contiguous layout because the linear
        // index loop below assumes flat memory layout. Plan() only generates
        // contiguous strides, but this guard catches misuse from other callers.
        if (!inputs[0].is_contiguous() || !inputs[1].is_contiguous() ||
            !outputs[0].is_contiguous()) {
            return Status::InvalidArgument(
                    "Add constant evaluator requires contiguous tensors");
        }

        return EvaluateAddByDType(dtype, inputs, outputs, inputs[0].numel());
    }
};

const AddConstEvaluator kAddEvaluator;

struct EvaluatorEntry {
    OpType op_type;
    const ConstEvaluator* evaluator;
};

constexpr EvaluatorEntry kEvaluators[] = {
        {.op_type = OpType::kAdd, .evaluator = &kAddEvaluator},
};

}// namespace

const ConstEvaluator* FindConstEvaluator(OpType op_type) noexcept {
    auto pred = [op_type](const EvaluatorEntry& entry) {
        return entry.op_type == op_type;
    };
    const auto it = std::ranges::find_if(kEvaluators, pred);
    return it == std::end(kEvaluators) ? nullptr : it->evaluator;
}

}// namespace aethermind
