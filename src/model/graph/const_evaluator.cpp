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
    return dtype == DataType::Float32() ||
           dtype == DataType::Double() ||
           dtype == DataType::BFloat(16) ||
           dtype == DataType::Int(32) ||
           dtype == DataType::Int(64);
}

bool ShapesEqual(std::span<const int64_t> lhs, std::span<const int64_t> rhs) {
    return std::ranges::equal(lhs, rhs);
}

StatusOr<std::vector<int64_t>> BroadcastShapes(std::span<const int64_t> lhs,
                                               std::span<const int64_t> rhs) {
    const size_t output_rank = std::max(lhs.size(), rhs.size());
    std::vector<int64_t> output_shape(output_rank, 1);
    const size_t lhs_axis_offset = output_rank - lhs.size();
    const size_t rhs_axis_offset = output_rank - rhs.size();
    for (size_t output_axis = 0; output_axis < output_rank; ++output_axis) {
        const int64_t lhs_dim = output_axis < lhs_axis_offset ? 1 : lhs[output_axis - lhs_axis_offset];
        const int64_t rhs_dim = output_axis < rhs_axis_offset ? 1 : rhs[output_axis - rhs_axis_offset];
        if (lhs_dim < 0 || rhs_dim < 0) {
            return Status::InvalidArgument("broadcast dimensions must be non-negative");
        }

        if (lhs_dim == 1) {
            output_shape[output_axis] = rhs_dim;
        } else if (rhs_dim == 1 || lhs_dim == rhs_dim) {
            output_shape[output_axis] = lhs_dim;
        } else {
            return Status::InvalidArgument("broadcast dimensions are incompatible");
        }
    }
    return output_shape;
}

StatusOr<std::vector<int64_t>> BroadcastInputStrides(std::span<const int64_t> input_shape,
                                                     std::span<const int64_t> input_strides,
                                                     std::span<const int64_t> output_shape) {
    if (input_shape.size() != input_strides.size() || input_shape.size() > output_shape.size()) {
        return Status::InvalidArgument("broadcast input metadata rank mismatch");
    }

    std::vector<int64_t> effective_strides(output_shape.size(), 0);
    const size_t axis_offset = output_shape.size() - input_shape.size();
    for (size_t output_axis = axis_offset; output_axis < output_shape.size(); ++output_axis) {
        const size_t input_axis = output_axis - axis_offset;
        const int64_t input_dim = input_shape[input_axis];
        const int64_t output_dim = output_shape[output_axis];
        if (input_dim == output_dim) {
            effective_strides[output_axis] = input_strides[input_axis];
        } else if (input_dim == 1) {
            effective_strides[output_axis] = 0;
        } else {
            return Status::InvalidArgument("input shape is not broadcast-compatible with output shape");
        }
    }
    return effective_strides;
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

    if (dtype == DataType::BFloat(16)) {
        return EvaluateAddTyped<BFloat16>(inputs, outputs, numel);
    }

    if (dtype == DataType::Int(32)) {
        return EvaluateAddTyped<int32_t>(inputs, outputs, numel);
    }

    if (dtype == DataType::Int(64)) {
        return EvaluateAddTyped<int64_t>(inputs, outputs, numel);
    }
    return Status::InvalidArgument("Add constant evaluator received unsupported dtype");
}

template<typename T>
Status EvaluateAddBroadcastTyped(std::span<const TensorView> inputs,
                                 std::span<MutableTensorView> outputs) {
    auto lhs_effective_strides = BroadcastInputStrides(inputs[0].shape(),
                                                       inputs[0].strides(),
                                                       outputs[0].shape());
    AM_RETURN_IF_ERROR(lhs_effective_strides.status());
    auto rhs_effective_strides = BroadcastInputStrides(inputs[1].shape(),
                                                       inputs[1].strides(),
                                                       outputs[0].shape());
    AM_RETURN_IF_ERROR(rhs_effective_strides.status());

    const auto* lhs = inputs[0].data<T>();
    const auto* rhs = inputs[1].data<T>();
    auto* out = outputs[0].data<T>();
    const auto output_shape = outputs[0].shape();
    std::vector<int64_t> coordinates(output_shape.size(), 0);
    int64_t lhs_offset = 0;
    int64_t rhs_offset = 0;
    for (int64_t output_index = 0; output_index < outputs[0].numel(); ++output_index) {
        AM_RETURN_IF_ERROR(AddScalar(lhs[lhs_offset], rhs[rhs_offset], out[output_index]));
        for (size_t remaining = output_shape.size(); remaining > 0U; --remaining) {
            const size_t axis = remaining - 1U;
            ++coordinates[axis];
            lhs_offset += (*lhs_effective_strides)[axis];
            rhs_offset += (*rhs_effective_strides)[axis];
            if (coordinates[axis] < output_shape[axis]) {
                break;
            }

            coordinates[axis] = 0;
            lhs_offset -= (*lhs_effective_strides)[axis] * output_shape[axis];
            rhs_offset -= (*rhs_effective_strides)[axis] * output_shape[axis];
        }
    }
    return Status::Ok();
}

Status EvaluateAddBroadcastByDType(const DataType& dtype,
                                   std::span<const TensorView> inputs,
                                   std::span<MutableTensorView> outputs) {
    if (dtype == DataType::Float32()) {
        return EvaluateAddBroadcastTyped<float>(inputs, outputs);
    }

    if (dtype == DataType::Double()) {
        return EvaluateAddBroadcastTyped<double>(inputs, outputs);
    }

    if (dtype == DataType::BFloat(16)) {
        return EvaluateAddBroadcastTyped<BFloat16>(inputs, outputs);
    }

    if (dtype == DataType::Int(32)) {
        return EvaluateAddBroadcastTyped<int32_t>(inputs, outputs);
    }

    if (dtype == DataType::Int(64)) {
        return EvaluateAddBroadcastTyped<int64_t>(inputs, outputs);
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
                    "Add constant evaluator only supports float32, float64, bfloat16, int32, and int64 tensors");
        }

        auto lhs_shape = ExtractStaticShape(lhs);
        AM_RETURN_IF_ERROR(lhs_shape.status());
        auto rhs_shape = ExtractStaticShape(rhs);
        AM_RETURN_IF_ERROR(rhs_shape.status());
        auto shape = ExtractStaticShape(output);
        AM_RETURN_IF_ERROR(shape.status());
        if (lhs_shape->empty() || rhs_shape->empty() || shape->empty()) {
            return Status::Unimplemented(
                    "Add constant evaluator requires non-scalar tensor shapes");
        }

        auto broadcast_shape = BroadcastShapes(*lhs_shape, *rhs_shape);
        if (!broadcast_shape.ok() || *broadcast_shape != *shape) {
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

        auto broadcast_shape = BroadcastShapes(inputs[0].shape(), inputs[1].shape());
        if (!broadcast_shape.ok() || !ShapesEqual(*broadcast_shape, outputs[0].shape())) {
            return Status::InvalidArgument(
                    "Add constant evaluator received mismatched shapes");
        }

        if (!outputs[0].is_contiguous()) {
            return Status::InvalidArgument(
                    "Add constant evaluator requires contiguous output tensor");
        }

        if (inputs[0].shape() == outputs[0].shape() &&
            inputs[1].shape() == outputs[0].shape() &&
            inputs[0].is_contiguous() && inputs[1].is_contiguous()) {
            return EvaluateAddByDType(dtype, inputs, outputs, outputs[0].numel());
        }

        return EvaluateAddBroadcastByDType(dtype, inputs, outputs);
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
