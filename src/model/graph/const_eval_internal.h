#ifndef AETHERMIND_MODEL_GRAPH_CONST_EVAL_INTERNAL_H
#define AETHERMIND_MODEL_GRAPH_CONST_EVAL_INTERNAL_H

#include "aethermind/base/status.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/dtypes/data_type.h"
#include "aethermind/model/graph/const_evaluator.h"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace aethermind::detail {

// ── Broadcast helpers (shared across evaluator implementations) ──
inline StatusOr<std::vector<int64_t>> BroadcastShapes(std::span<const int64_t> lhs_shape,
                                                      std::span<const int64_t> rhs_shape) {
    const size_t output_rank = std::max(lhs_shape.size(), rhs_shape.size());
    std::vector<int64_t> output_shape(output_rank, 1);
    const size_t lhs_axis_offset = output_rank - lhs_shape.size();
    const size_t rhs_axis_offset = output_rank - rhs_shape.size();
    for (size_t output_axis = 0; output_axis < output_rank; ++output_axis) {
        const int64_t lhs_dim = output_axis < lhs_axis_offset
                                        ? 1
                                        : lhs_shape[output_axis - lhs_axis_offset];
        const int64_t rhs_dim = output_axis < rhs_axis_offset
                                        ? 1
                                        : rhs_shape[output_axis - rhs_axis_offset];
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

inline StatusOr<std::vector<int64_t>> BroadcastInputStrides(std::span<const int64_t> input_shape,
                                                            std::span<const int64_t> input_strides,
                                                            std::span<const int64_t> output_shape) {
    if (input_shape.size() != input_strides.size() || input_shape.size() > output_shape.size()) {
        return Status::InvalidArgument("broadcast input metadata rank mismatch");
    }

    std::vector<int64_t> effective_strides(output_shape.size());
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
            return Status::InvalidArgument(
                    "input shape is not broadcast-compatible with output shape");
        }
    }
    return effective_strides;
}

// ── Shared binary kernel templates ──
template<typename Op, typename T>
concept BinaryScalarOp = requires(T lhs, T rhs, T& out) {
    { Op::template Apply<T>(lhs, rhs, out) } -> std::same_as<Status>;
};

template<typename Op, typename T>
    requires BinaryScalarOp<Op, T>
Status EvaluateBinaryFlatTyped(std::span<const TensorView> inputs,
                               std::span<MutableTensorView> outputs,
                               int64_t numel) {
    const auto* lhs = inputs[0].data<T>();
    const auto* rhs = inputs[1].data<T>();
    auto* out = outputs[0].data<T>();
    for (int64_t i = 0; i < numel; ++i) {
        AM_RETURN_IF_ERROR(Op::Apply(lhs[i], rhs[i], out[i]));
    }
    return Status::Ok();
}

template<typename Op>
Status EvaluateBinaryFlatByDType(const DataType& dtype,
                                 std::span<const TensorView> inputs,
                                 std::span<MutableTensorView> outputs,
                                 int64_t numel) {
    if (dtype == DataType::Float32()) {
        return EvaluateBinaryFlatTyped<Op, float>(inputs, outputs, numel);
    }

    if (dtype == DataType::Double()) {
        return EvaluateBinaryFlatTyped<Op, double>(inputs, outputs, numel);
    }

    if (dtype == DataType::BFloat(16)) {
        return EvaluateBinaryFlatTyped<Op, BFloat16>(inputs, outputs, numel);
    }

    if (dtype == DataType::Int(32)) {
        return EvaluateBinaryFlatTyped<Op, int32_t>(inputs, outputs, numel);
    }

    if (dtype == DataType::Int(64)) {
        return EvaluateBinaryFlatTyped<Op, int64_t>(inputs, outputs, numel);
    }
    return Status::InvalidArgument("binary constant evaluator received unsupported dtype");
}

template<typename Op, typename T>
    requires BinaryScalarOp<Op, T>
Status EvaluateBinaryStridedKernel(std::span<const TensorView> inputs,
                                   std::span<MutableTensorView> outputs,
                                   std::span<const int64_t> lhs_strides,
                                   std::span<const int64_t> rhs_strides) {
    const auto* lhs = inputs[0].data<T>();
    const auto* rhs = inputs[1].data<T>();
    auto* out = outputs[0].data<T>();
    const auto shape = outputs[0].shape();
    std::vector<int64_t> coordinates(shape.size());
    int64_t lhs_offset = 0;
    int64_t rhs_offset = 0;
    for (int64_t output_index = 0; output_index < outputs[0].numel(); ++output_index) {
        AM_RETURN_IF_ERROR(Op::Apply(lhs[lhs_offset], rhs[rhs_offset], out[output_index]));
        for (size_t remaining = shape.size(); remaining > 0U; --remaining) {
            const size_t axis = remaining - 1U;
            ++coordinates[axis];
            lhs_offset += lhs_strides[axis];
            rhs_offset += rhs_strides[axis];
            if (coordinates[axis] < shape[axis]) {
                break;
            }
            coordinates[axis] = 0;
            lhs_offset -= lhs_strides[axis] * shape[axis];
            rhs_offset -= rhs_strides[axis] * shape[axis];
        }
    }
    return Status::Ok();
}

template<typename Op>
Status EvaluateBinaryStridedByDType(const DataType& dtype,
                                    std::span<const TensorView> inputs,
                                    std::span<MutableTensorView> outputs,
                                    std::span<const int64_t> lhs_strides,
                                    std::span<const int64_t> rhs_strides) {
    if (dtype == DataType::Float32()) {
        return EvaluateBinaryStridedKernel<Op, float>(inputs, outputs, lhs_strides, rhs_strides);
    }

    if (dtype == DataType::Double()) {
        return EvaluateBinaryStridedKernel<Op, double>(inputs, outputs, lhs_strides, rhs_strides);
    }

    if (dtype == DataType::BFloat(16)) {
        return EvaluateBinaryStridedKernel<Op, BFloat16>(inputs, outputs, lhs_strides, rhs_strides);
    }

    if (dtype == DataType::Int(32)) {
        return EvaluateBinaryStridedKernel<Op, int32_t>(inputs, outputs, lhs_strides, rhs_strides);
    }

    if (dtype == DataType::Int(64)) {
        return EvaluateBinaryStridedKernel<Op, int64_t>(inputs, outputs, lhs_strides, rhs_strides);
    }
    return Status::InvalidArgument("binary constant evaluator received unsupported dtype");
}

// ── Shared unary kernel templates ──
template<typename Op, typename T>
concept UnaryScalarOp = requires(T input, T& output) {
    { Op::template Apply<T>(input, output) } -> std::same_as<Status>;
};

template<typename Op, typename T>
    requires UnaryScalarOp<Op, T>
Status EvaluateUnaryFlatTyped(std::span<const TensorView> inputs,
                              std::span<MutableTensorView> outputs,
                              int64_t numel) {
    const auto* in = inputs[0].data<T>();
    auto* out = outputs[0].data<T>();
    for (int64_t i = 0; i < numel; ++i) {
        AM_RETURN_IF_ERROR(Op::Apply(in[i], out[i]));
    }
    return Status::Ok();
}

template<typename Op, typename T>
    requires UnaryScalarOp<Op, T>
Status EvaluateUnaryStridedKernel(std::span<const TensorView> inputs,
                                  std::span<MutableTensorView> outputs,
                                  std::span<const int64_t> input_strides) {
    const auto* in = inputs[0].data<T>();
    auto* out = outputs[0].data<T>();
    const auto shape = outputs[0].shape();
    std::vector<int64_t> coordinates(shape.size());
    int64_t input_offset = 0;
    for (int64_t output_index = 0; output_index < outputs[0].numel(); ++output_index) {
        AM_RETURN_IF_ERROR(Op::Apply(in[input_offset], out[output_index]));
        for (size_t remaining = shape.size(); remaining > 0U; --remaining) {
            const size_t axis = remaining - 1U;
            ++coordinates[axis];
            input_offset += input_strides[axis];
            if (coordinates[axis] < shape[axis]) {
                break;
            }
            coordinates[axis] = 0;
            input_offset -= input_strides[axis] * shape[axis];
        }
    }
    return Status::Ok();
}

// ── Accessor declarations ──
const ConstEvaluator& GetAddConstEvaluator() noexcept;
const ConstEvaluator& GetMulConstEvaluator() noexcept;
const ConstEvaluator& GetSiluConstEvaluator() noexcept;

}// namespace aethermind::detail

#endif
