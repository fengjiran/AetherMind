#ifndef AETHERMIND_MODEL_GRAPH_OPTIMIZATION_CONST_EVAL_INTERNAL_H
#define AETHERMIND_MODEL_GRAPH_OPTIMIZATION_CONST_EVAL_INTERNAL_H

// Internal header shared by all const_eval_*.cpp TU-local implementations.
// Contains scalar-op concepts, typed flat/strided kernels,
// and accessor function declarations for the per-evaluator .cpp files.
//
// Broadcast helpers have been extracted to aethermind/shape_inference/broadcast.h.

#include "aethermind/base/status.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/dtypes/data_type.h"
#include "aethermind/model/graph/optimization/const_evaluator.h"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace aethermind::detail {

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

// Strided iteration: walks the output in linear order and computes input
// offsets via strides. An inner carry loop advances per-axis coordinates
// and resets on dimension boundaries, enabling non-contiguous access without
// materializing the full coordinate space.
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

// Same carry-loop strided iteration as the binary variant, but for
// single-input unary operations (e.g. Silu).
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
const ConstEvaluator& GetSiluMulConstEvaluator() noexcept;

}// namespace aethermind::detail

#endif
