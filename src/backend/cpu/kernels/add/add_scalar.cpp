// Scalar execution engine for the CPU Add kernel.
//
// Implements AddKernel_CPU_Scalar with TU-local dtype dispatch.
// - Flat path: when lhs/rhs/output share the same shape and are all
//   contiguous, a single linear pass indexes all three buffers directly.
// - Stride-aware path: for general broadcasts / non-contiguous output,
//   the output is iterated using output.strides() to compute the output
//   offset, while lhs/rhs offsets use broadcast-aware MapCoordToOffset.
//
// Integer addition uses CheckOverflowAdd; overflow returns kOverflow.
// No AVX2, no threading, no runtime allocation. The coordinate buffer is
// a fixed-size std::array on the stack, bounded by ShapeAndStride::kMaxRank.

#include "add_internal.h"
#include "aethermind/base/shape_and_stride.h"
#include "aethermind/utils/overflow_check.h"

#include <array>
#include <cstdint>
#include <span>
#include <type_traits>

namespace aethermind::cpu::detail {
namespace {

constexpr uint32_t kMaxRank = ShapeAndStride::kMaxRank;

// Maps an output coordinate to an input element offset using broadcast
// semantics. For input axes where input_shape[axis] == 1, the coordinate is
// forced to 0 so the single element is reused. Leading axes missing from
// the input (lower input rank) are skipped via axis_offset.
int64_t MapCoordToOffset(std::span<const int64_t> input_shape,
                         int32_t output_rank,
                         std::span<const int64_t> input_strides,
                         const std::array<int64_t, kMaxRank>& out_coord) noexcept {
    const auto input_rank = static_cast<int32_t>(input_shape.size());
    const int32_t axis_offset = output_rank - input_rank;
    int64_t offset = 0;
    for (int32_t axis = axis_offset; axis < output_rank; ++axis) {
        const int32_t input_axis = axis - axis_offset;
        const int64_t dim = input_shape[input_axis];
        // Broadcast: a size-1 axis always reads element 0 regardless of the
        // output coordinate.
        const int64_t coord = (dim == 1) ? int64_t{0} : out_coord[axis];
        offset += coord * input_strides[input_axis];
    }
    return offset;
}

// Returns true when the flat path is applicable: lhs/rhs/output must share
// the identical shape (so element-wise indexing matches) and all three must
// be contiguous (so flat indexing equals stride-based indexing).
bool CanUseFlatPath(const TensorView& lhs,
                    const TensorView& rhs,
                    const MutableTensorView& output) noexcept {
    return lhs.is_contiguous() && rhs.is_contiguous() && output.is_contiguous() &&
           lhs.shape() == output.shape() && rhs.shape() == output.shape();
}

template<typename T>
Status AddScalar(T lhs, T rhs, T& output) noexcept {
    if constexpr (std::is_integral_v<T>) {
        if (CheckOverflowAdd(lhs, rhs, &output)) {
            return Status::Overflow("CpuAddKernel integer overflow");
        }
    } else {
        output = lhs + rhs;
    }
    return Status::Ok();
}

// Flat contiguous same-shape path for a typed buffer pair.
template<typename T>
Status ExecuteTypedFlat(const T* lhs_data,
                        const T* rhs_data,
                        T* output_data,
                        int64_t numel) noexcept {
    for (int64_t i = 0; i < numel; ++i) {
        AM_RETURN_IF_ERROR(AddScalar(lhs_data[i], rhs_data[i], output_data[i]));
    }
    return Status::Ok();
}

// Stride-aware path for arbitrary shapes/strides (including non-contiguous output).
template<typename T>
Status ExecuteTypedStrided(const TensorView& lhs,
                           const TensorView& rhs,
                           const MutableTensorView& output,
                           int64_t numel) noexcept {
    const auto* const lhs_data = lhs.data<T>();
    const auto* const rhs_data = rhs.data<T>();
    auto* const output_data = output.data<T>();

    const auto lhs_shape = lhs.shape();
    const auto lhs_strides = lhs.strides();
    const auto rhs_shape = rhs.shape();
    const auto rhs_strides = rhs.strides();
    const auto out_shape = output.shape();
    const auto out_strides = output.strides();
    const int32_t output_rank = output.rank();

    // Stack-allocated coordinate buffer, bounded by kMaxRank. The entry
    // function has already rejected outputs whose rank exceeds kMaxRank.
    std::array<int64_t, kMaxRank> coord{};

    for (int64_t flat = 0; flat < numel; ++flat) {
        // Decode the flat index into a multi-axis coordinate by successive
        // division along the output shape (rightmost axis is fastest).
        int64_t remaining = flat;
        for (int32_t axis = output_rank - 1; axis >= 0; --axis) {
            coord[axis] = remaining % out_shape[axis];
            remaining /= out_shape[axis];
        }

        const int64_t lhs_offset = MapCoordToOffset(lhs_shape, output_rank, lhs_strides, coord);
        const int64_t rhs_offset = MapCoordToOffset(rhs_shape, output_rank, rhs_strides, coord);

        // Compute output offset from strides (may be non-contiguous).
        int64_t out_offset = 0;
        for (int32_t axis = 0; axis < output_rank; ++axis) {
            out_offset += coord[axis] * out_strides[axis];
        }

        AM_RETURN_IF_ERROR(AddScalar(lhs_data[lhs_offset], rhs_data[rhs_offset], output_data[out_offset]));
    }
    return Status::Ok();
}

template<typename T>
Status ExecuteTyped(const TensorView& lhs,
                    const TensorView& rhs,
                    const MutableTensorView& output,
                    int64_t numel) noexcept {
    if (CanUseFlatPath(lhs, rhs, output)) {
        return ExecuteTypedFlat<T>(lhs.data<T>(), rhs.data<T>(), output.data<T>(), numel);
    }
    return ExecuteTypedStrided<T>(lhs, rhs, output, numel);
}

}// namespace

Status AddKernel_CPU_Scalar(const TensorView& lhs,
                            const TensorView& rhs,
                            const MutableTensorView& output,
                            int64_t numel) noexcept {
    const DataType dtype = lhs.dtype();

    if (dtype == DataType::Float32()) {
        return ExecuteTyped<float>(lhs, rhs, output, numel);
    }
    if (dtype == DataType::Double()) {
        return ExecuteTyped<double>(lhs, rhs, output, numel);
    }
    if (dtype == DataType::BFloat(16)) {
        return ExecuteTyped<BFloat16>(lhs, rhs, output, numel);
    }
    if (dtype == DataType::Int(32)) {
        return ExecuteTyped<int32_t>(lhs, rhs, output, numel);
    }
    return ExecuteTyped<int64_t>(lhs, rhs, output, numel);
}

}// namespace aethermind::cpu::detail
