// Template-based scalar implementation for the CPU Add kernel.
//
// Included by add_scalar.cpp only (TU-local). Each template is instantiated
// for all five supported dtypes. When AVX2 support is added, a separate
// add_avx2.cpp will provide hand-vectorized specializations while this file
// remains the scalar fallback.
//
// Integer addition uses CheckOverflowAdd; overflow returns kOverflow.
// The coordinate buffer is bounded by ShapeAndStride::kMaxRank.

#pragma once

#include "add_internal.h"
#include "aethermind/utils/overflow_check.h"

#include <array>
#include <cstdint>
#include <span>
#include <type_traits>

namespace aethermind::cpu::detail {
namespace {

// Maps an output coordinate to an input element offset using broadcast
// semantics. For input axes where input_shape[axis] == 1, the coordinate is
// forced to 0 so the single element is reused. Leading axes missing from
// the input (lower input rank) are skipped via axis_offset.
inline int64_t MapCoordToOffset(std::span<const int64_t> input_shape,
                                int32_t output_rank,
                                std::span<const int64_t> input_strides,
                                const std::array<int64_t, kMaxRank>& out_coord) noexcept {
    const auto input_rank = static_cast<int32_t>(input_shape.size());
    const int32_t axis_offset = output_rank - input_rank;
    int64_t offset = 0;
    for (int32_t axis = axis_offset; axis < output_rank; ++axis) {
        const int32_t input_axis = axis - axis_offset;
        const int64_t dim = input_shape[input_axis];
        const int64_t coord = (dim == 1) ? int64_t{0} : out_coord[axis];
        offset += coord * input_strides[input_axis];
    }
    return offset;
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
Status ExecuteTypedFlat(const AddKernelArgs& args) noexcept {
    const auto* lhs_data = static_cast<const T*>(args.lhs_data);
    const auto* rhs_data = static_cast<const T*>(args.rhs_data);
    auto* output_data = static_cast<T*>(args.output_data);
    for (int64_t i = 0; i < args.numel; ++i) {
        AM_RETURN_IF_ERROR(AddScalar(lhs_data[i], rhs_data[i], output_data[i]));
    }
    return Status::Ok();
}

// Stride-aware path for arbitrary shapes/strides including non-contiguous output.
template<typename T>
Status ExecuteTypedStrided(const AddKernelArgs& args) noexcept {
    const auto* lhs_data = static_cast<const T*>(args.lhs_data);
    const auto* rhs_data = static_cast<const T*>(args.rhs_data);
    auto* output_data = static_cast<T*>(args.output_data);

    const std::span lhs_shape(args.lhs_shape.data(), args.lhs_rank);
    const std::span lhs_strides(args.lhs_strides.data(), args.lhs_rank);
    const std::span rhs_shape(args.rhs_shape.data(), args.rhs_rank);
    const std::span rhs_strides(args.rhs_strides.data(), args.rhs_rank);
    const std::span out_shape(args.output_shape.data(), args.output_rank);
    const std::span out_strides(args.output_strides.data(), args.output_rank);

    std::array<int64_t, kMaxRank> coord{};

    for (int64_t flat = 0; flat < args.numel; ++flat) {
        int64_t remaining = flat;
        for (int32_t axis = args.output_rank - 1; axis >= 0; --axis) {
            coord[axis] = remaining % out_shape[axis];
            remaining /= out_shape[axis];
        }

        const int64_t lhs_offset = MapCoordToOffset(lhs_shape, args.output_rank, lhs_strides, coord);
        const int64_t rhs_offset = MapCoordToOffset(rhs_shape, args.output_rank, rhs_strides, coord);

        int64_t out_offset = 0;
        for (int32_t axis = 0; axis < args.output_rank; ++axis) {
            out_offset += coord[axis] * out_strides[axis];
        }

        AM_RETURN_IF_ERROR(AddScalar(lhs_data[lhs_offset], rhs_data[rhs_offset], output_data[out_offset]));
    }
    return Status::Ok();
}

template<typename T>
Status ExecuteTyped(const AddKernelArgs& args) noexcept {
    if (args.is_flat) {
        return ExecuteTypedFlat<T>(args);
    }
    return ExecuteTypedStrided<T>(args);
}

}// namespace
}// namespace aethermind::cpu::detail
