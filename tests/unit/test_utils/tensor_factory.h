#ifndef AETHERMIND_TEST_UTILS_TENSOR_FACTORY_H
#define AETHERMIND_TEST_UTILS_TENSOR_FACTORY_H

#include "aethermind/base/tensor.h"
#include "aethermind/memory/buffer.h"
#include "aethermind/utils/overflow_check.h"
#include "container/array_view.h"
#include "shape_and_stride.h"
#include "utils/logging.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace aethermind::test_utils {

namespace detail {

inline void free_aligned_buffer(void*, void* ptr) noexcept {
    std::free(ptr);
}

inline Buffer make_buffer(size_t nbytes, size_t alignment = 64) {
    AM_CHECK(alignment != 0, "MakeBuffer requires non-zero alignment.");
    AM_CHECK((alignment & (alignment - 1)) == 0, "MakeBuffer requires power-of-two alignment.");

    const size_t bytes_to_allocate = nbytes == 0 ? 1 : nbytes;
    void* ptr = nullptr;
    const int rc = posix_memalign(&ptr, alignment, bytes_to_allocate);
    AM_CHECK(rc == 0 && ptr != nullptr, "posix_memalign failed in MakeBuffer.");

    return {nbytes, MemoryHandle(ptr, nullptr, &free_aligned_buffer, Device::CPU(), alignment)};
}

inline size_t compute_required_bytes(IntArrayView shape,
                                      IntArrayView strides,
                                      size_t itemsize,
                                      size_t byte_offset = 0) {
    ShapeAndStride shape_and_stride(shape, strides);

    size_t required = byte_offset;
    if (shape_and_stride.numel() == 0) {
        return required;
    }

    const int64_t max_offset = shape_and_stride.max_element_offset();
    AM_CHECK(max_offset >= 0, "ComputeRequiredBytes expects non-negative max offset.");

    const auto touched_elems = static_cast<size_t>(max_offset) + 1;
    size_t span_bytes = 0;
    AM_CHECK(!mul_overflow(touched_elems, itemsize, &span_bytes), "ComputeRequiredBytes span overflow.");
    AM_CHECK(!add_overflow(required, span_bytes, &required), "ComputeRequiredBytes total overflow.");
    return required;
}

}// namespace detail

inline Tensor MakeEmptyTensor(DataType dtype = DataType::Float32()) {
    static constexpr std::array<int64_t, 1> kShape{0};
    static constexpr std::array<int64_t, 1> kStride{1};

    return Tensor(detail::make_buffer(0),
                  0,
                  dtype,
                  IntArrayView{kShape.data(), kShape.size()},
                  IntArrayView{kStride.data(), kStride.size()});
}

inline Tensor MakeContiguousTensor(IntArrayView shape,
                                    DataType dtype = DataType::Float32(),
                                    size_t byte_offset = 0) {
    ShapeAndStride shape_and_stride;
    shape_and_stride.set_contiguous(shape);

    const size_t required_bytes = detail::compute_required_bytes(
            shape_and_stride.shape(), shape_and_stride.strides(), static_cast<size_t>(dtype.nbytes()), byte_offset);

    return Tensor(detail::make_buffer(required_bytes), byte_offset, dtype, shape_and_stride);
}

inline Tensor MakeTensor(IntArrayView shape,
                          IntArrayView strides,
                          DataType dtype = DataType::Float32(),
                          size_t byte_offset = 0) {
    const size_t required_bytes = detail::compute_required_bytes(
            shape, strides, static_cast<size_t>(dtype.nbytes()), byte_offset);

    return Tensor(detail::make_buffer(required_bytes), byte_offset, dtype, shape, strides);
}

}// namespace aethermind::test_utils

#endif