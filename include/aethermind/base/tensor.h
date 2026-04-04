#ifndef AETHERMIND_BASE_TENSOR_H
#define AETHERMIND_BASE_TENSOR_H

#include "aethermind/memory/buffer.h"
#include "container/array_view.h"
#include "data_type.h"
#include "macros.h"
#include "shape_and_stride.h"

#include <cstddef>
#include <cstdint>
#include <utility>

namespace aethermind {

class Tensor {
public:
    Tensor() noexcept = default;

    Tensor(Buffer buffer, size_t byte_offset, const DataType& dtype, const ShapeAndStride_bk& shape_and_strides)
        : buffer_(std::move(buffer)), byte_offset_(byte_offset), dtype_(dtype), shape_and_strides_(shape_and_strides) {
        validate();
    }

    Tensor(Buffer buffer, size_t byte_offset, const DataType& dtype, IntArrayView shape, IntArrayView strides)
        : buffer_(std::move(buffer)), byte_offset_(byte_offset), dtype_(dtype), shape_and_strides_(shape, strides) {
        validate();
    }


    AM_NODISCARD bool is_initialized() const noexcept {
        return buffer_.is_initialized();
    }

    explicit operator bool() const noexcept {
        return is_initialized();
    }

    AM_NODISCARD bool is_contiguous() const noexcept {
        return shape_and_strides_.is_contiguous();
    }

    AM_NODISCARD size_t byte_offset() const noexcept {
        return byte_offset_;
    }

    AM_NODISCARD DataType dtype() const noexcept {
        return dtype_;
    }

    AM_NODISCARD int32_t rank() const noexcept {
        return shape_and_strides_.size();
    }

    AM_NODISCARD const Buffer& buffer() const noexcept {
        return buffer_;
    }

    AM_NODISCARD IntArrayView shape() const noexcept {
        return shape_and_strides_.shape();
    }

    AM_NODISCARD IntArrayView strides() const noexcept {
        return shape_and_strides_.strides();
    }

    AM_NODISCARD int64_t shape(int32_t i) const noexcept {
        return shape_and_strides_.shape(i);
    }

    AM_NODISCARD int64_t stride(int32_t i) const noexcept {
        return shape_and_strides_.stride(i);
    }

    AM_NODISCARD int64_t numel() const noexcept {
        return shape_and_strides_.numel();
    }

    AM_NODISCARD size_t itemsize() const noexcept {
        return dtype_.nbytes();
    }

private:
    Buffer buffer_;
    size_t byte_offset_{0};
    DataType dtype_;
    ShapeAndStride_bk shape_and_strides_;

    void validate();
};

}// namespace aethermind


#endif// AETHERMIND_BASE_TENSOR_H
