#ifndef AETHERMIND_BASE_TENSOR_H
#define AETHERMIND_BASE_TENSOR_H

#include "aethermind/memory/buffer.h"
#include "data_type.h"
#include "macros.h"
#include "shape_and_stride.h"

#include <cstddef>
#include <cstdint>

namespace aethermind {

class Tensor_bk {
public:
    Tensor_bk() noexcept = default;

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

private:
    Buffer buffer_;
    size_t byte_offset_{0};
    DataType dtype_;
    ShapeAndStride_bk shape_and_strides_;
};

}// namespace aethermind


#endif// AETHERMIND_BASE_TENSOR_H
