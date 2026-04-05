#include "aethermind/base/tensor.h"
#include "aethermind/base/shape_and_stride.h"
#include "utils/logging.h"
#include <cstddef>

namespace aethermind {

void Tensor::validate() {
    AM_CHECK(is_initialized(), "Tensor buffer is not initialized.");
    AM_CHECK(byte_offset_ <= buffer_.nbytes(), "Tensor byte_offset out of buffer range.");
    const auto r = shape_and_strides_.size();
    AM_CHECK(r >= 0 && r <= ShapeAndStride::kMaxRank, "Invalid tensor rank.");
    AM_CHECK(itemsize() > 0, "Tensor dtype itemsize must be positive.");

    for (int i = 0; i < r; ++i) {
        AM_CHECK(shape_and_strides_.shape(i) >= 0, "Tensor shape must be non-negative.");
        AM_CHECK(shape_and_strides_.stride(i) >= 0, "Tensor stride requires non-negative.");
    }

    if (numel() == 0) {
        return;
    }

    const auto max_element_offset = shape_and_strides_.max_element_offset();
    AM_CHECK(max_element_offset >= 0, "Tensor max element offset must be non-negative");

    const auto required_span_bytes = static_cast<size_t>(max_element_offset + 1) * itemsize();
    AM_CHECK(byte_offset_ + required_span_bytes <= buffer_.nbytes(), "Tensor logical span exceeds visible buffer range.");
}

}// namespace aethermind
