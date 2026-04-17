#include "aethermind/base/tensor_view.h"

#include "aethermind/base/shape_and_stride.h"
#include "aethermind/base/tensor.h"
#include "aethermind/utils/overflow_check.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace aethermind {

namespace {

bool HasValidAlignment(const void* data, size_t alignment) noexcept {
    if (alignment == 0) {
        return true;// unknown / unspecified
    }

    if (!std::has_single_bit(alignment)) {
        return false;
    }

    if (data == nullptr) {
        return true;// empty tensor 允许 data 为空时，不检查指针对齐
    }

    const auto addr = reinterpret_cast<std::uintptr_t>(data);
    return (addr & (alignment - 1)) == 0;
}

bool HasValidMetadata(const DataType& dtype,
                      const IntArrayView shape,
                      const IntArrayView strides) noexcept {
    if (dtype.nbytes() <= 0) {
        return false;
    }

    if (shape.size() != strides.size()) {
        return false;
    }

    if (shape.empty()) {
        return false;
    }

    if (shape.size() > static_cast<size_t>(ShapeAndStride::kMaxRank)) {
        return false;
    }

    if (shape.data() == nullptr || strides.data() == nullptr) {
        return false;
    }

    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] < 0 || strides[i] < 0) {
            return false;
        }
    }

    return true;
}

}// namespace

TensorView::TensorView(const void* data,
                       const DataType dtype,
                       const IntArrayView shape,
                       const IntArrayView strides,
                       const size_t alignment) noexcept
    : data_(data),
      dtype_(dtype),
      shape_(shape),
      strides_(strides),
      alignment_(alignment) {
    AM_CHECK(is_valid(), "TensorView must borrow a valid Tensor-like state.");
}

TensorView::TensorView(const Tensor& tensor) noexcept
    : TensorView(tensor.data(), tensor.dtype(), tensor.shape(), tensor.strides(), tensor.alignment()) {}

bool TensorView::is_valid() const noexcept {
    if (!HasValidMetadata(dtype_, shape_, strides_)) {
        return false;
    }

    if (!HasValidAlignment(data_, alignment_)) {
        return false;
    }

    return true;
}

int64_t TensorView::numel() const noexcept {
    AM_DCHECK(is_valid(), "numel() requires a valid TensorView.");

    uint64_t product = 1;
    const bool overflow = SafeMultiplyU64(shape_, &product);
    constexpr auto kNumelMax = std::min<uint64_t>(
            std::numeric_limits<int64_t>::max(),
            std::numeric_limits<size_t>::max());

    AM_DCHECK(!overflow && product <= kNumelMax,
              "Integer multiplication overflow when computing TensorView numel.");
    return static_cast<int64_t>(product);
}

size_t TensorView::logical_nbytes() const noexcept {
    AM_DCHECK(is_valid(), "logical_nbytes() requires a valid TensorView.");

    const auto n = numel();
    AM_DCHECK(n >= 0);

    size_t logical_nbytes = 0;
    const bool overflow = CheckOverflowMul(static_cast<size_t>(n), itemsize(), &logical_nbytes);
    AM_DCHECK(!overflow, "TensorView logical_nbytes overflow.");
    return logical_nbytes;
}

bool TensorView::is_contiguous() const noexcept {
    AM_DCHECK(is_valid(), "is_contiguous() requires a valid TensorView.");

    ShapeAndStride metadata;
    metadata.set(shape_, strides_);
    return metadata.is_contiguous();
}


}// namespace aethermind
