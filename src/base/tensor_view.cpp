#include "aethermind/base/tensor_view.h"

#include "aethermind/base/shape_and_stride.h"
#include "aethermind/base/tensor.h"
#include "aethermind/utils/overflow_check.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace aethermind {

namespace {

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

int64_t ComputeNumel(const IntArrayView shape) noexcept {
    uint64_t product = 1;
    const bool overflow = SafeMultiplyU64(shape, &product);
    constexpr auto kNumelMax = std::min<uint64_t>(
            std::numeric_limits<int64_t>::max(),
            std::numeric_limits<size_t>::max());

    AM_DCHECK(!overflow && product <= kNumelMax,
              "Integer multiplication overflow when computing TensorView numel.");
    return static_cast<int64_t>(product);
}

size_t ComputeLogicalNBytes(const IntArrayView shape, const size_t itemsize) noexcept {
    const auto n = ComputeNumel(shape);
    AM_DCHECK(n >= 0);

    size_t logical_nbytes = 0;
    const bool overflow = CheckOverflowMul(static_cast<size_t>(n), itemsize, &logical_nbytes);
    AM_DCHECK(!overflow, "TensorView logical_nbytes overflow.");
    return logical_nbytes;
}

bool IsContiguous(const IntArrayView shape, const IntArrayView strides) noexcept {
    if (shape.empty()) {
        return false;
    }

    if (shape.size() != strides.size()) {
        return false;
    }

    int64_t expected_stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        if (shape[i] == 1) {
            continue;
        }

        if (strides[i] != expected_stride) {
            return false;
        }

        int64_t next_expected_stride = 0;
        AM_CHECK(!CheckOverflowMul(expected_stride, shape[i], &next_expected_stride),
                 "Stride validation overflow");
        expected_stride = next_expected_stride;
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
    return ComputeNumel(shape_);
}

size_t TensorView::logical_nbytes() const noexcept {
    AM_DCHECK(is_valid(), "logical_nbytes() requires a valid TensorView.");
    return ComputeLogicalNBytes(shape_, itemsize());
}

bool TensorView::is_contiguous() const noexcept {
    AM_DCHECK(is_valid(), "is_contiguous() requires a valid TensorView.");
    return IsContiguous(shape_, strides_);
}

MutableTensorView::MutableTensorView(void* data,
                                     const DataType dtype,
                                     const IntArrayView shape,
                                     const IntArrayView strides,
                                     const size_t alignment) noexcept
    : data_(data),
      dtype_(dtype),
      shape_(shape),
      strides_(strides),
      alignment_(alignment) {
    AM_CHECK(is_valid(), "MutableTensorView must borrow a valid Tensor-like state.");
}

MutableTensorView::MutableTensorView(Tensor& tensor) noexcept
    : MutableTensorView(tensor.mutable_data(),
                        tensor.dtype(),
                        tensor.shape(),
                        tensor.strides(),
                        tensor.alignment()) {}

bool MutableTensorView::is_valid() const noexcept {
    if (!HasValidMetadata(dtype_, shape_, strides_)) {
        return false;
    }

    if (!HasValidAlignment(data_, alignment_)) {
        return false;
    }

    return true;
}

int64_t MutableTensorView::numel() const noexcept {
    AM_DCHECK(is_valid(), "numel() requires a valid MutableTensorView.");
    return ComputeNumel(shape_);
}

size_t MutableTensorView::logical_nbytes() const noexcept {
    AM_DCHECK(is_valid(), "logical_nbytes() requires a valid MutableTensorView.");
    return ComputeLogicalNBytes(shape_, itemsize());
}

bool MutableTensorView::is_contiguous() const noexcept {
    AM_DCHECK(is_valid(), "is_contiguous() requires a valid MutableTensorView.");
    return IsContiguous(shape_, strides_);
}


}// namespace aethermind
