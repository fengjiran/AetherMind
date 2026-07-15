// Borrowed-view validation and query helpers for TensorView and MutableTensorView.
//
// This file implements the anonymous-namespace validators (metadata, alignment,
// element presence, numel, contiguity) shared by both view types, plus the
// public constructor/query methods.  All views borrow external storage —
// they neither own nor extend the lifetime of the data or metadata they reference.

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

// Returns true when dtype/shape/strides are internally consistent.
//
// Rejects scalable-vector dtypes, zero-byte dtypes, mismatched shape/stride
// ranks, overlong ranks, and negative dimensions/offsets.  Null metadata
// pointers are allowed only when the shape size is zero (rank-0 views), so
// callers that construct rank-0 views from raw parts can pass null pointers
// for empty shape/strides arrays.
bool HasValidMetadata(const DataType& dtype,
                      const IntArrayView shape,
                      const IntArrayView strides) noexcept {
    if (dtype.IsScalableVector()) {
        return false;
    }

    if (dtype.nbytes() <= 0) {
        return false;
    }

    if (shape.size() != strides.size()) {
        return false;
    }

    if (shape.size() > static_cast<size_t>(ShapeAndStride::kMaxRank)) {
        return false;
    }

    if (shape.size() == 0) {
        // Rank-0: empty shape/strides are valid.
        // Null metadata pointers are legal when size is 0.
        return true;
    }

    // Now that we know the shape is non-empty, null metadata pointers are
    // always invalid — a non-empty shape must have addressable dimensions.
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

// Returns true when `data` satisfies the `alignment` constraint.
//
// alignment == 0 means "unspecified" and always passes.  A null data pointer
// also passes because views with zero logical elements may have no storage,
// and there is no address to align.  Non-null pointers are checked against
// the power-of-two alignment mask.
bool HasValidAlignment(const void* data, size_t alignment) noexcept {
    if (alignment == 0) {
        return true;  // unknown / unspecified
    }

    if (!std::has_single_bit(alignment)) {
        return false;
    }

    // Null data is valid regardless of alignment — there is no address to
        // check.  This covers zero-element views that have no backing storage.
    if (data == nullptr) {
        return true;
    }

    const auto addr = reinterpret_cast<std::uintptr_t>(data);
    return (addr & (alignment - 1)) == 0;
}

// Returns true when the shape describes at least one logical element.
//
// An empty shape (rank-0) holds one element.  A shape containing at least
// one zero dimension holds zero elements.  This predicate is used to decide
// whether a null data pointer is acceptable: only zero-element shapes may
// have null data when constructing a view.
bool HasAnyElement(const IntArrayView shape) noexcept {
    if (shape.empty()) {
        // Rank-0: [] has exactly one logical element.
        return true;
    }

    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == 0) {
            return false;
        }
    }
    return true;
}

// Returns the number of elements described by `shape`.
//
// Delegates to SafeMultiplyU64 so overflow is detected even for extreme
// shapes.  The result is clamped to the smaller of int64_t and size_t
// maxima; the debug-only ASSERT catches callers that build views with
// shapes that exceed the representable range.
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

// Returns the number of bytes required to store the logical elements described
// by `shape` with the given `itemsize`.
//
// Used to compute view-level byte bounds without consulting a backing buffer.
// The overflow check is debug-only; production code is expected to have
// validated this at construction time.
size_t ComputeLogicalNBytes(const IntArrayView shape, const size_t itemsize) noexcept {
    const auto n = ComputeNumel(shape);
    AM_DCHECK(n >= 0);

    size_t logical_nbytes = 0;
    const bool overflow = CheckOverflowMul(static_cast<size_t>(n), itemsize, &logical_nbytes);
    AM_DCHECK(!overflow, "TensorView logical_nbytes overflow.");
    return logical_nbytes;
}

// Returns true when `strides` describe a row-major contiguous layout for `shape`.
//
// Rank-0 and rank-1 empty shapes are trivially contiguous.  The forward
// expected-stride loop checks that non-1 dimensions have the correct stride
// and uses CheckOverflowMul to guard the stride product — a contiguous
// N-element axis should always produce a stride that fits in int64_t when
// the corresponding shape dimension fits.
bool IsContiguous(const IntArrayView shape, const IntArrayView strides) noexcept {
    if (shape.empty()) {
        return true;
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

// Constructs an immutable view borrowing data, dtype, and metadata from the
// caller.  The caller is responsible for keeping the referenced storage alive
// for the lifetime of this view.
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

// Validates metadata consistency, data alignment, and the null-data rule.
//
// A view is valid when:
//   1. dtype/shape/strides pass HasValidMetadata (non-negative, same rank,
//      within bounds, no scalable vectors, etc.).
//   2. HasValidAlignment passes (alignment is zero / a power of two matching
//      the data pointer, or the pointer is null and no address check applies).
//   3. If the shape describes at least one logical element (HasAnyElement),
//      the data pointer must not be null.  Zero-element shapes may use null.
bool TensorView::is_valid() const noexcept {
    if (!HasValidMetadata(dtype_, shape_, strides_)) {
        return false;
    }

    if (!HasValidAlignment(data_, alignment_)) {
        return false;
    }

    if (HasAnyElement(shape_) && data_ == nullptr) {
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

// Constructs a mutable view borrowing data, dtype, and metadata from the
// caller.  Lifetime responsibility remains with the caller.
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

// Mirrors TensorView::is_valid with the same three-part check.
bool MutableTensorView::is_valid() const noexcept {
    if (!HasValidMetadata(dtype_, shape_, strides_)) {
        return false;
    }

    if (!HasValidAlignment(data_, alignment_)) {
        return false;
    }

    if (HasAnyElement(shape_) && data_ == nullptr) {
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
