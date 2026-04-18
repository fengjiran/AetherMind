/// \file
/// Non-owning borrowed tensor metadata and data view.
///
/// TensorView is the hot-path counterpart to Tensor:
/// - Tensor owns Buffer + ShapeAndStride
/// - TensorView borrows data pointer + shape/stride metadata
///
/// Design constraints:
/// - Strict borrowed view: does not own storage or metadata
/// - Shape and strides are expressed in elements, not bytes
/// - No metadata-transforming helpers (slice/narrow/contiguous construction)
/// - Lifetime must be managed by the caller

#ifndef AETHERMIND_TENSOR_VIEW_H
#define AETHERMIND_TENSOR_VIEW_H

#include "container/array_view.h"
#include "data_type.h"
#include "macros.h"

#include <cstddef>
#include <cstdint>

namespace aethermind {

class Tensor;

/// Non-owning immutable tensor view.
///
/// Lifetime:
/// - `data_` must remain valid for the lifetime of this view.
/// - `shape_` and `strides_` must refer to metadata owned elsewhere and remain
///   valid for the lifetime of this view.
/// - This type is intended for stack-bound hot-path use. It is not generally
///   safe to store unless the caller can guarantee the borrowed lifetimes.
///
/// Semantics:
/// - Strides are in elements, matching Tensor / ShapeAndStride semantics.
/// - `alignment_ == 0` means alignment is unknown or unspecified.
/// - `valid()` is intended to mirror a usable borrowed Tensor-like state rather
///   than act as a loose metadata envelope.
class TensorView {
public:
    TensorView() noexcept = default;

    /// Construct from borrowed raw parts.
    ///
    /// \pre `shape.size() == strides.size()`
    /// \pre Borrowed data and metadata must outlive this view.
    TensorView(const void* data,
               DataType dtype,
               IntArrayView shape,
               IntArrayView strides,
               size_t alignment = 0) noexcept;

    /// Construct by borrowing metadata and data from an existing Tensor.
    explicit TensorView(const Tensor& tensor) noexcept;

    AM_NODISCARD bool is_valid() const noexcept;

    explicit operator bool() const noexcept {
        return is_valid();
    }

    AM_NODISCARD const void* data() const noexcept {
        return data_;
    }

    template<typename T>
    AM_NODISCARD const T* data() const noexcept {
        AM_DCHECK(DataType::Make<T>() == dtype_);
        return static_cast<const T*>(data_);
    }

    AM_NODISCARD DataType dtype() const noexcept {
        return dtype_;
    }

    AM_NODISCARD IntArrayView shape() const noexcept {
        return shape_;
    }

    AM_NODISCARD IntArrayView strides() const noexcept {
        return strides_;
    }

    AM_NODISCARD size_t alignment() const noexcept {
        return alignment_;
    }

    AM_NODISCARD int32_t rank() const noexcept {
        return static_cast<int32_t>(shape_.size());
    }

    AM_NODISCARD int64_t dim(int32_t i) const noexcept {
        AM_DCHECK(i >= 0 && i < rank());
        return shape_[i];
    }

    AM_NODISCARD int64_t stride(int32_t i) const noexcept {
        AM_DCHECK(i >= 0 && i < rank());
        return strides_[i];
    }

    AM_NODISCARD int64_t numel() const noexcept;

    AM_NODISCARD size_t itemsize() const noexcept {
        return static_cast<size_t>(dtype_.nbytes());
    }

    AM_NODISCARD size_t logical_nbytes() const noexcept;

    AM_NODISCARD bool is_contiguous() const noexcept;

private:
    const void* data_ = nullptr;
    DataType dtype_{};
    IntArrayView shape_{};
    IntArrayView strides_{};
    size_t alignment_ = 0;
};

/// Non-owning mutable tensor view.
///
/// Lifetime and metadata borrowing semantics match TensorView, but the data
/// pointer is writable.
class MutableTensorView {
public:
    MutableTensorView() noexcept = default;

    /// Construct from borrowed raw parts.
    ///
    /// \pre `shape.size() == strides.size()`
    /// \pre Borrowed data and metadata must outlive this view.
    MutableTensorView(void* data,
                      DataType dtype,
                      IntArrayView shape,
                      IntArrayView strides,
                      size_t alignment = 0) noexcept;

    /// Construct by borrowing metadata and writable data from an existing
    /// Tensor.
    explicit MutableTensorView(Tensor& tensor) noexcept;

    AM_NODISCARD bool is_valid() const noexcept;

    explicit operator bool() const noexcept {
        return is_valid();
    }

    AM_NODISCARD void* data() const noexcept {
        return data_;
    }

    template<typename T>
    AM_NODISCARD T* data() const noexcept {
        AM_DCHECK(DataType::Make<T>() == dtype_);
        return static_cast<T*>(data_);
    }

    AM_NODISCARD DataType dtype() const noexcept {
        return dtype_;
    }

    AM_NODISCARD IntArrayView shape() const noexcept {
        return shape_;
    }

    AM_NODISCARD IntArrayView strides() const noexcept {
        return strides_;
    }

    AM_NODISCARD size_t alignment() const noexcept {
        return alignment_;
    }

    AM_NODISCARD int32_t rank() const noexcept {
        return static_cast<int32_t>(shape_.size());
    }

    AM_NODISCARD int64_t dim(int32_t i) const noexcept {
        AM_DCHECK(i >= 0 && i < rank());
        return shape_[i];
    }

    AM_NODISCARD int64_t stride(int32_t i) const noexcept {
        AM_DCHECK(i >= 0 && i < rank());
        return strides_[i];
    }

    AM_NODISCARD int64_t numel() const noexcept;

    AM_NODISCARD size_t itemsize() const noexcept {
        return static_cast<size_t>(dtype_.nbytes());
    }

    AM_NODISCARD size_t logical_nbytes() const noexcept;

    AM_NODISCARD bool is_contiguous() const noexcept;

private:
    void* data_ = nullptr;
    DataType dtype_{};
    IntArrayView shape_{};
    IntArrayView strides_{};
    size_t alignment_ = 0;
};

}// namespace aethermind

#endif// AETHERMIND_TENSOR_VIEW_H
