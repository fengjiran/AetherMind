#ifndef AETHERMIND_BASE_TENSOR_H
#define AETHERMIND_BASE_TENSOR_H

#include "aethermind/dtypes/data_type.h"
#include "aethermind/memory/buffer.h"
#include "container/array_view.h"
#include "device.h"
#include "macros.h"
#include "shape_and_stride.h"
#include "utils/logging.h"

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>

namespace aethermind {

class Allocator;
class Scalar;
class TensorView;
class MutableTensorView;

/// Owning dense tensor backed by a Buffer.
///
/// Rank-0 (scalar tensor): shape `[]`, strides `[]`, rank 0, numel 1,
/// contiguous=true. Distinct from default-uninitialized (numel 0,
/// !is_contiguous) and from `[0]` (rank 1, numel 0). Rank-0 participates
/// in the existing constant evaluation and graph folding paths. Runtime
/// operators (Linear, RmsNorm, Embedding, etc.) reject rank-0 through
/// their existing rank guards; slice/narrow require rank >= 1.
///
/// The explicit Scalar bridge (FromScalar/item/set_item) is CPU-only and
/// allocator-explicit. No implicit Scalar-to-Tensor conversion, inline
/// storage, or general eager arithmetic is supported.
///
/// Shallow copy/move assignment shares the underlying Buffer.
class Tensor {
public:
    Tensor() noexcept = default;

    Tensor(Buffer buffer, size_t byte_offset, const DataType& dtype,
           const ShapeAndStride& shape_and_strides)
        : buffer_(std::move(buffer)), byte_offset_(byte_offset), dtype_(dtype),
          shape_and_strides_(shape_and_strides) {
        validate();
    }

    Tensor(Buffer buffer, size_t byte_offset, const DataType& dtype,
           IntArrayView shape, IntArrayView strides)
        : buffer_(std::move(buffer)), byte_offset_(byte_offset), dtype_(dtype),
          shape_and_strides_(shape, strides) {
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

    AM_NODISCARD Device device() const noexcept {
        return buffer_.device();
    }

    AM_NODISCARD size_t alignment() const noexcept {
        const size_t base_align = buffer_.alignment();
        if (base_align == 0) {
            return 0;
        }

        if (byte_offset_ == 0) {
            return base_align;
        }

        return std::gcd(base_align, byte_offset_);
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

    AM_NODISCARD int64_t dim(int32_t i) const noexcept {
        return shape_and_strides_.dim(i);
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

    AM_NODISCARD const void* data() const noexcept {
        if (!is_initialized()) {
            return nullptr;
        }

        const auto* base = static_cast<const char*>(buffer_.data());
        return base + byte_offset_;
    }

    AM_NODISCARD void* mutable_data() noexcept {
        if (!is_initialized()) {
            return nullptr;
        }

        auto* base = static_cast<char*>(buffer_.mutable_data());
        return base + byte_offset_;
    }

    AM_NODISCARD TensorView view() const noexcept;

    AM_NODISCARD MutableTensorView mutable_view() noexcept;

    AM_NODISCARD int64_t max_touched_element_offset() const noexcept {
        return shape_and_strides_.max_element_offset();
    }

    AM_NODISCARD size_t logical_nbytes() const noexcept;

    AM_NODISCARD size_t max_touched_span_bytes() const noexcept;

    AM_NODISCARD bool storage_range_is_valid() const noexcept;

    AM_NODISCARD Tensor slice(int32_t dim_idx, int64_t start, int64_t end, int64_t step) const noexcept;

    AM_NODISCARD Tensor slice(int32_t dim_idx, int64_t start, int64_t end) const noexcept {
        return slice(dim_idx, start, end, 1);
    }

    AM_NODISCARD Tensor narrow(int32_t dim_idx, int64_t start, int64_t length) const noexcept {
        AM_CHECK(length >= 0);
        return slice(dim_idx, start, start + length);
    }

    AM_NODISCARD bool is_rank_zero() const noexcept {
        return is_initialized() && rank() == 0;
    }

    // ── Scalar-Tensor bridge ─────────────────────────────────

    /// Creates a rank-0 CPU Tensor from a Scalar value.
    /// The allocator must be a CPU allocator.
    /// The resulting Tensor owns its Buffer and has rank 0, numel 1.
    static Tensor FromScalar(const Scalar& scalar, Allocator& allocator);

    /// Extracts the single element as a Scalar.
    /// Requires: initialized CPU Tensor, numel() == 1, non-null data,
    /// and dtype().IsScalar().
    AM_NODISCARD Scalar item() const;

    /// Replaces the single element with a new value.
    /// Requires: initialized CPU Tensor, numel() == 1, non-null data,
    /// and dtype().IsScalar(). Converts via the checked Scalar::to<T>().
    void set_item(const Scalar& scalar);

private:
    Buffer buffer_;
    size_t byte_offset_{0};
    DataType dtype_;
    ShapeAndStride shape_and_strides_;

    void validate() const;
};

}// namespace aethermind


#endif// AETHERMIND_BASE_TENSOR_H
