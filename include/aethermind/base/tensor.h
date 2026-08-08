#ifndef AETHERMIND_BASE_TENSOR_H
#define AETHERMIND_BASE_TENSOR_H

/// @file tensor.h
/// @brief Owning dense tensor backed by a Buffer.
#include "aethermind/dtypes/data_type.h"
#include "aethermind/memory/buffer.h"
#include "container/array_view.h"
#include "aethermind/base/device.h"
#include "aethermind/base/macros.h"
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

/// @brief Owning dense tensor backed by a Buffer.
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
/// @note Shallow copy/move assignment shares the underlying Buffer.
class Tensor {
public:
    Tensor() noexcept = default;

    /// @brief Constructs a Tensor from a Buffer and explicit shape/strides.
    /// @param buffer            Owning buffer (moved into this Tensor).
    /// @param byte_offset       Byte offset into the buffer where data begins.
    /// @param dtype             Element data type.
    /// @param shape_and_strides Shape and stride metadata (strides in elements).
    /// @throws AM_CHECK failure if metadata validation fails.
    Tensor(Buffer buffer, size_t byte_offset, const DataType& dtype,
           const ShapeAndStride& shape_and_strides)
        : buffer_(std::move(buffer)), byte_offset_(byte_offset), dtype_(dtype),
          shape_and_strides_(shape_and_strides) {
        validate();
    }

    /// @brief Constructs a Tensor from a Buffer and separate shape/strides views.
    /// @param buffer      Owning buffer (moved into this Tensor).
    /// @param byte_offset Byte offset into the buffer where data begins.
    /// @param dtype       Element data type.
    /// @param shape       Dimension sizes (all non-negative).
    /// @param strides     Stride values in elements.
    /// @throws AM_CHECK failure if metadata validation fails.
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

    /// @brief Returns an immutable borrowed view over this Tensor's data.
    /// @return TensorView valid only while this Tensor is alive and unmodified.
    AM_NODISCARD TensorView view() const noexcept;

    /// @brief Returns a mutable borrowed view over this Tensor's data.
    /// @return MutableTensorView valid only while this Tensor is alive and unmodified.
    AM_NODISCARD MutableTensorView mutable_view() noexcept;

    /// @brief Returns the maximum element offset reachable via the strides.
    /// @return Offset in elements from the data pointer.
    AM_NODISCARD int64_t max_touched_element_offset() const noexcept {
        return shape_and_strides_.max_element_offset();
    }

    /// @brief Returns the byte size of the logical tensor (numel * itemsize).
    /// @return Byte count; 0 for an uninitialized Tensor.
    /// @throws AM_CHECK failure on overflow.
    AM_NODISCARD size_t logical_nbytes() const noexcept;

    /// @brief Returns the byte span from the data pointer through the last touched element.
    /// @return Byte count covering the addressed storage range; 0 if uninitialized or numel == 0.
    /// @throws AM_CHECK failure on overflow.
    AM_NODISCARD size_t max_touched_span_bytes() const noexcept;

    /// @brief Checks whether the addressed storage range fits within the backing Buffer.
    /// @return True if the buffer can hold the addressed range; false otherwise.
    AM_NODISCARD bool storage_range_is_valid() const noexcept;

    /// @brief Returns a Tensor sharing storage with a strided slice along one dimension.
    /// @param dim_idx Dimension to slice (0-based; negative wraps from the end).
    /// @param start   Start index (inclusive).
    /// @param end     End index (exclusive).
    /// @param step    Step between selected indices (must be > 0).
    /// @return A Tensor sharing this Tensor's Buffer with adjusted offset/strides.
    /// @pre rank() >= 1.
    /// @throws AM_CHECK failure on an uninitialized tensor, invalid dim, or step <= 0.
    AM_NODISCARD Tensor slice(int32_t dim_idx, int64_t start, int64_t end, int64_t step) const noexcept;

    /// @brief Convenience overload of slice() with step = 1.
    /// @param dim_idx Dimension to slice (0-based; negative wraps from the end).
    /// @param start   Start index (inclusive).
    /// @param end     End index (exclusive).
    /// @return A Tensor sharing this Tensor's Buffer with adjusted offset/strides.
    /// @pre rank() >= 1.
    AM_NODISCARD Tensor slice(int32_t dim_idx, int64_t start, int64_t end) const noexcept {
        return slice(dim_idx, start, end, 1);
    }

    /// @brief Returns a Tensor sharing storage with a sub-range along one dimension.
    /// @param dim_idx Dimension to narrow (0-based).
    /// @param start   Start index (inclusive).
    /// @param length  Number of elements to keep (must be >= 0).
    /// @return A Tensor sharing this Tensor's Buffer with adjusted offset/strides.
    /// @pre rank() >= 1.
    /// @throws AM_CHECK failure if length < 0.
    AM_NODISCARD Tensor narrow(int32_t dim_idx, int64_t start, int64_t length) const noexcept {
        AM_CHECK(length >= 0);
        return slice(dim_idx, start, start + length);
    }

    AM_NODISCARD bool is_rank_zero() const noexcept {
        return is_initialized() && rank() == 0;
    }

    // ── Scalar-Tensor bridge ─────────────────────────────────

    /// @brief Creates a rank-0 CPU Tensor from a Scalar value.
    /// @param scalar    The scalar value to wrap.
    /// @param allocator A CPU allocator used to allocate the backing buffer.
    /// @return A rank-0 Tensor owning its Buffer with numel 1.
    /// @pre allocator.device().is_cpu().
    static Tensor FromScalar(const Scalar& scalar, Allocator& allocator);

    /// @brief Extracts the single element as a Scalar.
    /// @return The scalar value stored in this tensor.
    /// @pre Initialized CPU Tensor, numel() == 1, non-null data, dtype().IsScalar().
    AM_NODISCARD Scalar item() const;

    /// @brief Replaces the single element with a new value.
    /// @param scalar The new scalar value to write.
    /// @pre Initialized CPU Tensor, numel() == 1, non-null data, dtype().IsScalar().
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
