/// \file
/// Owning tensor metadata container for shape and stride.
///
/// Phase 1 design constraints:
/// - Fixed max rank (kMaxRank = 8), fully inline storage, zero heap allocation.
/// - Shape and stride are updated together to avoid transient inconsistency.
/// - Strides are expressed in elements, not bytes.
///
/// This container is intended for owning Tensor metadata. For hot-path TensorView,
/// consider a non-owning view type instead.

#ifndef AETHERMIND_BASE_SHAPE_AND_STRIDE_H
#define AETHERMIND_BASE_SHAPE_AND_STRIDE_H

#include "aethermind/utils/overflow_check.h"
#include "container/array_view.h"
#include "macros.h"
#include "utils/logging.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace aethermind {

/// Owning tensor metadata container for shape and stride.
///
/// Invariants:
/// - 0 <= size_ <= kMaxRank
/// - shape_[i] >= 0 for all i < size_
/// - Unused slots (i >= size_) are zero-initialized
/// - Strides are in elements, not bytes
///
/// Thread-safety: Not thread-safe. External synchronization required if shared.
///
/// Rank-0 represents a scalar-like empty shape; numel() returns 1.
/// Rank is capped at kMaxRank as a Phase 1 implementation constraint.
///
/// Warning: mutable_shape_data()/mutable_stride_data() expose raw mutable storage.
/// Callers are responsible for preserving invariants when using these methods.
class ShapeAndStride {
public:
    ShapeAndStride() noexcept = default;

    /// Constructs from shape and strides arrays.
    /// \pre shape.size() == strides.size()
    /// \pre shape.size() <= kMaxRank
    /// \pre shape[i] >= 0 for all i
    ShapeAndStride(IntArrayView shape, IntArrayView strides) {
        set(shape, strides);
    }

    /// Sets shape and strides atomically.
    ///
    /// \pre shape.size() == strides.size()
    /// \pre shape.size() <= kMaxRank
    /// \pre shape[i] >= 0 for all i
    /// \post Unused slots are zero-initialized
    void set(IntArrayView shape, IntArrayView strides) {
        AM_CHECK(shape.size() == strides.size(), "shape/strides size mismatch");
        AM_CHECK(shape.size() <= static_cast<size_t>(kMaxRank), "rank exceeds kMaxRank");

        auto new_size = static_cast<int32_t>(shape.size());
        for (int32_t i = 0; i < new_size; ++i) {
            AM_CHECK(shape[i] >= 0, "shape dimensions must be non-negative");
        }

        size_ = new_size;
        std::ranges::copy(shape, shape_.begin());
        std::ranges::copy(strides, strides_.begin());

        for (uint32_t i = size_; i < kMaxRank; ++i) {
            shape_[i] = 0;
            strides_[i] = 0;
        }
    }

    /// Sets shape and computes contiguous row-major strides.
    ///
    /// \pre shape.size() <= kMaxRank
    /// \pre shape[i] >= 0 for all i
    /// \throws AM_CHECK failure on stride overflow
    void set_contiguous(IntArrayView shape) {
        AM_CHECK(shape.size() <= static_cast<size_t>(kMaxRank), "rank exceeds kMaxRank");
        auto new_size = static_cast<int32_t>(shape.size());
        for (int32_t i = 0; i < new_size; ++i) {
            AM_CHECK(shape[i] >= 0, "shape dimensions must be non-negative");
        }

        size_ = new_size;
        if (size_ == 0) {
            for (uint32_t i = 0; i < kMaxRank; ++i) {
                shape_[i] = 0;
                strides_[i] = 0;
            }
            return;
        }

        std::ranges::copy(shape, shape_.begin());

        strides_[size_ - 1] = 1;
        for (int32_t i = size_ - 2; i >= 0; --i) {
            AM_CHECK(!CheckOverflowMul(strides_[i + 1], shape_[i + 1], &strides_[i]),
                     "Stride calculation overflow");
        }

        for (uint32_t i = size_; i < kMaxRank; ++i) {
            shape_[i] = 0;
            strides_[i] = 0;
        }
    }

    /// Returns the number of dimensions (rank).
    AM_NODISCARD int32_t size() const noexcept {
        return size_;
    }

    /// Returns a view of the shape array.
    /// The view is valid only while this object is not modified or destroyed.
    AM_NODISCARD IntArrayView shape() const noexcept {
        return {shape_.data(), static_cast<size_t>(size_)};
    }

    /// Returns a view of the strides array.
    /// The view is valid only while this object is not modified or destroyed.
    AM_NODISCARD IntArrayView strides() const noexcept {
        return {strides_.data(), static_cast<size_t>(size_)};
    }

    /// Returns a pointer to the shape data (const).
    AM_NODISCARD const int64_t* shape_data() const noexcept {
        return shape_.data();
    }

    /// Returns a pointer to the stride data (const).
    AM_NODISCARD const int64_t* stride_data() const noexcept {
        return strides_.data();
    }

    /// Returns a mutable pointer to the shape data.
    /// \warning Caller is responsible for preserving invariants.
    AM_NODISCARD int64_t* mutable_shape_data() noexcept {
        return shape_.data();
    }

    /// Returns a mutable pointer to the stride data.
    /// \warning Caller is responsible for preserving invariants.
    AM_NODISCARD int64_t* mutable_stride_data() noexcept {
        return strides_.data();
    }

    /// Returns the i-th dimension size.
    /// \pre 0 <= i < size()
    AM_NODISCARD int64_t shape(int32_t i) const noexcept {
        AM_DCHECK(i >= 0 && i < size_);
        return shape_[i];
    }

    /// Returns the i-th stride (in elements).
    /// \pre 0 <= i < size()
    AM_NODISCARD int64_t stride(int32_t i) const noexcept {
        AM_DCHECK(i >= 0 && i < size_);
        return strides_[i];
    }

    /// Returns the total number of elements (product of shape).
    /// \throws AM_CHECK failure on overflow
    AM_NODISCARD int64_t numel() const noexcept {
        if (size_ == 0) {
            return 0;
        }

        uint64_t numel = 0;
        bool overflow = SafeMultiplyU64(shape(), &numel);
        constexpr auto kNumelMax = std::min<uint64_t>(
                std::numeric_limits<int64_t>::max(),
                std::numeric_limits<size_t>::max());

        overflow |= numel > kNumelMax;
        AM_CHECK(!overflow, "Integer multiplication overflow when compute numel.");
        return static_cast<int64_t>(numel);
    }

    /// Returns true if strides represent a contiguous row-major layout.
    /// Dimensions with shape[i] == 1 are ignored (broadcasting-friendly).
    AM_NODISCARD bool is_contiguous() const noexcept {
        if (size_ == 0) {
            return true;
        }

        int64_t expected_stride = 1;
        for (int i = size_ - 1; i >= 0; --i) {
            if (shape_[i] == 1) {
                continue;
            }

            if (strides_[i] != expected_stride) {
                return false;
            }

            int64_t next_expected_stride = 0;
            AM_CHECK(!CheckOverflowMul(expected_stride, shape_[i], &next_expected_stride),
                     "Stride validation overflow");
            expected_stride = next_expected_stride;
        }
        return true;
    }

    AM_NODISCARD int64_t max_element_offset() const {
        if (size_ == 0) {
            return 0;
        }

        for (int32_t i = 0; i < size_; ++i) {
            AM_CHECK(shape_[i] >= 0);
            AM_CHECK(strides_[i] >= 0);
            if (shape_[i] == 0) {
                return 0;
            }
        }

        int64_t offset = 0;
        for (int32_t i = 0; i < size_; ++i) {
            int64_t term = 0;
            AM_CHECK(!CheckOverflowMul((shape_[i] - 1), strides_[i], &term));
            AM_CHECK(term >= 0);
            AM_CHECK(offset <= std::numeric_limits<int64_t>::max() - term, "max_element_offset overflow.");
            offset += term;
        }

        return offset;
    }

    static constexpr uint32_t kMaxRank = 8;

private:
    int32_t size_ = 0;
    std::array<int64_t, kMaxRank> shape_{};
    std::array<int64_t, kMaxRank> strides_{};
};

}// namespace aethermind


#endif// AETHERMIND_BASE_SHAPE_AND_STRIDE_H
