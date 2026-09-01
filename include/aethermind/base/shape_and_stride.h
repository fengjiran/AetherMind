#ifndef AETHERMIND_BASE_SHAPE_AND_STRIDE_H
#define AETHERMIND_BASE_SHAPE_AND_STRIDE_H

/// @file shape_and_stride.h
/// @brief Owning tensor metadata container for shape and stride.
///
/// Design constraints:
/// - Fixed max rank (kMaxRank = 8), fully inline storage, zero heap allocation.
/// - Shape and stride are updated together to avoid transient inconsistency.
/// - Strides are expressed in elements, not bytes.
///
/// This container is intended for owning Tensor metadata. For hot-path TensorView,
/// consider a non-owning view type instead.
#include "aethermind/base/macros.h"
#include "container/array_view.h"
#include "utils/logging.h"

#include <algorithm>
#include <array>

namespace aethermind {

/// @brief Owning tensor metadata container for shape and stride.
///
/// Invariants:
/// - 0 <= size_ <= kMaxRank
/// - shape_[i] >= 0 for all i < size_
/// - Unused slots (i >= size_) are zero-initialized
/// - Strides are in elements, not bytes
/// - initialized_ is true after any successful set/set_contiguous, including
///   empty shape (rank-0)
///
/// Thread-safety: Not thread-safe. External synchronization required if shared.
///
/// Rank-0 is a valid explicit metadata state when initialized_ is true.
/// Default-constructed (uninitialized) metadata returns numel=0 and
/// is_contiguous=false. Explicit empty shape/set_contiguous({}) produces
/// rank-0: numel=1, is_contiguous=true, max_element_offset=0.
/// @warning mutable_shape_data()/mutable_stride_data() expose raw mutable storage.
///          Callers are responsible for preserving invariants when using these methods.
class ShapeAndStride {
public:
    ShapeAndStride() noexcept = default;

    /// @brief Constructs from shape and strides arrays.
    /// @param shape   Dimension sizes (all non-negative).
    /// @param strides Stride values in elements.
    /// @pre shape.size() == strides.size().
    /// @pre shape.size() <= kMaxRank.
    /// @pre shape[i] >= 0 for all i.
    ShapeAndStride(IntArrayView shape, IntArrayView strides) {
        set(shape, strides);
    }

    /// @brief Sets shape and strides atomically.
    /// @param shape   Dimension sizes (all non-negative).
    /// @param strides Stride values in elements.
    /// @pre shape.size() == strides.size().
    /// @pre shape.size() <= kMaxRank.
    /// @pre shape[i] >= 0 for all i.
    /// @post Unused slots are zero-initialized.
    void set(IntArrayView shape, IntArrayView strides);

    /// @brief Sets shape and computes contiguous row-major strides.
    /// @param shape Dimension sizes (all non-negative).
    /// @pre shape.size() <= kMaxRank.
    /// @pre shape[i] >= 0 for all i.
    /// @throws AM_CHECK failure on stride overflow.
    void set_contiguous(IntArrayView shape);

    /// @brief Returns the number of dimensions (rank).
    /// @return The rank (0 to kMaxRank).
    AM_NODISCARD int32_t size() const noexcept {
        return size_;
    }

    /// @brief Returns a view of the shape array.
    /// @return IntArrayView valid only while this object is not modified or destroyed.
    AM_NODISCARD IntArrayView shape() const noexcept {
        return {shape_.data(), static_cast<size_t>(size_)};
    }

    /// @brief Returns a view of the strides array.
    /// @return IntArrayView valid only while this object is not modified or destroyed.
    AM_NODISCARD IntArrayView strides() const noexcept {
        return {strides_.data(), static_cast<size_t>(size_)};
    }

    /// @brief Returns a pointer to the const shape data.
    /// @return Pointer to the internal shape storage.
    AM_NODISCARD const int64_t* shape_data() const noexcept {
        return shape_.data();
    }

    /// @brief Returns a pointer to the const stride data.
    /// @return Pointer to the internal stride storage.
    AM_NODISCARD const int64_t* stride_data() const noexcept {
        return strides_.data();
    }

    /// @brief Returns a mutable pointer to the shape data.
    /// @return Pointer to internal shape storage.
    /// @warning Caller is responsible for preserving invariants.
    AM_NODISCARD int64_t* mutable_shape_data() noexcept {
        return shape_.data();
    }

    /// @brief Returns a mutable pointer to the stride data.
    /// @return Pointer to internal stride storage.
    /// @warning Caller is responsible for preserving invariants.
    AM_NODISCARD int64_t* mutable_stride_data() noexcept {
        return strides_.data();
    }

    /// @brief Returns whether metadata has been explicitly initialized.
    /// @return False for default-constructed; true after any successful
    ///         set/set_contiguous call (including empty shape).
    AM_NODISCARD bool is_initialized() const noexcept {
        return initialized_;
    }

    /// @brief Returns the i-th dimension size.
    /// @param i Dimension index.
    /// @return The size of dimension i.
    /// @pre 0 <= i < size().
    AM_NODISCARD int64_t dim(int32_t i) const noexcept {
        AM_DCHECK(i >= 0 && i < size_);
        return shape_[i];
    }

    /// @brief Returns the i-th stride (in elements).
    /// @param i Dimension index.
    /// @return The stride of dimension i.
    /// @pre 0 <= i < size().
    AM_NODISCARD int64_t stride(int32_t i) const noexcept {
        AM_DCHECK(i >= 0 && i < size_);
        return strides_[i];
    }

    /// @brief Returns the total number of elements (product of shape).
    /// @return The element count; 0 if uninitialized, 1 for rank-0.
    /// @throws AM_CHECK failure on overflow.
    AM_NODISCARD int64_t numel() const noexcept;

    /// @brief Returns true if strides represent a contiguous row-major layout.
    /// @return True if contiguous; dimensions with shape[i] == 1 are ignored.
    AM_NODISCARD bool is_contiguous() const noexcept;

    /// @brief Returns the maximum element offset reachable via the strides.
    /// @return Offset in elements from the data pointer; 0 for rank-0.
    /// @throws AM_CHECK failure on negative shape/stride or offset overflow.
    AM_NODISCARD int64_t max_element_offset() const;

    /// @brief Maximum supported tensor rank.
    static constexpr uint32_t kMaxRank = 8;

private:
    int32_t size_ = 0;
    bool initialized_ = false;
    std::array<int64_t, kMaxRank> shape_{};
    std::array<int64_t, kMaxRank> strides_{};
};

} // namespace aethermind


#endif // AETHERMIND_BASE_SHAPE_AND_STRIDE_H
