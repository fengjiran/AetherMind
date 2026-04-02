#ifndef AETHERMIND_BASE_SHAPE_AND_STRIDE_H
#define AETHERMIND_BASE_SHAPE_AND_STRIDE_H

#include "container/array_view.h"
#include "utils/logging.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace aethermind {


/*!
 * \brief Owning tensor metadata container for shape and stride.
 *
 * \note
 * - Phase 1 design: fixed max rank, fully inline storage, zero heap allocation.
 * - Shape and stride are always updated together to avoid transient inconsistency.
 * - Strides are expressed in ELEMENTS, not bytes.
 * - This type is intended for owning Tensor metadata, not for hot-path TensorView.
 */
class ShapeAndStride_bk {
public:
    ShapeAndStride_bk() noexcept = default;

    ShapeAndStride_bk(IntArrayView shape, IntArrayView strides) {
        set(shape, strides);
    }

    void set(IntArrayView shape, IntArrayView strides) {
        AM_DCHECK(shape.size() == strides.size());
        AM_DCHECK(shape.size() <= static_cast<uint32_t>(kMaxRank));

        auto new_size = static_cast<int32_t>(shape.size());
        for (int32_t i = 0; i < new_size; ++i) {
            AM_DCHECK(shape[i] >= 0);
        }

        size_ = new_size;
        std::copy(shape.begin(), shape.end(), shape_);
        std::copy(strides.begin(), strides.end(), strides_);

        for (int32_t i = size_; i < kMaxRank; ++i) {
            shape_[i] = 0;
            strides_[i] = 0;
        }
    }

    void set_contiguous(IntArrayView shape) {
        AM_DCHECK(shape.size() <= static_cast<uint32_t>(kMaxRank));
        auto new_size = static_cast<int32_t>(shape.size());
        for (int32_t i = 0; i < new_size; ++i) {
            AM_DCHECK(shape[i] >= 0);
        }

        size_ = new_size;
        if (size_ == 0) {
            for (int32_t i = 0; i < kMaxRank; ++i) {
                shape_[i] = 0;
                strides_[i] = 0;
            }
            return;
        }

        std::copy(shape.begin(), shape.end(), shape_);

        strides_[size_ - 1] = 1;
        for (int32_t i = size_ - 2; i >= 0; --i) {
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        }

        for (int32_t i = size_; i < kMaxRank; ++i) {
            shape_[i] = 0;
            strides_[i] = 0;
        }
    }

    AM_NODISCARD int32_t size() const noexcept {
        return size_;
    }

    AM_NODISCARD IntArrayView shape() const noexcept {
        return IntArrayView(shape_, size_);
    }

    AM_NODISCARD IntArrayView strides() const noexcept {
        return IntArrayView(strides_, size_);
    }

    AM_NODISCARD const int64_t* shape_data() const noexcept {
        return shape_;
    }

    AM_NODISCARD const int64_t* stride_data() const noexcept {
        return strides_;
    }

    AM_NODISCARD int64_t* mutable_shape_data() noexcept {
        return shape_;
    }

    AM_NODISCARD int64_t* mutable_stride_data() noexcept {
        return strides_;
    }

    AM_NODISCARD int64_t dim(int32_t i) const noexcept {
        AM_DCHECK(i >= 0 && i < size_);
        return shape_[i];
    }

    AM_NODISCARD int64_t stride(int32_t i) const noexcept {
        AM_DCHECK(i >= 0 && i < size_);
        return strides_[i];
    }

private:
    constexpr static uint32_t kMaxRank = 8;

    int32_t size_ = 0;
    int64_t shape_[kMaxRank]{};
    int64_t strides_[kMaxRank]{};
};

}// namespace aethermind


#endif// AETHERMIND_BASE_SHAPE_AND_STRIDE_H
