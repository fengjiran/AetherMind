//
// Created by 赵丹 on 2025/8/26.
//

#ifndef AETHERMIND_SHAPE_AND_STRIDE_H
#define AETHERMIND_SHAPE_AND_STRIDE_H

#include "container/array_view.h"

#define MAX_INLINE_SIZE 5

namespace aethermind {

// Packed container for TensorImpl shape and strides.
// 1 size_t for the size
// 5 eightbytes of inline sizes and 5 eightbytes of inline strides, OR pointer
// to out-of-line array
class ShapeAndStride {
public:
    ShapeAndStride() {
        shape_at_uncheck(0) = 0;
        stride_at_uncheck(0) = 1;
    }

    ShapeAndStride(const ShapeAndStride& other) : size_(other.size_) {
        if (other.is_inline()) {
            copy_inline_data(other);
        } else {
            allocate_outline_storage(size_);
            copy_outline_data(other);
        }
    }

    ShapeAndStride(ShapeAndStride&& other) noexcept : size_(other.size_) {
        if (other.is_inline()) {
            std::memcpy(inline_storage_, other.inline_storage_, sizeof(inline_storage_));
        } else {
            outline_storage_ = other.outline_storage_;
            other.outline_storage_ = nullptr;
        }
        other.size_ = 0;
    }

    ShapeAndStride& operator=(const ShapeAndStride& rhs) {
        if (this == &rhs) {
            return *this;
        }

        if (rhs.is_inline()) {
            if (!is_inline()) {
                free(outline_storage_);
            }
            copy_inline_data(rhs);
        } else {
            if (is_inline()) {
                allocate_outline_storage(rhs.size_);
            } else {
                resize_outline_storage(rhs.size_);
            }
            copy_outline_data(rhs);
        }
        size_ = rhs.size_;
        return *this;
    }

    ShapeAndStride& operator=(ShapeAndStride&& rhs) noexcept {
        if (this == &rhs) {
            return *this;
        }

        if (rhs.is_inline()) {
            if (!is_inline()) {
                free(outline_storage_);
            }
            copy_inline_data(rhs);
        } else {
            if (!is_inline()) {
                free(outline_storage_);
            }
            outline_storage_ = rhs.outline_storage_;
            rhs.outline_storage_ = nullptr;
        }
        size_ = rhs.size_;
        rhs.size_ = 0;
        return *this;
    }

    bool operator==(const ShapeAndStride& other) const {
        if (size_ != other.size_) {
            return false;
        }

        bool res = is_inline() ? std::memcmp(inline_storage_, other.inline_storage_, sizeof(inline_storage_))
                               : std::memcmp(outline_storage_, other.outline_storage_, storage_bytes(size_));
        return !res;
    }

    ~ShapeAndStride() {
        if (!is_inline()) {
            free(outline_storage_);
        }
    }

    NODISCARD size_t size() const noexcept {
        return size_;
    }

    NODISCARD int64_t* shape_data() noexcept {
        return is_inline() ? &inline_storage_[0] : &outline_storage_[0];
    }

    NODISCARD const int64_t* shape_data() const noexcept {
        return is_inline() ? &inline_storage_[0] : &outline_storage_[0];
    }

    NODISCARD int64_t* stride_data() noexcept {
        return is_inline() ? &inline_storage_[MAX_INLINE_SIZE] : &outline_storage_[size()];
    }

    NODISCARD const int64_t* stride_data() const noexcept {
        return is_inline() ? &inline_storage_[MAX_INLINE_SIZE] : &outline_storage_[size()];
    }

    int64_t* shape_begin() noexcept {
        return shape_data();
    }

    NODISCARD const int64_t* shape_begin() const noexcept {
        return shape_data();
    }

    int64_t* shape_end() noexcept {
        return shape_data() + size();
    }

    NODISCARD const int64_t* shape_end() const noexcept {
        return shape_data() + size();
    }

    int64_t* stride_begin() noexcept {
        return stride_data();
    }

    NODISCARD const int64_t* stride_begin() const noexcept {
        return stride_data();
    }

    int64_t* stride_end() noexcept {
        return stride_data() + size();
    }

    NODISCARD const int64_t* stride_end() const noexcept {
        return stride_data() + size();
    }

    NODISCARD int64_t shape_at(size_t idx) const noexcept {
        AM_CHECK(idx < size());
        return shape_data()[idx];
    }

    NODISCARD int64_t& shape_at(size_t idx) noexcept {
        AM_CHECK(idx < size());
        return shape_data()[idx];
    }

    NODISCARD int64_t shape_at_uncheck(size_t idx) const noexcept {
        return shape_data()[idx];
    }

    NODISCARD int64_t& shape_at_uncheck(size_t idx) noexcept {
        return shape_data()[idx];
    }

    NODISCARD int64_t stride_at(size_t idx) const noexcept {
        AM_CHECK(idx < size());
        return stride_data()[idx];
    }

    NODISCARD int64_t& stride_at(size_t idx) noexcept {
        AM_CHECK(idx < size());
        return stride_data()[idx];
    }

    NODISCARD int64_t stride_at_uncheck(size_t idx) const noexcept {
        return stride_data()[idx];
    }

    NODISCARD int64_t& stride_at_uncheck(size_t idx) noexcept {
        return stride_data()[idx];
    }

    void set_shape(IntArrayView shape) {
        resize(shape.size());
        std::copy(shape.begin(), shape.end(), shape_begin());
    }

    void set_strides(IntArrayView strides) {
        AM_CHECK(strides.size() == size());
        std::copy(strides.begin(), strides.end(), stride_begin());
    }

    NODISCARD IntArrayView get_shape() const {
        return {shape_begin(), shape_end()};
    }

    NODISCARD IntArrayView get_strides() const {
        return {stride_begin(), stride_end()};
    }

    void resize(size_t new_size) {
        const auto old_size = size();
        if (new_size == old_size) {
            return;
        }

        if (new_size <= MAX_INLINE_SIZE && is_inline()) {
            if (old_size < new_size) {
                const auto bytes_to_zero = (new_size - old_size) * sizeof(inline_storage_[0]);
                memset(&inline_storage_[old_size], 0, bytes_to_zero);
                memset(&inline_storage_[MAX_INLINE_SIZE + old_size], 0, bytes_to_zero);
            }
            size_ = new_size;
        } else {
            resize_slow_path(new_size, old_size);
        }
    }

    void resize_slow_path(size_t new_size, size_t old_size) {
        if (new_size <= MAX_INLINE_SIZE) {
            AM_CHECK(!is_inline(), "resize slow path called when fast path should have been hit!");
            auto* tmp = outline_storage_;
            memcpy(&inline_storage_[0], &tmp[0], MAX_INLINE_SIZE * sizeof(inline_storage_[0]));
            memcpy(&inline_storage_[MAX_INLINE_SIZE], &tmp[old_size], MAX_INLINE_SIZE * sizeof(inline_storage_[0]));
            free(tmp);
        } else {
            if (is_inline()) {
                auto* tmp = static_cast<int64_t*>(malloc(storage_bytes(new_size)));
                AM_CHECK(tmp, "Could not allocate memory for Tensor ShapeAndStride.");
                const auto bytes_to_copy = old_size * sizeof(inline_storage_[0]);
                const auto bytes_to_zero = new_size > old_size ? (new_size - old_size) * sizeof(tmp[0]) : 0;
                memcpy(&tmp[0], &inline_storage_[0], bytes_to_copy);
                if (bytes_to_zero) {
                    memset(&tmp[old_size], 0, bytes_to_zero);
                }

                memcpy(&tmp[new_size], &inline_storage_[MAX_INLINE_SIZE], bytes_to_copy);
                if (bytes_to_zero) {
                    memset(&tmp[new_size + old_size], 0, bytes_to_zero);
                }

                outline_storage_ = tmp;
            } else {
                const bool is_growing = new_size > old_size;
                if (is_growing) {
                    resize_outline_storage(new_size);
                }

                memmove(outline_storage_ + new_size, outline_storage_ + old_size,
                        std::min(new_size, old_size) * sizeof(outline_storage_[0]));

                if (is_growing) {
                    const auto bytes_to_zero = (new_size - old_size) * sizeof(outline_storage_[0]);
                    memset(&outline_storage_[old_size], 0, bytes_to_zero);
                    memset(&outline_storage_[new_size + old_size], 0, bytes_to_zero);
                } else {
                    resize_outline_storage(new_size);
                }
            }
        }
        size_ = new_size;
    }

private:
    NODISCARD bool is_inline() const noexcept {
        return size_ <= MAX_INLINE_SIZE;
    }

    void copy_inline_data(const ShapeAndStride& other) {
        AM_CHECK(other.is_inline());
        memcpy(inline_storage_, other.inline_storage_, sizeof(inline_storage_));
    }

    static size_t storage_bytes(size_t size) noexcept {
        return size * 2 * sizeof(int64_t);
    }

    void allocate_outline_storage(size_t size) {
        outline_storage_ = static_cast<int64_t*>(malloc(storage_bytes(size)));
        AM_CHECK(outline_storage_, "Could not allocate memory for Tensor ShapeAndStride.");
    }

    void copy_outline_data(const ShapeAndStride& other) const noexcept {
        memcpy(outline_storage_, other.outline_storage_, storage_bytes(other.size_));
    }

    void resize_outline_storage(size_t new_size) {
        AM_CHECK(!is_inline());
        outline_storage_ = static_cast<int64_t*>(realloc(outline_storage_, storage_bytes(new_size)));
        AM_CHECK(outline_storage_, "Could not reallocate memory for Tensor ShapeAndStride.");
    }

    size_t size_{1};
    union {
        int64_t* outline_storage_;
        int64_t inline_storage_[MAX_INLINE_SIZE * 2];
    };
};

}// namespace aethermind
#endif//AETHERMIND_SHAPE_AND_STRIDE_H
