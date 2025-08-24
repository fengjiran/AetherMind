//
// Created by 赵丹 on 25-6-27.
//

#ifndef AETHERMIND_STORAGE_H
#define AETHERMIND_STORAGE_H

#include "storage_impl.h"

#define MAX_INLINE_SIZE 5

namespace aethermind {

class Storage {
public:
    Storage() = default;

    // Creates storage with pre-allocated memory buffer.
    Storage(size_t nbytes, DataPtr data_ptr, const std::unique_ptr<Allocator>& alloc)
        : impl_(make_object<StorageImpl>(nbytes, std::move(data_ptr), alloc)) {}

    // Allocates memory buffer using the given allocator and creates a storage with it
    Storage(size_t nbytes, const std::unique_ptr<Allocator>& alloc)
        : Storage(nbytes, alloc->allocate(nbytes), alloc) {}

    explicit Storage(ObjectPtr<StorageImpl> ptr) : impl_(std::move(ptr)) {}

    NODISCARD bool defined() const {
        return impl_;
    }

    NODISCARD size_t nbytes() const {
        return impl_->nbytes();
    }

    NODISCARD void* data() const {
        return impl_->get();
    }

    NODISCARD const void* const_data() const {
        return impl_->const_get();
    }

    NODISCARD DataPtr& data_ptr() const {
        // CHECK(impl_ != nullptr) << "Storage is not initialized";
        return impl_->data_ptr();
    }

    NODISCARD const DataPtr& const_data_ptr() const {
        // CHECK(impl_ != nullptr) << "Storage is not initialized";
        return impl_->data_ptr();
    }

    NODISCARD Device device() const {
        // CHECK(impl_ != nullptr) << "Storage is not initialized";
        return impl_->device();
    }

    operator bool() const {
        return impl_;
    }

    NODISCARD uint32_t use_count() const {
        return impl_.use_count();
    }

    NODISCARD bool unique() const {
        return use_count() == 1;
    }

private:
    ObjectPtr<StorageImpl> impl_;
};

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
            memcpy(inline_storage_, other.inline_storage_, sizeof(inline_storage_));
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
        CHECK(idx < size());
        return shape_data()[idx];
    }

    NODISCARD int64_t& shape_at(size_t idx) noexcept {
        CHECK(idx < size());
        return shape_data()[idx];
    }

    NODISCARD int64_t shape_at_uncheck(size_t idx) const noexcept {
        return shape_data()[idx];
    }

    NODISCARD int64_t& shape_at_uncheck(size_t idx) noexcept {
        return shape_data()[idx];
    }

    NODISCARD int64_t stride_at(size_t idx) const noexcept {
        CHECK(idx < size());
        return stride_data()[idx];
    }

    NODISCARD int64_t& stride_at(size_t idx) noexcept {
        CHECK(idx < size());
        return stride_data()[idx];
    }

    NODISCARD int64_t stride_at_uncheck(size_t idx) const noexcept {
        return stride_data()[idx];
    }

    NODISCARD int64_t& stride_at_uncheck(size_t idx) noexcept {
        return stride_data()[idx];
    }

    void set_shape(const std::vector<int64_t>& shape) {
        resize(shape.size());
        std::copy(shape.begin(), shape.end(), shape_begin());
    }

    void set_strides(const std::vector<int64_t>& strides) {
        CHECK(strides.size() == size());
        std::copy(strides.begin(), strides.end(), stride_begin());
    }

    NODISCARD std::vector<int64_t> get_shape() const {
        return {shape_begin(), shape_end()};
    }

    NODISCARD std::vector<int64_t> get_strides() const {
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
            CHECK(!is_inline()) << "resize slow path called when fast path should have been hit!";
            auto* tmp = outline_storage_;
            memcpy(&inline_storage_[0], &tmp[0], MAX_INLINE_SIZE * sizeof(inline_storage_[0]));
            memcpy(&inline_storage_[MAX_INLINE_SIZE], &tmp[old_size], MAX_INLINE_SIZE * sizeof(inline_storage_[0]));
            free(tmp);
        } else {
            if (is_inline()) {
                auto* tmp = static_cast<int64_t*>(malloc(storage_bytes(new_size)));
                CHECK(tmp) << "Could not allocate memory for Tensor ShapeAndStride.";
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
        CHECK(other.is_inline());
        memcpy(inline_storage_, other.inline_storage_, sizeof(inline_storage_));
    }

    static size_t storage_bytes(size_t size) noexcept {
        return size * 2 * sizeof(int64_t);
    }

    void allocate_outline_storage(size_t size) {
        outline_storage_ = static_cast<int64_t*>(malloc(storage_bytes(size)));
        CHECK(outline_storage_) << "Could not allocate memory for Tensor ShapeAndStride.";
    }

    void copy_outline_data(const ShapeAndStride& other) const noexcept {
        memcpy(outline_storage_, other.outline_storage_, storage_bytes(other.size_));
    }

    void resize_outline_storage(size_t new_size) {
        CHECK(!is_inline());
        outline_storage_ = static_cast<int64_t*>(realloc(outline_storage_, storage_bytes(new_size)));
        CHECK(outline_storage_) << "Could not reallocate memory for Tensor ShapeAndStride.";
    }

    size_t size_{1};
    union {
        int64_t* outline_storage_;
        int64_t inline_storage_[MAX_INLINE_SIZE * 2];
    };
};

}// namespace aethermind

#endif//AETHERMIND_STORAGE_H
