//
// Created by 赵丹 on 25-6-27.
//

#ifndef AETHERMIND_STORAGE_H
#define AETHERMIND_STORAGE_H

#include "data_ptr.h"
#include "device.h"
#include "macros.h"
#include "memory/storage_impl.h"
#include "object.h"
#include "object_allocator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace aethermind {

using memory_deleter_fn = void (*)(void* deleter_ctx, void* ptr);

inline void NoOpMemoryDeleter(void* deleter_ctx, void* ptr) noexcept {
    (void) deleter_ctx;
    (void) ptr;
}

class MemoryHandle {
public:
    MemoryHandle() noexcept = default;
    MemoryHandle(void* data,
                 void* deleter_ctx,
                 memory_deleter_fn deleter,
                 Device device,
                 size_t alignment = 0) noexcept
        : data_(data),
          ctx_(deleter_ctx),
          deleter_(deleter),
          alignment_(alignment),
          device_(device) {}

    ~MemoryHandle() noexcept {
        reset();
    }

    MemoryHandle(const MemoryHandle&) = delete;
    MemoryHandle& operator=(const MemoryHandle&) = delete;

    MemoryHandle(MemoryHandle&& other) noexcept {
        swap(other);
    }

    MemoryHandle& operator=(MemoryHandle&& other) noexcept {
        MemoryHandle(std::move(other)).swap(*this);
        return *this;
    }

    AM_NODISCARD void* get() noexcept {
        return data_;
    }

    AM_NODISCARD const void* get() const noexcept {
        return data_;
    }

    AM_NODISCARD Device device() const noexcept {
        return device_;
    }

    AM_NODISCARD size_t alignment() const noexcept {
        return alignment_;
    }

    AM_NODISCARD void* context() const noexcept {
        return ctx_;
    }

    AM_NODISCARD memory_deleter_fn deleter() const noexcept {
        return deleter_;
    }

    explicit operator bool() const noexcept {
        return data_ != nullptr;
    }

    void reset() noexcept {
        if (data_ != nullptr && deleter_ != nullptr) {
            deleter_(ctx_, data_);
        }

        data_ = nullptr;
        ctx_ = nullptr;
        deleter_ = nullptr;
        device_ = Device(kUndefined);
        alignment_ = 0;
    }

    void swap(MemoryHandle& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(ctx_, other.ctx_);
        std::swap(deleter_, other.deleter_);
        std::swap(device_, other.device_);
        std::swap(alignment_, other.alignment_);
    }

private:
    void* data_ = nullptr;
    void* ctx_ = nullptr;
    memory_deleter_fn deleter_ = nullptr;
    size_t alignment_ = 0;
    Device device_{kUndefined};
};

inline bool operator==(const MemoryHandle& h, std::nullptr_t) noexcept {
    return !static_cast<bool>(h);
}

inline bool operator==(std::nullptr_t, const MemoryHandle& h) noexcept {
    return !static_cast<bool>(h);
}

inline bool operator!=(const MemoryHandle& h, std::nullptr_t) noexcept {
    return static_cast<bool>(h);
}

inline bool operator!=(std::nullptr_t, const MemoryHandle& h) noexcept {
    return static_cast<bool>(h);
}

class BufferImpl : public Object {
public:
    BufferImpl() noexcept = default;

    ~BufferImpl() override = default;

    BufferImpl(size_t nbytes, MemoryHandle handle) noexcept
        : nbytes_(nbytes), handle_(std::move(handle)) {}

    AM_NODISCARD bool is_initialized() const noexcept {
        return static_cast<bool>(handle_);
    }

    /// Visible buffer size in bytes from `data()` base pointer.
    AM_NODISCARD size_t nbytes() const noexcept {
        return nbytes_;
    }

    AM_NODISCARD void* mutable_data() noexcept {
        return handle_.get();
    }

    AM_NODISCARD const void* data() const noexcept {
        return handle_.get();
    }

    AM_NODISCARD Device device() const noexcept {
        return handle_.device();
    }

    AM_NODISCARD size_t alignment() const noexcept {
        return handle_.alignment();
    }

    BufferImpl(const BufferImpl&) = delete;
    BufferImpl(BufferImpl&&) noexcept = delete;
    BufferImpl& operator=(const BufferImpl&) = delete;
    BufferImpl& operator=(BufferImpl&&) noexcept = delete;

private:
    size_t nbytes_{0};
    MemoryHandle handle_;
};

class Buffer : public ObjectRef {
public:
    Buffer() noexcept = default;

    /// Construct from a pre-created implementation object.
    explicit Buffer(ObjectPtr<BufferImpl> impl) noexcept
        : impl_(std::move(impl)) {}

    Buffer(size_t nbytes, MemoryHandle handle) noexcept
        : impl_(make_object<BufferImpl>(nbytes, std::move(handle))) {}

    AM_NODISCARD bool is_initialized() const noexcept {
        return impl_ && impl_->is_initialized();
    }

    AM_NODISCARD size_t nbytes() const noexcept {
        return impl_->nbytes();
    }

    AM_NODISCARD void* mutable_data() noexcept {
        return impl_->mutable_data();
    }

    AM_NODISCARD const void* data() const noexcept {
        return impl_->data();
    }

    AM_NODISCARD Device device() const noexcept {
        return impl_->device();
    }

    AM_NODISCARD size_t alignment() const noexcept {
        return impl_->alignment();
    }

    AM_NODISCARD uint32_t use_count() const noexcept {
        return impl_->use_count();
    }

    AM_NODISCARD bool unique() const noexcept {
        return use_count() == 1;
    }

    AM_NODISCARD const ObjectPtr<BufferImpl>& impl() const noexcept {
        return impl_;
    }


private:
    ObjectPtr<BufferImpl> impl_;
};

class Storage : public ObjectRef {
public:
    Storage() = default;

    // Creates storage with pre-allocated memory buffer.
    Storage(size_t nbytes, DataPtr data_ptr, const std::unique_ptr<Allocator>& alloc)
        : impl_(make_object<StorageImpl>(nbytes, std::move(data_ptr), alloc)) {}

    // Allocates memory buffer using the given allocator and creates a storage with it
    Storage(size_t nbytes, const std::unique_ptr<Allocator>& alloc)
        : Storage(nbytes, alloc->allocate(nbytes), alloc) {}

    explicit Storage(ObjectPtr<StorageImpl> ptr) : impl_(std::move(ptr)) {}

    AM_NODISCARD bool defined() const {
        return impl_;
    }

    AM_NODISCARD size_t nbytes() const {
        return impl_->nbytes();
    }

    AM_NODISCARD void* data() const {
        return impl_->get();
    }

    AM_NODISCARD const void* const_data() const {
        return impl_->const_get();
    }

    AM_NODISCARD DataPtr& data_ptr() const {
        // CHECK(impl_ != nullptr) << "Storage is not initialized";
        return impl_->data_ptr();
    }

    AM_NODISCARD const DataPtr& const_data_ptr() const {
        // CHECK(impl_ != nullptr) << "Storage is not initialized";
        return impl_->data_ptr();
    }

    AM_NODISCARD Device device() const {
        // CHECK(impl_ != nullptr) << "Storage is not initialized";
        return impl_->device();
    }

    operator bool() const {
        return impl_;
    }

    AM_NODISCARD uint32_t use_count() const {
        return impl_.use_count();
    }

    AM_NODISCARD bool unique() const {
        return use_count() == 1;
    }

private:
    ObjectPtr<StorageImpl> impl_;
};

}// namespace aethermind

#endif// AETHERMIND_STORAGE_H
