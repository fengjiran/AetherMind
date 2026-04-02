//
// Created by 赵丹 on 25-6-27.
//

#ifndef AETHERMIND_STORAGE_H
#define AETHERMIND_STORAGE_H

#include "macros.h"
#include "memory/data_ptr.h"
#include "memory/storage_impl.h"
#include "object.h"
#include "object_allocator.h"
#include <cstddef>
#include <cstdint>
#include <utility>

namespace aethermind {

class BufferImpl : public Object {
public:
    BufferImpl() noexcept = default;

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
    MemoryHandle handle_{};
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
