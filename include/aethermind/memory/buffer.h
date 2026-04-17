//
// Created by 赵丹 on 25-6-27.
//
#ifndef AETHERMIND_MEMORY_BUFFER_H
#define AETHERMIND_MEMORY_BUFFER_H

#include "macros.h"
#include "memory_handle.h"
#include "object.h"
#include "object_allocator.h"

#include <cstddef>
#include <cstdint>
#include <utility>

namespace aethermind {

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
        return is_initialized() && impl_->unique();
    }

    AM_NODISCARD const ObjectPtr<BufferImpl>& impl() const noexcept {
        return impl_;
    }

private:
    ObjectPtr<BufferImpl> impl_;
};

}// namespace aethermind

#endif// AETHERMIND_MEMORY_BUFFER_H
