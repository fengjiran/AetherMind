//
// Created by richard on 4/17/26.
//

#ifndef AETHERMIND_MEMORY_HANDLE_H
#define AETHERMIND_MEMORY_HANDLE_H

#include "device.h"

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

}// namespace aethermind

#endif// AETHERMIND_MEMORY_HANDLE_H
