#ifndef AETHERMIND_MEMORY_HANDLE_H
#define AETHERMIND_MEMORY_HANDLE_H

/// @file memory_handle.h
/// @brief Move-only raw memory pointer carrying device, alignment, and optional
///        destruction ownership.
///
/// A handle either owns its memory (non-null deleter) or borrows it (null
/// deleter or NoOpMemoryDeleter); borrowed storage must outlive every handle
/// and view that references it.

#include "aethermind/base/device.h"

#include <cstddef>
#include <utility>

namespace aethermind {

/// @brief Type of the release callback stored in a MemoryHandle.
///
/// The callback receives the handle's context pointer and the memory pointer,
/// is invoked at most once per owned pointer, and must not throw.
using memory_deleter_fn = void (*)(void* deleter_ctx, void* ptr) noexcept;

/// @brief No-op deleter that marks a handle as non-owning (borrowed memory).
///
/// @param deleter_ctx Ignored.
/// @param ptr Ignored.
inline void NoOpMemoryDeleter(void* deleter_ctx, void* ptr) noexcept {
    UNUSED(deleter_ctx);
    UNUSED(ptr);
}

/// @brief Move-only memory pointer with optional destruction ownership.
///
/// A null deleter or NoOpMemoryDeleter represents borrowed memory: the caller
/// must keep the storage alive through every use of the handle and its views.
/// A non-null deleter is invoked once for a non-null pointer and must not throw.
///
/// Empty handles (default-constructed, moved-from, or reset) have a null
/// context and deleter, kUndefined device, and zero alignment.
///
/// Thread-safety: distinct handles are independent. Concurrent access to the
/// same handle, including move/reset/swap, requires external synchronization.
class MemoryHandle {
public:
    MemoryHandle() noexcept = default;

    /// @brief Adopts a pointer and its device/alignment metadata.
    /// @param data Memory pointer; null represents an empty handle.
    /// @param deleter_ctx Borrowed context that must remain valid until deletion.
    ///        It is not freed separately; any context cleanup belongs to deleter.
    /// @param deleter Optional noexcept callback; it must not borrow an allocator
    ///        or other resource that can be destroyed before the handle.
    /// @param device Device containing the memory.
    /// @param alignment Guaranteed pointer alignment, or zero when unknown.
    /// @note Fields are stored verbatim; the constructor validates nothing.
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

    /// @brief Returns true if the handle references memory (get() != nullptr).
    explicit operator bool() const noexcept {
        return data_ != nullptr;
    }

    /// @brief Clears this handle, then invokes its previous deleter on the
    ///        released pointer.
    ///
    /// The handle is cleared before the deleter runs, so a deleter that
    /// reenters reset() observes an empty handle. A replacement installed by
    /// the deleter during an explicit reset is preserved and stays owned by
    /// this handle; installing one while the destructor runs leaks it.
    void reset() noexcept {
        void* const data = std::exchange(data_, nullptr);
        void* const ctx = std::exchange(ctx_, nullptr);
        const memory_deleter_fn deleter = std::exchange(deleter_, nullptr);
        device_ = Device(kUndefined);
        alignment_ = 0;

        if (data != nullptr && deleter != nullptr) {
            deleter(ctx, data);
        }
    }

    /// @brief Exchanges all state, including memory ownership, with `other`.
    ///
    /// @param other Handle to swap with.
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

} // namespace aethermind

#endif // AETHERMIND_MEMORY_HANDLE_H
