#ifndef AETHERMIND_MEMORY_BUFFER_H
#define AETHERMIND_MEMORY_BUFFER_H

#include "aethermind/base/macros.h"
#include "aethermind/base/object.h"
#include "aethermind/base/object_allocator.h"
#include "memory_handle.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace aethermind {

class BufferImpl : public Object {
public:
    BufferImpl() noexcept = default;

    ~BufferImpl() override = default;

    BufferImpl(size_t nbytes, MemoryHandle handle) noexcept
        : nbytes_(nbytes), handle_(std::move(handle)) {
        AM_CHECK(nbytes_ == 0 || handle_, "Non-empty Buffer requires a memory pointer");
        const DeviceType device_type = handle_.device().type();
        AM_CHECK(!handle_ || device_type == kCPU || device_type == kCUDA || device_type == kCANN,
                 "Initialized Buffer requires a valid device type");
        const size_t alignment = handle_.alignment();
        AM_CHECK(alignment == 0 || std::has_single_bit(alignment),
                 "Buffer alignment must be zero or a power of two");
        AM_CHECK(!handle_ || device_type != kCPU || alignment == 0 ||
                         reinterpret_cast<uintptr_t>(handle_.get()) % alignment == 0,
                 "CPU Buffer pointer does not satisfy its declared alignment");
    }

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

/// @brief Shared raw storage whose copies share both metadata and memory.
///
/// Writes through mutable_data() are visible to all copies. Const Buffer access
/// does not make the shared memory immutable. Default and moved-from buffers
/// are uninitialized; a non-null handle with nbytes() == 0 is initialized.
///
/// Distinct Buffer copies may manage their references concurrently. Access to
/// the same Buffer object and shared memory requires external synchronization
/// whenever a concurrent operation writes that object or memory.
class Buffer : public ObjectRef {
public:
    Buffer() noexcept = default;

    /// @brief Shares a pre-created implementation object.
    /// @param impl Implementation with validated metadata, or an empty pointer.
    explicit Buffer(ObjectPtr<BufferImpl> impl) noexcept
        : impl_(std::move(impl)) {}

    /// @brief Adopts a handle and allocates shared implementation metadata.
    /// @param nbytes Accessible byte count from the handle's base pointer.
    /// @param handle Memory ownership or borrowing contract to transfer.
    /// @pre Nonzero nbytes requires a non-null pointer; non-null pointers require
    ///      a valid device. Alignment is zero (unknown) or a power of two; CPU
    ///      pointers must satisfy it. The caller guarantees accessible capacity.
    /// @throws std::bad_alloc if implementation allocation fails. The transferred
    ///         handle is destroyed on failure, invoking its deleter if present.
    Buffer(size_t nbytes, MemoryHandle handle)
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

    /// @brief Observes implementation strong references, excluding raw aliases.
    /// @return A reference-count snapshot, or zero without an implementation.
    AM_NODISCARD uint32_t use_count() const noexcept {
        return impl_->use_count();
    }

    /// @brief Tests whether an initialized implementation has one strong owner.
    /// @return True for one strong owner, including borrowed memory. This does
    ///         not prove exclusive memory access or grant concurrent write access.
    AM_NODISCARD bool unique() const noexcept {
        return is_initialized() && impl_->unique();
    }

    /// @brief Borrows a read-only implementation without extending its lifetime.
    /// @return Implementation pointer, or null for default/moved-from Buffers.
    ///         It remains valid while a strong implementation owner survives.
    AM_NODISCARD const BufferImpl* impl() const noexcept {
        return impl_.get_or_null();
    }

private:
    ObjectPtr<BufferImpl> impl_;
};

} // namespace aethermind

#endif // AETHERMIND_MEMORY_BUFFER_H
