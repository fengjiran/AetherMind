#ifndef AETHERMIND_RUNTIME_RUNTIME_H
#define AETHERMIND_RUNTIME_RUNTIME_H

/// @file runtime.h
/// @brief Long-lived owner of backend, allocator, and KV-cache resources.

#include "aethermind/backend/backend_registry.h"
#include "aethermind/kv_cache/kv_cache_manager.h"
#include "aethermind/memory/allocator.h"

namespace aethermind {

/// @brief Long-lived owner of backend, allocator, and KV-cache resources.
///
/// A Runtime must outlive every ExecutionPlan resolved through its backends.
/// In particular, ResolvedKernel::name borrows backend-owned stable storage.
/// It also must outlive sessions that borrow its KVCacheManager storage.
/// Instances are created via RuntimeBuilder. The Runtime is not thread-safe;
/// concurrent access must be serialized externally, consistent with the
/// registries and KVCacheManager it owns.
class Runtime {
public:
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    Runtime(Runtime&&) noexcept = default;
    Runtime& operator=(Runtime&&) noexcept = default;
    ~Runtime() = default;

    /// @brief Returns the allocator for a device, creating it on first use.
    ///
    /// @param device Device to get an allocator for.
    /// @return Reference to the cached allocator owned by the Runtime.
    /// @note Aborts when no allocator provider is registered for the device
    ///       type.
    /// @see HasAllocatorProvider
    Allocator& GetAllocator(Device device) {
        return allocator_registry_.GetAllocator(device);
    }

    /// @brief Returns whether Runtime has an allocator provider for a device.
    ///
    /// @param type Device type to check.
    /// @return True if a provider is registered for the device type.
    AM_NODISCARD bool HasAllocatorProvider(DeviceType type) const noexcept {
        return allocator_registry_.HasProvider(type);
    }

    /// @brief Returns the backend for a device type, creating it on first use.
    ///
    /// @param type Device type to look up.
    /// @return Borrowed backend pointer owned by the Runtime on success, or
    ///         an error when no factory is registered for the device type or
    ///         backend creation fails.
    /// @note Backends are created lazily via their factories and cached;
    ///       repeated calls return the same instance.
    StatusOr<Backend*> GetBackend(DeviceType type) noexcept {
        return backend_registry_.GetBackend(type);
    }

    /// @brief Returns the KV cache manager, or nullptr when it is disabled.
    ///
    /// @return Borrowed pointer owned by the Runtime, or nullptr when KV
    ///         caching is not enabled.
    AM_NODISCARD KVCacheManager* GetKVCacheManager() noexcept {
        return kv_cache_manager_.is_initialized() ? &kv_cache_manager_ : nullptr;
    }

    /// @brief Returns the KV cache manager, or nullptr when it is disabled.
    ///
    /// @return Borrowed pointer owned by the Runtime, or nullptr when KV
    ///         caching is not enabled.
    AM_NODISCARD const KVCacheManager* GetKVCacheManager() const noexcept {
        return kv_cache_manager_.is_initialized() ? &kv_cache_manager_ : nullptr;
    }

private:
    Runtime(AllocatorRegistry allocator_registry, BackendRegistry backend_registry,
            KVCacheManager kv_cache_manager)
        : allocator_registry_(std::move(allocator_registry)),
          backend_registry_(std::move(backend_registry)),
          kv_cache_manager_(std::move(kv_cache_manager)) {}

    AllocatorRegistry allocator_registry_;
    BackendRegistry backend_registry_;
    KVCacheManager kv_cache_manager_{};

    friend class RuntimeBuilder;
};


} // namespace aethermind
#endif
