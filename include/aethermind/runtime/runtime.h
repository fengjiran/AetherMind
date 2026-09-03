#ifndef AETHERMIND_RUNTIME_RUNTIME_H
#define AETHERMIND_RUNTIME_RUNTIME_H

#include "aethermind/backend/backend_registry.h"
#include "aethermind/memory/allocator.h"
#include "aethermind/runtime/kv_cache_manager.h"

namespace aethermind {

/// @brief Long-lived owner of backend, allocator, and KV-cache resources.
///
/// A Runtime must outlive every ExecutionPlan resolved through its backends.
/// In particular, ResolvedKernel::name borrows backend-owned stable storage.
/// It also must outlive sessions that borrow its KVCacheManager storage.
class Runtime {
public:
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    Runtime(Runtime&&) noexcept = default;
    Runtime& operator=(Runtime&&) noexcept = default;
    ~Runtime() = default;

    Allocator& GetAllocator(Device device) {
        return allocator_registry_.GetAllocator(device);
    }

    StatusOr<Backend*> GetBackend(DeviceType type) noexcept {
        return backend_registry_.GetBackend(type);
    }

    AM_NODISCARD KVCacheManager* GetKVCacheManager() noexcept {
        return kv_cache_manager_.is_initialized() ? &kv_cache_manager_ : nullptr;
    }

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
