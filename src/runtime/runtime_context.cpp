#include "aethermind/runtime/runtime_context.h"
#include "aethermind/backend/backend_fwd.h"

namespace aethermind {

RuntimeContext::RuntimeContext(AllocatorRegistry allocator_registry,
                               BackendRegistry backend_registry,
                               KVCacheManager kv_cache_manager)
    : allocator_registry_(std::move(allocator_registry)),
      backend_registry_(std::move(backend_registry)),
      kv_cache_manager_(std::move(kv_cache_manager)) {}

Allocator& RuntimeContext::GetAllocator(Device device) {
    return allocator_registry_.GetAllocator(device);
}

StatusOr<Backend*> RuntimeContext::GetBackend(DeviceType type) {
    return backend_registry_.GetBackend(type);
}

KVCacheManager* RuntimeContext::GetKVCacheManager() noexcept {
    return kv_cache_manager_.is_initialized() ? &kv_cache_manager_ : nullptr;
}

const KVCacheManager* RuntimeContext::GetKVCacheManager() const noexcept {
    return kv_cache_manager_.is_initialized() ? &kv_cache_manager_ : nullptr;
}

}// namespace aethermind
