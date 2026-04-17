#ifndef AETHERMIND_RUNTIME_RUNTIME_CONTEXT_H
#define AETHERMIND_RUNTIME_RUNTIME_CONTEXT_H

#include "aethermind/backend/backend_registry.h"
#include "aethermind/execution/kv_cache_manager.h"
#include "aethermind/memory/allocator.h"

namespace aethermind {

class RuntimeContext {
public:
    Allocator& GetAllocator(Device device);
    StatusOr<Backend*> GetBackend(DeviceType type);
    AM_NODISCARD KVCacheManager* GetKVCacheManager() noexcept;
    AM_NODISCARD const KVCacheManager* GetKVCacheManager() const noexcept;

    RuntimeContext(const RuntimeContext&) = delete;
    RuntimeContext& operator=(const RuntimeContext&) = delete;
    RuntimeContext(RuntimeContext&&) noexcept = default;
    RuntimeContext& operator=(RuntimeContext&&) noexcept = default;
    ~RuntimeContext() = default;

private:
    explicit RuntimeContext(AllocatorRegistry allocator_registry,
                            BackendRegistry backend_registry,
                            KVCacheManager kv_cache_manager);

    AllocatorRegistry allocator_registry_;
    BackendRegistry backend_registry_;
    KVCacheManager kv_cache_manager_{};

    friend class RuntimeBuilder;
};


}// namespace aethermind
#endif
