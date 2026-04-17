#ifndef AETHERMIND_RUNTIME_RUNTIME_BUILDER_H
#define AETHERMIND_RUNTIME_RUNTIME_BUILDER_H

#include "aethermind/execution/kv_cache_manager.h"
#include "device.h"
#include "runtime_context.h"
#include "runtime_options.h"

#include <memory>
#include <vector>

namespace aethermind {

class RuntimeBuilder {
public:
    RuntimeBuilder& WithOptions(const RuntimeOptions& options);

    RuntimeBuilder& RegisterCustomAllocatorProvider(
            DeviceType type,
            std::unique_ptr<AllocatorProvider> provider);

    RuntimeBuilder& RegisterBackendFactory(
            DeviceType type,
            std::unique_ptr<BackendFactory> factory);

    RuntimeContext Build();

private:
    struct PendingCustomAllocatorProvider {
        DeviceType type;
        std::unique_ptr<AllocatorProvider> provider;
    };

    struct PendingBackendFactory {
        DeviceType type;
        std::unique_ptr<BackendFactory> factory;
    };

    RuntimeOptions options_;
    std::vector<PendingCustomAllocatorProvider> pending_custom_allocator_providers_;
    std::vector<PendingBackendFactory> pending_backend_factories_;

    AllocatorRegistry BuildAllocatorRegistry();
    BackendRegistry BuildBackendRegistry();
    KVCacheManager BuildKVCacheManager();
};


}// namespace aethermind

#endif
