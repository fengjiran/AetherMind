#include "aethermind/runtime/runtime_builder.h"
#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/memory/cann_allocator.h"
#include "aethermind/memory/cpu_allocator.h"
#include "aethermind/memory/cuda_allocator.h"
#include "aethermind/runtime/runtime_options.h"

#include <new>

namespace aethermind {

RuntimeBuilder& RuntimeBuilder::WithOptions(const RuntimeOptions& options) {
    options_ = options;
    return *this;
}

Runtime RuntimeBuilder::Build() {
    auto runtime = TryBuild();
    AM_CHECK(runtime.ok(), "Failed to build Runtime: {}",
             runtime.status().ToString().c_str());
    return std::move(runtime).value();
}

StatusOr<Runtime> RuntimeBuilder::TryBuild() try {
    auto allocator_registry = BuildAllocatorRegistry();
    auto backend_registry = BuildBackendRegistry();
    AM_ASSIGN_OR_RETURN(auto kv_cache_manager,
                        TryBuildKVCacheManager(allocator_registry));
    return Runtime{std::move(allocator_registry), std::move(backend_registry),
                   std::move(kv_cache_manager)};
} catch (const std::bad_alloc&) {
    // Registry node inserts and the pending-registration transfers allocate.
    // An empty message avoids allocating again while reporting OOM.
    return Status::ResourceExhausted({});
}

RuntimeBuilder& RuntimeBuilder::RegisterCustomAllocatorProvider(
        DeviceType type,
        std::unique_ptr<AllocatorProvider> provider) {
    AM_CHECK(provider != nullptr, "Allocator provider cannot be null");
    AM_CHECK(type != DeviceType::kUndefined,
             "Cannot register allocator provider for undefined device type");
    pending_custom_allocator_providers_.push_back({.type = type,
                                                   .provider = std::move(provider)});
    return *this;
}

RuntimeBuilder& RuntimeBuilder::RegisterBackendFactory(
        DeviceType type,
        std::unique_ptr<BackendFactory> factory) {
    AM_CHECK(factory != nullptr, "Backend factory cannot be null");
    AM_CHECK(type != DeviceType::kUndefined,
             "Cannot register backend factory for undefined device type");
    pending_backend_factories_.push_back({.type = type,
                                          .factory = std::move(factory)});
    return *this;
}

AllocatorRegistry RuntimeBuilder::BuildAllocatorRegistry() {
    AllocatorRegistry registry;
    const auto& [enable_cpu, enable_cuda, enable_cann] = options_.allocator;

    if (enable_cpu) {
        registry.SetProvider(DeviceType::kCPU,
                             std::make_unique<CPUAllocatorProvider>());
    }

    if (enable_cuda) {
        registry.SetProvider(DeviceType::kCUDA,
                             std::make_unique<CUDAAllocatorProvider>());
    }

    if (enable_cann) {
        registry.SetProvider(DeviceType::kCANN,
                             std::make_unique<CANNAllocatorProvider>());
    }

    // Pop before transferring: SetProvider allocates a registry node and can
    // throw, and a moved-from entry left in the pending list would trip its
    // non-null check on a later build. Entries already handed to the registry
    // are unrecoverable if a later step discards it.
    while (!pending_custom_allocator_providers_.empty()) {
        auto entry = std::move(pending_custom_allocator_providers_.front());
        pending_custom_allocator_providers_.erase(
                pending_custom_allocator_providers_.begin());
        registry.SetProvider(entry.type, std::move(entry.provider));
    }

    return registry;
}

BackendRegistry RuntimeBuilder::BuildBackendRegistry() {
    BackendRegistry registry;
    if (options_.backend.enable_cpu) {
        registry.SetFactory(DeviceType::kCPU,
                            std::make_unique<CpuBackendFactory>(
                                    options_.backend.cpu_feature_policy));
    }

    // Pop before transferring, as in BuildAllocatorRegistry.
    while (!pending_backend_factories_.empty()) {
        auto entry = std::move(pending_backend_factories_.front());
        pending_backend_factories_.erase(pending_backend_factories_.begin());
        registry.SetFactory(entry.type, std::move(entry.factory));
    }

    return registry;
}

StatusOr<KVCacheManager> RuntimeBuilder::TryBuildKVCacheManager(
        AllocatorRegistry& allocator_registry) {
    KVCacheManager manager;
    const KVCacheRuntimeOptions& options = options_.kv_cache;
    if (!options.enable_manager) {
        return manager;
    }

    // KV planes are charged to the Runtime ledger, so a CPU allocator must
    // exist before the manager can be initialized. Reported rather than
    // aborted: this is a configuration error the caller can act on.
    AM_ASSIGN_OR_RETURN(Allocator * allocator,
                        allocator_registry.TryGetAllocator(Device::CPU()));

    // Propagated verbatim: Init reports ResourceExhausted on allocation
    // failure, and formatting a new message would allocate while reporting it.
    AM_RETURN_IF_ERROR(manager.Init(*allocator,
                                    options.num_layers,
                                    options.num_kv_heads,
                                    options.max_tokens,
                                    options.head_dim,
                                    options.kv_dtype,
                                    options.alignment));
    return manager;
}

} // namespace aethermind
