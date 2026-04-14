#include "aethermind/runtime/runtime_builder.h"
#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/memory/cann_allocator.h"
#include "aethermind/memory/cpu_allocator.h"
#include "aethermind/memory/cuda_allocator.h"
#include "aethermind/runtime/runtime_options.h"

namespace aethermind {

RuntimeBuilder& RuntimeBuilder::WithOptions(const RuntimeOptions& options) {
    options_ = options;
    return *this;
}

RuntimeContext RuntimeBuilder::Build() {
    auto allocator_registry = BuildAllocatorRegistry();
    auto backend_registry = BuildBackendRegistry();
    return RuntimeContext(std::move(allocator_registry),
                          std::move(backend_registry));
}

RuntimeBuilder& RuntimeBuilder::RegisterCustomAllocatorProvider(
        DeviceType type,
        std::unique_ptr<AllocatorProvider> provider) {
    AM_CHECK(provider != nullptr, "Allocator provider cannot be null");
    AM_CHECK(type != DeviceType::kUndefined,
             "Cannot register allocator provider for undefined device type");
    pending_custom_allocator_providers_.push_back(
            PendingCustomAllocatorProvider{
                    .type = type,
                    .provider = std::move(provider)});
    return *this;
}

RuntimeBuilder& RuntimeBuilder::RegisterBackendFactory(
        DeviceType type,
        std::unique_ptr<BackendFactory> factory) {
    AM_CHECK(factory != nullptr, "Backend factory cannot be null");
    AM_CHECK(type != DeviceType::kUndefined,
             "Cannot register backend factory for undefined device type");
    pending_backend_factories_.push_back(
            PendingBackendFactory{
                    .type = type,
                    .factory = std::move(factory)});
    return *this;
}

AllocatorRegistry RuntimeBuilder::BuildAllocatorRegistry() {
    AllocatorRegistry registry;
    const auto& [enable_cpu, enable_cuda, enable_cann] = options_.allocator;

    if (enable_cpu) {
        registry.SetProvider(DeviceType::kCPU, std::make_unique<CPUAllocatorProvider>());
    }

    if (enable_cuda) {
        registry.SetProvider(DeviceType::kCUDA, std::make_unique<CUDAAllocatorProvider>());
    }

    if (enable_cann) {
        registry.SetProvider(DeviceType::kCANN, std::make_unique<CANNAllocatorProvider>());
    }

    for (auto& [type, provider]: pending_custom_allocator_providers_) {
        registry.SetProvider(type, std::move(provider));
    }
    pending_custom_allocator_providers_.clear();

    return registry;
}

BackendRegistry RuntimeBuilder::BuildBackendRegistry() {
    BackendRegistry registry;
    if (options_.backend.enable_cpu) {
        registry.SetFactory(DeviceType::kCPU, std::make_unique<CpuBackendFactory>());
    }

    for (auto& [type, factory]: pending_backend_factories_) {
        registry.SetFactory(type, std::move(factory));
    }
    pending_backend_factories_.clear();

    return registry;
}

}// namespace aethermind
