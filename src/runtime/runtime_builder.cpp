#include "aethermind/runtime/runtime_builder.h"
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
    return RuntimeContext(std::move(allocator_registry));
}

RuntimeBuilder& RuntimeBuilder::OverrideAllocatorProvider(
        DeviceType type,
        std::unique_ptr<AllocatorProvider> provider) {
    AM_CHECK(provider != nullptr, "Allocator provider cannot be null");
    AM_CHECK(type != DeviceType::kUndefined,
             "Cannot register allocator provider for undefined device type");
    allocator_state_.pending_registrations.push_back(
            PendingAllocatorProvider{
                    .type = type,
                    .provider = std::move(provider)});
    return *this;
}

AllocatorRegistry RuntimeBuilder::BuildAllocatorRegistry() {
    AllocatorRegistry registry;
    const auto& allocator_opts = options_.allocator;
    if (allocator_opts.enable_cpu) {
        registry.SetProvider(DeviceType::kCPU, std::make_unique<CPUAllocatorProvider>());
    }

    if (allocator_opts.enable_cuda) {
        registry.SetProvider(DeviceType::kCUDA, std::make_unique<CUDAAllocatorProvider>());
    }

    if (allocator_opts.enable_cann) {
        registry.SetProvider(DeviceType::kCANN, std::make_unique<CANNAllocatorProvider>());
    }

    for (auto& [type, provider]: allocator_state_.pending_registrations) {
        registry.SetProvider(type, std::move(provider));
    }

    return registry;
}

}// namespace aethermind
