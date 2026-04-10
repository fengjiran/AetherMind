//
// Created by richard on 4/9/26.
//
#include "aethermind/runtime/runtime_registration.h"
#include "aethermind/memory/cann_allocator.h"
#include "aethermind/memory/cpu_allocator.h"
#include "aethermind/memory/cuda_allocator.h"

namespace aethermind {
namespace internal {

void RegisterAllocatorProviders(
        AllocatorRegistry& registry,
        const AllocatorRuntimeOptions& options) {
    if (options.enable_cpu) {
        RegisterCPUAllocatorProvider(registry);
    }

    if (options.enable_cuda) {
        RegisterCUDAAllocatorProvider(registry);
    }

    if (options.enable_cann) {
        RegisterCANNAllocatorProvider(registry);
    }
}

void RegisterCPUAllocatorProvider(AllocatorRegistry& registry) {
    registry.SetProvider(DeviceType::kCPU, std::make_unique<CPUAllocatorProvider>());
}

void RegisterCUDAAllocatorProvider(AllocatorRegistry& registry) {
    registry.SetProvider(DeviceType::kCUDA, std::make_unique<CUDAAllocatorProvider>());
}

void RegisterCANNAllocatorProvider(AllocatorRegistry& registry) {
    registry.SetProvider(DeviceType::kCANN, std::make_unique<CANNAllocatorProvider>());
}

}// namespace internal
}// namespace aethermind
