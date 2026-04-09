//
// Created by richard on 4/9/26.
//
#include "aethermind/runtime/runtime_registration.h"

#include "aethermind/memory/cpu_allocator.h"
#include "aethermind/runtime/runtime_context.h"

#include <memory>

namespace aethermind {
namespace internal {

void RegisterAllocatorProviders(RuntimeContext* context) {
    RegisterCPUAllocatorProvider(context);
    RegisterCUDAAllocatorProvider(context);
    RegisterCANNAllocatorProvider(context);
}

void RegisterCPUAllocatorProvider(RuntimeContext* context) {
    context->RegisterAllocatorProvider(DeviceType::kCPU,
                                       std::make_unique<CPUAllocatorProvider>());
}

void RegisterCUDAAllocatorProvider(RuntimeContext*) {
}

void RegisterCANNAllocatorProvider(RuntimeContext*) {
}

}// namespace internal
}// namespace aethermind
