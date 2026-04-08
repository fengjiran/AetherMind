#include "aethermind/memory/allocator_registration.h"
#include "aethermind/memory/allocator.h"
#include "aethermind/memory/cpu_allocator.h"

namespace aethermind {
namespace internal {

void RegisterBuiltinAllocatorProviders(RuntimeContext* context) {
    RegisterCPUAllocatorProvider(context);
    RegisterCUDAAllocatorProvider(context);
    RegisterCANNAllocatorProvider(context);
}

void RegisterCPUAllocatorProvider(RuntimeContext* context) {
    context->RegisterAllocatorProvider(DeviceType::kCPU, std::make_unique<CPUAllocatorProvider>());
}

void RegisterCUDAAllocatorProvider(RuntimeContext*) {
}

void RegisterCANNAllocatorProvider(RuntimeContext*) {
}

}// namespace internal
}// namespace aethermind
