#ifndef AETHERMIND_RUNTIME_RUNTIME_REGISTRATION_H
#define AETHERMIND_RUNTIME_RUNTIME_REGISTRATION_H

#include "aethermind/memory/allocator.h"
#include "runtime_options.h"

namespace aethermind {

class RuntimeContext;

namespace internal {

void RegisterAllocatorProviders(
        AllocatorRegistry& registry,
        const AllocatorRuntimeOptions& options);

void RegisterCPUAllocatorProvider(AllocatorRegistry& registry);
void RegisterCUDAAllocatorProvider(AllocatorRegistry& registry);
void RegisterCANNAllocatorProvider(AllocatorRegistry& registry);

}// namespace internal
}// namespace aethermind

#endif
