#ifndef AETHERMIND_ALLOCATOR_REGISTRATION_H
#define AETHERMIND_ALLOCATOR_REGISTRATION_H

namespace aethermind {

class RuntimeContext;

namespace internal {

void RegisterBuiltinAllocatorProviders(RuntimeContext* context);

void RegisterCPUAllocatorProvider(RuntimeContext* context);
void RegisterCUDAAllocatorProvider(RuntimeContext* context);
void RegisterCANNAllocatorProvider(RuntimeContext* context);

}// namespace internal
}// namespace aethermind

#endif// AETHERMIND_ALLOCATOR_REGISTRATION_H
