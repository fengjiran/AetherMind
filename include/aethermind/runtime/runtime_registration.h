#ifndef AETHERMIND_RUNTIME_RUNTIME_REGISTRATION_H
#define AETHERMIND_RUNTIME_RUNTIME_REGISTRATION_H

namespace aethermind {

class RuntimeContext;

namespace internal {

void RegisterAllocatorProviders(RuntimeContext* context);

void RegisterCPUAllocatorProvider(RuntimeContext* context);
void RegisterCUDAAllocatorProvider(RuntimeContext* context);
void RegisterCANNAllocatorProvider(RuntimeContext* context);

}// namespace internal
}// namespace aethermind

#endif
