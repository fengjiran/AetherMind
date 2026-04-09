#include "aethermind/runtime/runtime_context.h"
#include "aethermind/runtime/runtime_registration.h"

namespace aethermind {

RuntimeContext::RuntimeContext() {
    internal::RegisterAllocatorProviders(this);
}

Allocator& RuntimeContext::GetAllocator(Device device) {
    return registry_.GetAllocator(device);
}

}// namespace aethermind
