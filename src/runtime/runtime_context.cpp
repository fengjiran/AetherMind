#include "aethermind/runtime/runtime_context.h"
#include "aethermind/runtime/runtime_builder.h"

namespace aethermind {

RuntimeContext::RuntimeContext(AllocatorRegistry registry)
    : allocator_registry_(std::move(registry)) {
}

Allocator& RuntimeContext::GetAllocator(Device device) {
    return allocator_registry_.GetAllocator(device);
}

}// namespace aethermind
