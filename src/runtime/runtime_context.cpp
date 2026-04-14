#include "aethermind/runtime/runtime_context.h"
#include "aethermind/backend/backend_fwd.h"

namespace aethermind {

RuntimeContext::RuntimeContext(AllocatorRegistry allocator_registry,
                               BackendRegistry backend_registry)
    : allocator_registry_(std::move(allocator_registry)),
      backend_registry_(std::move(backend_registry)) {}

Allocator& RuntimeContext::GetAllocator(Device device) {
    return allocator_registry_.GetAllocator(device);
}

StatusOr<Backend*> RuntimeContext::GetBackend(DeviceType type) {
    return backend_registry_.GetBackend(type);
}

}// namespace aethermind
