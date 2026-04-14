#ifndef AETHERMIND_RUNTIME_RUNTIME_CONTEXT_H
#define AETHERMIND_RUNTIME_RUNTIME_CONTEXT_H

#include "aethermind/backend/backend_registry.h"
#include "aethermind/memory/allocator.h"

namespace aethermind {

class RuntimeContext {
public:
    Allocator& GetAllocator(Device device);
    StatusOr<Backend*> GetBackend(DeviceType type);

    RuntimeContext(const RuntimeContext&) = delete;
    RuntimeContext& operator=(const RuntimeContext&) = delete;
    RuntimeContext(RuntimeContext&&) noexcept = default;
    RuntimeContext& operator=(RuntimeContext&&) noexcept = default;
    ~RuntimeContext() = default;

private:
    explicit RuntimeContext(AllocatorRegistry allocator_registry,
                            BackendRegistry backend_registry);

    AllocatorRegistry allocator_registry_;
    BackendRegistry backend_registry_;

    friend class RuntimeBuilder;
};


}// namespace aethermind
#endif
