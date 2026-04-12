#ifndef AETHERMIND_RUNTIME_RUNTIME_CONTEXT_H
#define AETHERMIND_RUNTIME_RUNTIME_CONTEXT_H

#include "aethermind/memory/allocator.h"

namespace aethermind {

class RuntimeContext {
public:
    Allocator& GetAllocator(Device device);

    RuntimeContext(const RuntimeContext&) = delete;
    RuntimeContext& operator=(const RuntimeContext&) = delete;

private:
    explicit RuntimeContext(AllocatorRegistry registry);

    AllocatorRegistry allocator_registry_;

    friend class RuntimeBuilder;
};


}// namespace aethermind
#endif
