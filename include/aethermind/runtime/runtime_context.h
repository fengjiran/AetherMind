#ifndef AETHERMIND_RUNTIME_RUNTIME_CONTEXT_H
#define AETHERMIND_RUNTIME_RUNTIME_CONTEXT_H

#include "aethermind/memory/allocator.h"

namespace aethermind {

class RuntimeContext {
public:
    RuntimeContext();

    // Registers an allocator provider for a specific device type.
    // RuntimeContext takes ownership of the provider.
    void RegisterAllocatorProvider(DeviceType type, std::unique_ptr<AllocatorProvider> provider) {
        registry_.RegisterProvider(type, std::move(provider));
    }

    Allocator& GetAllocator(Device device);

private:
    AllocatorRegistry registry_;
};


}// namespace aethermind
#endif
