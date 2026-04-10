#ifndef AETHERMIND_RUNTIME_RUNTIME_BUILDER_H
#define AETHERMIND_RUNTIME_RUNTIME_BUILDER_H

#include "device.h"
#include "runtime_context.h"
#include "runtime_options.h"

#include <memory>
#include <vector>

namespace aethermind {

struct PendingAllocatorProviderRegistration {
    DeviceType type;
    std::unique_ptr<AllocatorProvider> provider;
};

struct AllocatorBuilderState {
    std::vector<PendingAllocatorProviderRegistration> pending_registrations;
};

class RuntimeBuilder {
public:
    RuntimeBuilder& WithOptions(RuntimeOptions options);
    RuntimeBuilder& RegisterAllocatorProvider(DeviceType type,
                                              std::unique_ptr<AllocatorProvider> provider);
    RuntimeContext Build();

private:
    RuntimeOptions options_;
    AllocatorBuilderState allocator_state_;

    AllocatorRegistry BuildAllocatorRegistry();
};


}// namespace aethermind

#endif
