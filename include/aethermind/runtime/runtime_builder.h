#ifndef AETHERMIND_RUNTIME_RUNTIME_BUILDER_H
#define AETHERMIND_RUNTIME_RUNTIME_BUILDER_H

#include "device.h"
#include "runtime_context.h"
#include "runtime_options.h"

#include <memory>
#include <vector>

namespace aethermind {

class RuntimeBuilder {
public:
    RuntimeBuilder& WithOptions(const RuntimeOptions& options);
    RuntimeBuilder& OverrideAllocatorProvider(DeviceType type,
                                              std::unique_ptr<AllocatorProvider> provider);
    RuntimeContext Build();

private:
    struct PendingAllocatorProvider {
        DeviceType type;
        std::unique_ptr<AllocatorProvider> provider;
    };

    struct AllocatorBuilderState {
        std::vector<PendingAllocatorProvider> pending_registrations;
    };

    RuntimeOptions options_;

    AllocatorBuilderState allocator_state_;

    AllocatorRegistry BuildAllocatorRegistry();
};


}// namespace aethermind

#endif
