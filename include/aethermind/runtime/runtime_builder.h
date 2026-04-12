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
    RuntimeBuilder& RegisterCustomAllocatorProvider(DeviceType type,
                                              std::unique_ptr<AllocatorProvider> provider);
    RuntimeContext Build();

private:
    struct PendingCustomAllocatorProvider {
        DeviceType type;
        std::unique_ptr<AllocatorProvider> provider;
    };

    RuntimeOptions options_;
    std::vector<PendingCustomAllocatorProvider> pending_custom_allocator_providers_;

    AllocatorRegistry BuildAllocatorRegistry();
};


}// namespace aethermind

#endif
