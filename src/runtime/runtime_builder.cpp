#include "aethermind/runtime/runtime_builder.h"
#include "aethermind/runtime/runtime_options.h"
#include "aethermind/runtime/runtime_registration.h"

namespace aethermind {

RuntimeBuilder& RuntimeBuilder::WithOptions(RuntimeOptions options) {
    options_ = options;
    return *this;
}

RuntimeBuilder& RuntimeBuilder::RegisterAllocatorProvider(
        DeviceType type,
        std::unique_ptr<AllocatorProvider> provider) {
    AM_CHECK(provider != nullptr, "Allocator provider cannot be null");
    AM_CHECK(type != DeviceType::kUndefined,
             "Cannot register allocator provider for undefined device type");
    allocator_state_.pending_registrations.push_back(
            PendingAllocatorProviderRegistration{
                    .type = type,
                    .provider = std::move(provider)});
    return *this;
}

AllocatorRegistry RuntimeBuilder::BuildAllocatorRegistry() {
    AllocatorRegistry registry;
    internal::RegisterAllocatorProviders(registry, options_.allocator);
    for (auto& pending: allocator_state_.pending_registrations) {
        registry.SetProvider(pending.type, std::move(pending.provider));
    }
    return registry;
}

RuntimeContext RuntimeBuilder::Build() {
    return RuntimeContext(BuildAllocatorRegistry());
}

}// namespace aethermind
