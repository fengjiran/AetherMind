#ifndef AETHERMIND_RUNTIME_RUNTIME_BUILDER_H
#define AETHERMIND_RUNTIME_RUNTIME_BUILDER_H

/// @file runtime_builder.h
/// @brief Assembles a Runtime from options and custom provider registrations.

#include "aethermind/base/device.h"
#include "aethermind/base/status.h"
#include "aethermind/kv_cache/kv_cache_manager.h"
#include "runtime.h"
#include "runtime_options.h"

#include <memory>
#include <vector>

namespace aethermind {

/// @brief Assembles a Runtime from RuntimeOptions and custom registrations.
///
/// Default allocators and backends are derived from RuntimeOptions. Custom
/// allocator providers and backend factories added before Build() take
/// precedence over the defaults for the same device type. The builder is
/// not thread-safe.
class RuntimeBuilder {
public:
    /// @brief Replaces the options used by subsequent Build() calls.
    ///
    /// @param options Options replacing the builder's current configuration.
    /// @return `*this` for chaining.
    RuntimeBuilder& WithOptions(const RuntimeOptions& options);

    /// @brief Registers an allocator provider that overrides the default one.
    ///
    /// @param type Device type to register. Must not be `kUndefined`.
    /// @param provider Provider to register. Must be non-null. Ownership is
    ///        transferred to the builder.
    /// @return `*this` for chaining.
    /// @note Aborts when a precondition is violated.
    RuntimeBuilder& RegisterCustomAllocatorProvider(
            DeviceType type,
            std::unique_ptr<AllocatorProvider> provider);

    /// @brief Registers a backend factory that overrides the default one.
    ///
    /// @param type Device type to register. Must not be `kUndefined`.
    /// @param factory Factory to register. Must be non-null. Ownership is
    ///        transferred to the builder.
    /// @return `*this` for chaining.
    /// @note Aborts when a precondition is violated.
    RuntimeBuilder& RegisterBackendFactory(
            DeviceType type,
            std::unique_ptr<BackendFactory> factory);

    /// @brief Builds a Runtime from the current options and registrations.
    ///
    /// Initializes the KV cache manager when KV caching is enabled in the
    /// options.
    ///
    /// @return The assembled Runtime, which owns all created resources.
    /// @note Consumes the pending custom registrations; a later Build() reuses
    ///       only the current options and registrations added after this call.
    ///       Aborts when initialization fails, including on allocation failure.
    ///       Prefer TryBuild() when the failure must be recoverable.
    Runtime Build();

    /// @brief Builds a Runtime, reporting failures instead of aborting.
    ///
    /// Initializes the KV cache manager when KV caching is enabled in the
    /// options. A KV cache initialization failure and any allocation failure
    /// during registry assembly are returned to the caller; allocation failure
    /// reports ResourceExhausted with an empty message. Partially built
    /// registries are discarded.
    ///
    /// @return The assembled Runtime, or the first initialization failure.
    /// @note On success this consumes the pending custom registrations, as
    ///       Build() does. On failure the builder stays reusable: every pending
    ///       list is left holding only registrations that were never
    ///       transferred. A registration already moved into a discarded
    ///       registry is unrecoverable, so a retry proceeds without it.
    AM_NODISCARD StatusOr<Runtime> TryBuild();

private:
    struct PendingCustomAllocatorProvider {
        DeviceType type;
        std::unique_ptr<AllocatorProvider> provider;
    };

    struct PendingBackendFactory {
        DeviceType type;
        std::unique_ptr<BackendFactory> factory;
    };

    RuntimeOptions options_;
    std::vector<PendingCustomAllocatorProvider> pending_custom_allocator_providers_;
    std::vector<PendingBackendFactory> pending_backend_factories_;

    AllocatorRegistry BuildAllocatorRegistry();
    BackendRegistry BuildBackendRegistry();
    /// @brief Initializes the KV cache manager, allocating its planes through
    ///        `allocator_registry` so they charge the Runtime's ledger.
    StatusOr<KVCacheManager> TryBuildKVCacheManager(AllocatorRegistry& allocator_registry);
};


} // namespace aethermind

#endif
