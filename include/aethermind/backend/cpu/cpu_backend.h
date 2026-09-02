#ifndef AETHERMIND_BACKEND_CPU_CPU_BACKEND_H
#define AETHERMIND_BACKEND_CPU_CPU_BACKEND_H

/// @file cpu_backend.h
/// @brief CPU backend and its factory for kernel resolution.
///
/// Provides `CpuBackend`, which resolves kernels against the process-wide CPU
/// capability snapshot and `CpuBackendFactory`, which detects capabilities
/// through the status channel before constructing a backend instance.

#include "aethermind/backend/backend.h"
#include "aethermind/backend/backend_factory.h"
#include "aethermind/backend/cpu/cpu_capabilities.h"
#include "aethermind/backend/kernel_registry.h"

namespace aethermind {

/// @brief CPU backend that resolves kernels using the effective CPU feature set.
///
/// Holds an immutable `CpuCapabilities` snapshot captured at construction.
/// `PrepareKernel` filters registry candidates by `effective_features` and
/// priority, then freezes the selected kernel into a `ResolvedKernel`.
class CpuBackend final : public Backend {
public:
    /// @brief Constructs a backend with the default feature policy.
    CpuBackend();

    /// @brief Constructs a backend from an already-detected capability snapshot.
    ///
    /// @param capabilities Immutable snapshot to use for kernel selection.
    ///        The caller must ensure it was produced by `DetectCpuCapabilities`
    ///        and already has `effective_features` derived.
    explicit CpuBackend(const CpuCapabilities& capabilities);

    /// @brief Constructs a backend by detecting capabilities under a policy.
    ///
    /// Detection failures abort the process; supported platforms never take
    /// this path. Prefer `CpuBackendFactory` when a `Status` error is required.
    ///
    /// @param policy Feature policy that may only restrict usable features.
    explicit CpuBackend(const CpuFeaturePolicy& policy);

    /// @brief Returns the device type handled by this backend.
    ///
    /// @return `DeviceType::kCPU`.
    AM_NODISCARD DeviceType device_type() const noexcept override {
        return DeviceType::kCPU;
    }

    /// @brief Resolves a CPU kernel for the given operator and selector.
    ///
    /// Filters registry candidates by `effective_features` and priority, then
    /// freezes params and metadata builders into the returned kernel.
    ///
    /// @param op_type Operator type to resolve.
    /// @param selector Selector describing device, dtype, and layout constraints.
    ///        Must have `device_type == kCPU`.
    /// @param params Typed operator parameters for metadata derivation.
    /// @return Resolved kernel on success, or `NotFound`/`InvalidArgument`
    ///         when no eligible kernel exists or the selector is incompatible.
    AM_NODISCARD StatusOr<ResolvedKernel> PrepareKernel(
            OpType op_type,
            const KernelSelector& selector,
            const OpParams& params) const override;

    /// @brief Returns the global kernel registry for debug inspection.
    ///
    /// @return Non-null pointer to the global `KernelRegistry`.
    AM_NODISCARD const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override {
        return &KernelRegistry::Global();
    }

    /// @brief Returns the capability snapshot used for kernel selection.
    ///
    /// @return Reference to the immutable `CpuCapabilities` held by this backend.
    AM_NODISCARD const CpuCapabilities& cpu_capabilities() const noexcept {
        return capabilities_;
    }

private:
    const CpuCapabilities capabilities_;
};

/// @brief Factory that creates `CpuBackend` instances via capability detection.
///
/// Detection is performed through the `Status` channel so failures can be
/// propagated to `BackendRegistry` callers, unlike the aborting `CpuBackend`
/// constructors.
class CpuBackendFactory final : public BackendFactory {
public:
    /// @brief Constructs a factory with a feature policy.
    ///
    /// @param policy Feature policy forwarded to `DetectCpuCapabilities` on
    ///        each `Create` call. An empty policy exposes the full usable set.
    explicit CpuBackendFactory(const CpuFeaturePolicy& policy = {})
        : policy_(policy) {}

    /// @brief Returns the device type handled by this factory.
    ///
    /// @return `DeviceType::kCPU`.
    AM_NODISCARD DeviceType device_type() const noexcept override {
        return DeviceType::kCPU;
    }

    /// @brief Creates a CPU backend instance.
    ///
    /// Detects CPU capabilities under the stored policy and constructs a
    /// `CpuBackend`. Errors from detection are returned as `Status`.
    ///
    /// @return Backend instance on success, or an error status on detection
    ///         failure.
    AM_NODISCARD StatusOr<std::unique_ptr<Backend>> Create() const override;

private:
    CpuFeaturePolicy policy_{};
};

} // namespace aethermind
#endif
