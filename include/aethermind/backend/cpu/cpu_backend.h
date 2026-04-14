#ifndef AETHERMIND_BACKEND_CPU_CPU_BACKEND_H
#define AETHERMIND_BACKEND_CPU_CPU_BACKEND_H

#include "aethermind/backend/backend.h"
#include "aethermind/backend/backend_factory.h"
#include "aethermind/backend/cpu/cpu_capabilities.h"

namespace aethermind {

class CpuBackend final : public Backend {
public:
    CpuBackend() = default;
    AM_NODISCARD DeviceType device_type() const noexcept override;
    AM_NODISCARD const BackendCapabilities& capabilities() const noexcept override;
    AM_NODISCARD KernelFn ResolveKernel(const KernelKey& key) const noexcept override;
    AM_NODISCARD const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override;

private:
    CpuCapabilities capabilities_{};
};

class CpuBackendFactory final : public BackendFactory {
public:
    AM_NODISCARD DeviceType device_type() const noexcept override;
    AM_NODISCARD std::unique_ptr<Backend> Create() const override;
};

}// namespace aethermind
#endif
