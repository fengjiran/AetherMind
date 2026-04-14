#ifndef AETHERMIND_BACKEND_CPU_CPU_BACKEND_H
#define AETHERMIND_BACKEND_CPU_CPU_BACKEND_H

#include "aethermind/backend/backend.h"
#include "aethermind/backend/backend_factory.h"

namespace aethermind {

class CpuBackend final : public Backend {
public:
    CpuBackend();
    DeviceType device_type() const noexcept override;
    const BackendCapabilities& capabilities() const noexcept override;
    KernelFn ResolveKernel(const KernelKey& key) const noexcept override;
    const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override;

private:
    BackendCapabilities capabilities_{};
};


class CpuBackendFactory final : public BackendFactory {
public:
    DeviceType device_type() const noexcept override;
    std::unique_ptr<Backend> Create() const override;
};

}// namespace aethermind
#endif
