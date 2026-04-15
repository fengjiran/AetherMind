#ifndef AETHERMIND_BACKEND_CPU_CPU_BACKEND_H
#define AETHERMIND_BACKEND_CPU_CPU_BACKEND_H

#include "aethermind/backend/backend.h"
#include "aethermind/backend/backend_factory.h"
#include "aethermind/backend/cpu/cpu_capabilities.h"
#include "aethermind/backend/kernel_registry.h"
#include "aethermind/backend/kernel_selector.h"
#include "aethermind/operators/op_type.h"

namespace aethermind {

class CpuBackend final : public Backend {
public:
    CpuBackend();
    AM_NODISCARD DeviceType device_type() const noexcept override;
    AM_NODISCARD const BackendCapabilities& capabilities() const noexcept override;
    AM_NODISCARD KernelFunc ResolveKernel(OpType op_type,
                                          const KernelSelector& selector) const noexcept override;
    AM_NODISCARD const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override;

private:
    void RegisterBuiltinKernels();
    CpuCapabilities capabilities_{};
    KernelRegistry kernel_registry_{};
};

class CpuBackendFactory final : public BackendFactory {
public:
    AM_NODISCARD DeviceType device_type() const noexcept override;
    AM_NODISCARD std::unique_ptr<Backend> Create() const override;
};

}// namespace aethermind
#endif
