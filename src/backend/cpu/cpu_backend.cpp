#include "aethermind/backend/cpu/cpu_backend.h"

namespace aethermind {

DeviceType CpuBackend::device_type() const noexcept {
    return DeviceType::kCPU;
}

const BackendCapabilities& CpuBackend::capabilities() const noexcept {
    return capabilities_.base;
}

KernelFn CpuBackend::ResolveKernel(const KernelKey&) const noexcept {
    return nullptr;
}

const KernelRegistry* CpuBackend::TryGetKernelRegistryForDebug() const noexcept {
    return nullptr;
}

DeviceType CpuBackendFactory::device_type() const noexcept {
    return DeviceType::kCPU;
}

std::unique_ptr<Backend> CpuBackendFactory::Create() const {
    return std::make_unique<CpuBackend>();
}

}// namespace aethermind
