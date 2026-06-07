#include "aethermind/backend/cpu/cpu_backend.h"

#include "utils/logging.h"

namespace aethermind {

CpuBackend::CpuBackend() {
    const Status status = KernelRegistry::Global().Freeze();
    AM_CHECK(status.ok(), "Failed to freeze CPU kernel registry: {}", status.ToString().c_str());
}

DeviceType CpuBackend::device_type() const noexcept {
    return DeviceType::kCPU;
}

const BackendCapabilities& CpuBackend::capabilities() const noexcept {
    return capabilities_.base;
}

KernelFunc CpuBackend::ResolveKernel(OpType op_type,
                                     const KernelSelector& selector) const noexcept {
    if (selector.device_type != DeviceType::kCPU) {
        return nullptr;
    }

    const StatusOr<const KernelDescriptor*> descriptor = KernelRegistry::Global().Resolve(op_type, selector);
    if (!descriptor.ok()) {
        return nullptr;
    }
    return (*descriptor)->kernel_func;
}

StatusOr<ResolvedKernel> CpuBackend::ResolveKernelInfo(
        OpType op_type,
        const KernelSelector& selector) const noexcept {
    if (selector.device_type != DeviceType::kCPU) {
        return Status::InvalidArgument("CpuBackend cannot resolve non-CPU kernel selector");
    }

    const StatusOr<const KernelDescriptor*> descriptor = KernelRegistry::Global().Resolve(op_type, selector);
    if (!descriptor.ok()) {
        return descriptor.status();
    }

    return ResolvedKernel{
            .op_type = op_type,
            .fn = (*descriptor)->kernel_func,
            .attrs = {},
            .debug_name = (*descriptor)->name.c_str(),
    };
}

const KernelRegistry* CpuBackend::TryGetKernelRegistryForDebug() const noexcept {
    return &KernelRegistry::Global();
}

DeviceType CpuBackendFactory::device_type() const noexcept {
    return DeviceType::kCPU;
}

std::unique_ptr<Backend> CpuBackendFactory::Create() const {
    return std::make_unique<CpuBackend>();
}

}// namespace aethermind
