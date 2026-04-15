#include "aethermind/backend/cpu/cpu_backend.h"

namespace aethermind {
namespace {

Status FakeCpuKernel() noexcept {
    return Status::Ok();
}

OperatorName MakeFakeKernelOperatorName() {
    return OperatorName("test::fake_cpu_kernel", "");
}

}// namespace

CpuBackend::CpuBackend() {
    RegisterBuiltinKernels();
}

void CpuBackend::RegisterBuiltinKernels() {
    kernel_registry_.Register(
            KernelKey{
                    .device_type = DeviceType::kCPU,
                    .op_name = MakeFakeKernelOperatorName(),
            },
            &FakeCpuKernel);
}

DeviceType CpuBackend::device_type() const noexcept {
    return DeviceType::kCPU;
}

const BackendCapabilities& CpuBackend::capabilities() const noexcept {
    return capabilities_.base;
}

KernelFunc CpuBackend::ResolveKernel(const KernelKey& key) const noexcept {
    return kernel_registry_.Find(key);
}

const KernelRegistry* CpuBackend::TryGetKernelRegistryForDebug() const noexcept {
    return &kernel_registry_;
}

DeviceType CpuBackendFactory::device_type() const noexcept {
    return DeviceType::kCPU;
}

std::unique_ptr<Backend> CpuBackendFactory::Create() const {
    return std::make_unique<CpuBackend>();
}

}// namespace aethermind
