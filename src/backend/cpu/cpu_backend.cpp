#include "aethermind/backend/cpu/cpu_backend.h"

#include "data_type.h"

namespace aethermind {
namespace {

Status FakeCpuKernel() noexcept {
    return Status::Ok();
}

KernelSelector MakeDefaultCpuSelector() {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .isa = IsaLevel::kScalar,
            .phase = ExecPhase::kBoth,
    };
}

}// namespace

CpuBackend::CpuBackend() {
    RegisterBuiltinKernels();
}

void CpuBackend::RegisterBuiltinKernels() {
    const Status status = kernel_registry_.Register(KernelDescriptor{
            .op_type = OpType::kRMSNorm,
            .selector = MakeDefaultCpuSelector(),
            .kernel_func = &FakeCpuKernel,
            .name = "test::fake_cpu_kernel",
            .priority = 10,
    });
    AM_CHECK(status.ok(), "Failed to register builtin CPU kernels: {}", status.ToString().c_str());
}

DeviceType CpuBackend::device_type() const noexcept {
    return DeviceType::kCPU;
}

const BackendCapabilities& CpuBackend::capabilities() const noexcept {
    return capabilities_.base;
}

KernelFunc CpuBackend::ResolveKernel(OpType op_type,
                                     const KernelSelector& selector) const noexcept {
    const KernelDescriptor* descriptor = nullptr;
    const Status status = kernel_registry_.Resolve(op_type, selector, &descriptor);
    if (!status.ok()) {
        return nullptr;
    }
    return descriptor->kernel_func;
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
