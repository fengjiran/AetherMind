#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_invocation.h"
#include "aethermind/backend/op_kernel_context.h"
#include "aethermind/backend/workspace_types.h"
#include "data_type.h"

namespace aethermind {

namespace {

Status FakeCpuKernel(const KernelInvocation& invocation,
                     const OpKernelContext& op_ctx,
                     const WorkspaceBinding& workspace) noexcept {
    if (invocation.op_type != OpType::kRMSNorm) {
        return Status::InvalidArgument("CPU fake kernel expected RMSNorm invocation");
    }
    if (!op_ctx.device.is_cpu()) {
        return Status::InvalidArgument("CPU fake kernel expected CPU OpKernelContext device");
    }
    if (workspace.size == 0 && workspace.data != nullptr) {
        return Status::InvalidArgument("WorkspaceBinding with zero size must not carry data");
    }
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
    const Status freeze_status = kernel_registry_.Freeze();
    AM_CHECK(freeze_status.ok(),
             "Failed to freeze CPU kernel registry: {}",
             freeze_status.ToString().c_str());
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
    const StatusOr<const KernelDescriptor*> descriptor = kernel_registry_.Resolve(op_type, selector);
    if (!descriptor.ok()) {
        return nullptr;
    }
    return (*descriptor)->kernel_func;
}

StatusOr<ResolvedKernel> CpuBackend::ResolveKernelInfo(
        OpType op_type,
        const KernelSelector& selector) const noexcept {
    const StatusOr<const KernelDescriptor*> descriptor = kernel_registry_.Resolve(op_type, selector);
    if (!descriptor.ok()) {
        return descriptor.status();
    }

    return ResolvedKernel{
            .op_type = op_type,
            .fn = (*descriptor)->kernel_func,
            .attrs = {},
            .debug_name = (*descriptor)->name,
    };
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
