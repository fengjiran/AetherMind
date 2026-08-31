#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/cpu/cpu_info.h"
#include "aethermind/backend/cpu/cpu_weight_prepacker.h"

#include "utils/logging.h"

namespace aethermind {
namespace {

// 能力检测失败（当前仅在无 CPU 探测器的平台上可能发生）会导致进程中止：
// BackendFactory::Create() 返回 unique_ptr<Backend>，没有 Status 通道可传播错误。
// x86-64 / AArch64 等受支持平台不会走到该分支。
CpuCapabilities DetectCapabilitiesOrDie(const CpuFeaturePolicy& policy) {
    auto capabilities = cpu::DetectCpuCapabilities(policy);
    AM_CHECK(capabilities.ok(),
             "Failed to initialize CPU capabilities: {}",
             capabilities.status().ToString().c_str());
    return std::move(capabilities).value();
}

StatusOr<const KernelDescriptor*> ResolveEligibleDescriptor(
        const KernelRegistry& registry,
        OpType op_type,
        const KernelSelector& selector,
        const CpuFeatureSet& effective_features) {
    const auto candidates = registry.FindCandidates(op_type, selector);
    if (!candidates.ok()) {
        return candidates.status();
    }

    const KernelDescriptor* best = nullptr;
    for (const KernelDescriptor* descriptor: *candidates) {
        if (!effective_features.ContainsAll(descriptor->cpu_requirements.all_of)) {
            continue;
        }
        if (best == nullptr || descriptor->priority > best->priority) {
            best = descriptor;
        }
    }
    if (best == nullptr) {
        return Status::NotFound(
                "No eligible CPU kernel registered for op_type=" +
                std::string(ToString(op_type)) + ", selector=" + ToString(selector) +
                ", effective_features=" + ToString(effective_features));
    }
    return best;
}

}// namespace

CpuBackend::CpuBackend() : CpuBackend(CpuFeaturePolicy{}) {}

CpuBackend::CpuBackend(CpuFeaturePolicy policy)
    : capabilities_(DetectCapabilitiesOrDie(policy)) {
    const Status status = KernelRegistry::Global().Freeze();
    AM_CHECK(status.ok(), "Failed to freeze CPU kernel registry: {}", status.ToString().c_str());
}

DeviceType CpuBackend::device_type() const noexcept {
    return DeviceType::kCPU;
}

const BackendCapabilities& CpuBackend::capabilities() const noexcept {
    return capabilities_.base;
}

StatusOr<ResolvedKernel> CpuBackend::PrepareKernel(
        OpType op_type,
        const KernelSelector& selector,
        const OpParams& params) const {
    if (selector.device_type != DeviceType::kCPU) {
        return Status::InvalidArgument(
                "CpuBackend cannot prepare non-CPU kernel selector");
    }

    const StatusOr<const KernelDescriptor*> descriptor = ResolveEligibleDescriptor(
            KernelRegistry::Global(), op_type, selector,
            capabilities_.effective_features);
    if (!descriptor.ok()) {
        return descriptor.status();
    }

    ResolvedKernel resolved{
            .op_type = op_type,
            .fn = (*descriptor)->kernel_func,
            .attrs = {},
            .debug_name = (*descriptor)->name.c_str(),
            .params_builder = (*descriptor)->params_builder,
            .params_size = (*descriptor)->params_size,
            .workspace_requirement = {},
    };
    if (selector.weight_format == WeightFormat::kPacked) {
        // The kernel consumes the same packing layout the prepacker produces;
        // both derive from CpuWeightPrepacker so pack and consume stay in sync.
        resolved.expected_packing_recipe = CpuWeightPrepacker::RecipeFor(selector);
    }

    if ((*descriptor)->metadata_builder != nullptr) {
        AM_RETURN_IF_ERROR((*descriptor)->metadata_builder(params, resolved.attrs));
    }
    return resolved;
}

const KernelRegistry* CpuBackend::TryGetKernelRegistryForDebug() const noexcept {
    return &KernelRegistry::Global();
}

const CpuCapabilities& CpuBackend::cpu_capabilities() const noexcept {
    return capabilities_;
}

CpuBackendFactory::CpuBackendFactory(CpuFeaturePolicy policy)
    : policy_(std::move(policy)) {}

DeviceType CpuBackendFactory::device_type() const noexcept {
    return DeviceType::kCPU;
}

std::unique_ptr<Backend> CpuBackendFactory::Create() const {
    return std::make_unique<CpuBackend>(policy_);
}

}// namespace aethermind
