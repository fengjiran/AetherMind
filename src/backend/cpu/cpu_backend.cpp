#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/cpu/cpu_info.h"
#include "aethermind/backend/cpu/cpu_weight_prepacker.h"
#include "backend/cpu/cpu_backend_internal.h"
#include "utils/logging.h"

namespace aethermind {
namespace {

// Capability detection failure (only possible on platforms without a CPU
// detector) aborts the process. CpuBackend constructors have no Status
// channel to propagate the error, unlike CpuBackendFactory::Create().
// Supported platforms (x86-64 / AArch64) never take this path.
CpuCapabilities DetectCapabilitiesOrDie(const CpuFeaturePolicy& policy) {
    auto capabilities = cpu::DetectCpuCapabilities(policy);
    AM_CHECK(capabilities.ok(),
             "Failed to initialize CPU capabilities: {}",
             capabilities.status().ToString().c_str());
    return std::move(capabilities).value();
}

} // namespace

namespace cpu::internal {

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
    for (const auto* descriptor: *candidates) {
        if (!effective_features.ContainsAll(descriptor->cpu_requirements)) {
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

StatusOr<PackingRecipe> ResolvePackingRecipeFromRegistry(
        const KernelRegistry& registry,
        OpType op_type,
        const KernelSelector& selector,
        const CpuFeatureSet& effective_features) {
    if (selector.device_type != DeviceType::kCPU ||
        selector.weight_format != WeightFormat::kPacked) {
        return Status::InvalidArgument(
                "CPU packing recipe query requires a packed CPU selector");
    }
    AM_ASSIGN_OR_RETURN(const KernelDescriptor* descriptor,
                        ResolveEligibleDescriptor(
                                registry, op_type, selector, effective_features));
    if (descriptor->packing_recipe.layout.empty() ||
        descriptor->packing_recipe.alignment == 0) {
        return Status::Internal(
                "Packed CPU descriptor is missing its packing recipe");
    }
    return descriptor->packing_recipe;
}

} // namespace cpu::internal

CpuBackend::CpuBackend(const CpuCapabilities& capabilities) : capabilities_(capabilities) {
    const Status status = KernelRegistry::Global().Freeze();
    AM_CHECK(status.ok(), "Failed to freeze CPU kernel registry: {}", status.ToString().c_str());
}

CpuBackend::CpuBackend() : CpuBackend(CpuFeaturePolicy{}) {}

CpuBackend::CpuBackend(const CpuFeaturePolicy& policy)
    : CpuBackend(DetectCapabilitiesOrDie(policy)) {}

StatusOr<ResolvedKernel> CpuBackend::PrepareKernel(OpType op_type,
                                                   const KernelSelector& selector,
                                                   const OpParams& params) const {
    if (selector.device_type != DeviceType::kCPU) {
        return Status::InvalidArgument(
                "CpuBackend cannot prepare non-CPU kernel selector");
    }

    const StatusOr<const KernelDescriptor*> descriptor =
            cpu::internal::ResolveEligibleDescriptor(
                    KernelRegistry::Global(), op_type, selector, capabilities_.effective_features);
    if (!descriptor.ok()) {
        return descriptor.status();
    }

    ResolvedKernel resolved{
            .op_type = op_type,
            .fn = (*descriptor)->kernel_func,
            .attrs = {},
            .name = (*descriptor)->name.c_str(),
            .params_builder = (*descriptor)->params_builder,
            .params_size = (*descriptor)->params_size,
            .workspace_requirement = {},
    };

    if (selector.weight_format == WeightFormat::kPacked) {
        if ((*descriptor)->packing_recipe.layout.empty() ||
            (*descriptor)->packing_recipe.alignment == 0) {
            return Status::Internal(
                    "Packed CPU descriptor is missing its packing recipe");
        }
        resolved.expected_packing_recipe = (*descriptor)->packing_recipe;
    }

    if ((*descriptor)->metadata_builder != nullptr) {
        AM_RETURN_IF_ERROR((*descriptor)->metadata_builder(params, resolved.attrs));
    }
    return resolved;
}

StatusOr<PackingRecipe> CpuBackend::GetPackingRecipe(
        OpType op_type, const KernelSelector& selector) const {
    return cpu::internal::ResolvePackingRecipeFromRegistry(
            KernelRegistry::Global(), op_type, selector,
            capabilities_.effective_features);
}

StatusOr<std::unique_ptr<PackedWeights>> CpuBackend::PackWeights(
        OpType op_type,
        std::span<const TensorView> components,
        const KernelSelector& selector) const {
    AM_ASSIGN_OR_RETURN(const PackingRecipe recipe,
                        GetPackingRecipe(op_type, selector));
    return PackWeights(op_type, components, selector, recipe);
}

StatusOr<std::unique_ptr<PackedWeights>> CpuBackend::PackWeights(
        OpType op_type,
        std::span<const TensorView> components,
        const KernelSelector& selector,
        const PackingRecipe& recipe) const {
    AM_ASSIGN_OR_RETURN(const PackingRecipe selected_recipe,
                        GetPackingRecipe(op_type, selector));
    if (recipe != selected_recipe) {
        return Status::InvalidArgument(
                "Requested packing recipe is not selected by this CPU backend");
    }
    CpuWeightPrepacker prepacker;
    return prepacker.Pack(op_type, components, selector, recipe);
}

StatusOr<std::unique_ptr<Backend>> CpuBackendFactory::Create() const {
    auto capabilities = cpu::DetectCpuCapabilities(policy_);
    if (!capabilities.ok()) {
        return capabilities.status();
    }
    return std::make_unique<CpuBackend>(std::move(capabilities).value());
}

} // namespace aethermind
