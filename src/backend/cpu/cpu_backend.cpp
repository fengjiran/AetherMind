#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/cpu/cpu_bpanel_packing.h"
#include "aethermind/backend/cpu/cpu_identity_packing.h"
#include "aethermind/backend/cpu/cpu_info.h"
#include "aethermind/base/macros.h"
#include "backend/cpu/cpu_backend_internal.h"
#include "utils/logging.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace aethermind {
namespace {

void FreePackedCpuBuffer(void*, void* ptr) noexcept {
    std::free(ptr);
}

StatusOr<Buffer> AllocateCpuPackedBuffer(size_t nbytes, size_t alignment) {
    constexpr size_t kDefaultAlignment = 64;
    const size_t effective_alignment = alignment == 0 ? kDefaultAlignment : alignment;
    // posix_memalign requires a power-of-two alignment; any power of two at
    // least sizeof(void*) is also a multiple of sizeof(void*), satisfying both
    // constraints. Rejecting invalid values here keeps an EINVAL from being
    // misreported as resource exhaustion.
    if (effective_alignment < sizeof(void*) ||
        (effective_alignment & (effective_alignment - 1)) != 0) {
        return Status::InvalidArgument(
                "Packed CPU weight alignment must be a power of two and at "
                "least sizeof(void*)");
    }

    void* data = nullptr;
    if (const int rc = posix_memalign(&data, effective_alignment, nbytes == 0 ? 1 : nbytes);
        rc != 0 || data == nullptr) {
        return Status::ResourceExhausted(
                "Failed to allocate packed CPU weight storage");
    }

    try {
        return Buffer{nbytes,
                      MemoryHandle(data,
                                   nullptr,
                                   &FreePackedCpuBuffer,
                                   Device::CPU(),
                                   effective_alignment)};
    } catch (const std::bad_alloc&) {
        return Status::ResourceExhausted("Failed to allocate packed CPU weight buffer metadata");
    }
}

class CpuPackedWeight final : public PackedWeight {
public:
    CpuPackedWeight(OpType op_type,
                    KernelSelector selector,
                    PackingRecipe recipe,
                    DataType logical_dtype,
                    std::vector<int64_t> logical_shape,
                    Buffer storage) noexcept
        : op_type_(op_type),
          selector_(selector),
          recipe_(recipe),
          logical_dtype_(logical_dtype),
          logical_shape_(std::move(logical_shape)),
          storage_(std::move(storage)) {}

    AM_NODISCARD OpType op_type() const noexcept override {
        return op_type_;
    }

    AM_NODISCARD const KernelSelector& selector() const noexcept override {
        return selector_;
    }

    AM_NODISCARD const Buffer& storage() const noexcept override {
        return storage_;
    }

    AM_NODISCARD const PackingRecipe& recipe() const noexcept override {
        return recipe_;
    }

    AM_NODISCARD DataType logical_dtype() const noexcept override {
        return logical_dtype_;
    }

    AM_NODISCARD const std::vector<int64_t>& logical_shape() const noexcept override {
        return logical_shape_;
    }

private:
    OpType op_type_ = OpType::kUnknown;
    KernelSelector selector_{};
    PackingRecipe recipe_{};
    DataType logical_dtype_{};
    std::vector<int64_t> logical_shape_{};
    Buffer storage_{};
};

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

StatusOr<std::unique_ptr<PackedWeight>> PackWeightsWithRecipe(
        OpType op_type,
        std::span<const TensorView> components,
        const KernelSelector& selector,
        const PackingRecipe& recipe) noexcept {
    if (recipe != CpuIdentityPackingRecipe() && recipe != CpuBPanelF32V1Avx2Recipe()) {
        return Status::InvalidArgument(
                "CPU weight packing does not support the requested recipe");
    }

    if (op_type == OpType::kUnknown || selector.device_type != DeviceType::kCPU ||
        selector.weight_format != WeightFormat::kPacked) {
        return Status::InvalidArgument(
                "CPU weight packing requires a packed CPU request");
    }

    if (components.empty()) {
        return Status::InvalidArgument(
                "CPU weight packing requires at least one weight component");
    }

    if (recipe == CpuIdentityPackingRecipe()) {
        // A single component keeps the single-component rank contract: direct
        // bindings pack any valid rank (e.g. rank-1 norm weights).
        if (components.size() == 1U) {
            const TensorView& logical_weight = components.front();
            if (!logical_weight.is_valid()) {
                return Status::InvalidArgument(
                        "CPU weight packing requires a valid logical weight TensorView");
            }

            if (!logical_weight.is_contiguous()) {
                return Status::InvalidArgument(
                        "logical weights must be contiguous row-major views");
            }

            const size_t packed_nbytes = logical_weight.logical_nbytes();
            // Preserve a stronger source alignment while satisfying the recipe.
            AM_ASSIGN_OR_RETURN(
                    Buffer packed_storage,
                    AllocateCpuPackedBuffer(
                            packed_nbytes,
                            std::max(logical_weight.alignment(), recipe.alignment)));
            if (packed_nbytes > 0) {
                std::memcpy(packed_storage.mutable_data(), logical_weight.data(), packed_nbytes);
            }

            std::vector<int64_t> logical_shape(logical_weight.shape().begin(),
                                               logical_weight.shape().end());
            return std::make_unique<CpuPackedWeight>(
                    op_type, selector, recipe,
                    logical_weight.dtype(), std::move(logical_shape),
                    std::move(packed_storage));
        }

        // The backend owns the fused layout authority for composite bindings:
        // components must be contiguous rank-2 views sharing one dtype and
        // feature count, concatenated along axis 0 in recipe order.
        const DataType& dtype = components.front().dtype();
        int64_t feature_count = -1;
        int64_t total_rows = 0;
        size_t total_bytes = 0;
        size_t alignment = recipe.alignment;
        for (const TensorView& component: components) {
            if (!component.is_valid()) {
                return Status::InvalidArgument(
                        "CPU weight packing requires valid weight component views");
            }
            if (!component.is_contiguous()) {
                return Status::InvalidArgument(
                        "weight components must be contiguous row-major views");
            }
            if (component.rank() != 2) {
                return Status::InvalidArgument("weight components must be rank 2");
            }
            if (component.dtype() != dtype) {
                return Status::InvalidArgument(
                        "weight components must share one dtype");
            }
            if (feature_count < 0) {
                feature_count = component.dim(1);
            } else if (component.dim(1) != feature_count) {
                return Status::InvalidArgument(
                        "weight components must share a feature count");
            }
            const int64_t rows = component.dim(0);
            if (rows < 0) {
                return Status::InvalidArgument(
                        "weight component row count is negative");
            }
            if (rows > std::numeric_limits<int64_t>::max() - total_rows) {
                return Status::InvalidArgument(
                        "fused weight row count overflows");
            }
            total_rows += rows;
            const size_t nbytes = component.logical_nbytes();
            if (nbytes > std::numeric_limits<size_t>::max() - total_bytes) {
                return Status::InvalidArgument(
                        "fused weight byte count overflows");
            }
            total_bytes += nbytes;
            alignment = std::max(alignment, component.alignment());
        }

        AM_ASSIGN_OR_RETURN(Buffer packed_storage,
                            AllocateCpuPackedBuffer(total_bytes, alignment));

        char* out = static_cast<char*>(packed_storage.mutable_data());
        for (const TensorView& component: components) {
            const size_t nbytes = component.logical_nbytes();
            if (nbytes > 0) {
                std::memcpy(out, component.data(), nbytes);
                out += nbytes;
            }
        }

        return std::make_unique<CpuPackedWeight>(
                op_type, selector, recipe, dtype,
                std::vector<int64_t>{total_rows, feature_count},
                std::move(packed_storage));
    }

    // cpu_bpanel_f32_v1_avx2:
    if (!components.front().is_valid() || components.front().rank() != 2) {
        return Status::InvalidArgument(
                "cpu_bpanel_f32 requires valid rank-2 weight components");
    }

    const int64_t feature_count = components.front().dim(1);
    int64_t total_rows = 0;
    size_t alignment = kCpuBPanelF32V1Alignment;
    for (const TensorView& component: components) {
        if (!component.is_valid() || !component.is_contiguous() ||
            component.rank() != 2 || component.dtype() != DataType::Float32() ||
            component.dim(1) != feature_count || component.dim(0) < 0 ||
            (component.logical_nbytes() != 0 && component.data() == nullptr)) {
            return Status::InvalidArgument(
                    "cpu_bpanel_f32 requires contiguous rank-2 float32 weights with equal K");
        }
        if (component.dim(0) > std::numeric_limits<int64_t>::max() - total_rows) {
            return Status::Overflow("cpu_bpanel_f32 logical N overflows int64_t");
        }
        total_rows += component.dim(0);
        alignment = std::max(alignment, component.alignment());
    }
    if (feature_count < 0) {
        return Status::InvalidArgument("cpu_bpanel_f32 requires a non-negative K dimension");
    }

    AM_ASSIGN_OR_RETURN(const size_t packed_nbytes,
                        cpu::CpuBPanelF32V1PackedByteSize(total_rows, feature_count));
    AM_ASSIGN_OR_RETURN(Buffer packed_storage,
                        AllocateCpuPackedBuffer(packed_nbytes, alignment));
    if (packed_nbytes != 0) {
        std::memset(packed_storage.mutable_data(), 0, packed_nbytes);
        float* const packed_data = static_cast<float*>(packed_storage.mutable_data());
        const size_t n_blocks = static_cast<size_t>(
                total_rows / cpu::kCpuBPanelF32V1NR +
                (total_rows % cpu::kCpuBPanelF32V1NR != 0));
        size_t row_offset = 0;
        for (const TensorView& component: components) {
            const float* const src = component.data<float>();
            for (int64_t row = 0; row < component.dim(0); ++row) {
                const size_t logical_row = row_offset + static_cast<size_t>(row);
                for (int64_t k = 0; k < feature_count; ++k) {
                    const size_t panel = static_cast<size_t>(k / cpu::kCpuBPanelF32V1KC);
                    const size_t block = logical_row /
                                         static_cast<size_t>(cpu::kCpuBPanelF32V1NR);
                    const size_t panel_row = static_cast<size_t>(k % cpu::kCpuBPanelF32V1KC);
                    const size_t column = logical_row %
                                          static_cast<size_t>(cpu::kCpuBPanelF32V1NR);
                    const size_t packed_index =
                            (((panel * n_blocks + block) *
                                      static_cast<size_t>(cpu::kCpuBPanelF32V1KC) +
                              panel_row) *
                             static_cast<size_t>(cpu::kCpuBPanelF32V1NR)) +
                            column;
                    packed_data[packed_index] =
                            src[static_cast<size_t>(row) *
                                        static_cast<size_t>(feature_count) +
                                static_cast<size_t>(k)];
                }
            }
            row_offset += static_cast<size_t>(component.dim(0));
        }
    }

    std::vector<int64_t> logical_shape{total_rows, feature_count};
    return std::make_unique<CpuPackedWeight>(
            op_type, selector, recipe, DataType::Float32(),
            std::move(logical_shape), std::move(packed_storage));
}

StatusOr<const KernelDef*> ResolveEligibleDescriptor(
        const KernelRegistry& registry,
        OpType op_type,
        const KernelSelector& selector,
        const CpuFeatureSet& effective_features) {
    const auto candidates = registry.FindCandidates(op_type, selector);
    if (!candidates.ok()) {
        return candidates.status();
    }

    const KernelDef* best = nullptr;
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

StatusOr<PackingRecipe> ResolvePackingRecipeFromRegistry(const KernelRegistry& registry,
                                                         OpType op_type,
                                                         const KernelSelector& selector,
                                                         const CpuFeatureSet& effective_features) {
    if (selector.device_type != DeviceType::kCPU ||
        selector.weight_format != WeightFormat::kPacked) {
        return Status::InvalidArgument(
                "CPU packing recipe query requires a packed CPU selector");
    }

    AM_ASSIGN_OR_RETURN(
            const KernelDef* descriptor,
            ResolveEligibleDescriptor(registry, op_type, selector, effective_features));
    if (!IsValidPackingLayout(descriptor->packing_recipe.layout) ||
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

    const StatusOr<const KernelDef*> descriptor =
            cpu::internal::ResolveEligibleDescriptor(
                    KernelRegistry::Global(), op_type, selector, capabilities_.effective_features);
    if (!descriptor.ok()) {
        return descriptor.status();
    }

    ResolvedKernel resolved{
            .op_type = op_type,
            .fn = (*descriptor)->kernel_func,
            .attrs = {},
            .name = (*descriptor)->name,
            .params_builder = (*descriptor)->params_builder,
            .params_size = (*descriptor)->params_size,
            .workspace_requirement = {},
    };

    if (selector.weight_format == WeightFormat::kPacked) {
        if (!IsValidPackingLayout((*descriptor)->packing_recipe.layout) ||
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

StatusOr<std::unique_ptr<PackedWeight>> CpuBackend::PackWeights(
        OpType op_type,
        std::span<const TensorView> components,
        const KernelSelector& selector,
        const PackingRecipe& recipe) const {
    AM_ASSIGN_OR_RETURN(
            const PackingRecipe selected_recipe, GetPackingRecipe(op_type, selector));
    if (recipe != selected_recipe) {
        return Status::InvalidArgument(
                "Requested packing recipe is not selected by this CPU backend");
    }

    return cpu::internal::PackWeightsWithRecipe(op_type, components, selector, recipe);
}

StatusOr<std::unique_ptr<Backend>> CpuBackendFactory::Create() const {
    auto capabilities = cpu::DetectCpuCapabilities(policy_);
    if (!capabilities.ok()) {
        return capabilities.status();
    }
    return std::make_unique<CpuBackend>(std::move(capabilities).value());
}

} // namespace aethermind
