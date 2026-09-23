#include "aethermind/backend/cpu/cpu_weight_prepacker.h"
#include "aethermind/base/macros.h"
#include "aethermind/base/tensor_view.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <string>

namespace aethermind {

namespace {

void FreePackedCpuBuffer(void*, void* ptr) noexcept {
    std::free(ptr);
}

Buffer AllocateCpuPackedBuffer(size_t nbytes, size_t alignment) {
    void* data = nullptr;
    const size_t effective_alignment = alignment == 0 ? 64 : alignment;
    const int rc = posix_memalign(&data, effective_alignment, nbytes == 0 ? 1 : nbytes);
    if (rc != 0 || data == nullptr) {
        return {};
    }

    return Buffer{nbytes,
                  MemoryHandle(data,
                               nullptr,
                               &FreePackedCpuBuffer,
                               Device::CPU(),
                               effective_alignment)};
}

class CpuPackedWeights final : public PackedWeights {
public:
    CpuPackedWeights(OpType op_type,
                     KernelSelector selector,
                     PackingRecipe recipe,
                     DataType logical_dtype,
                     std::vector<int64_t> logical_shape,
                     Buffer storage) noexcept
        : op_type_(op_type),
          selector_(selector),
          recipe_(std::move(recipe)),
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

} // namespace

StatusOr<std::unique_ptr<PackedWeights>> CpuWeightPrepacker::Pack(
        OpType op_type,
        const Tensor& logical_weight,
        const KernelSelector& selector) const noexcept {
    if (op_type == OpType::kUnknown) {
        return Status::InvalidArgument("CpuWeightPrepacker requires a concrete op type");
    }

    if (selector.device_type != DeviceType::kCPU) {
        return Status::InvalidArgument("CpuWeightPrepacker only supports CPU selectors");
    }

    if (selector.weight_format != WeightFormat::kPacked) {
        return Status::InvalidArgument("CpuWeightPrepacker requires WeightFormat::kPacked");
    }

    if (!logical_weight.is_initialized()) {
        return Status::InvalidArgument("CpuWeightPrepacker requires initialized logical weights");
    }

    if (!logical_weight.device().is_cpu()) {
        return Status::InvalidArgument("CpuWeightPrepacker only supports CPU logical weights");
    }

    return Pack(op_type, logical_weight.view(), selector);
}

StatusOr<std::unique_ptr<PackedWeights>> CpuWeightPrepacker::Pack(
        OpType op_type,
        TensorView logical_weight,
        const KernelSelector& selector) const noexcept {
    if (op_type == OpType::kUnknown) {
        return Status::InvalidArgument("CpuWeightPrepacker requires a concrete op type");
    }

    if (selector.device_type != DeviceType::kCPU) {
        return Status::InvalidArgument("CpuWeightPrepacker only supports CPU selectors");
    }

    if (selector.weight_format != WeightFormat::kPacked) {
        return Status::InvalidArgument("CpuWeightPrepacker requires WeightFormat::kPacked");
    }

    if (!logical_weight.is_valid()) {
        return Status::InvalidArgument("CpuWeightPrepacker requires a valid logical weight TensorView");
    }

    const size_t packed_nbytes = logical_weight.logical_nbytes();
    // Identity consumers require at least the recipe alignment, while a
    // source view may carry a stronger alignment contract that callers retain.
    Buffer packed_storage = AllocateCpuPackedBuffer(
            packed_nbytes,
            std::max(logical_weight.alignment(), cpu::kCpuIdentityPackingAlignment));
    if (!packed_storage.is_initialized()) {
        return Status::ResourceExhausted("Failed to allocate packed CPU weight storage");
    }

    if (packed_nbytes > 0) {
        std::memcpy(packed_storage.mutable_data(), logical_weight.data(), packed_nbytes);
    }

    std::vector<int64_t> logical_shape(logical_weight.shape().begin(),
                                       logical_weight.shape().end());
    return std::make_unique<CpuPackedWeights>(
            op_type, selector, RecipeFor(selector),
            logical_weight.dtype(), std::move(logical_shape),
            std::move(packed_storage));
}

StatusOr<std::unique_ptr<PackedWeights>> CpuWeightPrepacker::Pack(
        OpType op_type,
        std::span<const TensorView> components,
        const KernelSelector& selector) const noexcept {
    if (op_type == OpType::kUnknown) {
        return Status::InvalidArgument("CpuWeightPrepacker requires a concrete op type");
    }

    if (selector.device_type != DeviceType::kCPU) {
        return Status::InvalidArgument("CpuWeightPrepacker only supports CPU selectors");
    }

    if (selector.weight_format != WeightFormat::kPacked) {
        return Status::InvalidArgument("CpuWeightPrepacker requires WeightFormat::kPacked");
    }

    if (components.empty()) {
        return Status::InvalidArgument(
                "CpuWeightPrepacker requires at least one weight component");
    }
    // A single component keeps the unrestricted single-view contract: direct
    // bindings pack any valid rank (e.g. rank-1 norm weights).
    if (components.size() == 1U) {
        return Pack(op_type, components.front(), selector);
    }

    // The backend owns the fused layout authority for composite bindings:
    // components must be contiguous rank-2 views sharing one dtype and feature
    // count, concatenated along axis 0 in recipe order.
    const DataType& dtype = components.front().dtype();
    int64_t feature_count = -1;
    int64_t total_rows = 0;
    size_t total_bytes = 0;
    size_t alignment = cpu::kCpuIdentityPackingAlignment;
    for (const TensorView& component: components) {
        if (!component.is_valid()) {
            return Status::InvalidArgument(
                    "CpuWeightPrepacker requires valid weight component views");
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

    Buffer packed_storage = AllocateCpuPackedBuffer(total_bytes, alignment);
    if (!packed_storage.is_initialized()) {
        return Status::ResourceExhausted("Failed to allocate packed CPU weight storage");
    }

    char* out = static_cast<char*>(packed_storage.mutable_data());
    for (const TensorView& component: components) {
        std::memcpy(out, component.data(), component.logical_nbytes());
        out += component.logical_nbytes();
    }

    return std::make_unique<CpuPackedWeights>(
            op_type, selector, RecipeFor(selector), dtype,
            std::vector<int64_t>{total_rows, feature_count},
            std::move(packed_storage));
}

StatusOr<std::unique_ptr<PackedWeights>> CpuWeightPrepacker::Pack(
        OpType op_type,
        TensorView logical_weight,
        const KernelSelector& selector,
        const PackingRecipe& recipe) const noexcept {
    if (recipe == CpuIdentityPackingRecipe()) {
        return Pack(op_type, logical_weight, selector);
    }
    if (recipe != cpu::CpuBPanelF32V1Avx2Recipe()) {
        return Status::InvalidArgument(
                "CpuWeightPrepacker does not support the requested recipe");
    }
    const std::array<TensorView, 1> components{logical_weight};
    return Pack(op_type, components, selector, recipe);
}

StatusOr<std::unique_ptr<PackedWeights>> CpuWeightPrepacker::Pack(
        OpType op_type,
        std::span<const TensorView> components,
        const KernelSelector& selector,
        const PackingRecipe& recipe) const noexcept {
    if (recipe == CpuIdentityPackingRecipe()) {
        return Pack(op_type, components, selector);
    }
    if (recipe != cpu::CpuBPanelF32V1Avx2Recipe()) {
        return Status::InvalidArgument(
                "CpuWeightPrepacker does not support the requested recipe");
    }
    if (op_type == OpType::kUnknown || selector.device_type != DeviceType::kCPU ||
        selector.weight_format != WeightFormat::kPacked || components.empty()) {
        return Status::InvalidArgument(
                "CpuWeightPrepacker requires a packed CPU request and weight components");
    }

    if (!components.front().is_valid() || components.front().rank() != 2) {
        return Status::InvalidArgument(
                "cpu_bpanel_f32 requires valid rank-2 weight components");
    }
    const int64_t feature_count = components.front().dim(1);
    int64_t total_rows = 0;
    size_t alignment = cpu::kCpuBPanelF32V1Alignment;
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
    Buffer packed_storage = AllocateCpuPackedBuffer(packed_nbytes, alignment);
    if (!packed_storage.is_initialized()) {
        return Status::ResourceExhausted("Failed to allocate packed CPU B-panel storage");
    }
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
    return std::make_unique<CpuPackedWeights>(
            op_type, selector, recipe, DataType::Float32(),
            std::move(logical_shape), std::move(packed_storage));
}

PackingRecipe cpu::CpuBPanelF32V1Avx2Recipe() {
    return PackingRecipe{
            .layout = std::string(cpu::kCpuBPanelF32V1Avx2Layout),
            .alignment = cpu::kCpuBPanelF32V1Alignment};
}

StatusOr<size_t> cpu::CpuBPanelF32V1PackedByteSize(
        int64_t n, int64_t k) noexcept {
    if (n < 0 || k < 0) {
        return Status::InvalidArgument(
                "cpu_bpanel_f32 dimensions must be non-negative");
    }
    const size_t n_blocks = static_cast<size_t>(
            n / cpu::kCpuBPanelF32V1NR +
            (n % cpu::kCpuBPanelF32V1NR != 0));
    const size_t k_panels = static_cast<size_t>(
            k / cpu::kCpuBPanelF32V1KC +
            (k % cpu::kCpuBPanelF32V1KC != 0));
    size_t elements = n_blocks;
    const size_t factors[] = {
            static_cast<size_t>(cpu::kCpuBPanelF32V1KC),
            static_cast<size_t>(cpu::kCpuBPanelF32V1NR),
            k_panels,
            sizeof(float),
    };
    for (const size_t factor: factors) {
        if (factor != 0 && elements > std::numeric_limits<size_t>::max() / factor) {
            return Status::Overflow(
                    "cpu_bpanel_f32 packed byte size overflows size_t");
        }
        elements *= factor;
    }
    return elements;
}

PackingRecipe CpuWeightPrepacker::RecipeFor(const KernelSelector& selector) noexcept {
    // Phase 1 packs by identity copy; the recipe records this canonical layout
    // so distinct packing variants of the same {binding, selector} stay
    // distinguishable once real tile-block layouts land.
    (void) selector;
    return PackingRecipe{.layout = std::string(cpu::kCpuIdentityPackingLayout),
                         .alignment = cpu::kCpuIdentityPackingAlignment};
}

} // namespace aethermind
