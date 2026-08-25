#include "aethermind/backend/cpu/cpu_weight_prepacker.h"
#include "aethermind/base/tensor_view.h"

#include <cstdlib>
#include <cstring>
#include <memory>

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

}// namespace

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
    Buffer packed_storage = AllocateCpuPackedBuffer(packed_nbytes, logical_weight.alignment());
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

PackingRecipe CpuWeightPrepacker::RecipeFor(const KernelSelector& selector) noexcept {
    // Phase 1 packs by identity copy; the recipe records this canonical layout
    // so distinct packing variants of the same {binding, selector} stay
    // distinguishable once real tile-block layouts land.
    (void) selector;
    return PackingRecipe{.layout = "cpu_identity", .alignment = 64};
}

}// namespace aethermind
