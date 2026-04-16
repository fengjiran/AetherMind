#include "aethermind/backend/cpu/cpu_weight_prepacker.h"

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
                     Buffer storage) noexcept
        : op_type_(op_type),
          selector_(selector),
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

private:
    OpType op_type_ = OpType::kUnknown;
    KernelSelector selector_{};
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

    const size_t packed_nbytes = logical_weight.logical_nbytes();
    Buffer packed_storage = AllocateCpuPackedBuffer(packed_nbytes, logical_weight.alignment());
    if (!packed_storage.is_initialized()) {
        return Status(StatusCode::kResourceExhausted, "Failed to allocate packed CPU weight storage");
    }

    if (packed_nbytes > 0) {
        std::memcpy(packed_storage.mutable_data(), logical_weight.data(), packed_nbytes);
    }

    return std::make_unique<CpuPackedWeights>(op_type, selector, std::move(packed_storage));
}

}// namespace aethermind
