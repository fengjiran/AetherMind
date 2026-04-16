#include "aethermind/model/backend_sidecar.h"

#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_selector.h"
#include "aethermind/backend/packed_weights.h"
#include "aethermind/base/status.h"
#include "aethermind/memory/buffer.h"
#include "aethermind/operators/op_type.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>

namespace aethermind {
namespace {

void FreeTestBuffer(void*, void* ptr) noexcept {
    std::free(ptr);
}

Buffer MakeTestBuffer(size_t nbytes, size_t alignment = 64) {
    void* ptr = nullptr;
    const int rc = posix_memalign(&ptr, alignment, nbytes == 0 ? 1 : nbytes);
    if (rc != 0 || ptr == nullptr) {
        return {};
    }
    return Buffer{nbytes, MemoryHandle(ptr, nullptr, &FreeTestBuffer, Device::CPU(), alignment)};
}

KernelSelector MakePackedCpuSelector() {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
            .isa = IsaLevel::kAVX2,
            .phase = ExecPhase::kBoth,
    };
}

class CountingPackedWeights final : public PackedWeights {
public:
    CountingPackedWeights(OpType op_type,
                          KernelSelector selector,
                          Buffer storage,
                          bool* destroyed_flag) noexcept
        : op_type_(op_type),
          selector_(selector),
          storage_(std::move(storage)),
          destroyed_flag_(destroyed_flag) {}

    ~CountingPackedWeights() override {
        if (destroyed_flag_ != nullptr) {
            *destroyed_flag_ = true;
        }
    }

    OpType op_type() const noexcept override {
        return op_type_;
    }

    const KernelSelector& selector() const noexcept override {
        return selector_;
    }

    const Buffer& storage() const noexcept override {
        return storage_;
    }

private:
    OpType op_type_ = OpType::kUnknown;
    KernelSelector selector_{};
    Buffer storage_{};
    bool* destroyed_flag_ = nullptr;
};

TEST(BackendSidecarOwnership, SidecarOwnsPackedWeightsUntilItIsDestroyed) {
    bool destroyed = false;
    const KernelSelector selector = MakePackedCpuSelector();

    {
        BackendSidecar sidecar;
        auto packed = std::make_unique<CountingPackedWeights>(
                OpType::kLinear,
                selector,
                MakeTestBuffer(256),
                &destroyed);
        const PackedWeights* raw_ptr = packed.get();

        ASSERT_TRUE(sidecar.Store(std::move(packed)).ok());
        EXPECT_FALSE(destroyed);

        const PackedWeights* found = sidecar.Find(OpType::kLinear, selector);
        ASSERT_NE(found, nullptr);
        EXPECT_EQ(found, raw_ptr);
        EXPECT_TRUE(found->storage().is_initialized());
    }

    EXPECT_TRUE(destroyed);
}

TEST(BackendSidecarOwnership, StoredPackedWeightsOutliveBackendInstance) {
    bool destroyed = false;
    const KernelSelector selector = MakePackedCpuSelector();

    BackendSidecar sidecar;
    {
        CpuBackend backend;
        (void) backend;

        auto packed = std::make_unique<CountingPackedWeights>(
                OpType::kLinear,
                selector,
                MakeTestBuffer(128),
                &destroyed);
        ASSERT_TRUE(sidecar.Store(std::move(packed)).ok());
    }

    const PackedWeights* found = sidecar.Find(OpType::kLinear, selector);
    ASSERT_NE(found, nullptr);
    EXPECT_FALSE(destroyed);
    EXPECT_TRUE(found->storage().device().is_cpu());
}

TEST(BackendSidecarOwnership, StoreRejectsDuplicatePackedWeightEntries) {
    BackendSidecar sidecar;
    const KernelSelector selector = MakePackedCpuSelector();

    ASSERT_TRUE(sidecar.Store(std::make_unique<CountingPackedWeights>(
                                    OpType::kLinear,
                                    selector,
                                    MakeTestBuffer(64),
                                    nullptr))
                        .ok());

    const Status duplicate_status = sidecar.Store(std::make_unique<CountingPackedWeights>(
            OpType::kLinear,
            selector,
            MakeTestBuffer(64),
            nullptr));

    ASSERT_FALSE(duplicate_status.ok());
    EXPECT_EQ(duplicate_status.code(), StatusCode::kAlreadyExists);
}

}// namespace
}// namespace aethermind
