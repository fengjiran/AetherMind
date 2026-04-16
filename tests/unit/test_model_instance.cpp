#include "aethermind/model/model_instance.h"

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

TEST(ModelInstance, StoresPackedWeightsInBackendSidecar) {
    ModelInstance model_instance;
    const KernelSelector selector = MakePackedCpuSelector();

    ASSERT_TRUE(model_instance.StorePackedWeights(std::make_unique<CountingPackedWeights>(
                                          OpType::kLinear,
                                          selector,
                                          MakeTestBuffer(256),
                                          nullptr))
                        .ok());

    const PackedWeights* found = model_instance.FindPackedWeights(OpType::kLinear, selector);
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->op_type(), OpType::kLinear);
    EXPECT_EQ(found->selector(), selector);
    EXPECT_TRUE(found->storage().is_initialized());
}

TEST(ModelInstance, PackedWeightsRemainAliveWhileModelInstanceLives) {
    bool destroyed = false;
    const KernelSelector selector = MakePackedCpuSelector();

    ModelInstance model_instance;
    {
        CpuBackend backend;
        (void) backend;

        ASSERT_TRUE(model_instance.StorePackedWeights(std::make_unique<CountingPackedWeights>(
                                              OpType::kLinear,
                                              selector,
                                              MakeTestBuffer(128),
                                              &destroyed))
                            .ok());
    }

    const PackedWeights* found = model_instance.FindPackedWeights(OpType::kLinear, selector);
    ASSERT_NE(found, nullptr);
    EXPECT_FALSE(destroyed);
    EXPECT_TRUE(found->storage().device().is_cpu());
}

TEST(ModelInstance, DestroyingModelInstanceReleasesPackedWeights) {
    bool destroyed = false;
    const KernelSelector selector = MakePackedCpuSelector();

    {
        ModelInstance model_instance;
        ASSERT_TRUE(model_instance.StorePackedWeights(std::make_unique<CountingPackedWeights>(
                                              OpType::kLinear,
                                              selector,
                                              MakeTestBuffer(64),
                                              &destroyed))
                            .ok());
        EXPECT_FALSE(destroyed);
    }

    EXPECT_TRUE(destroyed);
}

}// namespace
}// namespace aethermind
