#include "aethermind/backend/cpu/cpu_weight_prepacker.h"

#include "aethermind/backend/kernel_selector.h"
#include "aethermind/backend/packed_weights.h"
#include "aethermind/base/status.h"
#include "aethermind/base/tensor.h"
#include "aethermind/memory/buffer.h"
#include "aethermind/operators/op_type.h"

#include <gtest/gtest.h>

#include <cstdlib>

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

Tensor MakeLogicalWeightTensor(int64_t rows, int64_t cols) {
    ShapeAndStride shape_and_stride;
    shape_and_stride.set_contiguous({rows, cols});

    const size_t element_count = static_cast<size_t>(rows * cols);
    return Tensor(MakeTestBuffer(element_count * sizeof(float)),
                  0,
                  DataType::Float32(),
                  shape_and_stride);
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

TEST(CpuWeightPrepacker, PackBuildsPackedWeightsWithCpuStorageAndSelectorMetadata) {
    CpuWeightPrepacker prepacker;
    const Tensor logical_weight = MakeLogicalWeightTensor(4, 8);
    const KernelSelector selector = MakePackedCpuSelector();

    const StatusOr<std::unique_ptr<PackedWeights>> packed =
            prepacker.Pack(OpType::kLinear, logical_weight, selector);

    ASSERT_TRUE(packed.ok());
    ASSERT_NE(*packed, nullptr);
    EXPECT_EQ((*packed)->op_type(), OpType::kLinear);
    EXPECT_EQ((*packed)->selector(), selector);
    EXPECT_TRUE((*packed)->storage().is_initialized());
    EXPECT_TRUE((*packed)->storage().device().is_cpu());
    EXPECT_GT((*packed)->storage().nbytes(), 0U);
}

TEST(CpuWeightPrepacker, PackRejectsNonPackedWeightFormatRequests) {
    CpuWeightPrepacker prepacker;
    const Tensor logical_weight = MakeLogicalWeightTensor(2, 4);
    KernelSelector selector = MakePackedCpuSelector();
    selector.weight_format = WeightFormat::kPlain;

    const StatusOr<std::unique_ptr<PackedWeights>> packed =
            prepacker.Pack(OpType::kLinear, logical_weight, selector);

    ASSERT_FALSE(packed.ok());
    EXPECT_EQ(packed.status().code(), StatusCode::kInvalidArgument);
}

}// namespace
}// namespace aethermind
