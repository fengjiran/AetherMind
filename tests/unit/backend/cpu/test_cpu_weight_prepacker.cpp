#include "aethermind/backend/cpu/cpu_weight_prepacker.h"

#include "aethermind/backend/packed_weights.h"
#include "aethermind/base/kernel_selector.h"
#include "aethermind/base/status.h"
#include "aethermind/base/tensor.h"
#include "aethermind/memory/buffer.h"
#include "aethermind/operators/op_type.h"

#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>

#include <array>
#include <span>
#include <vector>

namespace {

using namespace aethermind;

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
    const std::array<int64_t, 2> shape = {rows, cols};
    ShapeAndStride shape_and_stride;
    shape_and_stride.set_contiguous(shape);

    const size_t element_count = static_cast<size_t>(rows * cols);
    return Tensor(MakeTestBuffer(element_count * sizeof(float)),
                  0,
                  DataType::Float32(),
                  shape_and_stride);
}

KernelSelector MakePackedCpuSelector() {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
            .phase = ExecPhase::kBoth,
    };
}

TEST(CpuWeightPrepacker, PackBuildsPackedWeightsWithCpuStorageAndSelectorMetadata) {
    CpuWeightPrepacker prepacker;
    const Tensor logical_weight = MakeLogicalWeightTensor(4, 8);
    const KernelSelector selector = MakePackedCpuSelector();

    const auto packed = prepacker.Pack(OpType::kLinear, logical_weight, selector);

    ASSERT_TRUE(packed.ok());
    ASSERT_NE(*packed, nullptr);
    EXPECT_EQ((*packed)->op_type(), OpType::kLinear);
    EXPECT_EQ((*packed)->selector(), selector);
    EXPECT_TRUE((*packed)->storage().is_initialized());
    EXPECT_TRUE((*packed)->storage().device().is_cpu());
    EXPECT_GT((*packed)->storage().nbytes(), 0U);
    EXPECT_EQ((*packed)->recipe(), CpuWeightPrepacker::RecipeFor(selector));
    EXPECT_FALSE((*packed)->recipe().layout.empty());
}

TEST(CpuWeightPrepacker, RecipeForIsDeterministicPerSelector) {
    const KernelSelector selector = MakePackedCpuSelector();
    EXPECT_EQ(CpuWeightPrepacker::RecipeFor(selector),
              CpuWeightPrepacker::RecipeFor(selector));
    EXPECT_FALSE(CpuWeightPrepacker::RecipeFor(selector).layout.empty());
}

TEST(CpuWeightPrepacker, IdentityPackingPreservesStrongerSourceAlignment) {
    constexpr std::array<int64_t, 2> shape = {2, 4};
    ShapeAndStride shape_and_stride;
    shape_and_stride.set_contiguous(shape);
    Tensor logical_weight(MakeTestBuffer(8 * sizeof(float), 128),
                          0,
                          DataType::Float32(),
                          shape_and_stride);
    ASSERT_TRUE(logical_weight.is_initialized());

    CpuWeightPrepacker prepacker;
    const auto packed = prepacker.Pack(
            OpType::kLinear, logical_weight, MakePackedCpuSelector());

    ASSERT_TRUE(packed.ok()) << packed.status().ToString();
    ASSERT_NE(*packed, nullptr);
    EXPECT_EQ((*packed)->recipe().layout, cpu::kCpuIdentityPackingLayout);
    EXPECT_EQ((*packed)->recipe().alignment, cpu::kCpuIdentityPackingAlignment);
    EXPECT_GE((*packed)->storage().alignment(), size_t{128});
}

TEST(CpuWeightPrepacker, PackRejectsNonPackedWeightFormatRequests) {
    CpuWeightPrepacker prepacker;
    const Tensor logical_weight = MakeLogicalWeightTensor(2, 4);
    KernelSelector selector = MakePackedCpuSelector();
    selector.weight_format = WeightFormat::kPlain;

    const auto packed = prepacker.Pack(OpType::kLinear, logical_weight, selector);

    ASSERT_FALSE(packed.ok());
    EXPECT_EQ(packed.status().code(), StatusCode::kInvalidArgument);
}

TEST(CpuWeightPrepacker, PackAcceptsLogicalWeightTensorView) {
    CpuWeightPrepacker prepacker;
    const Tensor logical_weight = MakeLogicalWeightTensor(2, 4);
    const KernelSelector selector = MakePackedCpuSelector();

    const auto packed = prepacker.Pack(OpType::kLinear, logical_weight.view(), selector);

    ASSERT_TRUE(packed.ok());
    ASSERT_NE(*packed, nullptr);
    EXPECT_EQ((*packed)->op_type(), OpType::kLinear);
    EXPECT_EQ((*packed)->selector(), selector);
    EXPECT_EQ((*packed)->storage().nbytes(), logical_weight.logical_nbytes());
}

TEST(CpuWeightPrepacker, PackComponentsConcatenatesRecipeOrderedViews) {
    CpuWeightPrepacker prepacker;
    const KernelSelector selector = MakePackedCpuSelector();

    std::vector<float> q_data(8), k_data(12), v_data(4);
    for (size_t i = 0; i < q_data.size(); ++i) q_data[i] = static_cast<float>(i);
    for (size_t i = 0; i < k_data.size(); ++i) k_data[i] = static_cast<float>(100 + i);
    for (size_t i = 0; i < v_data.size(); ++i) v_data[i] = static_cast<float>(200 + i);

    const int64_t q_shape[2] = {2, 4};
    const int64_t k_shape[2] = {3, 4};
    const int64_t v_shape[2] = {1, 4};
    const int64_t strides[2] = {4, 1};
    const TensorView components[] = {
            TensorView(q_data.data(), DataType::Float32(), q_shape, strides),
            TensorView(k_data.data(), DataType::Float32(), k_shape, strides),
            TensorView(v_data.data(), DataType::Float32(), v_shape, strides),
    };

    const auto packed = prepacker.Pack(OpType::kQkvLinear, components, selector);

    ASSERT_TRUE(packed.ok()) << packed.status().ToString();
    ASSERT_NE(*packed, nullptr);
    EXPECT_EQ((*packed)->op_type(), OpType::kQkvLinear);
    EXPECT_EQ((*packed)->selector(), selector);
    EXPECT_EQ((*packed)->recipe(), CpuWeightPrepacker::RecipeFor(selector));
    EXPECT_EQ((*packed)->logical_shape(), (std::vector<int64_t>{6, 4}));
    EXPECT_EQ((*packed)->storage().nbytes(), 6 * 4 * sizeof(float));

    // The fused payload is the exact recipe-ordered concatenation.
    const auto* fused = static_cast<const char*>((*packed)->storage().data());
    size_t cursor = 0;
    for (const TensorView& component: components) {
        EXPECT_EQ(std::memcmp(fused + cursor, component.data(),
                              component.logical_nbytes()),
                  0);
        cursor += component.logical_nbytes();
    }
}

TEST(CpuWeightPrepacker, PackComponentsRejectsDtypeMismatch) {
    CpuWeightPrepacker prepacker;
    const KernelSelector selector = MakePackedCpuSelector();

    std::vector<float> f32_data(8);
    std::vector<int32_t> i32_data(8);
    const int64_t shape[2] = {2, 4};
    const int64_t strides[2] = {4, 1};
    const TensorView components[] = {
            TensorView(f32_data.data(), DataType::Float32(), shape, strides),
            TensorView(i32_data.data(), DataType::Int(32), shape, strides),
    };

    const auto packed = prepacker.Pack(OpType::kQkvLinear, components, selector);

    ASSERT_FALSE(packed.ok());
    EXPECT_EQ(packed.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(packed.status().message().find("share one dtype"),
              std::string::npos);
}

TEST(CpuWeightPrepacker, PackComponentsRejectsFeatureCountMismatch) {
    CpuWeightPrepacker prepacker;
    const KernelSelector selector = MakePackedCpuSelector();

    std::vector<float> first(8), second(16);
    const int64_t first_shape[2] = {2, 4};
    const int64_t first_strides[2] = {4, 1};
    const int64_t second_shape[2] = {2, 8};
    const int64_t second_strides[2] = {8, 1};
    const TensorView components[] = {
            TensorView(first.data(), DataType::Float32(), first_shape, first_strides),
            TensorView(second.data(), DataType::Float32(), second_shape, second_strides),
    };

    const auto packed = prepacker.Pack(OpType::kGateUpLinear, components, selector);

    ASSERT_FALSE(packed.ok());
    EXPECT_EQ(packed.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(packed.status().message().find("feature count"),
              std::string::npos);
}

TEST(CpuWeightPrepacker, PackComponentsRejectsNonRank2View) {
    CpuWeightPrepacker prepacker;
    const KernelSelector selector = MakePackedCpuSelector();

    // Composite bindings demand rank-2 components; a rank-1 partner must be
    // rejected even when the other component is well-formed.
    std::vector<float> matrix(8), rank_one(4);
    const int64_t matrix_shape[2] = {2, 4};
    const int64_t matrix_strides[2] = {4, 1};
    const int64_t vector_shape[1] = {4};
    const int64_t vector_strides[1] = {1};
    const TensorView components[] = {
            TensorView(matrix.data(), DataType::Float32(), matrix_shape, matrix_strides),
            TensorView(rank_one.data(), DataType::Float32(), vector_shape, vector_strides),
    };

    const auto packed = prepacker.Pack(OpType::kQkvLinear, components, selector);

    ASSERT_FALSE(packed.ok());
    EXPECT_EQ(packed.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(packed.status().message().find("rank 2"), std::string::npos);
}

TEST(CpuWeightPrepacker, PackComponentsRejectsEmptyComponentList) {
    CpuWeightPrepacker prepacker;
    const KernelSelector selector = MakePackedCpuSelector();

    const auto packed = prepacker.Pack(
            OpType::kLinear, std::span<const TensorView>{}, selector);

    ASSERT_FALSE(packed.ok());
    EXPECT_EQ(packed.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(packed.status().message().find("at least one"),
              std::string::npos);
}

} // namespace
