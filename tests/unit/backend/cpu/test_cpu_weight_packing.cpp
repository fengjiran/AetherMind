#include "aethermind/backend/cpu/cpu_bpanel_packing.h"
#include "aethermind/backend/cpu/cpu_identity_packing.h"
#include "backend/cpu/cpu_backend_internal.h"

#include "aethermind/backend/packed_weight.h"
#include "aethermind/base/kernel_selector.h"
#include "aethermind/base/status.h"
#include "aethermind/base/tensor.h"
#include "aethermind/memory/buffer.h"
#include "aethermind/operators/op_type.h"
#include "inference/test_malloc_interposer.h"

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

TEST(CpuWeightPacking, PackBuildsPackedWeightWithCpuStorageAndSelectorMetadata) {
    const Tensor logical_weight = MakeLogicalWeightTensor(4, 8);
    ASSERT_TRUE(logical_weight.is_initialized());
    ASSERT_TRUE(logical_weight.device().is_cpu());
    const std::array<TensorView, 1> components{logical_weight.view()};
    const KernelSelector selector = MakePackedCpuSelector();

    const auto packed = cpu::internal::PackWeightsWithRecipe(OpType::kLinear, components, selector, CpuIdentityPackingRecipe());

    ASSERT_TRUE(packed.ok());
    ASSERT_NE(*packed, nullptr);
    EXPECT_EQ((*packed)->op_type(), OpType::kLinear);
    EXPECT_EQ((*packed)->selector(), selector);
    EXPECT_TRUE((*packed)->storage().is_initialized());
    EXPECT_TRUE((*packed)->storage().device().is_cpu());
    EXPECT_GT((*packed)->storage().nbytes(), 0U);
    EXPECT_EQ((*packed)->recipe(), CpuIdentityPackingRecipe());
    EXPECT_TRUE(IsValidPackingLayout((*packed)->recipe().layout));
}

TEST(CpuWeightPacking, BufferMetadataFailureReturnsResourceExhaustedAndReleasesPayload) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }
    const Tensor logical_weight = MakeLogicalWeightTensor(4, 8);
    ASSERT_TRUE(logical_weight.is_initialized());
    const std::array<TensorView, 1> components{logical_weight.view()};
    const KernelSelector selector = MakePackedCpuSelector();

    test::ScopedMallocFailure failure(sizeof(BufferImpl));
    const auto packed = cpu::internal::PackWeightsWithRecipe(
            OpType::kLinear, components, selector, CpuIdentityPackingRecipe());
    const auto result = failure.Stop();

    ASSERT_FALSE(packed.ok());
    EXPECT_EQ(packed.status().code(), StatusCode::kResourceExhausted);
    EXPECT_TRUE(result.allocation_failed);
    EXPECT_EQ(result.aligned_allocation_calls, 1U);
    EXPECT_TRUE(result.last_aligned_allocation_released);
}

TEST(CpuWeightPacking, IdentityPackingPreservesStrongerSourceAlignment) {
    constexpr std::array<int64_t, 2> shape = {2, 4};
    ShapeAndStride shape_and_stride;
    shape_and_stride.set_contiguous(shape);
    Tensor logical_weight(MakeTestBuffer(8 * sizeof(float), 128),
                          0,
                          DataType::Float32(),
                          shape_and_stride);
    ASSERT_TRUE(logical_weight.is_initialized());
    ASSERT_TRUE(logical_weight.device().is_cpu());
    const std::array<TensorView, 1> components{logical_weight.view()};

    const auto packed = cpu::internal::PackWeightsWithRecipe(
            OpType::kLinear, components, MakePackedCpuSelector(), CpuIdentityPackingRecipe());

    ASSERT_TRUE(packed.ok()) << packed.status().ToString();
    ASSERT_NE(*packed, nullptr);
    EXPECT_EQ((*packed)->recipe().layout, cpu::kCpuIdentityPackingLayout);
    EXPECT_EQ((*packed)->recipe().alignment, cpu::kCpuIdentityPackingAlignment);
    EXPECT_GE((*packed)->storage().alignment(), size_t{128});
}

TEST(CpuWeightPacking, IdentityPackingAcceptsRankOneNormWeight) {
    const std::array<float, 4> data = {1.0F, 2.0F, 3.0F, 4.0F};
    constexpr int64_t shape[] = {4};
    constexpr int64_t strides[] = {1};
    const std::array<TensorView, 1> components{
            TensorView(data.data(), DataType::Float32(), shape, strides)};

    const auto packed = cpu::internal::PackWeightsWithRecipe(
            OpType::kRmsNorm, components, MakePackedCpuSelector(), CpuIdentityPackingRecipe());

    ASSERT_TRUE(packed.ok()) << packed.status().ToString();
    EXPECT_EQ((*packed)->logical_shape(), (std::vector<int64_t>{4}));
    ASSERT_EQ((*packed)->storage().nbytes(), sizeof(data));
    EXPECT_EQ(std::memcmp((*packed)->storage().data(), data.data(), sizeof(data)), 0);
}

TEST(CpuWeightPacking, IdentityPackingAcceptsScalarWeight) {
    const float data = 2.5F;
    const std::array<TensorView, 1> components{
            TensorView(&data, DataType::Float32(), {}, {})};

    const auto packed = cpu::internal::PackWeightsWithRecipe(
            OpType::kLinear, components, MakePackedCpuSelector(), CpuIdentityPackingRecipe());

    ASSERT_TRUE(packed.ok()) << packed.status().ToString();
    EXPECT_TRUE((*packed)->logical_shape().empty());
    ASSERT_EQ((*packed)->storage().nbytes(), sizeof(data));
    EXPECT_EQ(std::memcmp((*packed)->storage().data(), &data, sizeof(data)), 0);
}

TEST(CpuWeightPacking, IdentityPackingAcceptsZeroSizedWeightWithNullData) {
    constexpr int64_t shape[] = {0};
    constexpr int64_t strides[] = {1};
    const std::array<TensorView, 1> components{
            TensorView(nullptr, DataType::Float32(), shape, strides)};

    const auto packed = cpu::internal::PackWeightsWithRecipe(
            OpType::kLinear, components, MakePackedCpuSelector(), CpuIdentityPackingRecipe());

    ASSERT_TRUE(packed.ok()) << packed.status().ToString();
    EXPECT_EQ((*packed)->logical_shape(), (std::vector<int64_t>{0}));
    EXPECT_EQ((*packed)->storage().nbytes(), 0U);
}

TEST(CpuWeightPacking, IdentityPackingRejectsNonContiguousSingleComponent) {
    const std::array<float, 6> data = {1.0F, 2.0F, -1.0F, 3.0F, 4.0F, -1.0F};
    constexpr int64_t shape[] = {2, 2};
    constexpr int64_t strides[] = {3, 1};
    const std::array<TensorView, 1> components{
            TensorView(data.data(), DataType::Float32(), shape, strides)};
    ASSERT_TRUE(components.front().is_valid());
    ASSERT_FALSE(components.front().is_contiguous());

    const auto packed = cpu::internal::PackWeightsWithRecipe(
            OpType::kLinear, components, MakePackedCpuSelector(), CpuIdentityPackingRecipe());

    ASSERT_FALSE(packed.ok());
    EXPECT_EQ(packed.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(packed.status().message().find("contiguous"), std::string::npos);
}

TEST(CpuWeightPacking, PackRejectsInvalidSingleComponent) {
    const std::array<TensorView, 1> components{};

    const auto packed = cpu::internal::PackWeightsWithRecipe(
            OpType::kLinear, components, MakePackedCpuSelector(), CpuIdentityPackingRecipe());

    ASSERT_FALSE(packed.ok());
    EXPECT_EQ(packed.status().code(), StatusCode::kInvalidArgument);
}

TEST(CpuWeightPacking, BpanelPacksLogicalMatrixAndZeroPadsEveryTail) {
    constexpr int64_t n = 17;
    constexpr int64_t k = 513;
    constexpr int64_t shape[2] = {n, k};
    constexpr int64_t strides[2] = {k, 1};
    std::vector<float> logical(static_cast<size_t>(n * k));
    for (size_t i = 0; i < logical.size(); ++i) {
        logical[i] = static_cast<float>(static_cast<int64_t>(i % 23U) - 11) * 0.25F;
    }
    const TensorView logical_view(
            logical.data(), DataType::Float32(), shape, strides);
    const std::array<TensorView, 1> components{logical_view};
    const KernelSelector selector = MakePackedCpuSelector();
    const PackingRecipe recipe = cpu::CpuBPanelF32V1Avx2Recipe();

    const auto required_bytes = cpu::CpuBPanelF32V1PackedByteSize(n, k);
    ASSERT_TRUE(required_bytes.ok()) << required_bytes.status().ToString();
    EXPECT_EQ(*required_bytes, size_t{131072});

    const auto packed = cpu::internal::PackWeightsWithRecipe(
            OpType::kLinear, components, selector, recipe);
    ASSERT_TRUE(packed.ok()) << packed.status().ToString();
    ASSERT_NE(*packed, nullptr);
    EXPECT_EQ((*packed)->recipe(), recipe);
    EXPECT_EQ((*packed)->logical_shape(), (std::vector<int64_t>{n, k}));
    EXPECT_EQ((*packed)->storage().nbytes(), *required_bytes);
    ASSERT_EQ((*packed)->storage().alignment(), recipe.alignment);

    const float* const data = static_cast<const float*>((*packed)->storage().data());
    constexpr int64_t n_blocks = 2;
    for (int64_t padded_k = 0; padded_k < 2 * cpu::kCpuBPanelF32V1KC;
         ++padded_k) {
        for (int64_t padded_n = 0; padded_n < n_blocks * cpu::kCpuBPanelF32V1NR;
             ++padded_n) {
            const size_t index =
                    (((static_cast<size_t>(padded_k / cpu::kCpuBPanelF32V1KC) *
                               static_cast<size_t>(n_blocks) +
                       static_cast<size_t>(padded_n / cpu::kCpuBPanelF32V1NR)) *
                              static_cast<size_t>(cpu::kCpuBPanelF32V1KC) +
                      static_cast<size_t>(padded_k % cpu::kCpuBPanelF32V1KC)) *
                     static_cast<size_t>(cpu::kCpuBPanelF32V1NR)) +
                    static_cast<size_t>(padded_n % cpu::kCpuBPanelF32V1NR);
            const float expected = padded_n < n && padded_k < k
                                           ? logical[static_cast<size_t>(padded_n * k + padded_k)]
                                           : 0.0F;
            EXPECT_EQ(data[index], expected)
                    << "logical N=" << padded_n << " K=" << padded_k;
        }
    }
}

TEST(CpuWeightPacking, BpanelRejectsUnknownRecipe) {
    const Tensor logical_weight = MakeLogicalWeightTensor(2, 4);
    ASSERT_TRUE(logical_weight.is_initialized());
    ASSERT_TRUE(logical_weight.device().is_cpu());
    const std::array<TensorView, 1> components{logical_weight.view()};
    PackingRecipe unknown{.layout = static_cast<PackingLayout>(0xFF), .alignment = 64};
    const auto packed = cpu::internal::PackWeightsWithRecipe(
            OpType::kLinear, components, MakePackedCpuSelector(), unknown);
    ASSERT_FALSE(packed.ok());
    EXPECT_EQ(packed.status().code(), StatusCode::kInvalidArgument);
}

TEST(CpuWeightPacking, PackRejectsNonPackedWeightFormatRequests) {
    const Tensor logical_weight = MakeLogicalWeightTensor(2, 4);
    ASSERT_TRUE(logical_weight.is_initialized());
    ASSERT_TRUE(logical_weight.device().is_cpu());
    const std::array<TensorView, 1> components{logical_weight.view()};
    KernelSelector selector = MakePackedCpuSelector();
    selector.weight_format = WeightFormat::kPlain;

    const auto packed = cpu::internal::PackWeightsWithRecipe(OpType::kLinear, components, selector, CpuIdentityPackingRecipe());

    ASSERT_FALSE(packed.ok());
    EXPECT_EQ(packed.status().code(), StatusCode::kInvalidArgument);
}

TEST(CpuWeightPacking, PackRejectsUnknownOpType) {
    const std::array<float, 4> data{};
    constexpr int64_t shape[] = {2, 2};
    constexpr int64_t strides[] = {2, 1};
    const std::array<TensorView, 1> components{
            TensorView(data.data(), DataType::Float32(), shape, strides)};

    const auto packed = cpu::internal::PackWeightsWithRecipe(
            OpType::kUnknown, components, MakePackedCpuSelector(), CpuIdentityPackingRecipe());

    ASSERT_FALSE(packed.ok());
    EXPECT_EQ(packed.status().code(), StatusCode::kInvalidArgument);
}

TEST(CpuWeightPacking, PackRejectsNonCpuSelector) {
    const std::array<float, 4> data{};
    constexpr int64_t shape[] = {2, 2};
    constexpr int64_t strides[] = {2, 1};
    const std::array<TensorView, 1> components{
            TensorView(data.data(), DataType::Float32(), shape, strides)};
    KernelSelector selector = MakePackedCpuSelector();
    selector.device_type = DeviceType::kCUDA;

    const auto packed = cpu::internal::PackWeightsWithRecipe(
            OpType::kLinear, components, selector, CpuIdentityPackingRecipe());

    ASSERT_FALSE(packed.ok());
    EXPECT_EQ(packed.status().code(), StatusCode::kInvalidArgument);
}

TEST(CpuWeightPacking, PackSingleComponentCopiesLogicalBytes) {
    Tensor logical_weight = MakeLogicalWeightTensor(2, 4);
    ASSERT_TRUE(logical_weight.is_initialized());
    ASSERT_TRUE(logical_weight.device().is_cpu());
    const std::array<TensorView, 1> components{logical_weight.view()};
    const KernelSelector selector = MakePackedCpuSelector();
    const std::array<float, 8> values = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    std::memcpy(logical_weight.mutable_data(), values.data(), sizeof(values));

    const auto packed = cpu::internal::PackWeightsWithRecipe(OpType::kLinear, components, selector, CpuIdentityPackingRecipe());

    ASSERT_TRUE(packed.ok());
    ASSERT_NE(*packed, nullptr);
    EXPECT_EQ((*packed)->op_type(), OpType::kLinear);
    EXPECT_EQ((*packed)->selector(), selector);
    EXPECT_EQ((*packed)->storage().nbytes(), logical_weight.logical_nbytes());
    EXPECT_EQ(std::memcmp((*packed)->storage().data(), values.data(), sizeof(values)), 0);
}

TEST(CpuWeightPacking, PackComponentsConcatenatesRecipeOrderedViews) {
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

    const auto packed = cpu::internal::PackWeightsWithRecipe(OpType::kQkvLinear, components, selector, CpuIdentityPackingRecipe());

    ASSERT_TRUE(packed.ok()) << packed.status().ToString();
    ASSERT_NE(*packed, nullptr);
    EXPECT_EQ((*packed)->op_type(), OpType::kQkvLinear);
    EXPECT_EQ((*packed)->selector(), selector);
    EXPECT_EQ((*packed)->recipe(), CpuIdentityPackingRecipe());
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

TEST(CpuWeightPacking, PackComponentsRejectsDtypeMismatch) {
    const KernelSelector selector = MakePackedCpuSelector();

    std::vector<float> f32_data(8);
    std::vector<int32_t> i32_data(8);
    const int64_t shape[2] = {2, 4};
    const int64_t strides[2] = {4, 1};
    const TensorView components[] = {
            TensorView(f32_data.data(), DataType::Float32(), shape, strides),
            TensorView(i32_data.data(), DataType::Int(32), shape, strides),
    };

    const auto packed = cpu::internal::PackWeightsWithRecipe(OpType::kQkvLinear, components, selector, CpuIdentityPackingRecipe());

    ASSERT_FALSE(packed.ok());
    EXPECT_EQ(packed.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(packed.status().message().find("share one dtype"),
              std::string::npos);
}

TEST(CpuWeightPacking, PackComponentsAcceptsZeroSizedComponentWithNullData) {
    const std::array<float, 4> data = {1.0F, 2.0F, 3.0F, 4.0F};
    constexpr int64_t empty_shape[] = {0, 4};
    constexpr int64_t shape[] = {1, 4};
    constexpr int64_t strides[] = {4, 1};
    const std::array<TensorView, 2> components{
            TensorView(nullptr, DataType::Float32(), empty_shape, strides),
            TensorView(data.data(), DataType::Float32(), shape, strides)};

    const auto packed = cpu::internal::PackWeightsWithRecipe(
            OpType::kGateUpLinear, components, MakePackedCpuSelector(), CpuIdentityPackingRecipe());

    ASSERT_TRUE(packed.ok()) << packed.status().ToString();
    EXPECT_EQ((*packed)->logical_shape(), (std::vector<int64_t>{1, 4}));
    ASSERT_EQ((*packed)->storage().nbytes(), sizeof(data));
    EXPECT_EQ(std::memcmp((*packed)->storage().data(), data.data(), sizeof(data)), 0);
}

TEST(CpuWeightPacking, PackComponentsRejectsFeatureCountMismatch) {
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

    const auto packed = cpu::internal::PackWeightsWithRecipe(OpType::kGateUpLinear, components, selector, CpuIdentityPackingRecipe());

    ASSERT_FALSE(packed.ok());
    EXPECT_EQ(packed.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(packed.status().message().find("feature count"),
              std::string::npos);
}

TEST(CpuWeightPacking, PackComponentsRejectsNonRank2View) {
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

    const auto packed = cpu::internal::PackWeightsWithRecipe(OpType::kQkvLinear, components, selector, CpuIdentityPackingRecipe());

    ASSERT_FALSE(packed.ok());
    EXPECT_EQ(packed.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(packed.status().message().find("rank 2"), std::string::npos);
}

TEST(CpuWeightPacking, PackComponentsRejectsEmptyComponentList) {
    const KernelSelector selector = MakePackedCpuSelector();

    const auto packed = cpu::internal::PackWeightsWithRecipe(
            OpType::kLinear, std::span<const TensorView>{}, selector, CpuIdentityPackingRecipe());

    ASSERT_FALSE(packed.ok());
    EXPECT_EQ(packed.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(packed.status().message().find("at least one"),
              std::string::npos);
}

} // namespace
