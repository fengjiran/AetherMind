#include "aethermind/inference/weight_binding_metadata.h"

#include "aethermind/base/tensor_view.h"
#include "aethermind/dtypes/data_type.h"

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <utility>
#include <vector>

namespace {

using namespace aethermind;

std::vector<int64_t> Copy(IntArrayView view) {
    return {view.begin(), view.end()};
}

TEST(WeightBindingMetadata, AddReturnsViewOverBorrowedData) {
    WeightBindingMetadata metadata;
    alignas(64) const float data[24] = {};

    const auto view_result = metadata.Add(data, DataType::Float32(), {2, 3, 4});
    ASSERT_TRUE(view_result.ok()) << view_result.status().ToString();
    const TensorView view = *view_result;

    EXPECT_EQ(metadata.size(), 1U);
    EXPECT_FALSE(metadata.empty());
    EXPECT_EQ(view.data(), data);
    EXPECT_EQ(view.dtype(), DataType::Float32());
    EXPECT_EQ(Copy(view.shape()), (std::vector<int64_t>{2, 3, 4}));
    EXPECT_EQ(Copy(view.strides()), (std::vector<int64_t>{12, 4, 1}));
    EXPECT_EQ(view.alignment(), alignof(float));
    EXPECT_TRUE(view.is_valid());
}

TEST(WeightBindingMetadata, RankOneShapeHasUnitStride) {
    WeightBindingMetadata metadata;
    alignas(64) const float data[8] = {};

    const auto view_result = metadata.Add(data, DataType::Float32(), {8});
    ASSERT_TRUE(view_result.ok()) << view_result.status().ToString();
    const TensorView view = *view_result;

    EXPECT_EQ(Copy(view.shape()), (std::vector<int64_t>{8}));
    EXPECT_EQ(Copy(view.strides()), (std::vector<int64_t>{1}));
    EXPECT_TRUE(view.is_valid());
}

TEST(WeightBindingMetadata, RejectsMisalignedDataWithoutPublishingEntry) {
    WeightBindingMetadata metadata;
    alignas(64) const std::byte bytes[128]{};
    const auto original = metadata.Add(bytes, DataType::Float32(), {2});
    ASSERT_TRUE(original.ok());
    const DataType types[]{
            DataType::Float32(),
            DataType::Double(),
            DataType::Float(16),
            DataType::BFloat(16),
            DataType::Int(16),
            DataType::UInt(32),
            DataType::Int(64),
            DataType::Make<complex<float>>(),
    };
    for (const auto dtype: types) {
        SCOPED_TRACE(ToString(dtype));
        const auto result = metadata.Add(bytes + 1, dtype, {2});
        EXPECT_EQ(result.status(), StatusCode::kInvalidArgument);
        EXPECT_EQ(metadata.size(), 1U);
        EXPECT_TRUE(original->is_valid());
        EXPECT_EQ(Copy(original->shape()), (std::vector<int64_t>{2}));
    }
    EXPECT_TRUE(metadata.Add(bytes, DataType::Float32(), {2}).ok());
    EXPECT_EQ(metadata.size(), 2U);
}

TEST(WeightBindingMetadata, RequiresScalarAlignmentWithoutRequiringSimdAlignment) {
    WeightBindingMetadata metadata;
    alignas(64) const std::byte bytes[64]{};
    const auto view = metadata.Add(bytes + sizeof(float), DataType::Float(32, 3), {1});
    ASSERT_TRUE(view.ok()) << view.status().ToString();
    EXPECT_EQ(view->alignment(), alignof(float));
    EXPECT_EQ(view->data(), bytes + sizeof(float));
}

TEST(WeightBindingMetadata, ByteWeightsAllowAnyAddress) {
    WeightBindingMetadata metadata;
    alignas(64) const std::byte bytes[8]{};
    const auto view = metadata.Add(bytes + 1, DataType::UInt(8), {2});
    ASSERT_TRUE(view.ok()) << view.status().ToString();
    EXPECT_EQ(view->alignment(), 1U);
}

TEST(WeightBindingMetadata, RankZeroShapeHasNoStrides) {
    WeightBindingMetadata metadata;
    alignas(64) const float data[1] = {};

    const auto view_result = metadata.Add(data, DataType::Float32(), {});
    ASSERT_TRUE(view_result.ok()) << view_result.status().ToString();
    const TensorView view = *view_result;

    EXPECT_EQ(view.rank(), 0);
    EXPECT_TRUE(Copy(view.shape()).empty());
    EXPECT_TRUE(Copy(view.strides()).empty());
}

TEST(WeightBindingMetadata, ShapeProductOverflowReturnsStatus) {
    WeightBindingMetadata metadata;
    const auto result = metadata.Add(nullptr, DataType::Float32(), {int64_t{1} << 62, 8});
    EXPECT_EQ(result.status(), StatusCode::kOverflow);
    EXPECT_TRUE(metadata.empty());
}

TEST(WeightBindingMetadata, InvalidShapesPreserveEarlierViewsAndAllowRetry) {
    WeightBindingMetadata metadata;
    const float data[6] = {};
    const auto original = metadata.Add(data, DataType::Float32(), {2, 3});
    ASSERT_TRUE(original.ok());
    const std::vector<int64_t> invalid_shapes[]{
            {0, std::numeric_limits<int64_t>::max(), 2},
            {std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max(), 0},
            {-1, 0},
    };
    for (const auto& shape: invalid_shapes) {
        const auto result = metadata.Add(nullptr, DataType::Float32(), shape);
        EXPECT_EQ(result.status(), shape[0] < 0 ? StatusCode::kInvalidArgument : StatusCode::kOverflow);
        EXPECT_EQ(metadata.size(), 1U);
        EXPECT_EQ(original->data(), data);
        EXPECT_EQ(Copy(original->shape()), (std::vector<int64_t>{2, 3}));
        EXPECT_EQ(Copy(original->strides()), (std::vector<int64_t>{3, 1}));
    }
    EXPECT_TRUE(metadata.Add(data, DataType::Float32(), {6}).ok());
    EXPECT_EQ(metadata.size(), 2U);
}

TEST(WeightBindingMetadata, EmptyTensorKeepsRepresentableStrides) {
    WeightBindingMetadata metadata;
    const auto view = metadata.Add(nullptr, DataType::Float32(), {2, 0, 3});
    ASSERT_TRUE(view.ok()) << view.status().ToString();
    EXPECT_TRUE(view->is_valid());
    EXPECT_EQ(view->numel(), 0);
    EXPECT_EQ(Copy(view->strides()), (std::vector<int64_t>{0, 3, 1}));
}

TEST(WeightBindingMetadata, EachEntryKeepsItsOwnMetadata) {
    WeightBindingMetadata metadata;
    alignas(64) const float matrix[6] = {};
    alignas(64) const float vector[3] = {};

    const auto first_result = metadata.Add(matrix, DataType::Float32(), {2, 3});
    ASSERT_TRUE(first_result.ok()) << first_result.status().ToString();
    const TensorView first = *first_result;
    const auto second_result = metadata.Add(vector, DataType::Float32(), {3});
    ASSERT_TRUE(second_result.ok()) << second_result.status().ToString();
    const TensorView second = *second_result;

    EXPECT_EQ(metadata.size(), 2U);
    EXPECT_EQ(first.data(), matrix);
    EXPECT_EQ(second.data(), vector);
    EXPECT_EQ(Copy(first.shape()), (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(Copy(second.shape()), (std::vector<int64_t>{3}));
    EXPECT_EQ(Copy(second.strides()), (std::vector<int64_t>{1}));
}

TEST(WeightBindingMetadata, EarlierViewsSurviveLaterAdditions) {
    WeightBindingMetadata metadata;
    alignas(64) const float data[6] = {};
    const auto view_result = metadata.Add(data, DataType::Float32(), {2, 3});
    ASSERT_TRUE(view_result.ok()) << view_result.status().ToString();
    const TensorView view = *view_result;
    const std::vector<int64_t> expected_shape = Copy(view.shape());
    const std::vector<int64_t> expected_strides = Copy(view.strides());

    // Grow well past the first reallocation so the entry vector must move its
    // elements; the inner shape/stride buffers keep their addresses.
    alignas(64) const float filler[1] = {};
    for (int i = 0; i < 64; ++i) {
        ASSERT_TRUE(metadata.Add(filler, DataType::Float32(), {1}).ok());
    }

    ASSERT_EQ(metadata.size(), 65U);
    EXPECT_EQ(view.data(), data);
    EXPECT_EQ(Copy(view.shape()), expected_shape);
    EXPECT_EQ(Copy(view.strides()), expected_strides);
    EXPECT_TRUE(view.is_valid());
}

TEST(WeightBindingMetadata, EarlierViewsSurviveMoveConstruction) {
    WeightBindingMetadata metadata;
    alignas(64) const float data[6] = {};
    const auto view_result = metadata.Add(data, DataType::Float32(), {2, 3});
    ASSERT_TRUE(view_result.ok()) << view_result.status().ToString();
    const TensorView view = *view_result;
    const std::vector<int64_t> expected_shape = Copy(view.shape());
    const std::vector<int64_t> expected_strides = Copy(view.strides());

    const WeightBindingMetadata moved(std::move(metadata));

    EXPECT_EQ(moved.size(), 1U);
    EXPECT_EQ(view.data(), data);
    EXPECT_EQ(Copy(view.shape()), expected_shape);
    EXPECT_EQ(Copy(view.strides()), expected_strides);
    EXPECT_TRUE(view.is_valid());
}

TEST(WeightBindingMetadata, EarlierViewsSurviveMoveAssignment) {
    WeightBindingMetadata metadata;
    alignas(64) const float data[6] = {};
    const auto view_result = metadata.Add(data, DataType::Float32(), {2, 3});
    ASSERT_TRUE(view_result.ok()) << view_result.status().ToString();
    const TensorView view = *view_result;
    const std::vector<int64_t> expected_shape = Copy(view.shape());
    const std::vector<int64_t> expected_strides = Copy(view.strides());

    WeightBindingMetadata target;
    alignas(64) const float stale[1] = {};
    ASSERT_TRUE(target.Add(stale, DataType::Float32(), {1}).ok());
    target = std::move(metadata);

    EXPECT_EQ(target.size(), 1U);
    EXPECT_EQ(view.data(), data);
    EXPECT_EQ(Copy(view.shape()), expected_shape);
    EXPECT_EQ(Copy(view.strides()), expected_strides);
    EXPECT_TRUE(view.is_valid());
}

TEST(WeightBindingMetadata, StartsEmpty) {
    const WeightBindingMetadata metadata;

    EXPECT_EQ(metadata.size(), 0U);
    EXPECT_TRUE(metadata.empty());
}

} // namespace
