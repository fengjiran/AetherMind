#include "aethermind/inference/weight_binding_metadata.h"

#include "aethermind/base/tensor_view.h"
#include "aethermind/dtypes/data_type.h"

#include <cstdint>
#include <gtest/gtest.h>
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

    const TensorView view = metadata.Add(data, DataType::Float32(), {2, 3, 4});

    EXPECT_EQ(metadata.size(), 1U);
    EXPECT_FALSE(metadata.empty());
    EXPECT_EQ(view.data(), data);
    EXPECT_EQ(view.dtype(), DataType::Float32());
    EXPECT_EQ(Copy(view.shape()), (std::vector<int64_t>{2, 3, 4}));
    EXPECT_EQ(Copy(view.strides()), (std::vector<int64_t>{12, 4, 1}));
    EXPECT_EQ(view.alignment(), 0U) << "raw weight sources carry no alignment";
    EXPECT_TRUE(view.is_valid());
}

TEST(WeightBindingMetadata, RankOneShapeHasUnitStride) {
    WeightBindingMetadata metadata;
    alignas(64) const float data[8] = {};

    const TensorView view = metadata.Add(data, DataType::Float32(), {8});

    EXPECT_EQ(Copy(view.shape()), (std::vector<int64_t>{8}));
    EXPECT_EQ(Copy(view.strides()), (std::vector<int64_t>{1}));
    EXPECT_TRUE(view.is_valid());
}

TEST(WeightBindingMetadata, RankZeroShapeHasNoStrides) {
    WeightBindingMetadata metadata;
    alignas(64) const float data[1] = {};

    const TensorView view = metadata.Add(data, DataType::Float32(), {});

    EXPECT_EQ(view.rank(), 0);
    EXPECT_TRUE(Copy(view.shape()).empty());
    EXPECT_TRUE(Copy(view.strides()).empty());
}

// Compact row-major strides accumulate the shape product in int64_t; a product
// that does not fit would silently produce wrong strides, so it is a fatal check.
TEST(WeightBindingMetadata, ShapeProductOverflowDeath) {
    WeightBindingMetadata metadata;
    alignas(64) const float data[1] = {};

    EXPECT_DEATH(metadata.Add(data, DataType::Float32(), {int64_t{1} << 62, 8}),
                 "Check failed");
}

TEST(WeightBindingMetadata, EachEntryKeepsItsOwnMetadata) {
    WeightBindingMetadata metadata;
    alignas(64) const float matrix[6] = {};
    alignas(64) const float vector[3] = {};

    const TensorView first = metadata.Add(matrix, DataType::Float32(), {2, 3});
    const TensorView second = metadata.Add(vector, DataType::Float32(), {3});

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
    const TensorView view = metadata.Add(data, DataType::Float32(), {2, 3});
    const std::vector<int64_t> expected_shape = Copy(view.shape());
    const std::vector<int64_t> expected_strides = Copy(view.strides());

    // Grow well past the first reallocation so the entry vector must move its
    // elements; the inner shape/stride buffers keep their addresses.
    alignas(64) const float filler[1] = {};
    for (int i = 0; i < 64; ++i) {
        metadata.Add(filler, DataType::Float32(), {1});
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
    const TensorView view = metadata.Add(data, DataType::Float32(), {2, 3});
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
    const TensorView view = metadata.Add(data, DataType::Float32(), {2, 3});
    const std::vector<int64_t> expected_shape = Copy(view.shape());
    const std::vector<int64_t> expected_strides = Copy(view.strides());

    WeightBindingMetadata target;
    alignas(64) const float stale[1] = {};
    target.Add(stale, DataType::Float32(), {1});
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
