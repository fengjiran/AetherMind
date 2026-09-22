#include "aethermind/inference/weight_binding_storage.h"

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

TEST(WeightBindingStorage, AddReturnsViewOverBorrowedData) {
    WeightBindingStorage storage;
    alignas(64) const float data[24] = {};

    const TensorView view = storage.Add(data, DataType::Float32(), {2, 3, 4});

    EXPECT_EQ(storage.size(), 1U);
    EXPECT_FALSE(storage.empty());
    EXPECT_EQ(view.data(), data);
    EXPECT_EQ(view.dtype(), DataType::Float32());
    EXPECT_EQ(Copy(view.shape()), (std::vector<int64_t>{2, 3, 4}));
    EXPECT_EQ(Copy(view.strides()), (std::vector<int64_t>{12, 4, 1}));
    EXPECT_EQ(view.alignment(), 0U) << "raw weight sources carry no alignment";
    EXPECT_TRUE(view.is_valid());
}

TEST(WeightBindingStorage, RankOneShapeHasUnitStride) {
    WeightBindingStorage storage;
    alignas(64) const float data[8] = {};

    const TensorView view = storage.Add(data, DataType::Float32(), {8});

    EXPECT_EQ(Copy(view.shape()), (std::vector<int64_t>{8}));
    EXPECT_EQ(Copy(view.strides()), (std::vector<int64_t>{1}));
    EXPECT_TRUE(view.is_valid());
}

TEST(WeightBindingStorage, RankZeroShapeHasNoStrides) {
    WeightBindingStorage storage;
    alignas(64) const float data[1] = {};

    const TensorView view = storage.Add(data, DataType::Float32(), {});

    EXPECT_EQ(view.rank(), 0);
    EXPECT_TRUE(Copy(view.shape()).empty());
    EXPECT_TRUE(Copy(view.strides()).empty());
}

TEST(WeightBindingStorage, EachEntryKeepsItsOwnMetadata) {
    WeightBindingStorage storage;
    alignas(64) const float matrix[6] = {};
    alignas(64) const float vector[3] = {};

    const TensorView first = storage.Add(matrix, DataType::Float32(), {2, 3});
    const TensorView second = storage.Add(vector, DataType::Float32(), {3});

    EXPECT_EQ(storage.size(), 2U);
    EXPECT_EQ(first.data(), matrix);
    EXPECT_EQ(second.data(), vector);
    EXPECT_EQ(Copy(first.shape()), (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(Copy(second.shape()), (std::vector<int64_t>{3}));
    EXPECT_EQ(Copy(second.strides()), (std::vector<int64_t>{1}));
}

TEST(WeightBindingStorage, EarlierViewsSurviveLaterAdditions) {
    WeightBindingStorage storage;
    alignas(64) const float data[6] = {};
    const TensorView view = storage.Add(data, DataType::Float32(), {2, 3});
    const std::vector<int64_t> expected_shape = Copy(view.shape());
    const std::vector<int64_t> expected_strides = Copy(view.strides());

    // Grow well past the first reallocation so the entry vector must move its
    // elements; the inner shape/stride buffers keep their addresses.
    alignas(64) const float filler[1] = {};
    for (int i = 0; i < 64; ++i) {
        storage.Add(filler, DataType::Float32(), {1});
    }

    ASSERT_EQ(storage.size(), 65U);
    EXPECT_EQ(view.data(), data);
    EXPECT_EQ(Copy(view.shape()), expected_shape);
    EXPECT_EQ(Copy(view.strides()), expected_strides);
    EXPECT_TRUE(view.is_valid());
}

TEST(WeightBindingStorage, EarlierViewsSurviveMoveConstruction) {
    WeightBindingStorage storage;
    alignas(64) const float data[6] = {};
    const TensorView view = storage.Add(data, DataType::Float32(), {2, 3});
    const std::vector<int64_t> expected_shape = Copy(view.shape());
    const std::vector<int64_t> expected_strides = Copy(view.strides());

    const WeightBindingStorage moved(std::move(storage));

    EXPECT_EQ(moved.size(), 1U);
    EXPECT_EQ(view.data(), data);
    EXPECT_EQ(Copy(view.shape()), expected_shape);
    EXPECT_EQ(Copy(view.strides()), expected_strides);
    EXPECT_TRUE(view.is_valid());
}

TEST(WeightBindingStorage, EarlierViewsSurviveMoveAssignment) {
    WeightBindingStorage storage;
    alignas(64) const float data[6] = {};
    const TensorView view = storage.Add(data, DataType::Float32(), {2, 3});
    const std::vector<int64_t> expected_shape = Copy(view.shape());
    const std::vector<int64_t> expected_strides = Copy(view.strides());

    WeightBindingStorage target;
    alignas(64) const float stale[1] = {};
    target.Add(stale, DataType::Float32(), {1});
    target = std::move(storage);

    EXPECT_EQ(target.size(), 1U);
    EXPECT_EQ(view.data(), data);
    EXPECT_EQ(Copy(view.shape()), expected_shape);
    EXPECT_EQ(Copy(view.strides()), expected_strides);
    EXPECT_TRUE(view.is_valid());
}

TEST(WeightBindingStorage, StartsEmpty) {
    const WeightBindingStorage storage;

    EXPECT_EQ(storage.size(), 0U);
    EXPECT_TRUE(storage.empty());
}

} // namespace
