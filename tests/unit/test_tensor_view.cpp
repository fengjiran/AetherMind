#include "aethermind/base/tensor_view.h"

#include "test_utils/tensor_random.h"

#include <gtest/gtest.h>

#include <array>

using namespace aethermind;
using namespace aethermind::test_utils;

namespace {

TEST(TensorView, BorrowsTensorMetadataAndData) {
    Tensor tensor = RandomUniformTensor({2, 3}, DataType::Float32(), -1.0, 1.0, 7);

    TensorView view(tensor);

    ASSERT_TRUE(view.is_valid());
    EXPECT_EQ(view.data(), tensor.data());
    EXPECT_EQ(view.dtype(), tensor.dtype());
    EXPECT_EQ(view.rank(), tensor.rank());
    EXPECT_EQ(view.numel(), tensor.numel());
    EXPECT_TRUE(view.is_contiguous());
}

TEST(TensorView, MutableTensorViewWritesThroughTensorStorage) {
    Tensor tensor = RandomUniformTensor({2, 3}, DataType::Float32(), -1.0, 1.0, 11);

    MutableTensorView view(tensor);

    ASSERT_TRUE(view.is_valid());
    ASSERT_NE(view.data(), nullptr);

    float* view_data = view.data<float>();
    view_data[0] = 42.0F;
    view_data[5] = -3.5F;

    auto* tensor_data = static_cast<float*>(tensor.mutable_data());
    EXPECT_FLOAT_EQ(tensor_data[0], 42.0F);
    EXPECT_FLOAT_EQ(tensor_data[5], -3.5F);
    EXPECT_TRUE(view.is_contiguous());
}

TEST(TensorView, MutableTensorViewSupportsBorrowedRawParts) {
    alignas(64) std::array<float, 6> storage{0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    constexpr std::array<int64_t, 2> kShape{2, 3};
    constexpr std::array<int64_t, 2> kStrides{3, 1};

    MutableTensorView view(storage.data(),
                           DataType::Float32(),
                           IntArrayView{kShape.data(), kShape.size()},
                           IntArrayView{kStrides.data(), kStrides.size()},
                           64);

    ASSERT_TRUE(view.is_valid());
    EXPECT_EQ(view.data(), storage.data());
    EXPECT_EQ(view.alignment(), 64U);
    EXPECT_EQ(view.numel(), 6);

    view.data<float>()[4] = 99.0F;
    EXPECT_FLOAT_EQ(storage[4], 99.0F);
}

}// namespace
