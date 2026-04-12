#include "aethermind/base/tensor.h"
#include "test_utils/tensor_assert.h"
#include "test_utils/tensor_factory.h"
#include "test_utils/tensor_random.h"

#include <gtest/gtest.h>

using namespace aethermind;
using namespace aethermind::test_utils;

namespace {

TEST(TensorRandomNew, UniformTensorDeterministicWithSameSeed) {
    auto a = RandomUniformTensor({128}, DataType::Float32(), -1.0, 1.0, 42);
    auto b = RandomUniformTensor({128}, DataType::Float32(), -1.0, 1.0, 42);

    ASSERT_TRUE(a.numel() == b.numel());
    const float* ad = static_cast<const float*>(a.data());
    const float* bd = static_cast<const float*>(b.data());
    for (int64_t i = 0; i < a.numel(); ++i) {
        EXPECT_EQ(ad[i], bd[i]);
    }
}

TEST(TensorRandomNew, UniformTensorHonorsRange) {
    auto t = RandomUniformTensor({256}, DataType::Float32(), -2.5, 3.5, 7);

    const float* data = static_cast<const float*>(t.data());
    for (int64_t i = 0; i < t.numel(); ++i) {
        EXPECT_GE(data[i], -2.5F);
        EXPECT_LT(data[i], 3.5F);
    }
}

TEST(TensorRandomNew, RandomIntTensorHonorsRange) {
    auto t = RandomIntTensor({256}, 10, 20, 9, DataType::Int(64));

    const int64_t* data = static_cast<const int64_t*>(t.data());
    for (int64_t i = 0; i < t.numel(); ++i) {
        EXPECT_GE(data[i], 10);
        EXPECT_LT(data[i], 20);
    }
}

TEST(TensorRandomNew, NormalTensorDeterministicWithSameSeed) {
    auto a = RandomNormalTensor({128}, DataType::Double(), 0.5, 2.0, 123);
    auto b = RandomNormalTensor({128}, DataType::Double(), 0.5, 2.0, 123);

    ASSERT_TRUE(a.numel() == b.numel());
    const double* ad = static_cast<const double*>(a.data());
    const double* bd = static_cast<const double*>(b.data());
    for (int64_t i = 0; i < a.numel(); ++i) {
        EXPECT_EQ(ad[i], bd[i]);
    }
}

TEST(TensorAllCloseNew, PassesForCloseFloat32Tensor) {
    auto a = RandomUniformTensor({64}, DataType::Float32(), -1.0, 1.0, 1);
    auto b = RandomUniformTensor({64}, DataType::Float32(), -1.0, 1.0, 1);
    static_cast<float*>(b.mutable_data())[0] += 1e-7F;
    EXPECT_TRUE(ExpectTensorAllClose(a, b, 1e-6, 1e-6));
}

TEST(TensorAllCloseNew, FailsForLargeDifferenceFloat32Tensor) {
    auto a = RandomUniformTensor({64}, DataType::Float32(), -1.0, 1.0, 2);
    auto b = RandomUniformTensor({64}, DataType::Float32(), -1.0, 1.0, 200);
    EXPECT_FALSE(ExpectTensorAllClose(a, b, 1e-6, 1e-6));
}

TEST(TensorAllCloseNew, DetectsShapeMismatchTensor) {
    auto a = RandomUniformTensor({4, 8}, DataType::Float32(), 0.0, 1.0, 3);
    auto b = RandomUniformTensor({8, 4}, DataType::Float32(), 0.0, 1.0, 3);
    EXPECT_FALSE(ExpectTensorAllClose(a, b, 1e-6, 1e-6));
}

TEST(TensorAllCloseNew, SupportsExactIntComparisonTensor) {
    auto a = RandomIntTensor({64}, 0, 100, 99, DataType::Int(32));
    auto b = RandomIntTensor({64}, 0, 100, 99, DataType::Int(32));
    EXPECT_TRUE(ExpectTensorEqual(a, b));

    auto c = RandomIntTensor({64}, 0, 100, 100, DataType::Int(32));
    EXPECT_FALSE(ExpectTensorEqual(a, c));
}

}// namespace