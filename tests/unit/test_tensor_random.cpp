#include "aethermind/base/tensor.h"
#include "test_utils/tensor_assert.h"
#include "test_utils/tensor_factory.h"
#include "test_utils/tensor_random.h"

#include <gtest/gtest.h>

using namespace aethermind;
using namespace aethermind::test_utils;

namespace {

TEST(TensorRandom, UniformDeterministicWithSameSeed) {
    auto a = RandomUniform({128}, DataType::Float32(), -1.0, 1.0, 42);
    auto b = RandomUniform({128}, DataType::Float32(), -1.0, 1.0, 42);

    ASSERT_EQ(a.numel(), b.numel());
    const float* ad = a.const_data_ptr<float>();
    const float* bd = b.const_data_ptr<float>();
    for (int64_t i = 0; i < a.numel(); ++i) {
        EXPECT_EQ(ad[i], bd[i]);
    }
}

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

TEST(TensorRandom, UniformHonorsRange) {
    auto t = RandomUniform({256}, DataType::Float32(), -2.5, 3.5, 7);

    const float* data = t.const_data_ptr<float>();
    for (int64_t i = 0; i < t.numel(); ++i) {
        EXPECT_GE(data[i], -2.5F);
        EXPECT_LT(data[i], 3.5F);
    }
}

TEST(TensorRandom, RandomIntHonorsRange) {
    auto t = RandomInt({256}, 10, 20, 9, DataType::Int(64));

    const int64_t* data = t.const_data_ptr<int64_t>();
    for (int64_t i = 0; i < t.numel(); ++i) {
        EXPECT_GE(data[i], 10);
        EXPECT_LT(data[i], 20);
    }
}

TEST(TensorRandom, NormalDeterministicWithSameSeed) {
    auto a = RandomNormal({128}, DataType::Double(), 0.5, 2.0, 123);
    auto b = RandomNormal({128}, DataType::Double(), 0.5, 2.0, 123);

    ASSERT_EQ(a.numel(), b.numel());
    const double* ad = a.const_data_ptr<double>();
    const double* bd = b.const_data_ptr<double>();
    for (int64_t i = 0; i < a.numel(); ++i) {
        EXPECT_EQ(ad[i], bd[i]);
    }
}

TEST(TensorAllClose, PassesForCloseFloat32) {
    auto a = RandomUniform({64}, DataType::Float32(), -1.0, 1.0, 1);
    auto b = RandomUniform({64}, DataType::Float32(), -1.0, 1.0, 1);
    b.data_ptr<float>()[0] += 1e-7F;
    EXPECT_TRUE(aethermind::test_utils::ExpectTensorAllClose(a, b, 1e-6, 1e-6));
}

TEST(TensorAllClose, FailsForLargeDifferenceFloat32) {
    auto a = RandomUniform({64}, DataType::Float32(), -1.0, 1.0, 2);
    auto b = RandomUniform({64}, DataType::Float32(), -1.0, 1.0, 200);
    EXPECT_FALSE(aethermind::test_utils::ExpectTensorAllClose(a, b, 1e-6, 1e-6));
}

TEST(TensorAllClose, DetectsShapeMismatch) {
    auto a = RandomUniform({4, 8}, DataType::Float32(), 0.0, 1.0, 3);
    auto b = RandomUniform({8, 4}, DataType::Float32(), 0.0, 1.0, 3);
    EXPECT_FALSE(aethermind::test_utils::ExpectTensorAllClose(a, b, 1e-6, 1e-6));
}

TEST(TensorAllClose, SupportsHalf) {
    Tensor_BK a({8}, 0, DataType::Make<Half>(), Device::CPU());
    Tensor_BK b({8}, 0, DataType::Make<Half>(), Device::CPU());

    for (int64_t i = 0; i < a.numel(); ++i) {
        a.data_ptr<Half>()[i] = Half(static_cast<float>(i) * 0.1F);
        b.data_ptr<Half>()[i] = Half(static_cast<float>(i) * 0.1F);
    }
    b.data_ptr<Half>()[2] = Half(static_cast<float>(b.data_ptr<Half>()[2]) + 1e-4F);
    EXPECT_TRUE(aethermind::test_utils::ExpectTensorAllClose(a, b, 1e-3, 1e-3));
}

TEST(TensorAllClose, SupportsBFloat16) {
    Tensor_BK a({8}, 0, DataType::Make<BFloat16>(), Device::CPU());
    Tensor_BK b({8}, 0, DataType::Make<BFloat16>(), Device::CPU());

    for (int64_t i = 0; i < a.numel(); ++i) {
        a.data_ptr<BFloat16>()[i] = BFloat16(static_cast<float>(i) * 0.2F);
        b.data_ptr<BFloat16>()[i] = BFloat16(static_cast<float>(i) * 0.2F);
    }
    b.data_ptr<BFloat16>()[5] = BFloat16(static_cast<float>(b.data_ptr<BFloat16>()[5]) + 1e-4F);
    EXPECT_TRUE(aethermind::test_utils::ExpectTensorAllClose(a, b, 1e-2, 1e-2));
}

TEST(TensorAllClose, SupportsExactIntComparison) {
    auto a = RandomInt({64}, 0, 100, 99, DataType::Int(32));
    auto b = RandomInt({64}, 0, 100, 99, DataType::Int(32));
    EXPECT_TRUE(aethermind::test_utils::ExpectTensorEqual(a, b));

    auto c = RandomInt({64}, 0, 100, 100, DataType::Int(32));
    EXPECT_FALSE(aethermind::test_utils::ExpectTensorEqual(a, c));
}

TEST(TensorAllClose, ReportsMultipleMismatches) {
    auto a = RandomInt({16}, 0, 4, 11, DataType::Int(64));
    auto b = RandomInt({16}, 0, 4, 11, DataType::Int(64));
    b.data_ptr<int64_t>()[1] += 10;
    b.data_ptr<int64_t>()[3] += 10;
    b.data_ptr<int64_t>()[5] += 10;

    auto result = aethermind::test_utils::ExpectTensorEqual(a, b, 2);
    EXPECT_FALSE(result);
}

TEST(TensorAllClose, RejectsIntDtype) {
    auto a = RandomInt({8}, 0, 10, 1, DataType::Int(32));
    auto b = RandomInt({8}, 0, 10, 1, DataType::Int(32));
    EXPECT_FALSE(aethermind::test_utils::ExpectTensorAllClose(a, b, 1e-6, 1e-6));
}

}// namespace
