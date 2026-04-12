//
// Batch 4: Tensor compatibility layer tests
//

#include "aethermind/migration/tensor_compat.h"
#include "test_utils/tensor_factory.h"
#include "test_utils/tensor_random.h"
#include "gtest/gtest.h"
#include "tensor_bk.h"

using namespace aethermind;
using namespace aethermind::test_utils;

namespace {

TEST(TensorCompat, TensorFromLegacyEmpty) {
    Tensor_BK legacy_empty;
    Tensor converted = TensorFromLegacy(legacy_empty);
    EXPECT_TRUE(!converted.is_initialized());
}

TEST(TensorCompat, TensorFromLegacyDefined) {
    Tensor_BK legacy = RandomUniform({3, 10}, DataType::Float32());
    Tensor converted = TensorFromLegacy(legacy);
    
    EXPECT_TRUE(converted.is_initialized());
    EXPECT_TRUE(converted.dtype() == legacy.dtype());
    EXPECT_TRUE(converted.shape() == legacy.shape());
    EXPECT_TRUE(converted.numel() == legacy.numel());
    EXPECT_TRUE(converted.device().is_cpu());
    
    const float* legacy_data = legacy.const_data_ptr<float>();
    const float* new_data = static_cast<const float*>(converted.data());
    EXPECT_TRUE(new_data != nullptr);
    
    for (int64_t i = 0; i < std::min(converted.numel(), int64_t{10}); ++i) {
        EXPECT_FLOAT_EQ(new_data[i], legacy_data[i]);
    }
}

TEST(TensorCompat, LegacyTensorFromTensorEmpty) {
    Tensor new_empty;
    Tensor_BK converted = LegacyTensorFromTensor(new_empty);
    EXPECT_TRUE(!converted.defined());
}

TEST(TensorCompat, LegacyTensorFromTensorDefined) {
    Tensor new_tensor = RandomUniformTensor({3, 10}, DataType::Float32());
    Tensor_BK converted = LegacyTensorFromTensor(new_tensor);
    
    EXPECT_TRUE(converted.defined());
    EXPECT_TRUE(converted.dtype() == new_tensor.dtype());
    EXPECT_TRUE(converted.shape() == new_tensor.shape());
    EXPECT_TRUE(converted.numel() == new_tensor.numel());
    EXPECT_TRUE(converted.device().is_cpu());
    
    const float* new_data = static_cast<const float*>(new_tensor.data());
    const float* converted_data = converted.const_data_ptr<float>();
    
    for (int64_t i = 0; i < std::min(new_tensor.numel(), int64_t{10}); ++i) {
        EXPECT_FLOAT_EQ(converted_data[i], new_data[i]);
    }
}

TEST(TensorCompat, RoundTripIdentity) {
    Tensor_BK original = RandomUniform({3, 10}, DataType::Float32());
    
    Tensor intermediate = TensorFromLegacy(original);
    Tensor_BK roundtrip = LegacyTensorFromTensor(intermediate);
    
    EXPECT_TRUE(roundtrip.dtype() == original.dtype());
    EXPECT_TRUE(roundtrip.shape() == original.shape());
    EXPECT_TRUE(roundtrip.numel() == original.numel());
    
    const float* original_data = original.const_data_ptr<float>();
    const float* roundtrip_data = roundtrip.const_data_ptr<float>();
    
    for (int64_t i = 0; i < original.numel(); ++i) {
        EXPECT_FLOAT_EQ(roundtrip_data[i], original_data[i]);
    }
}

TEST(TensorCompat, DifferentDtypes) {
    Tensor_BK int_tensor = RandomInt({5, 5}, 0, 100, 12345, DataType::Int(64));
    Tensor converted = TensorFromLegacy(int_tensor);
    
    EXPECT_TRUE(converted.dtype() == DataType::Int(64));
    EXPECT_TRUE(converted.numel() == int_tensor.numel());
    
    const int64_t* int_data = static_cast<const int64_t*>(converted.data());
    const int64_t* legacy_data = int_tensor.const_data_ptr<int64_t>();
    
    for (int64_t i = 0; i < converted.numel(); ++i) {
        EXPECT_EQ(int_data[i], legacy_data[i]);
    }
}

TEST(TensorCompat, ZeroSizedTensor) {
    Tensor_BK zero_tensor(std::vector<int64_t>{0}, 0, DataType::Float32(), Device::CPU());
    Tensor converted = TensorFromLegacy(zero_tensor);
    
    EXPECT_TRUE(converted.is_initialized());
    EXPECT_TRUE(converted.numel() == 0);
    EXPECT_TRUE(converted.shape().size() == 1);
    EXPECT_TRUE(converted.shape()[0] == 0);
}

TEST(TensorCompat, BufferFromLegacyStorage) {
    Tensor_BK legacy = RandomNormal({5, 5}, DataType::Double());
    Storage storage = legacy.get_impl_ptr_unsafe()->storage();
    
    Buffer buffer = BufferFromLegacyStorage(storage);
    
    EXPECT_TRUE(buffer.is_initialized());
    EXPECT_TRUE(buffer.nbytes() == storage.nbytes());
    EXPECT_TRUE(buffer.device().is_cpu());
    
    const double* storage_data = static_cast<const double*>(storage.data());
    const double* buffer_data = static_cast<const double*>(buffer.data());
    
    EXPECT_TRUE(buffer_data == storage_data);
}

}// namespace