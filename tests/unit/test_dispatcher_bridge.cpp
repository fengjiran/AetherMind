#include "aethermind/backend/dispatcher_bridge.h"

#include "data_type.h"

#include <gtest/gtest.h>

using namespace aethermind;

namespace {

TEST(DispatcherBridge, ToOpTypeMapsCanonicalOperatorName) {
    auto op_type = ToOpType(OperatorName("aethermind::nn::linear", "Tensor"));

    ASSERT_TRUE(op_type.ok());
    EXPECT_EQ(op_type.value(), OpType::kLinear);
}

TEST(DispatcherBridge, ToOpTypeRejectsUnknownOperatorName) {
    auto op_type = ToOpType(OperatorName("test::unknown", ""));

    EXPECT_FALSE(op_type.ok());
    EXPECT_EQ(op_type.status().code(), StatusCode::kNotFound);
}

TEST(DispatcherBridge, MakeKernelSelectorBuildsTransitionSelector) {
    const KernelSelector selector = MakeKernelSelector(DeviceType::kCPU,
                                                       DataType::Float32(),
                                                       DataType::Float32(),
                                                       WeightFormat::kPlain,
                                                       IsaLevel::kAVX2,
                                                       ExecPhase::kDecode);

    EXPECT_EQ(selector.device_type, DeviceType::kCPU);
    EXPECT_EQ(selector.activation_dtype, DataType::Float32());
    EXPECT_EQ(selector.weight_dtype, DataType::Float32());
    EXPECT_EQ(selector.weight_format, WeightFormat::kPlain);
    EXPECT_EQ(selector.isa, IsaLevel::kAVX2);
    EXPECT_EQ(selector.phase, ExecPhase::kDecode);
}

}// namespace
