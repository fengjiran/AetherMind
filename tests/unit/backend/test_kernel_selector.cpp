#include "aethermind/base/kernel_selector.h"

#include "aethermind/dtypes/data_type.h"

#include <gtest/gtest.h>

namespace {
using namespace aethermind;

TEST(KernelSelector, EqualityMatchesIdenticalSelectors) {
    const KernelSelector lhs{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kDecode,
    };
    const KernelSelector rhs = lhs;

    EXPECT_EQ(lhs, rhs);
}

TEST(KernelSelector, EqualityDetectsDifferentPhase) {
    const KernelSelector lhs{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kPrefill,
    };
    const KernelSelector rhs{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kDecode,
    };

    EXPECT_NE(lhs, rhs);
}

TEST(ExecPhaseToString, AllKnownPhasesHaveStringRepresentation) {
    EXPECT_STREQ(ToString(ExecPhase::kPrefill), "Prefill");
    EXPECT_STREQ(ToString(ExecPhase::kDecode), "Decode");
    EXPECT_STREQ(ToString(ExecPhase::kBoth), "Both");
}

TEST(ExecPhaseToString, InvalidValueReturnsUnknown) {
    const ExecPhase invalid = static_cast<ExecPhase>(255);
    EXPECT_STREQ(ToString(invalid), "Unknown");
}

TEST(WeightFormatToString, AllKnownFormatsHaveStringRepresentation) {
    EXPECT_STREQ(ToString(WeightFormat::kPlain), "Plain");
    EXPECT_STREQ(ToString(WeightFormat::kPacked), "Packed");
    EXPECT_STREQ(ToString(WeightFormat::kQuantizedInt8), "QuantizedInt8");
    EXPECT_STREQ(ToString(WeightFormat::kQuantizedInt4), "QuantizedInt4");
}

TEST(WeightFormatToString, InvalidValueReturnsUnknown) {
    const WeightFormat invalid = static_cast<WeightFormat>(255);
    EXPECT_STREQ(ToString(invalid), "Unknown");
}

TEST(KernelSelectorToString, ReturnsDescriptiveString) {
    KernelSelector selector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float(16),
            .weight_format = WeightFormat::kQuantizedInt8,
            .phase = ExecPhase::kPrefill,
    };

    const std::string str = ToString(selector);
    EXPECT_TRUE(str.find("CPU") != std::string::npos);
    EXPECT_TRUE(str.find("32bit") != std::string::npos);
    EXPECT_TRUE(str.find("16bit") != std::string::npos);
    EXPECT_TRUE(str.find("QuantizedInt8") != std::string::npos);
    EXPECT_TRUE(str.find("Prefill") != std::string::npos);
}

TEST(KernelSelectorToString, ContainsAllFieldNames) {
    KernelSelector selector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kBoth,
    };

    const std::string str = ToString(selector);
    EXPECT_TRUE(str.find("device=") != std::string::npos);
    EXPECT_TRUE(str.find("activation_dtype=") != std::string::npos);
    EXPECT_TRUE(str.find("weight_dtype=") != std::string::npos);
    EXPECT_TRUE(str.find("weight_format=") != std::string::npos);
    EXPECT_TRUE(str.find("phase=") != std::string::npos);
}

} // namespace
