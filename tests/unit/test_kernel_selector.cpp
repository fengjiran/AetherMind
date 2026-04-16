#include "aethermind/backend/kernel_selector.h"

#include "data_type.h"

#include <gtest/gtest.h>

using namespace aethermind;

namespace {

TEST(KernelSelector, EqualityMatchesIdenticalSelectors) {
    const KernelSelector lhs{
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .isa = IsaLevel::kAVX2,
            .phase = ExecPhase::kDecode,
    };
    const KernelSelector rhs = lhs;

    EXPECT_EQ(lhs, rhs);
}

TEST(KernelSelector, EqualityDetectsDifferentPhase) {
    const KernelSelector lhs{
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .isa = IsaLevel::kAVX2,
            .phase = ExecPhase::kPrefill,
    };
    const KernelSelector rhs{
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .isa = IsaLevel::kAVX2,
            .phase = ExecPhase::kDecode,
    };

    EXPECT_NE(lhs, rhs);
}

TEST(IsaLevelToString, AllKnownLevelsHaveStringRepresentation) {
    EXPECT_STREQ(ToString(IsaLevel::kScalar), "Scalar");
    EXPECT_STREQ(ToString(IsaLevel::kAVX2), "AVX2");
    EXPECT_STREQ(ToString(IsaLevel::kAVX512), "AVX512");
    EXPECT_STREQ(ToString(IsaLevel::kAMX), "AMX");
}

TEST(IsaLevelToString, InvalidValueReturnsUnknown) {
    const IsaLevel invalid = static_cast<IsaLevel>(255);
    EXPECT_STREQ(ToString(invalid), "Unknown");
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
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float(16),
            .weight_format = WeightFormat::kQuantizedInt8,
            .isa = IsaLevel::kAVX2,
            .phase = ExecPhase::kPrefill,
    };

    const std::string str = ToString(selector);
    EXPECT_TRUE(str.find("CPU") != std::string::npos);
    EXPECT_TRUE(str.find("32bit") != std::string::npos);
    EXPECT_TRUE(str.find("16bit") != std::string::npos);
    EXPECT_TRUE(str.find("QuantizedInt8") != std::string::npos);
    EXPECT_TRUE(str.find("AVX2") != std::string::npos);
    EXPECT_TRUE(str.find("Prefill") != std::string::npos);
}

TEST(KernelSelectorToString, ContainsAllFieldNames) {
    KernelSelector selector{
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .isa = IsaLevel::kScalar,
            .phase = ExecPhase::kBoth,
    };

    const std::string str = ToString(selector);
    EXPECT_TRUE(str.find("device=") != std::string::npos);
    EXPECT_TRUE(str.find("activation_dtype=") != std::string::npos);
    EXPECT_TRUE(str.find("weight_dtype=") != std::string::npos);
    EXPECT_TRUE(str.find("weight_format=") != std::string::npos);
    EXPECT_TRUE(str.find("isa=") != std::string::npos);
    EXPECT_TRUE(str.find("phase=") != std::string::npos);
}

}// namespace
