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

TEST(KernelSelectorState, EqualityDetectsDifferentStateDType) {
    KernelSelector lhs{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .state_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kBoth,
    };
    KernelSelector rhs = lhs;
    rhs.state_dtype = DataType::Float(16);

    EXPECT_NE(lhs, rhs);
}

TEST(KernelSelectorState, MatchesRequiresExactStateDType) {
    KernelSelector f32_state{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .state_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kBoth,
    };
    KernelSelector f16_state = f32_state;
    f16_state.state_dtype = DataType::Float(16);
    KernelSelector bf16_state = f32_state;
    bf16_state.state_dtype = DataType::BFloat(16);

    EXPECT_TRUE(SelectorMatches(f32_state, f32_state));
    EXPECT_FALSE(SelectorMatches(f32_state, f16_state));
    EXPECT_FALSE(SelectorMatches(f16_state, f32_state));
    EXPECT_FALSE(SelectorMatches(f16_state, bf16_state));
}

TEST(KernelSelectorState, UndefinedStateIsNotWildcard) {
    KernelSelector stateless{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kBoth,
    };
    KernelSelector stateful = stateless;
    stateful.state_dtype = DataType::Float32();

    EXPECT_TRUE(stateless.state_dtype.IsUndefined());
    EXPECT_TRUE(SelectorMatches(stateless, stateless));
    EXPECT_FALSE(SelectorMatches(stateless, stateful));
    EXPECT_FALSE(SelectorMatches(stateful, stateless));
}

TEST(KernelSelectorState, HashConsistentWithEquality) {
    KernelSelector lhs{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .state_dtype = DataType::BFloat(16),
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kBoth,
    };
    KernelSelector rhs = lhs;
    EXPECT_EQ(std::hash<KernelSelector>{}(lhs), std::hash<KernelSelector>{}(rhs));

    rhs.state_dtype = DataType::Float(16);
    EXPECT_NE(lhs, rhs);
    EXPECT_NE(std::hash<KernelSelector>{}(lhs), std::hash<KernelSelector>{}(rhs));
}

TEST(KernelSelectorToString, ContainsStateDType) {
    KernelSelector selector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .state_dtype = DataType::Float(16),
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kBoth,
    };

    const std::string str = ToString(selector);
    EXPECT_TRUE(str.find("state_dtype=") != std::string::npos);
}

} // namespace
