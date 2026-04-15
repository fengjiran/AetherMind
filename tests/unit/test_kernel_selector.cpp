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

}// namespace
