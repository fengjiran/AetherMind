#include "aethermind/backend/cpu/cpu_capabilities.h"

#include <gtest/gtest.h>

namespace aethermind {
namespace {

TEST(CpuCapabilities, DefaultValues) {
    CpuCapabilities caps;
    EXPECT_EQ(caps.base.device_type, DeviceType::kCPU);
    EXPECT_TRUE(caps.supports_inline_execution);
}

TEST(CpuCapabilities, ExplicitInitialization) {
    CpuCapabilities caps{
            .base = {.device_type = DeviceType::kCPU},
            .supports_inline_execution = false};
    EXPECT_EQ(caps.base.device_type, DeviceType::kCPU);
    EXPECT_FALSE(caps.supports_inline_execution);
}

}// namespace
}// namespace aethermind
