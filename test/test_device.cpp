//
// Created by 赵丹 on 2025/8/15.
//
#include "device.h"
#include "container/string.h"
#include <gtest/gtest.h>

using namespace aethermind;

namespace {

TEST(Device, device_type) {
    EXPECT_TRUE(IsValidDeviceType(kCPU));
    EXPECT_TRUE(IsValidDeviceType(kCUDA));
    EXPECT_TRUE(IsValidDeviceType(kCANN));
    EXPECT_FALSE(IsValidDeviceType(kUndefined));

    EXPECT_EQ(DeviceType2Str(DeviceType::kCPU), "CPU");
    EXPECT_EQ(DeviceType2Str(DeviceType::kCUDA), "CUDA");
    EXPECT_EQ(DeviceType2Str(DeviceType::kCANN), "CANN");
    EXPECT_EQ(DeviceType2Str(DeviceType::kCPU, true), "cpu");
    EXPECT_EQ(DeviceType2Str(DeviceType::kCUDA, true), "cuda");
    EXPECT_EQ(DeviceType2Str(DeviceType::kCANN, true), "cann");

    std::cout << kCPU;
    std::cout << kCUDA;
    std::cout << kCANN;
}

TEST(Device, device) {
    Device dev1(kCPU);
    EXPECT_TRUE(dev1.is_cpu());
    EXPECT_FALSE(dev1.has_index());
    EXPECT_EQ(dev1.type(), kCPU);
    EXPECT_EQ(dev1.index(), -1);
    EXPECT_EQ(dev1.str(), "cpu");
}

}// namespace