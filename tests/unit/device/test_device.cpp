#include "device.h"

#include <gtest/gtest.h>

using namespace aethermind;

namespace {

TEST(Device, DeviceTypeValidation) {
    EXPECT_TRUE(IsValidDeviceType(kCPU));
    EXPECT_TRUE(IsValidDeviceType(kCUDA));
    EXPECT_TRUE(IsValidDeviceType(kCANN));
    EXPECT_FALSE(IsValidDeviceType(kUndefined));
}

TEST(Device, DeviceKindSupport) {
    EXPECT_TRUE(IsDeviceKindSupported(kCPU));
    EXPECT_FALSE(IsDeviceKindSupported(kCUDA));
    EXPECT_FALSE(IsDeviceKindSupported(kCANN));
    EXPECT_FALSE(IsDeviceKindSupported(kUndefined));
}

TEST(Device, DeviceTypeString) {
    EXPECT_EQ(DeviceType2Str(DeviceType::kCPU), "CPU");
    EXPECT_EQ(DeviceType2Str(DeviceType::kCUDA), "CUDA");
    EXPECT_EQ(DeviceType2Str(DeviceType::kCANN), "CANN");
    EXPECT_EQ(DeviceType2Str(DeviceType::kCPU, true), "cpu");
    EXPECT_EQ(DeviceType2Str(DeviceType::kCUDA, true), "cuda");
    EXPECT_EQ(DeviceType2Str(DeviceType::kCANN, true), "cann");
}

TEST(Device, CpuFactoryDefault) {
    Device dev = Device::CPU();
    EXPECT_TRUE(dev.is_cpu());
    EXPECT_FALSE(dev.has_index());
    EXPECT_EQ(dev.type(), kCPU);
    EXPECT_EQ(dev.index(), -1);
    EXPECT_EQ(dev.ToString(), "cpu");
}

TEST(Device, MakeCpuValidIndices) {
    StatusOr<Device> current = Device::Make(kCPU, -1);
    ASSERT_TRUE(current.ok());
    EXPECT_EQ(current->ToString(), "cpu");

    StatusOr<Device> concrete = Device::Make(kCPU, 0);
    ASSERT_TRUE(concrete.ok());
    EXPECT_EQ(concrete->ToString(), "cpu:0");
}

TEST(Device, MakeRejectsInvalidIndex) {
    StatusOr<Device> negative = Device::Make(kCPU, -2);
    EXPECT_FALSE(negative.ok());
    EXPECT_EQ(negative.status().code(), StatusCode::kInvalidArgument);

    StatusOr<Device> positive = Device::Make(kCPU, 1);
    EXPECT_FALSE(positive.ok());
    EXPECT_EQ(positive.status().code(), StatusCode::kInvalidArgument);
}

TEST(Device, MakeRejectsUndefinedType) {
    StatusOr<Device> undefined = Device::Make(kUndefined, -1);
    EXPECT_FALSE(undefined.ok());
    EXPECT_EQ(undefined.status().code(), StatusCode::kInvalidArgument);
}

TEST(Device, MakeAcceptsKnownKinds) {
    StatusOr<Device> cuda = Device::Make(kCUDA, 0);
    ASSERT_TRUE(cuda.ok());
    EXPECT_TRUE(cuda->is_cuda());

    StatusOr<Device> cann = Device::Make(kCANN, 0);
    ASSERT_TRUE(cann.ok());
    EXPECT_TRUE(cann->is_cann());
}

TEST(Device, FromStringParsesCpu) {
    StatusOr<Device> current = Device::FromString("cpu");
    ASSERT_TRUE(current.ok());
    EXPECT_EQ(current->type(), kCPU);
    EXPECT_EQ(current->index(), -1);

    StatusOr<Device> concrete = Device::FromString("cpu:0");
    ASSERT_TRUE(concrete.ok());
    EXPECT_EQ(concrete->type(), kCPU);
    EXPECT_EQ(concrete->index(), 0);

    StatusOr<Device> cuda = Device::FromString("cuda:0");
    ASSERT_TRUE(cuda.ok());
    EXPECT_TRUE(cuda->is_cuda());

    StatusOr<Device> cann = Device::FromString("cann");
    ASSERT_TRUE(cann.ok());
    EXPECT_TRUE(cann->is_cann());
}

TEST(Device, FromStringRejectsBadInput) {
    EXPECT_FALSE(Device::FromString("").ok());
    EXPECT_FALSE(Device::FromString("undefined").ok());
    EXPECT_FALSE(Device::FromString("cpu:").ok());
    EXPECT_FALSE(Device::FromString("cpu:x").ok());
    EXPECT_FALSE(Device::FromString("cpu:1").ok());
    EXPECT_FALSE(Device::FromString("npu:0").ok());
}

TEST(Device, EqualityAndHash) {
    Device a = Device::CPU();
    Device b = Device::CPU();
    EXPECT_EQ(a, b);
    EXPECT_EQ(std::hash<Device>()(a), std::hash<Device>()(b));

    StatusOr<Device> concrete = Device::Make(kCPU, 0);
    ASSERT_TRUE(concrete.ok());
    EXPECT_NE(a, *concrete);
}

}// namespace
