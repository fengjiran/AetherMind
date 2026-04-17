#include "aethermind/backend/op_kernel_context.h"

#include <gtest/gtest.h>

namespace aethermind {
namespace {

TEST(OpKernelContext, DefaultsAreNarrowAndNull) {
    OpKernelContext ctx;

    EXPECT_TRUE(ctx.device.is_cpu());
    EXPECT_EQ(ctx.stream, nullptr);
    EXPECT_EQ(ctx.workspace, nullptr);
    EXPECT_EQ(ctx.tracing, nullptr);
    EXPECT_EQ(ctx.caps, nullptr);
    EXPECT_EQ(ctx.backend_resources.opaque_backend_resources, nullptr);
}

TEST(OpKernelContext, CanCarryMinimalExecutionHandles) {
    CpuInlineStream stream;
    WorkspaceArena* workspace = nullptr;
    BackendCapabilities caps{.device_type = DeviceType::kCPU};
    int fake_resource = 7;

    OpKernelContext ctx{
            .device = Device::CPU(),
            .stream = &stream,
            .workspace = workspace,
            .tracing = nullptr,
            .caps = &caps,
            .backend_resources = {.opaque_backend_resources = &fake_resource},
    };

    EXPECT_TRUE(ctx.device.is_cpu());
    EXPECT_EQ(ctx.stream, &stream);
    EXPECT_EQ(ctx.workspace, workspace);
    ASSERT_NE(ctx.caps, nullptr);
    EXPECT_EQ(ctx.caps->device_type, DeviceType::kCPU);
    EXPECT_EQ(ctx.backend_resources.opaque_backend_resources, &fake_resource);
}

}// namespace
}// namespace aethermind
