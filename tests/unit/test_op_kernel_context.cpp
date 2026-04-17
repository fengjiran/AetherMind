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
    EXPECT_EQ(ctx.packed_params, nullptr);
    EXPECT_TRUE(ctx.attrs.empty());
    EXPECT_EQ(ctx.debug_name, nullptr);
    EXPECT_EQ(ctx.backend_resources.opaque_backend_resources, nullptr);
}

TEST(OpKernelContext, CanCarryMinimalExecutionHandles) {
    CpuInlineStream stream;
    WorkspaceArena* workspace = nullptr;
    BackendCapabilities caps{.device_type = DeviceType::kCPU};
    int fake_resource = 7;
    int fake_params = 11;
    const std::byte attrs_storage[4]{};

    OpKernelContext ctx{
            .device = Device::CPU(),
            .stream = &stream,
            .workspace = workspace,
            .tracing = nullptr,
            .caps = &caps,
            .packed_params = &fake_params,
            .attrs = std::span<const std::byte>(attrs_storage, sizeof(attrs_storage)),
            .debug_name = "test::kernel",
            .backend_resources = {.opaque_backend_resources = &fake_resource},
    };

    EXPECT_TRUE(ctx.device.is_cpu());
    EXPECT_EQ(ctx.stream, &stream);
    EXPECT_EQ(ctx.workspace, workspace);
    ASSERT_NE(ctx.caps, nullptr);
    EXPECT_EQ(ctx.caps->device_type, DeviceType::kCPU);
    EXPECT_EQ(ctx.packed_params, &fake_params);
    EXPECT_EQ(ctx.attrs.size(), sizeof(attrs_storage));
    EXPECT_STREQ(ctx.debug_name, "test::kernel");
    EXPECT_EQ(ctx.backend_resources.opaque_backend_resources, &fake_resource);
}

}// namespace
}// namespace aethermind
