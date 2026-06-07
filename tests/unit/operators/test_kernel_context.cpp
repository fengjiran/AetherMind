#include "aethermind/backend/kernel_context.h"

#include <gtest/gtest.h>

namespace aethermind {
namespace {

TEST(KernelContext, DefaultsAreNarrowAndNull) {
    KernelContext ctx;

    EXPECT_EQ(ctx.stream, nullptr);
    EXPECT_EQ(ctx.workspace, nullptr);
    EXPECT_EQ(ctx.workspace_binding.data, nullptr);
    EXPECT_EQ(ctx.workspace_binding.size, 0U);
    EXPECT_EQ(ctx.packed_weights, nullptr);
    EXPECT_EQ(ctx.kernel_params, nullptr);
    EXPECT_TRUE(ctx.attrs.empty());
}

TEST(KernelContext, CanCarryMinimalExecutionHandles) {
    CpuInlineStream stream;
    WorkspaceArena* workspace = nullptr;
    int fake_weights = 11;
    int fake_params = 13;
    const std::byte attrs_storage[4]{};

    KernelContext ctx{
            .stream = &stream,
            .workspace = workspace,
            .workspace_binding = {},
            .packed_weights = &fake_weights,
            .kernel_params = &fake_params,
            .attrs = std::span<const std::byte>(attrs_storage, sizeof(attrs_storage)),
    };

    EXPECT_EQ(ctx.stream, &stream);
    EXPECT_EQ(ctx.workspace, workspace);
    EXPECT_EQ(ctx.packed_weights, &fake_weights);
    EXPECT_EQ(ctx.kernel_params, &fake_params);
    EXPECT_EQ(ctx.attrs.size(), sizeof(attrs_storage));
}

}// namespace
}// namespace aethermind
