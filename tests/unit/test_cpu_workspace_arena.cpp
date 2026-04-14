#include "aethermind/backend/cpu/cpu_workspace_arena.h"

#include <cstdint>
#include <gtest/gtest.h>

namespace {

using namespace aethermind;

TEST(CpuWorkspaceArena, SuccessfulBindWithinRange) {
    alignas(16) uint8_t buffer[1024];
    CpuWorkspaceArena arena(buffer, sizeof(buffer));

    WorkspaceRequirement req{
            .bytes = 256,
            .alignment = 16,
            .offset = 0};

    auto binding = arena.Bind(req);
    EXPECT_EQ(binding.data, buffer);
    EXPECT_EQ(binding.size, 256);

    WorkspaceRequirement req2{
            .bytes = 128,
            .alignment = 8,
            .offset = 512};

    auto binding2 = arena.Bind(req2);
    EXPECT_EQ(binding2.data, buffer + 512);
    EXPECT_EQ(binding2.size, 128);
}

TEST(CpuWorkspaceArena, NullBaseReturnsEmptyBinding) {
    CpuWorkspaceArena arena(nullptr, 1024);
    WorkspaceRequirement req{.bytes = 256, .alignment = 1, .offset = 0};

    auto binding = arena.Bind(req);
    EXPECT_EQ(binding.data, nullptr);
    EXPECT_EQ(binding.size, 0);
}

TEST(CpuWorkspaceArena, OutOfRangeBindReturnsEmptyBinding) {
    uint8_t buffer[1024];
    CpuWorkspaceArena arena(buffer, sizeof(buffer));

    WorkspaceRequirement req1{.bytes = 1, .alignment = 1, .offset = 1024};
    EXPECT_EQ(arena.Bind(req1).data, nullptr);

    WorkspaceRequirement req2{.bytes = 513, .alignment = 1, .offset = 512};
    EXPECT_EQ(arena.Bind(req2).data, nullptr);
}

TEST(CpuWorkspaceArena, MisalignedBindReturnsEmptyBinding) {
    alignas(16) uint8_t buffer[1024];
    CpuWorkspaceArena arena(buffer, sizeof(buffer));

    WorkspaceRequirement req{
            .bytes = 16,
            .alignment = 16,
            .offset = 1};

    auto binding = arena.Bind(req);
    EXPECT_EQ(binding.data, nullptr);
    EXPECT_EQ(binding.size, 0);
}

TEST(CpuWorkspaceArena, ResetPreservesReusableBindingContract) {
    alignas(16) uint8_t buffer[1024];
    CpuWorkspaceArena arena(buffer, sizeof(buffer));

    WorkspaceRequirement req{
            .bytes = 256,
            .alignment = 16,
            .offset = 0};

    auto binding1 = arena.Bind(req);
    ASSERT_NE(binding1.data, nullptr);

    arena.Reset();

    auto binding2 = arena.Bind(req);
    EXPECT_EQ(binding2.data, binding1.data);
    EXPECT_EQ(binding2.size, binding1.size);
}

TEST(CpuWorkspaceArena, ZeroBytesReturnsEmptyBinding) {
    uint8_t buffer[1024];
    CpuWorkspaceArena arena(buffer, sizeof(buffer));
    WorkspaceRequirement req{.bytes = 0, .alignment = 1, .offset = 0};

    EXPECT_EQ(arena.Bind(req).data, nullptr);
}

TEST(CpuWorkspaceArena, ZeroAlignmentReturnsEmptyBinding) {
    uint8_t buffer[1024];
    CpuWorkspaceArena arena(buffer, sizeof(buffer));
    WorkspaceRequirement req{.bytes = 16, .alignment = 0, .offset = 0};

    EXPECT_EQ(arena.Bind(req).data, nullptr);
}

}// namespace
