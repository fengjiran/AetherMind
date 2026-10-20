#include "aethermind/runtime/workspace_planner.h"
#include "inference/test_malloc_interposer.h"

#include <array>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace {

using namespace aethermind;

TEST(WorkspaceRequirementPlanning, PlansAlignedSequentialOffsets) {
    std::vector<WorkspaceRequirement> requirements = {
            {.bytes = 32, .alignment = 16},
            {.bytes = 1, .alignment = 64},
            {.bytes = 7, .alignment = 8},
    };

    const StatusOr<WorkspacePlanLayout> layout = PlanWorkspaceRequirements(requirements);

    ASSERT_TRUE(layout.ok());
    ASSERT_EQ(requirements.size(), 3U);
    EXPECT_EQ(requirements[0].offset, 0U);
    EXPECT_EQ(requirements[1].offset, 64U);
    EXPECT_EQ(requirements[2].offset, 72U);
    EXPECT_EQ(layout->total_bytes, 79U);
    EXPECT_EQ(layout->required_alignment, 64U);
}

TEST(WorkspaceRequirementPlanning, ZeroByteRequirementDoesNotConsumeWorkspace) {
    std::vector<WorkspaceRequirement> requirements = {
            {.bytes = 32, .alignment = 16},
            {.bytes = 0, .alignment = 64},
            {.bytes = 16, .alignment = 8},
    };

    const StatusOr<WorkspacePlanLayout> layout = PlanWorkspaceRequirements(requirements);

    ASSERT_TRUE(layout.ok());
    EXPECT_EQ(requirements[0].offset, 0U);
    EXPECT_EQ(requirements[1].offset, 32U);
    EXPECT_EQ(requirements[2].offset, 32U);
    EXPECT_EQ(layout->total_bytes, 48U);
    EXPECT_EQ(layout->required_alignment, 64U);
}

TEST(WorkspaceRequirementPlanning, EmptyRequirementsUseEmptyLayout) {
    const StatusOr<WorkspacePlanLayout> layout = PlanWorkspaceRequirements({});

    ASSERT_TRUE(layout.ok());
    EXPECT_EQ(layout->total_bytes, 0U);
    EXPECT_EQ(layout->required_alignment, 1U);
}

TEST(WorkspaceRequirementPlanning, ZeroByteRequirementsTrackLargestAlignment) {
    constexpr size_t kLargestAlignment = size_t{1} << (std::numeric_limits<size_t>::digits - 1);
    std::array<WorkspaceRequirement, 2> requirements = {{
            {.bytes = 0, .alignment = 64, .offset = 11},
            {.bytes = 0, .alignment = kLargestAlignment, .offset = 22},
    }};

    const StatusOr<WorkspacePlanLayout> layout = PlanWorkspaceRequirements(requirements);

    ASSERT_TRUE(layout.ok());
    EXPECT_EQ(requirements[0].offset, 0U);
    EXPECT_EQ(requirements[1].offset, 0U);
    EXPECT_EQ(layout->total_bytes, 0U);
    EXPECT_EQ(layout->required_alignment, kLargestAlignment);
}

TEST(WorkspaceRequirementPlanning, AcceptsExactSizeCapacityAndTrailingZeroByteRequirement) {
    constexpr size_t kMaxSize = std::numeric_limits<size_t>::max();
    std::array<WorkspaceRequirement, 2> requirements = {{
            {.bytes = kMaxSize, .alignment = 1},
            {.bytes = 0, .alignment = 64},
    }};

    const StatusOr<WorkspacePlanLayout> layout = PlanWorkspaceRequirements(requirements);

    ASSERT_TRUE(layout.ok());
    EXPECT_EQ(requirements[0].offset, 0U);
    EXPECT_EQ(requirements[1].offset, kMaxSize);
    EXPECT_EQ(layout->total_bytes, kMaxSize);
    EXPECT_EQ(layout->required_alignment, 64U);
}

TEST(WorkspaceRequirementPlanning, SuccessfulPlanningDoesNotAllocate) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }
    std::array<WorkspaceRequirement, 3> requirements = {{
            {.bytes = 32, .alignment = 16},
            {.bytes = 0, .alignment = 64},
            {.bytes = 7, .alignment = 8},
    }};

    test::BeginMallocCallCounting();
    const StatusOr<WorkspacePlanLayout> layout = PlanWorkspaceRequirements(requirements);
    const test::MallocCallCounts counts = test::EndMallocCallCounting();

    EXPECT_TRUE(layout.ok());
    EXPECT_EQ(counts.malloc_calls, 0U);
    EXPECT_EQ(counts.calloc_calls, 0U);
    EXPECT_EQ(counts.realloc_calls, 0U);
    EXPECT_EQ(counts.free_calls, 0U);
    EXPECT_EQ(counts.aligned_alloc_calls, 0U);
    EXPECT_EQ(counts.posix_memalign_calls, 0U);
    EXPECT_EQ(counts.memalign_calls, 0U);
}

TEST(WorkspaceRequirementPlanning, RequirementDefaultsMatchOperatorContract) {
    const WorkspaceRequirement requirement;

    EXPECT_EQ(requirement.bytes, 0U);
    EXPECT_EQ(requirement.alignment, 64U);
    EXPECT_EQ(requirement.lifetime, WorkspaceLifetime::kNone);
    EXPECT_TRUE(requirement.reusable);
    EXPECT_EQ(requirement.offset, 0U);
    EXPECT_TRUE(requirement.empty());
}

TEST(WorkspaceRequirementPlanning, RequirementCanDescribeLifetimeAndReuse) {
    const WorkspaceRequirement requirement{
            .bytes = 128,
            .alignment = 64,
            .lifetime = WorkspaceLifetime::kPerLayer,
            .reusable = false,
    };

    EXPECT_FALSE(requirement.empty());
    EXPECT_EQ(requirement.lifetime, WorkspaceLifetime::kPerLayer);
    EXPECT_FALSE(requirement.reusable);
}

TEST(WorkspaceRequirementPlanning, RejectsInvalidAlignment) {
    static_assert(noexcept(PlanWorkspaceRequirements({})));

    std::vector<WorkspaceRequirement> zero_alignment = {
            {.bytes = 32, .alignment = 0},
    };

    const StatusOr<WorkspacePlanLayout> zero_result =
            PlanWorkspaceRequirements(zero_alignment);

    ASSERT_FALSE(zero_result.ok());
    EXPECT_EQ(zero_result.status().code(), StatusCode::kInvalidArgument);

    std::vector<WorkspaceRequirement> non_power_of_two = {
            {.bytes = 32, .alignment = 24},
    };

    const StatusOr<WorkspacePlanLayout> invalid_result =
            PlanWorkspaceRequirements(non_power_of_two);

    ASSERT_FALSE(invalid_result.ok());
    EXPECT_EQ(invalid_result.status().code(), StatusCode::kInvalidArgument);
}

TEST(WorkspaceRequirementPlanning, DetectsOffsetOverflowDuringPlanning) {
    std::vector<WorkspaceRequirement> requirements = {
            {.bytes = std::numeric_limits<size_t>::max() - 3, .alignment = 1, .offset = 11},
            {.bytes = 0, .alignment = 64, .offset = 22},
            {.bytes = 8, .alignment = 8, .offset = 33},
    };

    const StatusOr<WorkspacePlanLayout> layout = PlanWorkspaceRequirements(requirements);

    ASSERT_FALSE(layout.ok());
    EXPECT_EQ(layout.status().code(), StatusCode::kOverflow);
    EXPECT_EQ(requirements[0].offset, 11U);
    EXPECT_EQ(requirements[1].offset, 22U);
    EXPECT_EQ(requirements[2].offset, 33U);
}

TEST(WorkspaceRequirementPlanning, ByteOverflowPreservesAllOffsets) {
    std::array<WorkspaceRequirement, 3> requirements = {{
            {.bytes = 8, .alignment = 1, .offset = 11},
            {.bytes = 0, .alignment = 64, .offset = 22},
            {.bytes = std::numeric_limits<size_t>::max(), .alignment = 1, .offset = 33},
    }};

    const StatusOr<WorkspacePlanLayout> layout = PlanWorkspaceRequirements(requirements);

    ASSERT_FALSE(layout.ok());
    EXPECT_EQ(layout.status().code(), StatusCode::kOverflow);
    EXPECT_EQ(requirements[0].offset, 11U);
    EXPECT_EQ(requirements[1].offset, 22U);
    EXPECT_EQ(requirements[2].offset, 33U);
}

TEST(WorkspaceRequirementPlanning, LateInvalidAlignmentPreservesAllOffsets) {
    std::array<WorkspaceRequirement, 3> requirements = {{
            {.bytes = 8, .alignment = 1, .offset = 11},
            {.bytes = 0, .alignment = 64, .offset = 22},
            {.bytes = 8, .alignment = 24, .offset = 33},
    }};

    const StatusOr<WorkspacePlanLayout> layout = PlanWorkspaceRequirements(requirements);

    ASSERT_FALSE(layout.ok());
    EXPECT_EQ(layout.status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(requirements[0].offset, 11U);
    EXPECT_EQ(requirements[1].offset, 22U);
    EXPECT_EQ(requirements[2].offset, 33U);
}

TEST(WorkspaceRequirementPlanning, InvalidAlignmentTakesPrecedenceOverEarlierOverflow) {
    std::array<WorkspaceRequirement, 3> requirements = {{
            {.bytes = std::numeric_limits<size_t>::max(), .alignment = 1, .offset = 11},
            {.bytes = 1, .alignment = 1, .offset = 22},
            {.bytes = 0, .alignment = 0, .offset = 33},
    }};

    const StatusOr<WorkspacePlanLayout> layout = PlanWorkspaceRequirements(requirements);

    ASSERT_FALSE(layout.ok());
    EXPECT_EQ(layout.status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(requirements[0].offset, 11U);
    EXPECT_EQ(requirements[1].offset, 22U);
    EXPECT_EQ(requirements[2].offset, 33U);
}

enum class PlanningFailure {
    kInvalidAlignment,
    kAlignmentOverflow,
    kByteOverflow,
};

class WorkspaceRequirementPlanningDiagnosticFailure : public ::testing::TestWithParam<PlanningFailure> {};

TEST_P(WorkspaceRequirementPlanningDiagnosticFailure, ReturnsResourceExhaustedAndPreservesOffsets) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }
    std::array<WorkspaceRequirement, 3> requirements = {{
            {.bytes = 8, .alignment = 1, .offset = 11},
            {.bytes = 0, .alignment = 64, .offset = 22},
            {.bytes = 8, .alignment = 1, .offset = 33},
    }};
    switch (GetParam()) {
        case PlanningFailure::kInvalidAlignment:
            requirements[2].alignment = 24;
            break;
        case PlanningFailure::kAlignmentOverflow:
            requirements[0].bytes = std::numeric_limits<size_t>::max() - 3;
            requirements[2].alignment = 8;
            break;
        case PlanningFailure::kByteOverflow:
            requirements[2].bytes = std::numeric_limits<size_t>::max();
            break;
    }

    test::ScopedMallocFailure failure(0);
    const StatusOr<WorkspacePlanLayout> layout = PlanWorkspaceRequirements(requirements);
    const test::MallocFailureResult injection = failure.Stop();

    EXPECT_TRUE(injection.allocation_failed);
    ASSERT_FALSE(layout.ok());
    EXPECT_EQ(layout.status().code(), StatusCode::kResourceExhausted);
    EXPECT_TRUE(layout.status().message().empty());
    EXPECT_EQ(requirements[0].offset, 11U);
    EXPECT_EQ(requirements[1].offset, 22U);
    EXPECT_EQ(requirements[2].offset, 33U);
}

INSTANTIATE_TEST_SUITE_P(
        FailurePaths,
        WorkspaceRequirementPlanningDiagnosticFailure,
        ::testing::Values(PlanningFailure::kInvalidAlignment,
                          PlanningFailure::kAlignmentOverflow,
                          PlanningFailure::kByteOverflow),
        [](const ::testing::TestParamInfo<PlanningFailure>& info) {
            switch (info.param) {
                case PlanningFailure::kInvalidAlignment:
                    return "InvalidAlignment";
                case PlanningFailure::kAlignmentOverflow:
                    return "AlignmentOverflow";
                case PlanningFailure::kByteOverflow:
                    return "ByteOverflow";
            }
            return "Unknown";
        });

} // namespace
