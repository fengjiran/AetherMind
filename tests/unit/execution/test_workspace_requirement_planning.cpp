#include "aethermind/execution/workspace_types.h"

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

TEST(WorkspaceRequirementPlanning, RejectsInvalidAlignment) {
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
            {.bytes = std::numeric_limits<size_t>::max() - 3, .alignment = 1},
            {.bytes = 8, .alignment = 8},
    };

    const StatusOr<WorkspacePlanLayout> layout = PlanWorkspaceRequirements(requirements);

    ASSERT_FALSE(layout.ok());
    EXPECT_EQ(layout.status().code(), StatusCode::kOutOfRange);
}

}// namespace
