#include "utils/logging.h"

#include <gtest/gtest.h>

namespace {

TEST(CheckMacro, EvaluatesSuccessfulConditionOnceOutsideAethermindNamespace) {
    int evaluations = 0;
    AM_CHECK(++evaluations == 1);
    EXPECT_EQ(evaluations, 1);
}

TEST(CheckMacro, DoesNotEvaluateDiagnosticArgumentsOnSuccess) {
    int evaluations = 0;
    AM_CHECK(true, "diagnostic {}", ++evaluations);
    EXPECT_EQ(evaluations, 0);
}

TEST(CheckMacro, PreservesOuterElseBinding) {
    int branch = 0;
    if (false)
        AM_CHECK(false);
    else
        branch = 1;
    EXPECT_EQ(branch, 1);
}

TEST(CheckMacroDeathTest, FailsWithStableDiagnosticPrefix) {
    EXPECT_DEATH(AM_CHECK(false), "Check failed");
}

TEST(CheckMacroDeathTest, FormatsDiagnosticArgumentsOutsideAethermindNamespace) {
    EXPECT_DEATH(AM_CHECK(false, "value {}", 42), "Check failed:.*value 42");
}

TEST(DebugCheckMacro, EvaluatesConditionOnlyInDebugBuilds) {
    int evaluations = 0;
    AM_DCHECK(++evaluations == 1);
#ifdef NDEBUG
    EXPECT_EQ(evaluations, 0);
#else
    EXPECT_EQ(evaluations, 1);
#endif
}

TEST(DebugCheckMacro, PreservesOuterElseBinding) {
    int branch = 0;
    if (false)
        AM_DCHECK(false);
    else
        branch = 1;
    EXPECT_EQ(branch, 1);
}

#ifdef NDEBUG
TEST(DebugCheckMacro, DoesNotEvaluateDiagnosticArgumentsInReleaseBuilds) {
    int evaluations = 0;
    AM_DCHECK(false, "diagnostic {}", ++evaluations);
    EXPECT_EQ(evaluations, 0);
}
#else
TEST(DebugCheckMacroDeathTest, FailsWithStableDiagnosticPrefix) {
    EXPECT_DEATH(AM_DCHECK(false), "Check failed");
}
#endif

} // namespace
