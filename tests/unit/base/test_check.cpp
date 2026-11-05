#include "aethermind/base/check.h"

#include <gtest/gtest.h>

#include <stdexcept>

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

TEST(CheckMacroDeathTest, InvalidDynamicWidthDeath) {
    int width = -1;
    EXPECT_DEATH(AM_CHECK(false, "{:{}}", 7, width), "Check failed: \\(false\\)");
}

TEST(CheckMacroDeathTest, ThrowingDiagnosticArgumentDeath) {
    auto diagnostic = []() -> int { throw std::runtime_error("unavailable"); };
    EXPECT_DEATH(AM_CHECK(false, "{}", diagnostic()), "Check failed: \\(false\\)");
}

TEST(CheckMacro, UsesExplicitBoolConversion) {
    struct Condition {
        int& evaluations;
        explicit operator bool() const {
            ++evaluations;
            return true;
        }
        bool operator!() const = delete;
    };
    int evaluations = 0;
    Condition condition{evaluations};
    AM_CHECK(condition);
    EXPECT_EQ(evaluations, 1);
    AM_DCHECK(condition);
#ifdef NDEBUG
    EXPECT_EQ(evaluations, 1);
#else
    EXPECT_EQ(evaluations, 2);
#endif
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
