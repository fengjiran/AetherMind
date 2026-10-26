#include <gtest/gtest.h>

TEST(RequiredExecutionProbe, Skips) {
    GTEST_SKIP() << "Intentional skip for the CTest required-execution contract";
}
