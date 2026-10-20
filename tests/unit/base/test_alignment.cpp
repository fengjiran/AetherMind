#include "aethermind/base/alignment.h"
#include "inference/test_malloc_interposer.h"

#include <gtest/gtest.h>
#include <limits>

namespace {

using namespace aethermind;

TEST(OffsetAlignment, AlignsOffsetUpToBoundary) {
    const StatusOr<size_t> offset = AlignOffsetUp(17, 16);

    ASSERT_TRUE(offset.ok());
    EXPECT_EQ(*offset, 32U);
}

TEST(OffsetAlignment, ReturnsAlreadyAlignedOffsetUnchanged) {
    const StatusOr<size_t> offset = AlignOffsetUp(32, 16);

    ASSERT_TRUE(offset.ok());
    EXPECT_EQ(*offset, 32U);
}

TEST(OffsetAlignment, KeepsZeroOffsetAligned) {
    const StatusOr<size_t> offset = AlignOffsetUp(0, 16);

    ASSERT_TRUE(offset.ok());
    EXPECT_EQ(*offset, 0U);
}

TEST(OffsetAlignment, AcceptsLargestRepresentableAlignedOffset) {
    constexpr size_t kAlignment = 16;
    constexpr size_t kOffset =
            std::numeric_limits<size_t>::max() - (kAlignment - 1);

    const StatusOr<size_t> offset = AlignOffsetUp(kOffset, kAlignment);

    ASSERT_TRUE(offset.ok());
    EXPECT_EQ(*offset, kOffset);
}

TEST(OffsetAlignment, SuccessfulAlignmentDoesNotAllocate) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }

    test::ScopedMallocFailure failure(0);
    const StatusOr<size_t> offset = AlignOffsetUp(17, 16);
    const test::MallocFailureResult injection = failure.Stop();

    EXPECT_FALSE(injection.allocation_failed);
    ASSERT_TRUE(offset.ok());
    EXPECT_EQ(*offset, 32U);
}

TEST(OffsetAlignment, AcceptsMaximumOffsetAtByteAlignment) {
    constexpr size_t kMaxSize = std::numeric_limits<size_t>::max();

    const StatusOr<size_t> offset = AlignOffsetUp(kMaxSize, 1);

    ASSERT_TRUE(offset.ok());
    EXPECT_EQ(*offset, kMaxSize);
}

TEST(OffsetAlignment, AcceptsLargestPowerOfTwoAlignment) {
    constexpr size_t kLargestAlignment =
            size_t{1} << (std::numeric_limits<size_t>::digits - 1);

    const StatusOr<size_t> offset = AlignOffsetUp(1, kLargestAlignment);

    ASSERT_TRUE(offset.ok());
    EXPECT_EQ(*offset, kLargestAlignment);
}

TEST(OffsetAlignment, RejectsZeroAndNonPowerOfTwoAlignment) {
    static_assert(noexcept(AlignOffsetUp(0, 1)));

    for (const size_t alignment: {0U, 24U}) {
        const StatusOr<size_t> offset = AlignOffsetUp(8, alignment);

        ASSERT_FALSE(offset.ok());
        EXPECT_EQ(offset.status().code(), StatusCode::kInvalidArgument);
    }
}

TEST(OffsetAlignment, DetectsAlignmentOverflow) {
    const StatusOr<size_t> offset =
            AlignOffsetUp(std::numeric_limits<size_t>::max() - 3, 8);

    ASSERT_FALSE(offset.ok());
    EXPECT_EQ(offset.status().code(), StatusCode::kOverflow);
}

class OffsetAlignmentDiagnosticFailure : public ::testing::TestWithParam<bool> {};

TEST_P(OffsetAlignmentDiagnosticFailure, ReturnsResourceExhaustedWithEmptyMessage) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }
    const size_t input_offset = GetParam() ? std::numeric_limits<size_t>::max() - 3 : 8;
    const size_t alignment = GetParam() ? 8 : 24;

    test::ScopedMallocFailure failure(0);
    const StatusOr<size_t> offset = AlignOffsetUp(input_offset, alignment);
    const test::MallocFailureResult injection = failure.Stop();

    EXPECT_TRUE(injection.allocation_failed);
    ASSERT_FALSE(offset.ok());
    EXPECT_EQ(offset.status().code(), StatusCode::kResourceExhausted);
    EXPECT_TRUE(offset.status().message().empty());
}

INSTANTIATE_TEST_SUITE_P(
        FailurePaths,
        OffsetAlignmentDiagnosticFailure,
        ::testing::Bool(),
        [](const ::testing::TestParamInfo<bool>& info) {
            return info.param ? "AlignmentOverflow" : "InvalidAlignment";
        });

} // namespace
