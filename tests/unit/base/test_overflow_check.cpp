#include "utils/overflow_check.h"

#include <cstddef>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

namespace {

using aethermind::CheckNarrow;

template<typename To, typename From>
concept CanCheckNarrow = requires(From value, To* out) {
    CheckNarrow(value, out);
};

template<typename T>
concept SupportedConversionType = CanCheckNarrow<T, int> && CanCheckNarrow<int, T>;

static_assert(SupportedConversionType<signed char>);
static_assert(SupportedConversionType<unsigned char>);
static_assert(SupportedConversionType<short>);
static_assert(SupportedConversionType<unsigned short>);
static_assert(SupportedConversionType<int>);
static_assert(SupportedConversionType<unsigned int>);
static_assert(SupportedConversionType<long>);
static_assert(SupportedConversionType<unsigned long>);
static_assert(SupportedConversionType<long long>);
static_assert(SupportedConversionType<unsigned long long>);

enum class ScopedInteger { kValue };
enum UnscopedInteger { kValue };

template<typename T>
concept RejectedConversionType = !CanCheckNarrow<T, int> && !CanCheckNarrow<int, T>;

static_assert(RejectedConversionType<bool>);
static_assert(RejectedConversionType<char>);
static_assert(RejectedConversionType<wchar_t>);
static_assert(RejectedConversionType<char8_t>);
static_assert(RejectedConversionType<char16_t>);
static_assert(RejectedConversionType<char32_t>);
static_assert(RejectedConversionType<ScopedInteger>);
static_assert(RejectedConversionType<UnscopedInteger>);
static_assert(RejectedConversionType<float>);
static_assert(RejectedConversionType<double>);
static_assert(RejectedConversionType<int*>);
static_assert(!CanCheckNarrow<const int, int>);
static_assert(!CanCheckNarrow<volatile int, int>);
static_assert(!CanCheckNarrow<const volatile int, int>);
static_assert(noexcept(CheckNarrow(1, static_cast<unsigned*>(nullptr))));

static_assert([] {
    uint32_t result = 9;
    return !CheckNarrow(int64_t{0}, &result) && result == 0;
}());

static_assert([] {
    uint32_t result = 9;
    return CheckNarrow(int64_t{-1}, &result) && result == 9;
}());

static_assert([] {
    int8_t result = 9;
    return CheckNarrow(int64_t{128}, &result) && result == 9;
}());

TEST(CheckNarrow, SignedNarrowingAcceptsBounds) {
    for (const int32_t input: {int32_t{std::numeric_limits<int16_t>::min()},
                               int32_t{std::numeric_limits<int16_t>::max()}}) {
        int16_t result = 7;

        EXPECT_FALSE(CheckNarrow(input, &result));
        EXPECT_EQ(result, input);
    }
}

TEST(CheckNarrow, SignedNarrowingRejectsOutsideBoundsWithoutWriting) {
    for (const int32_t input: {int32_t{std::numeric_limits<int16_t>::min()} - 1,
                               int32_t{std::numeric_limits<int16_t>::max()} + 1}) {
        int16_t result = 7;

        EXPECT_TRUE(CheckNarrow(input, &result));
        EXPECT_EQ(result, 7);
    }
}

TEST(CheckNarrow, UnsignedNarrowingAcceptsBounds) {
    for (const uint64_t input: {uint64_t{0}, uint64_t{std::numeric_limits<uint32_t>::max()}}) {
        uint32_t result = 7;

        EXPECT_FALSE(CheckNarrow(input, &result));
        EXPECT_EQ(result, input);
    }
}

TEST(CheckNarrow, UnsignedNarrowingRejectsAboveMaximumWithoutWriting) {
    uint32_t result = 7;
    constexpr uint64_t kInput = uint64_t{std::numeric_limits<uint32_t>::max()} + 1;

    EXPECT_TRUE(CheckNarrow(kInput, &result));
    EXPECT_EQ(result, 7U);
}

TEST(CheckNarrow, SignedToUnsignedRejectsNegativeValuesWithoutWriting) {
    for (const int64_t input: {int64_t{-1}, std::numeric_limits<int64_t>::min()}) {
        uint64_t result = 7;

        EXPECT_TRUE(CheckNarrow(input, &result));
        EXPECT_EQ(result, 7U);
    }
}

TEST(CheckNarrow, SignedToUnsignedAcceptsNonnegativeValues) {
    for (const int32_t input: {int32_t{0}, std::numeric_limits<int32_t>::max()}) {
        uint32_t result = 7;

        EXPECT_FALSE(CheckNarrow(input, &result));
        EXPECT_EQ(result, static_cast<uint32_t>(input));
    }
}

TEST(CheckNarrow, UnsignedToSignedAcceptsMaximum) {
    int32_t result = 7;
    constexpr uint32_t kInput = std::numeric_limits<int32_t>::max();

    EXPECT_FALSE(CheckNarrow(kInput, &result));
    EXPECT_EQ(result, std::numeric_limits<int32_t>::max());
}

TEST(CheckNarrow, UnsignedToSignedRejectsAboveMaximumWithoutWriting) {
    for (const uint32_t input: {uint32_t{std::numeric_limits<int32_t>::max()} + 1,
                                std::numeric_limits<uint32_t>::max()}) {
        int32_t result = 7;

        EXPECT_TRUE(CheckNarrow(input, &result));
        EXPECT_EQ(result, 7);
    }
}

TEST(CheckNarrow, SignedWideningPreservesBounds) {
    for (const int32_t input: {std::numeric_limits<int32_t>::min(),
                               std::numeric_limits<int32_t>::max()}) {
        int64_t result = 7;

        EXPECT_FALSE(CheckNarrow(input, &result));
        EXPECT_EQ(result, input);
    }
}

TEST(CheckNarrow, UnsignedWideningPreservesBounds) {
    for (const uint32_t input: {uint32_t{0}, std::numeric_limits<uint32_t>::max()}) {
        uint64_t result = 7;

        EXPECT_FALSE(CheckNarrow(input, &result));
        EXPECT_EQ(result, input);
    }
}

TEST(CheckNarrow, UnsignedToWiderSignedPreservesFullSourceRange) {
    int64_t result = 7;
    constexpr uint32_t kInput = std::numeric_limits<uint32_t>::max();

    EXPECT_FALSE(CheckNarrow(kInput, &result));
    EXPECT_EQ(result, int64_t{4294967295});
}

TEST(CheckNarrow, SameTypeConversionPreservesBounds) {
    for (const int64_t input: {std::numeric_limits<int64_t>::min(),
                               std::numeric_limits<int64_t>::max()}) {
        int64_t result = 7;

        EXPECT_FALSE(CheckNarrow(input, &result));
        EXPECT_EQ(result, input);
    }
}

TEST(CheckNarrow, AcceptsConstSourceByValue) {
    const int32_t input = 42;
    size_t result = 7;

    EXPECT_FALSE(CheckNarrow(input, &result));
    EXPECT_EQ(result, 42U);
}

TEST(CheckNarrow, SizeTConversionChecksSignedSourceRange) {
    size_t result = 7;
    constexpr int64_t kInput = std::numeric_limits<int64_t>::max();

    if constexpr (std::numeric_limits<size_t>::digits < std::numeric_limits<int64_t>::digits) {
        EXPECT_TRUE(CheckNarrow(kInput, &result));
        EXPECT_EQ(result, 7U);
    } else {
        EXPECT_FALSE(CheckNarrow(kInput, &result));
        EXPECT_EQ(result, static_cast<size_t>(kInput));
    }
}

TEST(CheckNarrow, SizeTConversionChecksUnsignedSourceRange) {
    size_t result = 7;
    constexpr uint64_t kInput = std::numeric_limits<uint64_t>::max();

    if constexpr (std::numeric_limits<size_t>::digits < std::numeric_limits<uint64_t>::digits) {
        EXPECT_TRUE(CheckNarrow(kInput, &result));
        EXPECT_EQ(result, 7U);
    } else {
        EXPECT_FALSE(CheckNarrow(kInput, &result));
        EXPECT_EQ(result, static_cast<size_t>(kInput));
    }
}

} // namespace
