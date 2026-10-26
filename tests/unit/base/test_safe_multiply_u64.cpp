#include "utils/overflow_check.h"

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

namespace {

using aethermind::SafeMultiplyU64;

template<typename Container>
void ExpectProduct(const Container& values, bool expected_overflow, uint64_t expected_product) {
    {
        SCOPED_TRACE("iterator overload");
        uint64_t result = ~expected_product;
        EXPECT_EQ(SafeMultiplyU64(values.begin(), values.end(), &result), expected_overflow);
        EXPECT_EQ(result, expected_product);
    }
    {
        SCOPED_TRACE("container overload");
        uint64_t result = ~expected_product;
        EXPECT_EQ(SafeMultiplyU64(values, &result), expected_overflow);
        EXPECT_EQ(result, expected_product);
    }
}

TEST(SafeMultiplyU64, EmptyRangeWritesMultiplicativeIdentity) {
    const std::array<uint64_t, 0> values{};

    ExpectProduct(values, false, 1);
}

TEST(SafeMultiplyU64, MultipliesNonnegativeSignedValues) {
    const std::vector<int64_t> values{2, 3, 7};

    ExpectProduct(values, false, 42);
}

TEST(SafeMultiplyU64, MultipliesUnsignedValues) {
    const std::array<uint32_t, 3> values{3, 5, 7};

    ExpectProduct(values, false, 105);
}

TEST(SafeMultiplyU64, ZeroBeforeLargeFactorsDoesNotOverflow) {
    constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
    const std::array<uint64_t, 3> values{0, kMax, 2};

    ExpectProduct(values, false, 0);
}

TEST(SafeMultiplyU64, AcceptsProductAtUint64Maximum) {
    const std::array<uint64_t, 7> values{3, 5, 17, 257, 65537, 641, 6700417};

    ExpectProduct(values, false, std::numeric_limits<uint64_t>::max());
}

TEST(SafeMultiplyU64, OverflowWritesWrappingProduct) {
    constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
    const std::array<uint64_t, 2> values{kMax, 2};

    ExpectProduct(values, true, kMax - 1);
}

TEST(SafeMultiplyU64, ContinuesMultiplyingAfterOverflow) {
    constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
    const std::array<uint64_t, 4> values{kMax, 2, 3, 5};

    ExpectProduct(values, true, kMax - 29);
}

TEST(SafeMultiplyU64, RetainsOverflowAfterNonoverflowingMultiplications) {
    const std::array<uint64_t, 4> values{uint64_t{1} << 63, 2, 3, 5};

    ExpectProduct(values, true, 0);
}

TEST(SafeMultiplyU64, RetainsOverflowAfterZero) {
    constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
    const std::array<uint64_t, 4> values{kMax, 2, 0, 3};

    ExpectProduct(values, true, 0);
}

#ifndef NDEBUG
TEST(SafeMultiplyU64, NegativeElementDeath) {
    const std::array<int64_t, 2> values{0, -1};
    uint64_t result = 7;

    EXPECT_DEATH(SafeMultiplyU64(values.begin(), values.end(), &result), "Check failed");
    EXPECT_DEATH(SafeMultiplyU64(values, &result), "Check failed");
}

TEST(SafeMultiplyU64, NullOutputDeath) {
    const std::array<uint64_t, 0> values{};

    EXPECT_DEATH(SafeMultiplyU64(values.begin(), values.end(), nullptr), "Check failed");
    EXPECT_DEATH(SafeMultiplyU64(values, nullptr), "Check failed");
}
#endif

} // namespace
