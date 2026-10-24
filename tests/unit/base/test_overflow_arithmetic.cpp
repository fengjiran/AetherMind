#include "utils/overflow_check.h"

#include <array>
#include <cstdint>
#include <limits>
#include <random>

#include <gtest/gtest.h>

namespace {

using namespace aethermind;

template<typename T>
concept CanCheckOverflow = requires(T value, T* out) {
    CheckOverflowAdd(value, value, out);
    CheckOverflowSub(value, value, out);
    CheckOverflowMul(value, value, out);
};

static_assert(!CanCheckOverflow<bool>);
static_assert(!CanCheckOverflow<const int>);
static_assert(!CanCheckOverflow<volatile int>);
static_assert(noexcept(detail::CheckOverflowAddPortable(0, 0, static_cast<int*>(nullptr))));
static_assert(noexcept(detail::CheckOverflowSubPortable(0, 0, static_cast<int*>(nullptr))));
static_assert(noexcept(detail::CheckOverflowMulPortable(0, 0, static_cast<int*>(nullptr))));

static_assert([] {
    int64_t result = 1;
    return detail::CheckOverflowAddPortable(std::numeric_limits<int64_t>::min(),
                                            std::numeric_limits<int64_t>::min(), &result) &&
           result == 0;
}());

static_assert([] {
    int64_t result = 1;
    return detail::CheckOverflowMulPortable(std::numeric_limits<int64_t>::max(), int64_t{2}, &result) &&
           result == -2;
}());

static_assert([] {
    uint16_t result = 0;
    return detail::CheckOverflowMulPortable(uint16_t{65535}, uint16_t{65535}, &result) &&
           result == 1;
}());

template<typename T>
class OverflowPortable : public testing::Test {};

using ArithmeticTypes = testing::Types<int8_t, uint8_t, int16_t, uint16_t,
                                       int32_t, uint32_t, int64_t, uint64_t>;
TYPED_TEST_SUITE(OverflowPortable, ArithmeticTypes);

TYPED_TEST(OverflowPortable, AdditionChecksBoundsAndAlwaysWritesModuloResult) {
    using T = TypeParam;
    constexpr T kMax = std::numeric_limits<T>::max();
    constexpr T kMin = std::numeric_limits<T>::min();
    T result = 7;

    EXPECT_TRUE(detail::CheckOverflowAddPortable(kMax, T{1}, &result));
    EXPECT_EQ(result, kMin);
    EXPECT_FALSE(detail::CheckOverflowAddPortable(kMax, T{0}, &result));
    EXPECT_EQ(result, kMax);

    if constexpr (std::is_signed_v<T>) {
        EXPECT_TRUE(detail::CheckOverflowAddPortable(kMin, T{-1}, &result));
        EXPECT_EQ(result, kMax);
        EXPECT_TRUE(detail::CheckOverflowAddPortable(kMin, kMin, &result));
        EXPECT_EQ(result, 0);
        EXPECT_FALSE(detail::CheckOverflowAddPortable(kMin, kMax, &result));
        EXPECT_EQ(result, -1);
    }
}

TYPED_TEST(OverflowPortable, SubtractionChecksBoundsAndAlwaysWritesModuloResult) {
    using T = TypeParam;
    constexpr T kMax = std::numeric_limits<T>::max();
    constexpr T kMin = std::numeric_limits<T>::min();
    T result = 7;

    EXPECT_TRUE(detail::CheckOverflowSubPortable(kMin, T{1}, &result));
    EXPECT_EQ(result, kMax);
    EXPECT_FALSE(detail::CheckOverflowSubPortable(kMin, kMin, &result));
    EXPECT_EQ(result, 0);

    if constexpr (std::is_signed_v<T>) {
        EXPECT_TRUE(detail::CheckOverflowSubPortable(kMax, T{-1}, &result));
        EXPECT_EQ(result, kMin);
        EXPECT_TRUE(detail::CheckOverflowSubPortable(T{0}, kMin, &result));
        EXPECT_EQ(result, kMin);
    }
}

TYPED_TEST(OverflowPortable, MultiplicationChecksBoundsAndAlwaysWritesModuloResult) {
    using T = TypeParam;
    constexpr T kMax = std::numeric_limits<T>::max();
    constexpr T kMin = std::numeric_limits<T>::min();
    T result = 7;

    EXPECT_FALSE(detail::CheckOverflowMulPortable(kMax, T{0}, &result));
    EXPECT_EQ(result, 0);
    EXPECT_FALSE(detail::CheckOverflowMulPortable(kMin, T{1}, &result));
    EXPECT_EQ(result, kMin);
    EXPECT_TRUE(detail::CheckOverflowMulPortable(kMax, kMax, &result));
    EXPECT_EQ(result, 1);

    if constexpr (std::is_signed_v<T>) {
        EXPECT_FALSE(detail::CheckOverflowMulPortable(T{-2}, T{3}, &result));
        EXPECT_EQ(result, -6);
        EXPECT_FALSE(detail::CheckOverflowMulPortable(T{-2}, T{-3}, &result));
        EXPECT_EQ(result, 6);
        EXPECT_TRUE(detail::CheckOverflowMulPortable(kMax, T{2}, &result));
        EXPECT_EQ(result, -2);
        EXPECT_TRUE(detail::CheckOverflowMulPortable(kMin, T{-1}, &result));
        EXPECT_EQ(result, kMin);
        EXPECT_TRUE(detail::CheckOverflowMulPortable(T{-1}, kMin, &result));
        EXPECT_EQ(result, kMin);
    } else {
        EXPECT_TRUE(detail::CheckOverflowMulPortable(kMax, T{2}, &result));
        EXPECT_EQ(result, static_cast<T>(kMax - 1));
    }
}

TYPED_TEST(OverflowPortable, OutputMayAliasEitherOperand) {
    using T = TypeParam;
    constexpr T kMax = std::numeric_limits<T>::max();
    constexpr T kMin = std::numeric_limits<T>::min();
    T result = kMax;

    EXPECT_TRUE(detail::CheckOverflowAddPortable(result, T{1}, &result));
    EXPECT_EQ(result, kMin);
    EXPECT_EQ(detail::CheckOverflowSubPortable(T{0}, T{1}, &result), std::is_unsigned_v<T>);
    EXPECT_EQ(result, T(-1));
    result = kMax;
    EXPECT_TRUE(detail::CheckOverflowMulPortable(T{2}, result, &result));
    EXPECT_EQ(result, std::is_signed_v<T> ? T(-2) : static_cast<T>(kMax - 1));

    result = kMax;
    EXPECT_TRUE(CheckOverflowAdd(result, T{1}, &result));
    EXPECT_EQ(result, kMin);
    result = kMin;
    EXPECT_TRUE(CheckOverflowSub(result, T{1}, &result));
    EXPECT_EQ(result, kMax);
    result = kMax;
    EXPECT_TRUE(CheckOverflowMul(T{2}, result, &result));
    EXPECT_EQ(result, std::is_signed_v<T> ? T(-2) : static_cast<T>(kMax - 1));
}

template<typename T>
void ExhaustEightBitPairs() {
    constexpr int kMin = std::numeric_limits<T>::min();
    constexpr int kMax = std::numeric_limits<T>::max();
    for (int a = kMin; a <= kMax; ++a) {
        for (int b = kMin; b <= kMax; ++b) {
            T result = 0;
            const int sum = a + b;
            const int difference = a - b;
            const int product = a * b;
            const auto check = [&](bool overflowed, int exact) {
                if (overflowed != (exact < kMin || exact > kMax) || result != static_cast<T>(exact)) {
                    ADD_FAILURE() << "a=" << a << ", b=" << b << ", exact=" << exact
                                  << ", result=" << +result << ", overflowed=" << overflowed;
                    return false;
                }
                return true;
            };
            if (!check(detail::CheckOverflowAddPortable(T(a), T(b), &result), sum) ||
                !check(detail::CheckOverflowSubPortable(T(a), T(b), &result), difference) ||
                !check(detail::CheckOverflowMulPortable(T(a), T(b), &result), product)) {
                return;
            }
        }
    }
}

TEST(OverflowPortable, ExhaustiveSignedEightBitPairsMatchExactArithmetic) {
    ExhaustEightBitPairs<int8_t>();
}

TEST(OverflowPortable, ExhaustiveUnsignedEightBitPairsMatchExactArithmetic) {
    ExhaustEightBitPairs<uint8_t>();
}

#if defined(__GNUC__) || defined(__clang__)
template<typename T>
void CompareWithBuiltins(T a, T b) {
    SCOPED_TRACE(testing::Message() << "a=" << +a << ", b=" << +b);
    T expected = 0;
    T result = 0;
    EXPECT_EQ(detail::CheckOverflowAddPortable(a, b, &result), __builtin_add_overflow(a, b, &expected));
    EXPECT_EQ(result, expected);
    EXPECT_EQ(detail::CheckOverflowSubPortable(a, b, &result), __builtin_sub_overflow(a, b, &expected));
    EXPECT_EQ(result, expected);
    EXPECT_EQ(detail::CheckOverflowMulPortable(a, b, &result), __builtin_mul_overflow(a, b, &expected));
    EXPECT_EQ(result, expected);
}

TYPED_TEST(OverflowPortable, BoundaryPairsMatchCompilerBuiltins) {
    using T = TypeParam;
    constexpr T kMax = std::numeric_limits<T>::max();
    constexpr T kMin = std::numeric_limits<T>::min();
    constexpr std::array values{kMin, T(kMin + 1), T(kMax / 2), T(kMax / 2 + 1),
                                T(kMax - 1), kMax, T{0}, T{1}, T{2}, T(-1), T(-2)};
    for (const T a: values) {
        for (const T b: values) {
            CompareWithBuiltins(a, b);
        }
    }
}

TEST(OverflowPortable, DeterministicWideInputsMatchCompilerBuiltins) {
    std::mt19937_64 random(0x61ca2e59);
    for (size_t i = 0; i < 4096; ++i) {
        const uint64_t a = random();
        const uint64_t b = random();
        CompareWithBuiltins(a, b);
        CompareWithBuiltins(static_cast<int64_t>(a), static_cast<int64_t>(b));
    }
}
#endif

} // namespace
