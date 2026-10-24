#include "utils/hash.h"

#include "container/detail/fibonacci_hash.h"

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace hash_test {

struct DualCustomization {
    static size_t hash(const DualCustomization&) { return 101; }
};

struct StandardCustomization {};

struct CopyTracked {
    int value;
    static inline int copies = 0;

    explicit CopyTracked(int v) : value(v) {}
    CopyTracked(const CopyTracked& other) : value(other.value) { ++copies; }
    CopyTracked(CopyTracked&&) = default;

    static size_t hash(const CopyTracked& value) {
        return static_cast<size_t>(value.value);
    }
};

enum class Enumeration : uint64_t { kValue = 4294967295ULL };

} // namespace hash_test

namespace std {

template<>
struct hash<hash_test::DualCustomization> {
    size_t operator()(const hash_test::DualCustomization&) const { return 202; }
};

template<>
struct hash<hash_test::StandardCustomization> {
    size_t operator()(const hash_test::StandardCustomization&) const { return 303; }
};

} // namespace std

namespace {

static_assert(noexcept(aethermind::details::FibonacciHash(1, 63)));
static_assert(aethermind::details::FibonacciHash(1, 63) == 1);
static_assert(aethermind::details::FibonacciHash(2, 63) == 0);
static_assert(aethermind::details::FibonacciHash(1, 64) == 0);

TEST(HashUtils, StaticCustomizationTakesPrecedenceOverStandardCustomization) {
    EXPECT_EQ(aethermind::hash<hash_test::DualCustomization>{}({}), 101U);
}

TEST(HashUtils, UsesStandardCustomizationWhenNoStaticCustomizationExists) {
    EXPECT_EQ(aethermind::hash<hash_test::StandardCustomization>{}({}), 303U);
}

TEST(HashUtils, EnumerationUsesItsUnderlyingValue) {
    const auto value = hash_test::Enumeration::kValue;
    EXPECT_EQ(aethermind::hash<hash_test::Enumeration>{}(value),
              std::hash<uint64_t>{}(static_cast<uint64_t>(value)));
}

TEST(HashUtils, EmptyTupleUsesZeroSeed) {
    EXPECT_EQ(aethermind::hash<std::tuple<>>{}({}), 0U);
}

TEST(HashUtils, TupleReferencesMatchOwningValues) {
    const int number = 17;
    const std::string text = "value";
    auto values = std::make_tuple(number, text);
    auto references = std::tie(number, text);

    EXPECT_EQ(aethermind::hash<decltype(values)>{}(values),
              aethermind::hash<decltype(references)>{}(references));
    EXPECT_EQ(aethermind::get_hash(number, text),
              aethermind::hash<decltype(values)>{}(values));
}

TEST(HashUtils, PairMatchesTupleWithTheSameElements) {
    std::pair<int, std::string> pair{17, "value"};
    EXPECT_EQ(aethermind::hash<decltype(pair)>{}(pair),
              aethermind::get_hash(pair.first, pair.second));
}

TEST(HashUtils, PairHashingDoesNotCopyElements) {
    std::pair<hash_test::CopyTracked, hash_test::CopyTracked> pair{
            hash_test::CopyTracked{17}, hash_test::CopyTracked{19}};
    hash_test::CopyTracked::copies = 0;

    EXPECT_EQ(aethermind::hash<decltype(pair)>{}(pair),
              aethermind::get_hash(pair.first, pair.second));
    EXPECT_EQ(hash_test::CopyTracked::copies, 0);
}

TEST(HashUtils, PairHashingAcceptsMoveOnlyElements) {
    std::pair<std::unique_ptr<int>, int> pair{std::make_unique<int>(17), 19};

    EXPECT_EQ(aethermind::hash<decltype(pair)>{}(pair),
              aethermind::get_hash(pair.first, pair.second));
    ASSERT_NE(pair.first, nullptr);
    EXPECT_EQ(*pair.first, 17);
}

TEST(HashUtils, VectorHashesElementsInOrder) {
    const std::vector<int> values{17, 19};
    EXPECT_EQ(aethermind::hash<std::vector<int>>{}(values),
              aethermind::get_hash(17, 19));
    EXPECT_EQ(aethermind::hash<std::vector<int>>{}({}), 0U);
    EXPECT_NE(aethermind::hash<std::vector<int>>{}(values),
              aethermind::hash<std::vector<int>>{}({19, 17}));
}

TEST(HashUtils, ComplexHashesRealAndImaginaryParts) {
    const std::complex<double> value{1.5, -2.0};
    EXPECT_EQ(aethermind::hash<std::complex<double>>{}(value),
              aethermind::get_hash(value.real(), value.imag()));
}

TEST(HashUtils, StandardContainersAcceptExplicitCompositeHasher) {
    using Key = std::tuple<int, std::string>;
    std::unordered_map<Key, int, aethermind::hash<Key>> map;
    map.emplace(Key{17, "value"}, 19);
    EXPECT_EQ(map.at(Key{17, "value"}), 19);

    std::unordered_set<std::vector<int>, aethermind::hash<std::vector<int>>> set;
    set.emplace(std::vector<int>{17, 19});
    EXPECT_TRUE(set.contains({17, 19}));
    EXPECT_FALSE(set.contains({19, 17}));
}

TEST(HashUtils, CombineConvertsSignedInputBeforeUnsignedArithmetic) {
    constexpr long long kInput = std::numeric_limits<long long>::max();
    EXPECT_EQ(aethermind::hash_combine(size_t{17}, kInput),
              aethermind::hash_combine(size_t{17}, static_cast<size_t>(kInput)));
}

TEST(FibonacciHash, PreservesKnownBucketIndices) {
    EXPECT_EQ(aethermind::details::FibonacciHash(0, 60), 0U);
    EXPECT_EQ(aethermind::details::FibonacciHash(1, 60), 9U);
    EXPECT_EQ(aethermind::details::FibonacciHash(2, 60), 3U);
    EXPECT_EQ(aethermind::details::FibonacciHash(3, 60), 13U);
}

TEST(FibonacciHash, KeepsTheProductAt64BitsFor32BitHashValues) {
    EXPECT_EQ(aethermind::details::FibonacciHash(uint32_t{0xffffffffU}, 32),
              size_t{3776119387U});
}

TEST(FibonacciHash, ZeroShiftReturnsTheProductInSizeT) {
    EXPECT_EQ(aethermind::details::FibonacciHash(1, 0),
              static_cast<size_t>(uint64_t{0x9e3779b97f4a7c15ULL}));
}

TEST(FibonacciHash, FullWidthAndLargerShiftsSelectTheSingleBucket) {
    for (const uint32_t shift: {uint32_t{64}, uint32_t{65},
                                std::numeric_limits<uint32_t>::max()}) {
        EXPECT_EQ(aethermind::details::FibonacciHash(std::numeric_limits<size_t>::max(), shift), 0U);
    }
}

} // namespace
