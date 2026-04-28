#include <gtest/gtest.h>

#include "amstring/basic_string.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

template<typename CharT>
class BasicStringDifferentialTest : public ::testing::Test {
protected:
    using CharType = CharT;
    using AmString = aethermind::BasicString<CharT>;
    using StdString = std::basic_string<CharT>;

    static std::size_t inline_capacity() {
        return AmString{}.capacity();
    }

    static StdString make_string(std::size_t count, char first = 'a') {
        StdString result;
        result.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            result.push_back(static_cast<CharT>(first + static_cast<char>(i % 23)));
        }
        return result;
    }

    static std::vector<std::size_t> interesting_sizes() {
        const std::size_t cap = inline_capacity();
        std::vector<std::size_t> sizes = {0, 1};
        if (cap > 0) {
            sizes.push_back(cap - 1);
            sizes.push_back(cap);
            sizes.push_back(cap + 1);
        }
        sizes.push_back(cap + 5);
        sizes.push_back(cap * 2 + 3);
        std::sort(sizes.begin(), sizes.end());
        sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
        return sizes;
    }

    static void expect_equivalent(const AmString& actual, const StdString& expected) {
        EXPECT_EQ(actual.size(), expected.size());
        EXPECT_EQ(actual.length(), expected.size());
        EXPECT_EQ(actual.empty(), expected.empty());
        EXPECT_GE(actual.capacity(), actual.size());
        ASSERT_NE(actual.data(), nullptr);
        ASSERT_EQ(actual.c_str(), actual.data());
        for (std::size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(actual.data()[i], expected[i]);
            EXPECT_EQ(actual[i], expected[i]);
        }
        EXPECT_EQ(actual.data()[actual.size()], CharT{});
    }

    static void expect_pointer_length_constructor(std::size_t count) {
        const StdString expected = make_string(count, 'a');
        const AmString actual(expected.data(), expected.size());
        expect_equivalent(actual, expected);
    }

    static void expect_c_string_constructor(std::size_t count) {
        const StdString expected = make_string(count, 'b');
        const AmString actual(expected.c_str());
        expect_equivalent(actual, expected);
    }

    static void expect_string_view_constructor(std::size_t count) {
        const StdString expected = make_string(count, 'c');
        const std::basic_string_view<CharT> view(expected.data(), expected.size());
        const AmString actual(view);
        expect_equivalent(actual, expected);
    }

    static void expect_count_constructor(std::size_t count) {
        const CharT fill = static_cast<CharT>('q');
        const StdString expected(count, fill);
        const AmString actual(count, fill);
        expect_equivalent(actual, expected);
    }

    static void expect_copy_and_move(std::size_t count) {
        const StdString source_std = make_string(count, 'd');
        const AmString source_am(source_std.data(), source_std.size());

        const AmString copied(source_am);
        expect_equivalent(copied, source_std);

        AmString moved_source(source_std.data(), source_std.size());
        AmString moved(std::move(moved_source));
        expect_equivalent(moved, source_std);
        ASSERT_NE(moved_source.data(), nullptr);
        EXPECT_GE(moved_source.capacity(), moved_source.size());
    }

    static void expect_assign_and_append(std::size_t count) {
        const StdString initial = make_string(std::max<std::size_t>(count, 3), 'e');
        const StdString rhs = make_string(count, 'm');
        const CharT fill = static_cast<CharT>('z');

        AmString actual(initial.data(), initial.size());
        StdString expected = initial;

        actual.assign(rhs.data(), rhs.size());
        expected.assign(rhs.data(), rhs.size());
        expect_equivalent(actual, expected);

        actual.assign(std::basic_string_view<CharT>(initial.data(), initial.size()));
        expected.assign(initial.data(), initial.size());
        expect_equivalent(actual, expected);

        actual.assign(count, fill);
        expected.assign(count, fill);
        expect_equivalent(actual, expected);

        actual.append(rhs.data(), rhs.size());
        expected.append(rhs.data(), rhs.size());
        expect_equivalent(actual, expected);

        actual.append(std::basic_string_view<CharT>(initial.data(), initial.size()));
        expected.append(initial.data(), initial.size());
        expect_equivalent(actual, expected);

        actual.append(count, fill);
        expected.append(count, fill);
        expect_equivalent(actual, expected);
    }

    static void expect_resize_reserve_and_shrink(std::size_t count) {
        const StdString initial = make_string(std::max<std::size_t>(count, 3), 'f');
        const CharT fill = static_cast<CharT>('r');
        const std::size_t grow_to = std::max<std::size_t>(initial.size() + 2, inline_capacity() + 3);

        AmString actual(initial.data(), initial.size());
        StdString expected = initial;

        const std::size_t reserve_to = grow_to + inline_capacity() + 5;
        actual.reserve(reserve_to);
        expected.reserve(reserve_to);
        expect_equivalent(actual, expected);
        EXPECT_GE(actual.capacity(), reserve_to);

        const std::size_t shrink_to = initial.size() > 0 ? initial.size() - 1 : 0;
        actual.resize(shrink_to);
        expected.resize(shrink_to);
        expect_equivalent(actual, expected);

        actual.resize(grow_to);
        expected.resize(grow_to);
        expect_equivalent(actual, expected);

        actual.resize(grow_to + 3, fill);
        expected.resize(grow_to + 3, fill);
        expect_equivalent(actual, expected);

        actual.shrink_to_fit();
        expected.shrink_to_fit();
        expect_equivalent(actual, expected);
    }

    static void expect_push_pop_and_clear(std::size_t count) {
        const StdString initial = make_string(count, 'g');
        const CharT ch = static_cast<CharT>('x');

        AmString actual(initial.data(), initial.size());
        StdString expected = initial;

        actual.push_back(ch);
        expected.push_back(ch);
        expect_equivalent(actual, expected);

        actual.pop_back();
        expected.pop_back();
        expect_equivalent(actual, expected);

        actual.clear();
        expected.clear();
        expect_equivalent(actual, expected);
    }

    static void run_random_sequence(std::uint32_t seed, std::size_t steps) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> op_dist(0, 7);
        std::uniform_int_distribution<int> first_dist(0, 20);
        std::uniform_int_distribution<int> extra_dist(0, static_cast<int>(inline_capacity() + 8));
        std::uniform_int_distribution<int> char_dist(0, 25);

        AmString actual;
        StdString expected;

        for (std::size_t step = 0; step < steps; ++step) {
            const int op = op_dist(rng);
            const std::size_t count = static_cast<std::size_t>(extra_dist(rng));
            const CharT fill = static_cast<CharT>('a' + char_dist(rng));
            const StdString payload = make_string(count, static_cast<char>('a' + first_dist(rng)));

            SCOPED_TRACE(::testing::Message() << "seed=" << seed << ", step=" << step << ", op=" << op << ", count=" << count);

            switch (op) {
                case 0:
                    actual.assign(payload.data(), payload.size());
                    expected.assign(payload.data(), payload.size());
                    break;
                case 1:
                    actual.assign(count, fill);
                    expected.assign(count, fill);
                    break;
                case 2:
                    actual.append(payload.data(), payload.size());
                    expected.append(payload.data(), payload.size());
                    break;
                case 3:
                    actual.append(count, fill);
                    expected.append(count, fill);
                    break;
                case 4:
                    actual.resize(count);
                    expected.resize(count);
                    break;
                case 5:
                    actual.resize(count, fill);
                    expected.resize(count, fill);
                    break;
                case 6:
                    actual.reserve(count + inline_capacity());
                    expected.reserve(count + inline_capacity());
                    break;
                case 7:
                    actual.shrink_to_fit();
                    expected.shrink_to_fit();
                    break;
                default:
                    break;
            }

            expect_equivalent(actual, expected);
        }
    }
};

using AmstringDifferentialCharTypes = ::testing::Types<char, char8_t, char16_t, char32_t, wchar_t>;
TYPED_TEST_SUITE(BasicStringDifferentialTest, AmstringDifferentialCharTypes);

TYPED_TEST(BasicStringDifferentialTest, ConstructorsMatchStdBasicString) {
    for (const std::size_t count: TestFixture::interesting_sizes()) {
        TestFixture::expect_pointer_length_constructor(count);
        TestFixture::expect_c_string_constructor(count);
        TestFixture::expect_string_view_constructor(count);
        TestFixture::expect_count_constructor(count);
        TestFixture::expect_copy_and_move(count);
    }
}

TYPED_TEST(BasicStringDifferentialTest, AssignAndAppendMatchStdBasicString) {
    for (const std::size_t count: TestFixture::interesting_sizes()) {
        TestFixture::expect_assign_and_append(count);
    }
}

TYPED_TEST(BasicStringDifferentialTest, ResizeReserveAndShrinkMatchStdBasicString) {
    for (const std::size_t count: TestFixture::interesting_sizes()) {
        TestFixture::expect_resize_reserve_and_shrink(count);
    }
}

TYPED_TEST(BasicStringDifferentialTest, PushPopAndClearMatchStdBasicString) {
    for (const std::size_t count: TestFixture::interesting_sizes()) {
        TestFixture::expect_push_pop_and_clear(count);
    }
}

TYPED_TEST(BasicStringDifferentialTest, RandomOperationSequenceMatchesStdBasicString) {
    constexpr std::array<std::uint32_t, 4> kSeeds = {7U, 19U, 97U, 211U};
    for (const std::uint32_t seed: kSeeds) {
        TestFixture::run_random_sequence(seed, 250);
    }
}

}// namespace
