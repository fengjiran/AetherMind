#include <gtest/gtest.h>

#include "amstring/basic_string.hpp"

#include <cstddef>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace {

template<typename T>
struct StatefulAllocator {
    using value_type = T;

    int id = 0;

    StatefulAllocator() noexcept = default;

    explicit StatefulAllocator(int allocator_id) noexcept
        : id(allocator_id) {}

    template<typename U>
    StatefulAllocator(const StatefulAllocator<U>& other) noexcept
        : id(other.id) {}

    T* allocate(std::size_t n) {
        return std::allocator<T>{}.allocate(n);
    }

    void deallocate(T* ptr, std::size_t n) noexcept {
        std::allocator<T>{}.deallocate(ptr, n);
    }

    template<typename U>
    bool operator==(const StatefulAllocator<U>& other) const noexcept {
        return id == other.id;
    }

    template<typename U>
    bool operator!=(const StatefulAllocator<U>& other) const noexcept {
        return !(*this == other);
    }
};

template<typename CharT>
class BasicStringSkeletonTest : public ::testing::Test {
protected:
    using CharType = CharT;
    using String = aethermind::BasicString<CharT>;

    static std::basic_string<CharT> make_string(std::size_t count, char first = 'a') {
        std::basic_string<CharT> result;
        result.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            result.push_back(static_cast<CharT>(first + static_cast<char>(i % 23)));
        }
        return result;
    }

    static void expect_content(const String& string, std::basic_string_view<CharT> expected) {
        EXPECT_EQ(string.size(), expected.size());
        EXPECT_EQ(string.length(), expected.size());
        EXPECT_GE(string.capacity(), string.size());
        EXPECT_EQ(string.empty(), expected.empty());
        ASSERT_NE(string.data(), nullptr);
        EXPECT_EQ(string.c_str(), string.data());
        for (std::size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(string.data()[i], expected[i]);
            EXPECT_EQ(string[i], expected[i]);
        }
        EXPECT_EQ(string.data()[string.size()], CharT{});
    }
};

using AmstringPublicCharTypes = ::testing::Types<char, char8_t, char16_t, char32_t, wchar_t>;
TYPED_TEST_SUITE(BasicStringSkeletonTest, AmstringPublicCharTypes);

TYPED_TEST(BasicStringSkeletonTest, CharTMatrixIsAvailable) {
    using CharT = TypeParam;

    static_assert(std::is_trivial_v<CharT>);
    static_assert(std::is_standard_layout_v<CharT>);

    SUCCEED() << "M0 skeleton: BasicString tests are parameterized by CharT.";
}

TYPED_TEST(BasicStringSkeletonTest, StandardTypeAliasesAreAvailable) {
    using CharT = TypeParam;
    using String = aethermind::BasicString<CharT>;

    static_assert(std::is_same_v<typename String::value_type, CharT>);
    static_assert(std::is_same_v<typename String::traits_type, std::char_traits<CharT>>);
    static_assert(std::is_same_v<typename String::allocator_type, std::allocator<CharT>>);
    static_assert(std::is_same_v<typename String::size_type, std::size_t>);
    static_assert(std::is_same_v<typename String::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<typename String::reference, CharT&>);
    static_assert(std::is_same_v<typename String::const_reference, const CharT&>);
    static_assert(std::is_same_v<typename String::pointer, CharT*>);
    static_assert(std::is_same_v<typename String::const_pointer, const CharT*>);
    static_assert(std::is_same_v<typename String::iterator, CharT*>);
    static_assert(std::is_same_v<typename String::const_iterator, const CharT*>);
}

TYPED_TEST(BasicStringSkeletonTest, DefaultConstructedStringIsEmpty) {
    using String = typename TestFixture::String;
    using CharT = typename TestFixture::CharType;

    String string;

    EXPECT_EQ(string.size(), 0U);
    EXPECT_EQ(string.length(), 0U);
    EXPECT_TRUE(string.empty());
    EXPECT_GE(string.capacity(), string.size());
    ASSERT_NE(string.data(), nullptr);
    EXPECT_EQ(string.c_str(), string.data());
    EXPECT_EQ(string.data()[0], CharT{});
}

TYPED_TEST(BasicStringSkeletonTest, PointerAndLengthConstructorExposesPublicSemantics) {
    using String = typename TestFixture::String;
    using CharT = typename TestFixture::CharType;

    const auto expected = TestFixture::make_string(5);
    String string(expected.data(), expected.size());

    TestFixture::expect_content(string, expected);
    EXPECT_FALSE(string.empty());
}

TYPED_TEST(BasicStringSkeletonTest, CStringConstructorExposesPublicSemantics) {
    using String = typename TestFixture::String;

    const auto expected = TestFixture::make_string(4);
    String string(expected.c_str());

    TestFixture::expect_content(string, expected);
}

TYPED_TEST(BasicStringSkeletonTest, StringViewConstructorExposesPublicSemantics) {
    using String = typename TestFixture::String;
    using CharT = typename TestFixture::CharType;

    const auto expected = TestFixture::make_string(6);
    const std::basic_string_view<CharT> view(expected.data(), expected.size());
    String string(view);

    TestFixture::expect_content(string, view);
}

TYPED_TEST(BasicStringSkeletonTest, CountAndCharConstructorFillsContent) {
    using String = typename TestFixture::String;
    using CharT = typename TestFixture::CharType;

    const CharT fill = static_cast<CharT>('q');
    String string(7, fill);

    EXPECT_EQ(string.size(), 7U);
    for (std::size_t i = 0; i < string.size(); ++i) {
        EXPECT_EQ(string.data()[i], fill);
    }
    EXPECT_EQ(string.data()[string.size()], CharT{});
}

TYPED_TEST(BasicStringSkeletonTest, CStringMatchesData) {
    using String = typename TestFixture::String;
    using CharT = typename TestFixture::CharType;

    const auto expected = TestFixture::make_string(5);
    String string(expected.data(), expected.size());

    EXPECT_EQ(string.c_str(), string.data());
    EXPECT_EQ(string.c_str()[string.size()], CharT{});
}

TYPED_TEST(BasicStringSkeletonTest, CapacityIsAtLeastSize) {
    using String = typename TestFixture::String;

    String empty;
    const auto small = TestFixture::make_string(3);
    const auto external = TestFixture::make_string(128);
    String small_string(small.data(), small.size());
    String external_string(external.data(), external.size());

    EXPECT_GE(empty.capacity(), empty.size());
    EXPECT_GE(small_string.capacity(), small_string.size());
    EXPECT_GE(external_string.capacity(), external_string.size());
}

TYPED_TEST(BasicStringSkeletonTest, DataReturnsWritableBuffer) {
    using String = typename TestFixture::String;
    using CharT = typename TestFixture::CharType;

    const auto expected = TestFixture::make_string(3);
    String string(expected.data(), expected.size());
    const CharT replacement = static_cast<CharT>('z');

    string.data()[1] = replacement;

    EXPECT_EQ(string.data()[1], replacement);
    EXPECT_EQ(string.data()[string.size()], CharT{});
}

TYPED_TEST(BasicStringSkeletonTest, ConstDataReturnsReadableBuffer) {
    using String = typename TestFixture::String;

    const auto expected = TestFixture::make_string(3);
    const String string(expected.data(), expected.size());

    TestFixture::expect_content(string, expected);
}

TYPED_TEST(BasicStringSkeletonTest, IteratorsSpanVisibleContent) {
    using String = typename TestFixture::String;

    const auto expected = TestFixture::make_string(8);
    String string(expected.data(), expected.size());

    EXPECT_EQ(static_cast<std::size_t>(std::distance(string.begin(), string.end())), expected.size());
    EXPECT_EQ(static_cast<std::size_t>(std::distance(string.cbegin(), string.cend())), expected.size());
    std::size_t index = 0;
    for (auto it = string.begin(); it != string.end(); ++it, ++index) {
        EXPECT_EQ(*it, expected[index]);
    }
}

TYPED_TEST(BasicStringSkeletonTest, CopyConstructorCopiesPublicState) {
    using String = typename TestFixture::String;

    const auto expected = TestFixture::make_string(64);
    const String original(expected.data(), expected.size());
    String copy(original);

    TestFixture::expect_content(copy, expected);
    TestFixture::expect_content(original, expected);
}

TYPED_TEST(BasicStringSkeletonTest, MoveConstructorPreservesPublicUsability) {
    using String = typename TestFixture::String;

    const auto expected = TestFixture::make_string(64);
    String source(expected.data(), expected.size());
    String moved(std::move(source));

    TestFixture::expect_content(moved, expected);
    ASSERT_NE(source.data(), nullptr);
    EXPECT_GE(source.capacity(), source.size());
    source.clear();
    EXPECT_TRUE(source.empty());
}

TYPED_TEST(BasicStringSkeletonTest, CopyAssignmentReplacesContent) {
    using String = typename TestFixture::String;

    const auto initial = TestFixture::make_string(3, 'x');
    const auto expected = TestFixture::make_string(20, 'a');
    String lhs(initial.data(), initial.size());
    const String rhs(expected.data(), expected.size());

    lhs = rhs;

    TestFixture::expect_content(lhs, expected);
    TestFixture::expect_content(rhs, expected);
}

TYPED_TEST(BasicStringSkeletonTest, MoveAssignmentTransfersContent) {
    using String = typename TestFixture::String;

    const auto initial = TestFixture::make_string(3, 'x');
    const auto expected = TestFixture::make_string(20, 'a');
    String lhs(initial.data(), initial.size());
    String rhs(expected.data(), expected.size());

    lhs = std::move(rhs);

    TestFixture::expect_content(lhs, expected);
    ASSERT_NE(rhs.data(), nullptr);
    EXPECT_GE(rhs.capacity(), rhs.size());
    rhs.clear();
    EXPECT_TRUE(rhs.empty());
}

TYPED_TEST(BasicStringSkeletonTest, AssignFromCStringReplacesContent) {
    using String = typename TestFixture::String;

    const auto initial = TestFixture::make_string(8, 'x');
    const auto expected = TestFixture::make_string(5, 'a');
    String string(initial.data(), initial.size());

    string = expected.c_str();

    TestFixture::expect_content(string, expected);
}

TYPED_TEST(BasicStringSkeletonTest, AssignFromStringViewReplacesContent) {
    using String = typename TestFixture::String;
    using CharT = typename TestFixture::CharType;

    const auto initial = TestFixture::make_string(8, 'x');
    const auto expected = TestFixture::make_string(5, 'a');
    const std::basic_string_view<CharT> view(expected.data(), expected.size());
    String string(initial.data(), initial.size());

    string = view;

    TestFixture::expect_content(string, view);
}

TYPED_TEST(BasicStringSkeletonTest, AssignFromCharReplacesWithSingleCharacter) {
    using String = typename TestFixture::String;
    using CharT = typename TestFixture::CharType;

    const auto initial = TestFixture::make_string(8, 'x');
    const CharT ch = static_cast<CharT>('m');
    String string(initial.data(), initial.size());

    string = ch;

    EXPECT_EQ(string.size(), 1U);
    EXPECT_EQ(string.data()[0], ch);
    EXPECT_EQ(string.data()[1], CharT{});
}

TYPED_TEST(BasicStringSkeletonTest, ClearResetsVisibleContent) {
    using String = typename TestFixture::String;
    using CharT = typename TestFixture::CharType;

    const auto expected = TestFixture::make_string(32);
    String string(expected.data(), expected.size());
    const auto old_capacity = string.capacity();

    string.clear();

    EXPECT_EQ(string.size(), 0U);
    EXPECT_TRUE(string.empty());
    EXPECT_EQ(string.capacity(), old_capacity);
    EXPECT_EQ(string.data()[0], CharT{});
}

TYPED_TEST(BasicStringSkeletonTest, ReservePreservesPublicContent) {
    using String = typename TestFixture::String;

    const auto expected = TestFixture::make_string(6);
    String string(expected.data(), expected.size());

    string.reserve(96);

    TestFixture::expect_content(string, expected);
    EXPECT_GE(string.capacity(), 96U);
}

TYPED_TEST(BasicStringSkeletonTest, ResizeShrinksVisibleLength) {
    using String = typename TestFixture::String;
    using CharT = typename TestFixture::CharType;

    const auto initial = TestFixture::make_string(12);
    const std::basic_string_view<CharT> expected(initial.data(), 5);
    String string(initial.data(), initial.size());

    string.resize(expected.size());

    TestFixture::expect_content(string, expected);
}

TYPED_TEST(BasicStringSkeletonTest, ResizeExpandsWithNullCharacter) {
    using String = typename TestFixture::String;
    using CharT = typename TestFixture::CharType;

    const auto initial = TestFixture::make_string(3);
    String string(initial.data(), initial.size());

    string.resize(9);

    EXPECT_EQ(string.size(), 9U);
    for (std::size_t i = 0; i < initial.size(); ++i) {
        EXPECT_EQ(string.data()[i], initial[i]);
    }
    for (std::size_t i = initial.size(); i < string.size(); ++i) {
        EXPECT_EQ(string.data()[i], CharT{});
    }
    EXPECT_EQ(string.data()[string.size()], CharT{});
}

TYPED_TEST(BasicStringSkeletonTest, ResizeExpandsWithFillCharacter) {
    using String = typename TestFixture::String;
    using CharT = typename TestFixture::CharType;

    const auto initial = TestFixture::make_string(3);
    const CharT fill = static_cast<CharT>('r');
    String string(initial.data(), initial.size());

    string.resize(9, fill);

    EXPECT_EQ(string.size(), 9U);
    for (std::size_t i = 0; i < initial.size(); ++i) {
        EXPECT_EQ(string.data()[i], initial[i]);
    }
    for (std::size_t i = initial.size(); i < string.size(); ++i) {
        EXPECT_EQ(string.data()[i], fill);
    }
    EXPECT_EQ(string.data()[string.size()], CharT{});
}

TYPED_TEST(BasicStringSkeletonTest, ShrinkToFitPreservesVisibleContent) {
    using String = typename TestFixture::String;

    const auto expected = TestFixture::make_string(64);
    String string(expected.data(), expected.size());
    string.reserve(256);

    string.shrink_to_fit();

    TestFixture::expect_content(string, expected);
}

TYPED_TEST(BasicStringSkeletonTest, AssignPointerAndLengthReplacesContent) {
    using String = typename TestFixture::String;

    const auto initial = TestFixture::make_string(8, 'x');
    const auto expected = TestFixture::make_string(5, 'a');
    String string(initial.data(), initial.size());

    String& result = string.assign(expected.data(), expected.size());

    EXPECT_EQ(&result, &string);
    TestFixture::expect_content(string, expected);
}

TYPED_TEST(BasicStringSkeletonTest, AssignStringViewReplacesContent) {
    using String = typename TestFixture::String;
    using CharT = typename TestFixture::CharType;

    const auto initial = TestFixture::make_string(8, 'x');
    const auto expected = TestFixture::make_string(5, 'a');
    const std::basic_string_view<CharT> view(expected.data(), expected.size());
    String string(initial.data(), initial.size());

    String& result = string.assign(view);

    EXPECT_EQ(&result, &string);
    TestFixture::expect_content(string, view);
}

TYPED_TEST(BasicStringSkeletonTest, AssignCountAndCharReplacesContent) {
    using String = typename TestFixture::String;
    using CharT = typename TestFixture::CharType;

    const auto initial = TestFixture::make_string(8, 'x');
    const CharT fill = static_cast<CharT>('s');
    String string(initial.data(), initial.size());

    String& result = string.assign(6, fill);

    EXPECT_EQ(&result, &string);
    EXPECT_EQ(string.size(), 6U);
    for (std::size_t i = 0; i < string.size(); ++i) {
        EXPECT_EQ(string.data()[i], fill);
    }
    EXPECT_EQ(string.data()[string.size()], CharT{});
}

TYPED_TEST(BasicStringSkeletonTest, AppendBasicStringAppendsContent) {
    using String = typename TestFixture::String;

    const auto lhs_text = TestFixture::make_string(4, 'a');
    const auto rhs_text = TestFixture::make_string(5, 'k');
    const auto expected = lhs_text + rhs_text;
    String lhs(lhs_text.data(), lhs_text.size());
    const String rhs(rhs_text.data(), rhs_text.size());

    String& result = lhs.append(rhs);

    EXPECT_EQ(&result, &lhs);
    TestFixture::expect_content(lhs, expected);
}

TYPED_TEST(BasicStringSkeletonTest, AppendPointerAndLengthAppendsContent) {
    using String = typename TestFixture::String;

    const auto lhs_text = TestFixture::make_string(4, 'a');
    const auto rhs_text = TestFixture::make_string(5, 'k');
    const auto expected = lhs_text + rhs_text;
    String lhs(lhs_text.data(), lhs_text.size());

    String& result = lhs.append(rhs_text.data(), rhs_text.size());

    EXPECT_EQ(&result, &lhs);
    TestFixture::expect_content(lhs, expected);
}

TYPED_TEST(BasicStringSkeletonTest, AppendCStringAppendsContent) {
    using String = typename TestFixture::String;

    const auto lhs_text = TestFixture::make_string(4, 'a');
    const auto rhs_text = TestFixture::make_string(5, 'k');
    const auto expected = lhs_text + rhs_text;
    String lhs(lhs_text.data(), lhs_text.size());

    String& result = lhs.append(rhs_text.c_str());

    EXPECT_EQ(&result, &lhs);
    TestFixture::expect_content(lhs, expected);
}

TYPED_TEST(BasicStringSkeletonTest, AppendStringViewAppendsContent) {
    using String = typename TestFixture::String;
    using CharT = typename TestFixture::CharType;

    const auto lhs_text = TestFixture::make_string(4, 'a');
    const auto rhs_text = TestFixture::make_string(5, 'k');
    const auto expected = lhs_text + rhs_text;
    const std::basic_string_view<CharT> view(rhs_text.data(), rhs_text.size());
    String lhs(lhs_text.data(), lhs_text.size());

    String& result = lhs.append(view);

    EXPECT_EQ(&result, &lhs);
    TestFixture::expect_content(lhs, expected);
}

TYPED_TEST(BasicStringSkeletonTest, AppendCountAndCharAppendsFillCharacters) {
    using String = typename TestFixture::String;
    using CharT = typename TestFixture::CharType;

    const auto prefix = TestFixture::make_string(4, 'a');
    const CharT fill = static_cast<CharT>('z');
    String string(prefix.data(), prefix.size());

    String& result = string.append(5, fill);

    EXPECT_EQ(&result, &string);
    EXPECT_EQ(string.size(), prefix.size() + 5);
    for (std::size_t i = 0; i < prefix.size(); ++i) {
        EXPECT_EQ(string.data()[i], prefix[i]);
    }
    for (std::size_t i = prefix.size(); i < string.size(); ++i) {
        EXPECT_EQ(string.data()[i], fill);
    }
    EXPECT_EQ(string.data()[string.size()], CharT{});
}

TYPED_TEST(BasicStringSkeletonTest, AppendOperationsSupportChaining) {
    using String = typename TestFixture::String;
    using CharT = typename TestFixture::CharType;

    const auto first = TestFixture::make_string(2, 'a');
    const auto second = TestFixture::make_string(2, 'm');
    const CharT last = static_cast<CharT>('z');
    const auto expected = first + second + std::basic_string<CharT>(3, last);
    String string(first.data(), first.size());

    String& result = string.append(second.data(), second.size()).append(3, last);

    EXPECT_EQ(&result, &string);
    TestFixture::expect_content(string, expected);
}

TYPED_TEST(BasicStringSkeletonTest, PlusEqualAppendsContent) {
    using String = typename TestFixture::String;
    using CharT = typename TestFixture::CharType;

    const auto first = TestFixture::make_string(2, 'a');
    const auto second = TestFixture::make_string(3, 'm');
    const CharT ch = static_cast<CharT>('z');
    const auto expected = first + second + std::basic_string<CharT>(1, ch);
    String string(first.data(), first.size());
    const String suffix(second.data(), second.size());

    string += suffix;
    string += ch;

    TestFixture::expect_content(string, expected);
}

TYPED_TEST(BasicStringSkeletonTest, PushBackAppendsSingleCharacter) {
    using String = typename TestFixture::String;
    using CharT = typename TestFixture::CharType;

    const auto prefix = TestFixture::make_string(4, 'a');
    const CharT ch = static_cast<CharT>('p');
    String string(prefix.data(), prefix.size());

    string.push_back(ch);

    EXPECT_EQ(string.size(), prefix.size() + 1);
    for (std::size_t i = 0; i < prefix.size(); ++i) {
        EXPECT_EQ(string.data()[i], prefix[i]);
    }
    EXPECT_EQ(string.data()[prefix.size()], ch);
    EXPECT_EQ(string.data()[string.size()], CharT{});
}

TYPED_TEST(BasicStringSkeletonTest, PopBackRemovesLastCharacter) {
    using String = typename TestFixture::String;
    using CharT = typename TestFixture::CharType;

    const auto initial = TestFixture::make_string(6, 'a');
    const std::basic_string_view<CharT> expected(initial.data(), initial.size() - 1);
    String string(initial.data(), initial.size());

    string.pop_back();

    TestFixture::expect_content(string, expected);
}

TYPED_TEST(BasicStringSkeletonTest, PopBackOnEmptyIsNoop) {
    using String = typename TestFixture::String;
    using CharT = typename TestFixture::CharType;

    String string;

    string.pop_back();

    EXPECT_EQ(string.size(), 0U);
    EXPECT_TRUE(string.empty());
    EXPECT_EQ(string.data()[0], CharT{});
}

TYPED_TEST(BasicStringSkeletonTest, SwapExchangesVisibleContent) {
    using String = typename TestFixture::String;

    const auto lhs_text = TestFixture::make_string(4, 'a');
    const auto rhs_text = TestFixture::make_string(64, 'k');
    String lhs(lhs_text.data(), lhs_text.size());
    String rhs(rhs_text.data(), rhs_text.size());

    lhs.swap(rhs);

    TestFixture::expect_content(lhs, rhs_text);
    TestFixture::expect_content(rhs, lhs_text);
}

TYPED_TEST(BasicStringSkeletonTest, GetAllocatorReturnsAllocator) {
    using CharT = TypeParam;
    using Allocator = StatefulAllocator<CharT>;
    using String = aethermind::BasicString<CharT, std::char_traits<CharT>, Allocator>;

    const Allocator allocator(42);
    String string(allocator);

    EXPECT_EQ(string.get_allocator().id, allocator.id);
}

TYPED_TEST(BasicStringSkeletonTest, SwapNoexceptFollowsCore) {
    using CharT = TypeParam;
    using String = aethermind::BasicString<CharT>;

    static_assert(noexcept(std::declval<String&>().swap(std::declval<String&>())));
}

TYPED_TEST(BasicStringSkeletonTest, SmallStringIsTransparentToUser) {
    using String = typename TestFixture::String;

    const auto expected = TestFixture::make_string(3, 'a');
    String string(expected.data(), expected.size());

    string.push_back(static_cast<typename TestFixture::CharType>('z'));
    string.pop_back();

    TestFixture::expect_content(string, expected);
}

TYPED_TEST(BasicStringSkeletonTest, ExternalStringIsTransparentToUser) {
    using String = typename TestFixture::String;

    const auto expected = TestFixture::make_string(128, 'a');
    String string(expected.data(), expected.size());

    string.push_back(static_cast<typename TestFixture::CharType>('z'));
    string.pop_back();

    TestFixture::expect_content(string, expected);
}

TYPED_TEST(BasicStringSkeletonTest, SamePublicOperationBehavesSameAcrossStorageSizes) {
    using String = typename TestFixture::String;
    using CharT = typename TestFixture::CharType;

    const auto small_text = TestFixture::make_string(3, 'a');
    const auto external_text = TestFixture::make_string(128, 'a');
    const CharT fill = static_cast<CharT>('x');
    String small(small_text.data(), small_text.size());
    String external(external_text.data(), external_text.size());

    small.append(2, fill).resize(4);
    external.append(2, fill).resize(129);

    EXPECT_EQ(small.size(), 4U);
    EXPECT_EQ(small.data()[3], fill);
    EXPECT_EQ(small.data()[small.size()], CharT{});
    EXPECT_EQ(external.size(), 129U);
    EXPECT_EQ(external.data()[128], fill);
    EXPECT_EQ(external.data()[external.size()], CharT{});
}

}// namespace
