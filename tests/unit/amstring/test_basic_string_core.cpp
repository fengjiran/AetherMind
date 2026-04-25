#include <gtest/gtest.h>

#include "amstring/core.hpp"
#include "amstring/generic_layout_policy.hpp"

#include <array>
#include <cstddef>
#include <cstring>
#include <memory>
#include <string>

namespace {

using namespace aethermind;

template<typename CharT>
class BasicStringCoreLifecycleTest : public ::testing::Test {
protected:
    using Core = BasicStringCore<CharT>;
    using Policy = GenericLayoutPolicy<CharT>;
    using CharType = CharT;

    static std::basic_string<CharT> make_test_string(std::size_t len) {
        std::basic_string<CharT> s;
        for (std::size_t i = 0; i < len; ++i) {
            s.push_back(static_cast<CharT>('a' + (i % 26)));
        }
        return s;
    }

    static CharT* allocate_buffer(std::size_t capacity) {
        return new CharT[capacity + 1];
    }

    static void deallocate_buffer(CharT* ptr, std::size_t capacity) {
        delete[] ptr;
    }
};

using AmstringCoreCharTypes = ::testing::Types<char, char8_t, char16_t, char32_t, wchar_t>;
TYPED_TEST_SUITE(BasicStringCoreLifecycleTest, AmstringCoreCharTypes);

TYPED_TEST(BasicStringCoreLifecycleTest, DefaultConstructCreatesEmptySmallString) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;
    using CharT = typename TestFixture::CharType;

    Core core;

    EXPECT_TRUE(core.empty());
    EXPECT_EQ(core.size(), 0u);
    EXPECT_EQ(core.capacity(), Policy::kSmallCapacity);
    EXPECT_NE(core.data(), nullptr);
    EXPECT_EQ(core.data()[0], CharT{});
    EXPECT_TRUE(core.is_small());
}

TYPED_TEST(BasicStringCoreLifecycleTest, ConstructSmallStringFromPointerAndLength) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;
    using CharT = typename TestFixture::CharType;

    constexpr std::size_t kLen = Policy::kSmallCapacity > 3 ? 3 : 1;
    const auto test_str = TestFixture::make_test_string(kLen);

    Core core(test_str.data(), kLen);

    EXPECT_FALSE(core.empty());
    EXPECT_EQ(core.size(), kLen);
    EXPECT_EQ(core.capacity(), Policy::kSmallCapacity);
    EXPECT_TRUE(core.is_small());

    for (std::size_t i = 0; i < kLen; ++i) {
        EXPECT_EQ(core.data()[i], test_str[i]);
    }
    EXPECT_EQ(core.data()[kLen], CharT{});
}

TYPED_TEST(BasicStringCoreLifecycleTest, ConstructExternalStringFromPointerAndLength) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;
    using CharT = typename TestFixture::CharType;

    constexpr std::size_t kLen = Policy::kSmallCapacity + 10;
    const auto test_str = TestFixture::make_test_string(kLen);

    Core core(test_str.data(), kLen);

    EXPECT_FALSE(core.empty());
    EXPECT_EQ(core.size(), kLen);
    EXPECT_GE(core.capacity(), kLen);
    EXPECT_TRUE(core.is_external());

    for (std::size_t i = 0; i < kLen; ++i) {
        EXPECT_EQ(core.data()[i], test_str[i]);
    }
    EXPECT_EQ(core.data()[kLen], CharT{});
}

TYPED_TEST(BasicStringCoreLifecycleTest, CopySmallString) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;

    constexpr std::size_t kLen = Policy::kSmallCapacity > 3 ? 3 : 1;
    const auto test_str = TestFixture::make_test_string(kLen);

    Core original(test_str.data(), kLen);
    Core copy(original);

    EXPECT_EQ(copy.size(), original.size());
    EXPECT_EQ(copy.capacity(), original.capacity());
    EXPECT_TRUE(copy.is_small());

    for (std::size_t i = 0; i < kLen; ++i) {
        EXPECT_EQ(copy.data()[i], original.data()[i]);
    }

    EXPECT_NE(copy.data(), original.data());
}

TYPED_TEST(BasicStringCoreLifecycleTest, CopyExternalStringDeepCopy) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;

    constexpr std::size_t kLen = Policy::kSmallCapacity + 10;
    const auto test_str = TestFixture::make_test_string(kLen);

    Core original(test_str.data(), kLen);
    Core copy(original);

    EXPECT_EQ(copy.size(), original.size());
    EXPECT_EQ(copy.capacity(), original.capacity());
    EXPECT_TRUE(copy.is_external());

    for (std::size_t i = 0; i < kLen; ++i) {
        EXPECT_EQ(copy.data()[i], original.data()[i]);
    }

    EXPECT_NE(copy.data(), original.data());
}

TYPED_TEST(BasicStringCoreLifecycleTest, MoveSmallStringSourceRestoresToEmpty) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;
    using CharT = typename TestFixture::CharType;

    constexpr std::size_t kLen = Policy::kSmallCapacity > 3 ? 3 : 1;
    const auto test_str = TestFixture::make_test_string(kLen);

    Core original(test_str.data(), kLen);
    Core moved(std::move(original));

    EXPECT_EQ(moved.size(), kLen);
    EXPECT_TRUE(moved.is_small());
    for (std::size_t i = 0; i < kLen; ++i) {
        EXPECT_EQ(moved.data()[i], test_str[i]);
    }

    EXPECT_TRUE(original.empty());
    EXPECT_EQ(original.size(), 0u);
    EXPECT_NE(original.data(), nullptr);
    EXPECT_EQ(original.data()[0], CharT{});
}

TYPED_TEST(BasicStringCoreLifecycleTest, MoveExternalStringSourceRestoresToEmpty) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;
    using CharT = typename TestFixture::CharType;

    constexpr std::size_t kLen = Policy::kSmallCapacity + 10;
    const auto test_str = TestFixture::make_test_string(kLen);

    Core original(test_str.data(), kLen);
    const CharT* original_ptr = original.data();

    Core moved(std::move(original));

    EXPECT_EQ(moved.size(), kLen);
    EXPECT_TRUE(moved.is_external());
    EXPECT_EQ(moved.data(), original_ptr);

    EXPECT_TRUE(original.empty());
    EXPECT_EQ(original.size(), 0u);
    EXPECT_TRUE(original.is_small());
    EXPECT_NE(original.data(), nullptr);
    EXPECT_EQ(original.data()[0], CharT{});
}

TYPED_TEST(BasicStringCoreLifecycleTest, DestructorReleasesExternalBuffer) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;
    using CharT = typename TestFixture::CharType;

    constexpr std::size_t kLen = Policy::kSmallCapacity + 10;
    const auto test_str = TestFixture::make_test_string(kLen);

    CharT* allocated_ptr = nullptr;
    std::size_t allocated_capacity = 0;

    {
        Core core(test_str.data(), kLen);
        allocated_ptr = core.data();
        allocated_capacity = core.capacity();
        EXPECT_TRUE(core.is_external());
        EXPECT_NE(allocated_ptr, nullptr);
    }

    CharT* new_ptr = TestFixture::allocate_buffer(allocated_capacity);
    EXPECT_NE(new_ptr, nullptr);
    TestFixture::deallocate_buffer(new_ptr, allocated_capacity);
}

TYPED_TEST(BasicStringCoreLifecycleTest, ReadOnlyInterfacesWorkCorrectly) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;

    constexpr std::size_t kLen = Policy::kSmallCapacity > 5 ? 5 : 1;
    const auto test_str = TestFixture::make_test_string(kLen);

    Core core(test_str.data(), kLen);

    EXPECT_NE(core.data(), nullptr);
    for (std::size_t i = 0; i < kLen; ++i) {
        EXPECT_EQ(core.data()[i], test_str[i]);
    }

    EXPECT_EQ(core.c_str(), core.data());
    EXPECT_EQ(core.size(), kLen);
    EXPECT_GE(core.capacity(), kLen);
    EXPECT_FALSE(core.empty());

    Core empty_core;
    EXPECT_TRUE(empty_core.empty());
    EXPECT_EQ(empty_core.size(), 0u);
    EXPECT_NE(empty_core.data(), nullptr);
    EXPECT_EQ(empty_core.c_str(), empty_core.data());
}

TYPED_TEST(BasicStringCoreLifecycleTest, SelectOnContainerCopyConstruction) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;
    using CharT = typename TestFixture::CharType;

    constexpr std::size_t kLen = Policy::kSmallCapacity + 10;
    const auto test_str = TestFixture::make_test_string(kLen);

    std::allocator<CharT> alloc;
    Core original(test_str.data(), kLen, alloc);

    Core copy(original);

    EXPECT_EQ(copy.size(), original.size());
    EXPECT_TRUE(copy.is_external());
    EXPECT_NE(copy.data(), original.data());
}

TYPED_TEST(BasicStringCoreLifecycleTest, ExternalBufferAllocatesCapacityPlusOne) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;
    using CharT = typename TestFixture::CharType;

    constexpr std::size_t kLen = Policy::kSmallCapacity + 10;
    const auto test_str = TestFixture::make_test_string(kLen);

    Core core(test_str.data(), kLen);

    std::size_t cap = core.capacity();
    EXPECT_GE(cap, kLen);

    EXPECT_EQ(core.data()[kLen], CharT{});

    CharT* ptr = core.data();
    EXPECT_EQ(ptr[kLen], CharT{});
}

}// namespace
