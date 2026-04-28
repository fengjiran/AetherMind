#include "amstring/config.hpp"
#include "amstring/core.hpp"

#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace {

using namespace aethermind;

template<typename T>
struct StatefulAllocator {
    using value_type = T;

    int id = 0;

    StatefulAllocator() noexcept = default;

    explicit StatefulAllocator(int allocator_id) noexcept : id(allocator_id) {}

    template<typename U>
    StatefulAllocator(const StatefulAllocator<U>& other) noexcept : id(other.id) {}

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

template<typename T>
struct ThrowingSwapAllocator : StatefulAllocator<T> {
    using value_type = T;
    using propagate_on_container_swap = std::true_type;

    ThrowingSwapAllocator() noexcept = default;

    explicit ThrowingSwapAllocator(int allocator_id) noexcept : StatefulAllocator<T>(allocator_id) {}
};

template<typename T>
void swap(ThrowingSwapAllocator<T>& lhs, ThrowingSwapAllocator<T>& rhs) noexcept(false) {
    using std::swap;
    swap(lhs.id, rhs.id);
}

struct AllocationBudget {
    std::size_t remaining_allocations = std::numeric_limits<std::size_t>::max();
};

template<typename T>
struct FailingAllocator {
    using value_type = T;

    std::shared_ptr<AllocationBudget> budget = std::make_shared<AllocationBudget>();

    FailingAllocator() = default;

    explicit FailingAllocator(std::shared_ptr<AllocationBudget> allocation_budget) : budget(std::move(allocation_budget)) {}

    template<typename U>
    FailingAllocator(const FailingAllocator<U>& other) noexcept : budget(other.budget) {}

    T* allocate(std::size_t n) {
        if (budget && budget->remaining_allocations == 0) {
            throw std::bad_alloc();
        }
        if (budget) {
            --budget->remaining_allocations;
        }
        return std::allocator<T>{}.allocate(n);
    }

    void deallocate(T* ptr, std::size_t n) noexcept {
        std::allocator<T>{}.deallocate(ptr, n);
    }

    template<typename U>
    bool operator==(const FailingAllocator<U>& other) const noexcept {
        return budget == other.budget;
    }

    template<typename U>
    bool operator!=(const FailingAllocator<U>& other) const noexcept {
        return !(*this == other);
    }
};

template<typename CharT>
class BasicStringCoreLifecycleTest : public ::testing::Test {
protected:
    using Core = BasicStringCore<CharT>;
    using Policy = typename Core::LayoutPolicyType;
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

TYPED_TEST(BasicStringCoreLifecycleTest, GetAllocatorReturnsStoredAllocator) {
    using CharT = typename TestFixture::CharType;
    using Allocator = StatefulAllocator<CharT>;
    using Core = BasicStringCore<CharT, std::char_traits<CharT>, Allocator>;

    const Allocator allocator(42);
    Core core(allocator);

    EXPECT_EQ(core.get_allocator().id, allocator.id);
}

TYPED_TEST(BasicStringCoreLifecycleTest, SwapNoexceptReflectsAllocatorSwap) {
    using CharT = typename TestFixture::CharType;
    using DefaultCore = BasicStringCore<CharT>;
    using ThrowingSwapCore = BasicStringCore<CharT, std::char_traits<CharT>, ThrowingSwapAllocator<CharT>>;

    static_assert(noexcept(std::declval<DefaultCore&>().swap(std::declval<DefaultCore&>())));
    static_assert(!noexcept(std::declval<ThrowingSwapCore&>().swap(std::declval<ThrowingSwapCore&>())));
}

TYPED_TEST(BasicStringCoreLifecycleTest, ConstructorRejectsCapacityBeyondLayoutLimit) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;

    static_assert(Policy::max_external_capacity() < std::numeric_limits<std::size_t>::max());
    const std::size_t too_large = Policy::max_external_capacity() + 1;

    EXPECT_THROW((void) Core(nullptr, too_large), aethermind::Error);
}

TYPED_TEST(BasicStringCoreLifecycleTest, ReserveRejectsCapacityBeyondLayoutLimit) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;

    static_assert(Policy::max_external_capacity() < std::numeric_limits<std::size_t>::max());
    Core core;

    EXPECT_THROW(core.reserve(Policy::max_external_capacity() + 1), aethermind::Error);
}

TYPED_TEST(BasicStringCoreLifecycleTest, AppendRejectsSizeOverflow) {
    using Core = typename TestFixture::Core;
    using CharT = typename TestFixture::CharType;

    const CharT ch = static_cast<CharT>('x');
    Core core(&ch, 1);

    EXPECT_THROW(core.append(core.data(), std::numeric_limits<std::size_t>::max()), aethermind::Error);
    EXPECT_EQ(core.size(), 1U);
    EXPECT_EQ(core.data()[0], ch);
    EXPECT_EQ(core.data()[1], CharT{});
}

TYPED_TEST(BasicStringCoreLifecycleTest, ClearKeepsExternalCapacityAndTerminator) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;
    using CharT = typename TestFixture::CharType;

    constexpr std::size_t kLen = Policy::kSmallCapacity + 3;
    const auto test_str = TestFixture::make_test_string(kLen);
    Core core(test_str.data(), test_str.size());
    const auto original_capacity = core.capacity();
    const auto* original_data = core.data();

    core.clear();

    EXPECT_TRUE(core.empty());
    EXPECT_TRUE(core.is_external());
    EXPECT_EQ(core.capacity(), original_capacity);
    EXPECT_EQ(core.data(), original_data);
    EXPECT_EQ(core.data()[0], CharT{});
}

TYPED_TEST(BasicStringCoreLifecycleTest, ReserveGrowsAndPreservesContent) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;
    using CharT = typename TestFixture::CharType;

    constexpr std::size_t kLen = Policy::kSmallCapacity > 3 ? 3 : 1;
    constexpr std::size_t kRequestedCapacity = Policy::kSmallCapacity + 8;
    const auto test_str = TestFixture::make_test_string(kLen);
    Core core(test_str.data(), test_str.size());

    core.reserve(kRequestedCapacity);

    EXPECT_EQ(core.size(), kLen);
    EXPECT_TRUE(core.is_external());
    EXPECT_GE(core.capacity(), kRequestedCapacity);
    for (std::size_t i = 0; i < kLen; ++i) {
        EXPECT_EQ(core.data()[i], test_str[i]);
    }
    EXPECT_EQ(core.data()[kLen], CharT{});
}

TYPED_TEST(BasicStringCoreLifecycleTest, ResizeTruncatesAndMaintainsTerminator) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;
    using CharT = typename TestFixture::CharType;

    constexpr std::size_t kInitialLen = Policy::kSmallCapacity + 3;
    constexpr std::size_t kNewLen = Policy::kSmallCapacity > 3 ? 3 : 1;
    const auto test_str = TestFixture::make_test_string(kInitialLen);
    Core core(test_str.data(), test_str.size());
    const auto original_capacity = core.capacity();

    core.resize(kNewLen);

    EXPECT_EQ(core.size(), kNewLen);
    EXPECT_EQ(core.capacity(), original_capacity);
    for (std::size_t i = 0; i < kNewLen; ++i) {
        EXPECT_EQ(core.data()[i], test_str[i]);
    }
    EXPECT_EQ(core.data()[kNewLen], CharT{});
}

TYPED_TEST(BasicStringCoreLifecycleTest, ResizeExpandsAndFillsWithCharacter) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;
    using CharT = typename TestFixture::CharType;

    constexpr std::size_t kInitialLen = Policy::kSmallCapacity > 3 ? 3 : 1;
    constexpr std::size_t kNewLen = Policy::kSmallCapacity + 2;
    const auto test_str = TestFixture::make_test_string(kInitialLen);
    const CharT fill = static_cast<CharT>('r');
    Core core(test_str.data(), test_str.size());

    core.resize(kNewLen, fill);

    EXPECT_EQ(core.size(), kNewLen);
    EXPECT_TRUE(core.is_external());
    for (std::size_t i = 0; i < kInitialLen; ++i) {
        EXPECT_EQ(core.data()[i], test_str[i]);
    }
    for (std::size_t i = kInitialLen; i < kNewLen; ++i) {
        EXPECT_EQ(core.data()[i], fill);
    }
    EXPECT_EQ(core.data()[kNewLen], CharT{});
}

TYPED_TEST(BasicStringCoreLifecycleTest, AssignPointerAndLengthReplacesContent) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;
    using CharT = typename TestFixture::CharType;

    constexpr std::size_t kInitialLen = Policy::kSmallCapacity + 5;
    constexpr std::size_t kReplacementLen = Policy::kSmallCapacity > 3 ? 3 : 1;
    const auto initial = TestFixture::make_test_string(kInitialLen);
    const auto replacement = TestFixture::make_test_string(kReplacementLen);
    Core core(initial.data(), initial.size());

    core.assign(replacement.data(), replacement.size());

    EXPECT_EQ(core.size(), replacement.size());
    EXPECT_TRUE(core.is_small());
    for (std::size_t i = 0; i < replacement.size(); ++i) {
        EXPECT_EQ(core.data()[i], replacement[i]);
    }
    EXPECT_EQ(core.data()[replacement.size()], CharT{});
}

TYPED_TEST(BasicStringCoreLifecycleTest, AppendPointerAndLengthAppendsCharacters) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;
    using CharT = typename TestFixture::CharType;

    constexpr std::size_t kInitialLen = Policy::kSmallCapacity > 3 ? 3 : 1;
    constexpr std::size_t kSuffixLen = 2;
    const auto initial = TestFixture::make_test_string(kInitialLen);
    const auto suffix = TestFixture::make_test_string(kSuffixLen);
    const std::basic_string<CharT> expected = initial + suffix;
    Core core(initial.data(), initial.size());

    core.append(suffix.data(), suffix.size());

    EXPECT_EQ(core.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(core.data()[i], expected[i]);
    }
    EXPECT_EQ(core.data()[expected.size()], CharT{});
}

TYPED_TEST(BasicStringCoreLifecycleTest, ShrinkToFitMovesSmallContentBackInline) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;
    using CharT = typename TestFixture::CharType;

    constexpr std::size_t kInitialLen = Policy::kSmallCapacity + 8;
    constexpr std::size_t kShrunkLen = Policy::kSmallCapacity > 3 ? 3 : 1;
    const auto test_str = TestFixture::make_test_string(kInitialLen);
    Core core(test_str.data(), test_str.size());

    core.resize(kShrunkLen);
    core.shrink_to_fit();

    EXPECT_EQ(core.size(), kShrunkLen);
    EXPECT_TRUE(core.is_small());
    EXPECT_EQ(core.capacity(), Policy::kSmallCapacity);
    for (std::size_t i = 0; i < kShrunkLen; ++i) {
        EXPECT_EQ(core.data()[i], test_str[i]);
    }
    EXPECT_EQ(core.data()[kShrunkLen], CharT{});
}

TYPED_TEST(BasicStringCoreLifecycleTest, ShrinkToFitReducesExternalCapacity) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;
    using CharT = typename TestFixture::CharType;

    constexpr std::size_t kLen = Policy::kSmallCapacity + 4;
    constexpr std::size_t kReservedCapacity = kLen + 20;
    const auto test_str = TestFixture::make_test_string(kLen);
    Core core(test_str.data(), test_str.size());
    core.reserve(kReservedCapacity);

    core.shrink_to_fit();

    EXPECT_EQ(core.size(), kLen);
    EXPECT_TRUE(core.is_external());
    EXPECT_EQ(core.capacity(), kLen);
    for (std::size_t i = 0; i < kLen; ++i) {
        EXPECT_EQ(core.data()[i], test_str[i]);
    }
    EXPECT_EQ(core.data()[kLen], CharT{});
}

TYPED_TEST(BasicStringCoreLifecycleTest, ShrinkToFitUsesPageRoundedCapacityForLargeExternal) {
    using Core = typename TestFixture::Core;
    using CharT = typename TestFixture::CharType;

    static_assert(config::kPageSizeBytes % sizeof(CharT) == 0);
    constexpr std::size_t kLargeSize = (config::kPageSizeBytes * 2) / sizeof(CharT);
    constexpr std::size_t kReservedCapacity = (config::kPageSizeBytes * 6) / sizeof(CharT) - 1;
    const CharT fill = static_cast<CharT>('s');
    Core core;
    core.append(kLargeSize, fill);
    core.reserve(kReservedCapacity);
    ASSERT_EQ((core.capacity() + 1) * sizeof(CharT), config::kPageSizeBytes * 6);

    core.shrink_to_fit();

    const std::size_t allocation_bytes = (core.capacity() + 1) * sizeof(CharT);
    EXPECT_EQ(core.size(), kLargeSize);
    EXPECT_TRUE(core.is_external());
    EXPECT_GE(core.capacity(), core.size());
    EXPECT_EQ(allocation_bytes, config::kPageSizeBytes * 3);
    EXPECT_EQ(core.data()[core.size()], CharT{});
}

TYPED_TEST(BasicStringCoreLifecycleTest, SwapExchangesContentAndStorageCategory) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;
    using CharT = typename TestFixture::CharType;

    constexpr std::size_t kSmallLen = Policy::kSmallCapacity > 3 ? 3 : 1;
    constexpr std::size_t kExternalLen = Policy::kSmallCapacity + 4;
    const auto small = TestFixture::make_test_string(kSmallLen);
    const auto external = TestFixture::make_test_string(kExternalLen);
    Core lhs(small.data(), small.size());
    Core rhs(external.data(), external.size());

    lhs.swap(rhs);

    EXPECT_EQ(lhs.size(), external.size());
    EXPECT_TRUE(lhs.is_external());
    for (std::size_t i = 0; i < external.size(); ++i) {
        EXPECT_EQ(lhs.data()[i], external[i]);
    }
    EXPECT_EQ(lhs.data()[external.size()], CharT{});

    EXPECT_EQ(rhs.size(), small.size());
    EXPECT_TRUE(rhs.is_small());
    for (std::size_t i = 0; i < small.size(); ++i) {
        EXPECT_EQ(rhs.data()[i], small[i]);
    }
    EXPECT_EQ(rhs.data()[small.size()], CharT{});
}

TYPED_TEST(BasicStringCoreLifecycleTest, StateTransitionMatrixCoversMutatingOperations) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;
    using CharT = typename TestFixture::CharType;

    const CharT marker = static_cast<CharT>('m');

    {
        Core core;
        core.push_back(marker);

        EXPECT_EQ(core.size(), 1U);
        EXPECT_TRUE(core.is_small());
        EXPECT_EQ(core.data()[0], marker);
        EXPECT_EQ(core.data()[1], CharT{});
    }

    {
        Core core;
        core.resize(Policy::kSmallCapacity + 1, marker);

        EXPECT_EQ(core.size(), Policy::kSmallCapacity + 1);
        EXPECT_TRUE(core.is_external());
        EXPECT_EQ(core.data()[core.size()], CharT{});
    }

    {
        constexpr std::size_t kInitialLen = Policy::kSmallCapacity + 2;
        const auto initial = TestFixture::make_test_string(kInitialLen);
        Core core(initial.data(), initial.size());
        core.append(2, marker);

        EXPECT_EQ(core.size(), kInitialLen + 2);
        EXPECT_TRUE(core.is_external());
        EXPECT_EQ(core.data()[kInitialLen], marker);
        EXPECT_EQ(core.data()[kInitialLen + 1], marker);
        EXPECT_EQ(core.data()[core.size()], CharT{});
    }

    {
        constexpr std::size_t kInitialLen = Policy::kSmallCapacity + 2;
        const auto initial = TestFixture::make_test_string(kInitialLen);
        Core core(initial.data(), initial.size());
        core.assign(1, marker);

        EXPECT_EQ(core.size(), 1U);
        EXPECT_TRUE(core.is_small());
        EXPECT_EQ(core.data()[0], marker);
        EXPECT_EQ(core.data()[1], CharT{});
    }
}

TYPED_TEST(BasicStringCoreLifecycleTest, LargeAppendRoundsCapacityToPageAndRemainsExternal) {
    using Core = typename TestFixture::Core;
    using CharT = typename TestFixture::CharType;

    static_assert(config::kPageSizeBytes % sizeof(CharT) == 0);
    constexpr std::size_t kLargeAppendCount = config::kPageSizeBytes / sizeof(CharT);
    const CharT fill = static_cast<CharT>('l');
    Core core;

    core.append(kLargeAppendCount, fill);

    const std::size_t allocation_bytes = (core.capacity() + 1) * sizeof(CharT);
    EXPECT_EQ(core.size(), kLargeAppendCount);
    EXPECT_TRUE(core.is_external());
    EXPECT_GE(core.capacity(), kLargeAppendCount);
    EXPECT_EQ(allocation_bytes % config::kPageSizeBytes, 0U);
    EXPECT_EQ(core.data()[core.size()], CharT{});
}

TYPED_TEST(BasicStringCoreLifecycleTest, LargeGrowthUsesQuarterIncrementBeforePageRounding) {
    using Core = typename TestFixture::Core;
    using CharT = typename TestFixture::CharType;

    static_assert(config::kPageSizeBytes % sizeof(CharT) == 0);
    constexpr std::size_t kInitialSize = (config::kPageSizeBytes * 2) / sizeof(CharT);
    const CharT fill = static_cast<CharT>('g');
    Core core;
    core.append(kInitialSize, fill);
    const std::size_t initial_allocation_bytes = (core.capacity() + 1) * sizeof(CharT);
    ASSERT_EQ(initial_allocation_bytes, config::kPageSizeBytes * 3);

    core.append(core.capacity() - core.size() + 1, fill);

    const std::size_t grown_allocation_bytes = (core.capacity() + 1) * sizeof(CharT);
    EXPECT_TRUE(core.is_external());
    EXPECT_EQ(grown_allocation_bytes, config::kPageSizeBytes * 4);
    EXPECT_EQ(core.data()[core.size()], CharT{});
}

TYPED_TEST(BasicStringCoreLifecycleTest, AssignPreservesOriginalWhenAllocationFails) {
    using Policy = typename TestFixture::Policy;
    using CharT = typename TestFixture::CharType;
    using Allocator = FailingAllocator<CharT>;
    using Core = BasicStringCore<CharT, std::char_traits<CharT>, Allocator>;

    constexpr std::size_t kInitialLen = Policy::kSmallCapacity + 4;
    constexpr std::size_t kReplacementLen = Policy::kSmallCapacity + 6;
    const auto initial = TestFixture::make_test_string(kInitialLen);
    const auto replacement = TestFixture::make_test_string(kReplacementLen);
    auto budget = std::make_shared<AllocationBudget>();
    budget->remaining_allocations = 1;
    Core core(initial.data(), initial.size(), Allocator(budget));

    budget->remaining_allocations = 0;

    EXPECT_THROW(core.assign(replacement.data(), replacement.size()), std::bad_alloc);
    EXPECT_EQ(core.size(), initial.size());
    EXPECT_TRUE(core.is_external());
    for (std::size_t i = 0; i < initial.size(); ++i) {
        EXPECT_EQ(core.data()[i], initial[i]);
    }
    EXPECT_EQ(core.data()[initial.size()], CharT{});
}

TYPED_TEST(BasicStringCoreLifecycleTest, AssignSelfExternalSubrangeUsesSourceBeforeRelease) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;
    using CharT = typename TestFixture::CharType;

    constexpr std::size_t kLen = Policy::kSmallCapacity + 10;
    const auto test_str = TestFixture::make_test_string(kLen);
    Core core(test_str.data(), kLen);

    constexpr std::size_t kOffset = 2;
    constexpr std::size_t kAssignedLen = Policy::kSmallCapacity > 4 ? 4 : 1;
    const std::basic_string<CharT> expected(test_str.data() + kOffset, kAssignedLen);

    core.assign(core.data() + kOffset, kAssignedLen);

    EXPECT_EQ(core.size(), kAssignedLen);
    EXPECT_TRUE(core.is_small());
    for (std::size_t i = 0; i < kAssignedLen; ++i) {
        EXPECT_EQ(core.data()[i], expected[i]);
    }
    EXPECT_EQ(core.data()[kAssignedLen], CharT{});
}

TYPED_TEST(BasicStringCoreLifecycleTest, AssignSelfSmallWholeRangeKeepsContent) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;
    using CharT = typename TestFixture::CharType;

    constexpr std::size_t kLen = Policy::kSmallCapacity > 5 ? 5 : 1;
    const auto test_str = TestFixture::make_test_string(kLen);
    Core core(test_str.data(), kLen);

    core.assign(core.data(), core.size());

    EXPECT_EQ(core.size(), kLen);
    EXPECT_TRUE(core.is_small());
    for (std::size_t i = 0; i < kLen; ++i) {
        EXPECT_EQ(core.data()[i], test_str[i]);
    }
    EXPECT_EQ(core.data()[kLen], CharT{});
}

TYPED_TEST(BasicStringCoreLifecycleTest, AssignStringViewReplacesContent) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;
    using CharT = typename TestFixture::CharType;

    constexpr std::size_t kInitialLen = Policy::kSmallCapacity + 6;
    constexpr std::size_t kAssignedLen = Policy::kSmallCapacity > 4 ? 4 : 1;
    const auto initial = TestFixture::make_test_string(kInitialLen);
    const auto replacement = TestFixture::make_test_string(kAssignedLen);
    Core core(initial.data(), initial.size());

    core.assign(std::basic_string_view<CharT>(replacement.data(), replacement.size()));

    EXPECT_EQ(core.size(), replacement.size());
    EXPECT_TRUE(core.is_small());
    for (std::size_t i = 0; i < replacement.size(); ++i) {
        EXPECT_EQ(core.data()[i], replacement[i]);
    }
    EXPECT_EQ(core.data()[core.size()], CharT{});
}

TYPED_TEST(BasicStringCoreLifecycleTest, AssignCountAndCharFillsReplacement) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;
    using CharT = typename TestFixture::CharType;

    constexpr std::size_t kInitialLen = Policy::kSmallCapacity > 3 ? 3 : 1;
    constexpr std::size_t kAssignedLen = Policy::kSmallCapacity + 3;
    const auto initial = TestFixture::make_test_string(kInitialLen);
    const CharT fill = static_cast<CharT>('q');
    Core core(initial.data(), initial.size());

    core.assign(kAssignedLen, fill);

    EXPECT_EQ(core.size(), kAssignedLen);
    EXPECT_TRUE(core.is_external());
    for (std::size_t i = 0; i < kAssignedLen; ++i) {
        EXPECT_EQ(core.data()[i], fill);
    }
    EXPECT_EQ(core.data()[core.size()], CharT{});
}

TYPED_TEST(BasicStringCoreLifecycleTest, AssignCountZeroClearsToSmall) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;
    using CharT = typename TestFixture::CharType;

    constexpr std::size_t kInitialLen = Policy::kSmallCapacity + 6;
    const auto initial = TestFixture::make_test_string(kInitialLen);
    Core core(initial.data(), initial.size());

    core.assign(0, static_cast<CharT>('q'));

    EXPECT_TRUE(core.empty());
    EXPECT_EQ(core.size(), 0U);
    EXPECT_TRUE(core.is_small());
    EXPECT_EQ(core.data()[0], CharT{});
}

TYPED_TEST(BasicStringCoreLifecycleTest, AppendSelfSmallRangeRelocatesSourceWhenGrowing) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;
    using CharT = typename TestFixture::CharType;

    constexpr std::size_t kLen = Policy::kSmallCapacity;
    const auto test_str = TestFixture::make_test_string(kLen);
    Core core(test_str.data(), kLen);
    const std::basic_string<CharT> expected = test_str + test_str;

    core.append(core.data(), core.size());

    EXPECT_EQ(core.size(), expected.size());
    EXPECT_TRUE(core.is_external());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(core.data()[i], expected[i]);
    }
    EXPECT_EQ(core.data()[expected.size()], CharT{});
}

TYPED_TEST(BasicStringCoreLifecycleTest, AppendSelfExternalSubrangeKeepsBufferWhenCapacityAllows) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;
    using CharT = typename TestFixture::CharType;

    constexpr std::size_t kLen = Policy::kSmallCapacity + 4;
    const auto test_str = TestFixture::make_test_string(kLen);
    Core core(test_str.data(), kLen);
    const CharT* original_ptr = core.data();
    const std::size_t original_cap = core.capacity();

    constexpr std::size_t kOffset = 1;
    constexpr std::size_t kAppendLen = 3;
    const std::basic_string<CharT> expected = test_str + std::basic_string<CharT>(test_str.data() + kOffset, kAppendLen);

    core.append(core.data() + kOffset, kAppendLen);

    EXPECT_EQ(core.size(), expected.size());
    EXPECT_EQ(core.data(), original_ptr);
    EXPECT_EQ(core.capacity(), original_cap);
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(core.data()[i], expected[i]);
    }
    EXPECT_EQ(core.data()[expected.size()], CharT{});
}

TYPED_TEST(BasicStringCoreLifecycleTest, AppendStringViewAppendsCharacters) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;
    using CharT = typename TestFixture::CharType;

    constexpr std::size_t kInitialLen = Policy::kSmallCapacity > 3 ? 3 : 1;
    constexpr std::size_t kAppendLen = 2;
    const auto initial = TestFixture::make_test_string(kInitialLen);
    const auto suffix = TestFixture::make_test_string(kAppendLen);
    const std::basic_string<CharT> expected = initial + suffix;
    Core core(initial.data(), initial.size());

    core.append(std::basic_string_view<CharT>(suffix.data(), suffix.size()));

    EXPECT_EQ(core.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(core.data()[i], expected[i]);
    }
    EXPECT_EQ(core.data()[expected.size()], CharT{});
}

TYPED_TEST(BasicStringCoreLifecycleTest, AppendCountAndCharFillsNewRange) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;
    using CharT = typename TestFixture::CharType;

    constexpr std::size_t kInitialLen = Policy::kSmallCapacity;
    constexpr std::size_t kAppendLen = 4;
    const auto initial = TestFixture::make_test_string(kInitialLen);
    const CharT fill = static_cast<CharT>('z');
    Core core(initial.data(), initial.size());

    core.append(kAppendLen, fill);

    EXPECT_EQ(core.size(), kInitialLen + kAppendLen);
    EXPECT_TRUE(core.is_external());
    for (std::size_t i = 0; i < kInitialLen; ++i) {
        EXPECT_EQ(core.data()[i], initial[i]);
    }
    for (std::size_t i = 0; i < kAppendLen; ++i) {
        EXPECT_EQ(core.data()[kInitialLen + i], fill);
    }
    EXPECT_EQ(core.data()[core.size()], CharT{});
}

TYPED_TEST(BasicStringCoreLifecycleTest, PushBackAppendsOneCharacter) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;
    using CharT = typename TestFixture::CharType;

    constexpr std::size_t kInitialLen = Policy::kSmallCapacity;
    const auto initial = TestFixture::make_test_string(kInitialLen);
    const CharT pushed = static_cast<CharT>('p');
    Core core(initial.data(), initial.size());

    core.push_back(pushed);

    EXPECT_EQ(core.size(), kInitialLen + 1);
    EXPECT_TRUE(core.is_external());
    for (std::size_t i = 0; i < kInitialLen; ++i) {
        EXPECT_EQ(core.data()[i], initial[i]);
    }
    EXPECT_EQ(core.data()[kInitialLen], pushed);
    EXPECT_EQ(core.data()[core.size()], CharT{});
}

TYPED_TEST(BasicStringCoreLifecycleTest, PopBackRemovesLastCharacterAndMaintainsTerminator) {
    using Core = typename TestFixture::Core;
    using Policy = typename TestFixture::Policy;
    using CharT = typename TestFixture::CharType;

    constexpr std::size_t kInitialLen = Policy::kSmallCapacity > 4 ? 4 : 1;
    const auto initial = TestFixture::make_test_string(kInitialLen);
    Core core(initial.data(), initial.size());

    core.pop_back();

    EXPECT_EQ(core.size(), kInitialLen - 1);
    EXPECT_TRUE(core.is_small());
    for (std::size_t i = 0; i < core.size(); ++i) {
        EXPECT_EQ(core.data()[i], initial[i]);
    }
    EXPECT_EQ(core.data()[core.size()], CharT{});
}

TYPED_TEST(BasicStringCoreLifecycleTest, PopBackOnEmptyIsNoop) {
    using Core = typename TestFixture::Core;
    using CharT = typename TestFixture::CharType;

    Core core;

    core.pop_back();

    EXPECT_TRUE(core.empty());
    EXPECT_TRUE(core.is_small());
    EXPECT_EQ(core.data()[0], CharT{});
}

}// namespace
