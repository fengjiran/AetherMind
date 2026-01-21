//
// Created by richard on 9/2/25.
//

#ifndef AETHERMIND_CONTAINER_UTILS_H
#define AETHERMIND_CONTAINER_UTILS_H

#include "macros.h"

#include <cstdint>
#include <iterator>
#include <optional>
#include <numeric>

namespace aethermind {

struct MapMagicConstants {
    // 0b11111111 represent that the slot is empty
    static constexpr auto kEmptySlot = std::byte{0xFF};
    // 0b11111110 represent that the slot is tombstone
    static constexpr auto kTombStoneSlot = std::byte{0xFE};
    // Number of probing choices available
    static constexpr int kNumOffsetDists = 126;
    // head flag
    static constexpr auto kHeadFlag = std::byte{0x00};
    // tail flag
    static constexpr auto kTailFlag = std::byte{0x80};
    // head flag mask
    static constexpr auto kHeadFlagMask = std::byte{0x80};
    // offset mask
    static constexpr auto kOffsetIdxMask = std::byte{0x7F};
    // default fib shift
    static constexpr uint32_t kDefaultFibShift = 63;

    // The number of elements in a memory block.
    static constexpr uint8_t kSlotsPerBlock = 16;

    // Max load factor of hash table
    static constexpr double kMaxLoadFactor = 0.75;

    static constexpr size_t kIncFactor = 2;

    // Index indicator to indicate an invalid index.
    static constexpr size_t kInvalidIndex = std::numeric_limits<size_t>::max();

    // next probe position offset
    static const size_t NextProbePosOffset[kNumOffsetDists];
};


namespace details {

#ifdef CPP20
template<typename T>
concept is_valid_array_type = std::default_initializable<T>;

template<typename Iter, typename T>
concept is_valid_iterator_v = std::convertible_to<typename std::iterator_traits<Iter>::iterator_category, std::input_iterator_tag> &&
                              (requires(Iter it) { requires std::same_as<std::remove_cv_t<std::remove_reference_t<decltype(*it)>>, T>; } ||//
                               requires(Iter it) { requires std::derived_from<std::remove_cv_t<std::remove_reference_t<decltype(*it)>>, T>; });


template<typename InputIter>
concept is_valid_iter = requires(InputIter t) {
    requires std::input_iterator<InputIter>;
    ++t;
    --t;
};

#else
template<typename Iter, typename T>
struct is_valid_iterator {
    static constexpr bool value = std::is_convertible_v<typename std::iterator_traits<Iter>::iterator_category, std::input_iterator_tag> &&
                                  (std::is_same_v<T, std::remove_cv_t<std::remove_reference_t<decltype(*std::declval<Iter>())>>> ||
                                   std::is_base_of_v<T, std::remove_cv_t<std::remove_reference_t<decltype(*std::declval<Iter>())>>>);
};

template<typename Iter, typename T>
struct is_valid_iterator<Iter, std::optional<T>> : is_valid_iterator<Iter, T> {};

template<typename Iter, typename T>
inline constexpr bool is_valid_iterator_v = is_valid_iterator<Iter, T>::value;

#endif

#ifdef CPP20
template<std::unsigned_integral T>
#else
template<typename T, std::enable_if_t<std::is_unsigned_v<T>>* = nullptr>
#endif
unsigned int GetDigitNumOfUnsigned(T val, int base = 10) noexcept {
    unsigned int n = 1;
    const unsigned int b2 = base * base;
    const unsigned int b3 = b2 * base;
    const unsigned long b4 = b3 * base;

    while (true) {
        if (val < static_cast<unsigned int>(base)) {
            return n;
        }

        if (val < b2) {
            return n + 1;
        }

        if (val < b3) {
            return n + 2;
        }

        if (val < b4) {
            return n + 3;
        }

        val /= b4;
        n += 4;
    }
}

#ifdef CPP20
template<std::unsigned_integral T>
#else
template<typename T, std::enable_if_t<std::is_unsigned_v<T>>* = nullptr>
#endif
void UnsignedToDigitChar(char* p, unsigned int len, T val) noexcept {
    constexpr char digits[201] =
            "0001020304050607080910111213141516171819"
            "2021222324252627282930313233343536373839"
            "4041424344454647484950515253545556575859"
            "6061626364656667686970717273747576777879"
            "8081828384858687888990919293949596979899";
    unsigned int pos = len - 1;
    while (val >= 100) {
        const auto idx = (val % 100) * 2;
        val /= 100;
        p[pos] = digits[idx + 1];
        p[pos - 1] = digits[idx];
        pos -= 2;
    }

    if (val >= 10) {
        const auto idx = val * 2;
        p[1] = digits[idx + 1];
        p[0] = digits[idx];
    } else {
        p[0] = '0' + val;
    }
}

// IteratorAdapter is a wrapper around an iterator that converts the value
// type of the iterator to another type.
// \tparam Iter The type of the iterator.
// \tparam Container The value container.
template<typename Iter, typename Container>
#ifdef CPP20
    requires requires { typename Container::Converter; }
#endif
class IteratorAdapter {
public:
    using traits_type = std::iterator_traits<Iter>;
    using value_type = traits_type::value_type;
    using pointer = traits_type::pointer;
    using reference = traits_type::reference;
    using iterator_category = traits_type::iterator_category;
    using difference_type = traits_type::difference_type;

    using Converter = Container::Converter;

    explicit IteratorAdapter(Container* ptr, Iter iter) : ptr_(ptr), iter_(iter) {}

#ifdef CPP20
    template<typename Iter1>
        requires std::convertible_to<Iter1, Iter>
#else
    template<typename Iter1,
             typename = std::enable_if_t<std::is_convertible_v<Iter1, Iter>>>
#endif
    IteratorAdapter(const IteratorAdapter<Iter1, Container>& other) : ptr_(other.ptr_), iter_(other.iter_) {//NOLINT
    }

    IteratorAdapter(const IteratorAdapter& other) : ptr_(other.ptr_), iter_(other.iter_) {}

    IteratorAdapter& operator=(const IteratorAdapter& other) {
        IteratorAdapter(other).swap(*this);
        return *this;
    }

    IteratorAdapter& operator++() {
        ++iter_;
        return *this;
    }

    IteratorAdapter& operator--() {
        --iter_;
        return *this;
    }

    IteratorAdapter operator++(int) {
        IteratorAdapter tmp = *this;
        ++iter_;
        return tmp;
    }

    IteratorAdapter operator--(int) {
        IteratorAdapter tmp = *this;
        --iter_;
        return tmp;
    }

    IteratorAdapter operator+(difference_type offset) const {
        return IteratorAdapter(ptr_, iter_ + offset);
    }

    IteratorAdapter operator-(difference_type offset) const {
        return IteratorAdapter(ptr_, iter_ - offset);
    }

    IteratorAdapter& operator+=(difference_type offset) {
        iter_ += offset;
        return *this;
    }

    IteratorAdapter& operator-=(difference_type offset) {
        iter_ -= offset;
        return *this;
    }

    const Iter& base() const noexcept {
        return iter_;
    }

    void swap(IteratorAdapter& other) noexcept {
        std::swap(ptr_, other.ptr_);
        std::swap(iter_, other.iter_);
    }

    decltype(auto) operator*() {
        return Converter::convert(ptr_, iter_);
    }

private:
    Container* ptr_;
    Iter iter_;

    template<typename T1, typename T2>
#ifdef CPP20
        requires requires { typename T2::Converter; }
#endif
    friend class IteratorAdapter;
};

template<typename IterL, typename IterR, typename Container>
bool operator==(const IteratorAdapter<IterL, Container>& lhs,
                const IteratorAdapter<IterR, Container>& rhs) noexcept {
    return lhs.base() == rhs.base();
}

template<typename Iter, typename Container>
bool operator==(const IteratorAdapter<Iter, Container>& lhs,
                const IteratorAdapter<Iter, Container>& rhs) noexcept {
    return lhs.base() == rhs.base();
}

template<typename IterL, typename IterR, typename Container>
bool operator!=(const IteratorAdapter<IterL, Container>& lhs,
                const IteratorAdapter<IterR, Container>& rhs) noexcept {
    return lhs.base() != rhs.base();
}

template<typename Iter, typename Container>
bool operator!=(const IteratorAdapter<Iter, Container>& lhs,
                const IteratorAdapter<Iter, Container>& rhs) noexcept {
    return lhs.base() != rhs.base();
}

template<typename IterL, typename IterR, typename Container>
bool operator>(const IteratorAdapter<IterL, Container>& lhs,
               const IteratorAdapter<IterR, Container>& rhs) noexcept {
    return lhs.base() > rhs.base();
}

template<typename Iter, typename Container>
bool operator>(const IteratorAdapter<Iter, Container>& lhs,
               const IteratorAdapter<Iter, Container>& rhs) noexcept {
    return lhs.base() > rhs.base();
}

template<typename IterL, typename IterR, typename Container>
bool operator>=(const IteratorAdapter<IterL, Container>& lhs,
                const IteratorAdapter<IterR, Container>& rhs) noexcept {
    return lhs.base() >= rhs.base();
}

template<typename Iter, typename Container>
bool operator>=(const IteratorAdapter<Iter, Container>& lhs,
                const IteratorAdapter<Iter, Container>& rhs) noexcept {
    return lhs.base() >= rhs.base();
}

template<typename IterL, typename IterR, typename Container>
bool operator<(const IteratorAdapter<IterL, Container>& lhs,
               const IteratorAdapter<IterR, Container>& rhs) noexcept {
    return lhs.base() < rhs.base();
}

template<typename Iter, typename Container>
bool operator<(const IteratorAdapter<Iter, Container>& lhs,
               const IteratorAdapter<Iter, Container>& rhs) noexcept {
    return lhs.base() < rhs.base();
}

template<typename IterL, typename IterR, typename Container>
bool operator<=(const IteratorAdapter<IterL, Container>& lhs,
                const IteratorAdapter<IterR, Container>& rhs) noexcept {
    return lhs.base() <= rhs.base();
}

template<typename Iter, typename Container>
bool operator<=(const IteratorAdapter<Iter, Container>& lhs,
                const IteratorAdapter<Iter, Container>& rhs) noexcept {
    return lhs.base() <= rhs.base();
}

template<typename IterL, typename IterR, typename Container>
auto operator-(const IteratorAdapter<IterL, Container>& lhs,
               const IteratorAdapter<IterR, Container>& rhs) noexcept
        -> decltype(lhs.base() - rhs.base()) {
    return lhs.base() - rhs.base();
}

template<typename Iter, typename Container>
IteratorAdapter<Iter, Container>::difference_type
operator-(const IteratorAdapter<Iter, Container>& lhs,
          const IteratorAdapter<Iter, Container>& rhs) noexcept {
    return lhs.base() - rhs.base();
}

template<typename Iter, typename Container>
IteratorAdapter<Iter, Container> operator+(typename IteratorAdapter<Iter, Container>::difference_type n,
                                           const IteratorAdapter<Iter, Container>& rhs) noexcept {
    return rhs + n;
}

template<typename Iter, typename Container>
#ifdef CPP20
    requires requires { typename Container::Converter; }
#endif
class ReverseIteratorAdapter {
public:
    using traits_type = std::iterator_traits<Iter>;
    using value_type = traits_type::value_type;
    using pointer = traits_type::pointer;
    using reference = traits_type::reference;
    using iterator_category = traits_type::iterator_category;
    using difference_type = traits_type::difference_type;

    using Converter = Container::Converter;

    explicit ReverseIteratorAdapter(Container* ptr, Iter iter) : ptr_(ptr), iter_(iter) {}

    ReverseIteratorAdapter& operator++() {
        --iter_;
        return *this;
    }

    ReverseIteratorAdapter& operator--() {
        ++iter_;
        return *this;
    }

    ReverseIteratorAdapter operator++(int) {
        ReverseIteratorAdapter tmp = *this;
        --iter_;
        return tmp;
    }

    ReverseIteratorAdapter operator--(int) {
        ReverseIteratorAdapter tmp = *this;
        ++iter_;
        return tmp;
    }

    ReverseIteratorAdapter operator+(difference_type offset) const {
        return ReverseIteratorAdapter(ptr_, iter_ - offset);
    }

    ReverseIteratorAdapter operator-(difference_type offset) const {
        return ReverseIteratorAdapter(ptr_, iter_ + offset);
    }

    ReverseIteratorAdapter& operator+=(difference_type offset) {
        iter_ -= offset;
        return *this;
    }

    ReverseIteratorAdapter& operator-=(difference_type offset) {
        iter_ += offset;
        return *this;
    }

    template<typename T = ReverseIteratorAdapter,
             typename R = std::enable_if_t<std::is_same_v<iterator_category,
                                                          std::random_access_iterator_tag>,
                                           typename T::difference_type>>
    R operator-(const ReverseIteratorAdapter& other) const {
        return other.iter_ - iter_;
    }

    const Iter& base() const noexcept {
        return iter_;
    }

    void swap(ReverseIteratorAdapter& other) noexcept {
        std::swap(ptr_, other.ptr_);
        std::swap(iter_, other.iter_);
    }

    decltype(auto) operator*() {
        return Converter::convert(ptr_, iter_);
    }

private:
    Container* ptr_;
    Iter iter_;

    template<typename T1, typename T2>
#ifdef CPP20
        requires requires { typename T2::Converter; }
#endif
    friend class ReverseIteratorAdapter;
};

template<typename IterL, typename IterR, typename Container>
bool operator==(const ReverseIteratorAdapter<IterL, Container>& lhs,
                const ReverseIteratorAdapter<IterR, Container>& rhs) noexcept {
    return lhs.base() == rhs.base();
}

template<typename Iter, typename Container>
bool operator==(const ReverseIteratorAdapter<Iter, Container>& lhs,
                const ReverseIteratorAdapter<Iter, Container>& rhs) noexcept {
    return lhs.base() == rhs.base();
}


template<typename IterL, typename IterR, typename Container>
bool operator!=(const ReverseIteratorAdapter<IterL, Container>& lhs,
                const ReverseIteratorAdapter<IterR, Container>& rhs) noexcept {
    return lhs.base() != rhs.base();
}

template<typename Iter, typename Container>
bool operator!=(const ReverseIteratorAdapter<Iter, Container>& lhs,
                const ReverseIteratorAdapter<Iter, Container>& rhs) noexcept {
    return lhs.base() != rhs.base();
}

template<typename IterL, typename IterR, typename Container>
bool operator>(const ReverseIteratorAdapter<IterL, Container>& lhs,
               const ReverseIteratorAdapter<IterR, Container>& rhs) noexcept {
    return rhs.base() > lhs.base();
}

template<typename Iter, typename Container>
bool operator>(const ReverseIteratorAdapter<Iter, Container>& lhs,
               const ReverseIteratorAdapter<Iter, Container>& rhs) noexcept {
    return rhs.base() > lhs.base();
}


template<typename IterL, typename IterR, typename Container>
bool operator>=(const ReverseIteratorAdapter<IterL, Container>& lhs,
                const ReverseIteratorAdapter<IterR, Container>& rhs) noexcept {
    return rhs.base() >= lhs.base();
}

template<typename Iter, typename Container>
bool operator>=(const ReverseIteratorAdapter<Iter, Container>& lhs,
                const ReverseIteratorAdapter<Iter, Container>& rhs) noexcept {
    return rhs.base() >= lhs.base();
}

template<typename IterL, typename IterR, typename Container>
bool operator<(const ReverseIteratorAdapter<IterL, Container>& lhs,
               const ReverseIteratorAdapter<IterR, Container>& rhs) noexcept {
    return rhs.base() < lhs.base();
}

template<typename Iter, typename Container>
bool operator<(const ReverseIteratorAdapter<Iter, Container>& lhs,
               const ReverseIteratorAdapter<Iter, Container>& rhs) noexcept {
    return rhs.base() < lhs.base();
}

template<typename IterL, typename IterR, typename Container>
bool operator<=(const ReverseIteratorAdapter<IterL, Container>& lhs,
                const ReverseIteratorAdapter<IterR, Container>& rhs) noexcept {
    return rhs.base() <= lhs.base();
}

template<typename Iter, typename Container>
bool operator<=(const ReverseIteratorAdapter<Iter, Container>& lhs,
                const ReverseIteratorAdapter<Iter, Container>& rhs) noexcept {
    return rhs.base() <= lhs.base();
}

template<typename IterL, typename IterR, typename Container>
auto operator-(const ReverseIteratorAdapter<IterL, Container>& lhs,
               const ReverseIteratorAdapter<IterR, Container>& rhs) noexcept
        -> decltype(rhs.base() - lhs.base()) {
    return rhs.base() - lhs.base();
}

template<typename Iter, typename Container>
ReverseIteratorAdapter<Iter, Container>::difference_type
operator-(const ReverseIteratorAdapter<Iter, Container>& lhs,
          const ReverseIteratorAdapter<Iter, Container>& rhs) noexcept {
    return rhs.base() - lhs.base();
}

template<typename Iter, typename Container>
ReverseIteratorAdapter<Iter, Container> operator+(typename ReverseIteratorAdapter<Iter, Container>::difference_type n,
                                                  const ReverseIteratorAdapter<Iter, Container>& rhs) noexcept {
    return rhs + n;
}

}// namespace details
}// namespace aethermind

#endif//AETHERMIND_CONTAINER_UTILS_H
