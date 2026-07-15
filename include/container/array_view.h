/// \file
/// Non-owning array view types and utility functions.
///
/// Provides two implementations selected by the USE_SPAN_AS_ARRAY_VIEW macro:
///   - Defined (default): ArrayView<T> and MutableArrayView<T> are type aliases
///     for std::span<const T> and std::span<T>, delegating all functionality to
///     the standard library.
///   - Undefined: A custom ArrayView<T> class with full iterator support, slice
///     operations, and std::hash integration for use with unordered containers.
///
/// ArrayView carries no ownership; the referenced memory must outlive the view.
/// Where possible, prefer the std::span-based path for ABI compatibility with
/// standard types.

#ifndef AETHERMIND_ARRAY_REF_H
#define AETHERMIND_ARRAY_REF_H

#include "macros.h"
#include "utils/hash.h"

#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <ostream>
#include <ranges>
#include <span>
#include <type_traits>
#include <vector>

// Default: ArrayView aliases to std::span.  Undefine before inclusion to use
// the custom ArrayView class instead.
#define USE_SPAN_AS_ARRAY_VIEW

#ifdef USE_SPAN_AS_ARRAY_VIEW

namespace aethermind {

/// Non-owning view of a contiguous const array.  Alias for std::span<const T>.
template<typename T>
using ArrayView = std::span<const T>;

/// Non-owning view of a contiguous mutable array.  Alias for std::span<T>.
template<typename T>
using MutableArrayView = std::span<T>;

/// Convenience alias for int64_t array views (the most common use case).
using IntArrayView = ArrayView<int64_t>;
using MutableIntArrayView = MutableArrayView<int64_t>;

/// Formats an ArrayView as "[e0, e1, ...]" into the given ostream.
template<typename T>
std::ostream& print_array_view(std::ostream& os, ArrayView<T> array) {
    os << "[";
    auto i = array.size();
    for (const auto& e: array) {
        os << e << (--i > 0 ? ", " : "");
    }
    os << "]";
    return os;
}

/// Wraps a single element into a one-element ArrayView.
template<typename T>
AM_NODISCARD constexpr ArrayView<T> make_array_view(const T& elem) noexcept {
    return ArrayView<T>(&elem, 1);
}

/// Wraps a pointer+size pair into an ArrayView.
template<typename T>
AM_NODISCARD constexpr ArrayView<T> make_array_view(const T* data, size_t size) noexcept {
    return ArrayView<T>(data, size);
}

/// Copies the view contents into a new vector, stripping cv-qualifiers.
template<typename T>
AM_NODISCARD std::vector<std::remove_cv_t<T>> to_vector(ArrayView<T> array) {
    return std::vector<std::remove_cv_t<T>>(array.begin(), array.end());
}

/// Element-wise equality comparison for ArrayView.
template<typename T>
AM_NODISCARD bool operator==(ArrayView<T> a, ArrayView<T> b) noexcept {
    return std::ranges::equal(a, b);
}

template<typename T>
AM_NODISCARD bool operator!=(ArrayView<T> a, ArrayView<T> b) noexcept {
    return !(a == b);
}

}// namespace aethermind

/// Stream formatting for IntArrayView (global scope for ADL).
inline std::ostream& operator<<(std::ostream& os, aethermind::IntArrayView array) {
    return aethermind::print_array_view(os, array);
}

/// Stream formatting for ArrayView<T> (global scope for ADL).
template<typename T>
std::ostream& operator<<(std::ostream& os, aethermind::ArrayView<T> array) {
    return aethermind::print_array_view(os, array);
}

#else

namespace aethermind {

/// Non-owning view of a contiguous const array.
///
/// ArrayView carries no ownership; the referenced memory must outlive the view.
/// It is not safe to store an ArrayView beyond the lifetime of the backing storage.
/// (Only used when USE_SPAN_AS_ARRAY_VIEW is undefined; the default path uses std::span.)
template<typename T>
class ArrayView final {
public:
    using value_type = T;
    using size_type = size_t;
    using iterator = const T*;
    using const_iterator = const T*;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    ArrayView() : data_(nullptr), size_(0) {}

    explicit ArrayView(const T& e) : ArrayView(&e, 1) {}// NOLINT

    ArrayView(const T* begin, const T* end) : ArrayView(begin, end - begin) {}

#ifdef CPP20
    template<typename Container>
        requires requires(Container container) {
            { container.data() } -> std::convertible_to<const T*>;
            container.size();
        }
#else
    template<typename Container,
             typename U = decltype(std::declval<Container>().data()),
             typename = std::enable_if_t<std::is_same_v<U, T*> || std::is_same_v<U, const T*>>>
#endif
    ArrayView(const Container& container) : ArrayView(container.data(), container.size()) {
    }// NOLINT

    ArrayView(const std::vector<T>& vec) : ArrayView(vec.data(), vec.size()) {// NOLINT
        static_assert(!std::is_same_v<T, bool>, "ArrayView<bool> cannot be constructed from a std::vector<bool> bitfield.");
    }

    template<size_type N>
    ArrayView(const std::array<T, N>& arr) : ArrayView(arr.data(), N) {}// NOLINT

    template<size_type N>
    ArrayView(const T (&arr)[N]) : ArrayView(arr, N) {}// NOLINT

    ArrayView(const std::initializer_list<T>& list)
        : ArrayView(list.begin() == list.end() ? static_cast<T*>(nullptr) : list.begin(), list.size()) {}

    ArrayView(const T* data, size_type size) : data_(data), size_(size) {
        check();
    }

    // Validates that nullptr is only paired with size 0.
    // std::optional<ArrayView> uses the nullptr+zero representation for its
    // null state, so the combination must always be legal.
    void check() const {
        AM_CHECK(data_ != nullptr || size_ == 0, "created ArrayRef with nullptr and non-zero length! std::optional relies on this being illegal");
    }

    const T* data() const {
        return data_;
    }

    AM_NODISCARD size_type size() const {
        return size_;
    }

    AM_NODISCARD bool empty() const {
        return size_ == 0;
    }

    iterator begin() const {
        return data_;
    }

    iterator end() const {
        return data_ + size_;
    }

    const_iterator cbegin() const {
        return data_;
    }

    const_iterator cend() const {
        return data_ + size_;
    }

    reverse_iterator rbegin() const {
        return reverse_iterator(end());
    }

    reverse_iterator rend() const {
        return reverse_iterator(begin());
    }

    const T& front() const {
        AM_CHECK(!empty(), "ArrayView front() must not be empty");
        return data()[0];
    }

    const T& back() const {
        AM_CHECK(!empty(), "ArrayView back() must not be empty");
        return data()[size() - 1];
    }

    /// Element-wise equality.  Cheaper than constructing a vector for the comparison.
    bool equals(ArrayView other) const {
        return size() == other.size() && std::equal(cbegin(), cend(), other.cbegin());
    }

    // Returns true if all elements in the array match the given predicate.
    bool all_match(bool (*pred)(const T&)) const {
        return std::all_of(cbegin(), cend(), pred);
    }

    /// Returns a sub-view starting at `offset` with `n` elements.
    /// \pre offset + n <= size()
    ArrayView slice(size_type offset, size_type n) const {
        AM_CHECK(offset + n <= size(), "ArrayView slice out of bounds!");
        return ArrayView(data() + offset, n);
    }

    /// Returns a sub-view of the first `n` elements.
    ArrayView slice(size_type n) const {
        return slice(0, n);
    }

    const T& operator[](size_type idx) const {
        return data()[idx];
    }

    const T& at(size_type idx) const {
        AM_CHECK(idx < size(), "ArrayView::at() index out of bounds!");
        return data()[idx];
    }

    // ArrayView is read-only; assignment is deleted to prevent accidental mutation.
    ArrayView& operator=(T&&) = delete;
    ArrayView& operator=(std::initializer_list<T>) = delete;

    std::vector<T> vec() const {
        return std::vector<T>(data(), data() + size());
    }

private:
    const T* data_;
    size_type size_;
};

/// Formats an ArrayView as "[e0, e1, ...]" into the given ostream.
template<typename T>
std::ostream& print_array_view(std::ostream& os, ArrayView<T> array) {
    os << "[";
    auto i = array.size();
    for (const auto& e: array) {
        os << e << (--i > 0 ? ", " : "");
    }
    os << "]";
    return os;
}

/// Stream formatting for ArrayView<T> (namespace scope, found via ADL).
template<typename T>
std::ostream& operator<<(std::ostream& os, ArrayView<T> array) {
    return print_array_view(os, array);
}

/// Convenience alias for int64_t array views (the most common use case).
using IntArrayView = ArrayView<int64_t>;

/// Wraps a single element into a one-element ArrayView.
template<typename T>
AM_NODISCARD ArrayView<T> make_array_view(const T& elem) {
    return ArrayView<T>(elem);
}

/// Wraps a pointer+size pair into an ArrayView.
template<typename T>
ArrayView<T> make_array_view(const T* data, size_t size) {
    return ArrayView<T>(data, size);
}

/// Wraps a pointer range [begin, end) into an ArrayView.
template<typename T>
ArrayView<T> make_array_view(const T* begin, const T* end) {
    return ArrayView<T>(begin, end);
}

/// Implicit conversion from std::vector.
template<typename T>
ArrayView<T> make_array_view(const std::vector<T>& vec) {
    return vec;
}

/// Implicit conversion from std::array.
template<typename T, size_t N>
ArrayView<T> make_array_view(const std::array<T, N>& arr) {
    return arr;
}

/// Implicit conversion from a C array.
template<typename T, size_t N>
ArrayView<T> make_array_view(const T (&arr)[N]) {
    return arr;
}

/// Copies the view contents into a new vector, stripping cv-qualifiers.
template<typename T>
AM_NODISCARD std::vector<std::remove_cv_t<T>> to_vector(ArrayView<T> array) {
    return std::vector<std::remove_cv_t<T>>(array.begin(), array.end());
}

/// Element-wise equality comparison for ArrayView.
template<typename T>
bool operator==(ArrayView<T> a, ArrayView<T> b) {
    return a.equals(b);
}

template<typename T>
bool operator!=(ArrayView<T> a, ArrayView<T> b) {
    return !a.equals(b);
}

/// Element-wise equality between ArrayView and std::vector.
template<typename T>
bool operator==(const std::vector<T>& a, ArrayView<T> b) {
    return ArrayView<T>(a).equals(b);
}

template<typename T>
bool operator!=(const std::vector<T>& a, ArrayView<T> b) {
    return !ArrayView<T>(a).equals(b);
}

template<typename T>
bool operator==(ArrayView<T> a, const std::vector<T>& b) {
    return a.equals(ArrayView<T>(b));
}

template<typename T>
bool operator!=(ArrayView<T> a, const std::vector<T>& b) {
    return !a.equals(ArrayView<T>(b));
}

}// namespace aethermind

namespace std {

/// std::hash specialization for ArrayView, enabling use in unordered containers.
template<typename T>
struct hash<aethermind::ArrayView<T>> {
    size_t operator()(aethermind::ArrayView<T> v) const {
        size_t seed = 0;
        for (const auto& elem: v) {
            seed = aethermind::hash_combine(seed, aethermind::details::simple_get_hash(elem));
        }
        return seed;
    }
};

}// namespace std

#endif

#endif// AETHERMIND_ARRAY_REF_H
