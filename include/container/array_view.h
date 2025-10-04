//
// Created by 赵丹 on 2025/8/25.
//

#ifndef AETHERMIND_ARRAY_REF_H
#define AETHERMIND_ARRAY_REF_H

#include "macros.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <glog/logging.h>
#include <initializer_list>
#include <iterator>
#include <ostream>
#include <type_traits>
#include <vector>

namespace aethermind {

/*!
 * \brief ArrayView is a view of an array.
 *
 * \tparam T the type of the array elements.
 *
 * \note ArrayView is a view of an array, it does not own the underlying array memory.
 * It is expected to be used in scenarios where the array memory is owned by someone else,
 * whose lifetime extends past that of the ArrayView. For this reason, it is not in general
 * safe to store an ArrayRef.
 */
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

    template<typename Container,
             typename U = decltype(std::declval<Container>().data()),
             typename = std::enable_if_t<std::is_same_v<U, T*> || std::is_same_v<U, const T*>>>
    ArrayView(const Container& container) : ArrayView(container.data(), container.size()) {}// NOLINT

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

    void check() const {
        CHECK(data_ != nullptr || size_ == 0)
                << "created ArrayRef with nullptr and non-zero length! std::optional relies on this being illegal";
    }

    const T* data() const {
        return data_;
    }

    NODISCARD size_type size() const {
        return size_;
    }

    NODISCARD bool empty() const {
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
        CHECK(!empty()) << "ArrayView front() must not be empty";
        return data()[0];
    }

    const T& back() const {
        CHECK(!empty()) << "ArrayView back() must not be empty";
        return data()[size() - 1];
    }

    bool equals(ArrayView other) const {
        return size() == other.size() && std::equal(cbegin(), cend(), other.cbegin());
    }

    // Returns true if all elements in the array match the given predicate.
    bool all_match(bool (*pred)(const T&)) const {
        return std::all_of(cbegin(), cend(), pred);
    }

    ArrayView slice(size_type offset, size_type n) const {
        CHECK(offset + n <= size()) << n << "ArrayView slice out of bounds!";
        return ArrayView(data() + offset, n);
    }

    ArrayView slice(size_type n) const {
        return slice(0, n);
    }

    const T& operator[](size_type idx) const {
        return data()[idx];
    }

    const T& at(size_type idx) const {
        CHECK(idx < size()) << "ArrayView::at() index out of bounds!";
        return data()[idx];
    }

    ArrayView& operator=(T&&) = delete;
    ArrayView& operator=(std::initializer_list<T>) = delete;

    std::vector<T> vec() const {
        return std::vector<T>(data(), data() + size());
    }

private:
    const T* data_;
    size_type size_;
};

template<typename T>
std::ostream& operator<<(std::ostream& os, ArrayView<T> array) {
    os << "[";
    auto i = array.size();
    for (const auto& e: array) {
        os << e << (--i > 0 ? ", " : "");
    }
    os << "]";
    return os;
}

// Returns an ArrayView of a single element.
template<typename T>
ArrayView<T> make_array_view(const T& elem) {
    return ArrayView<T>(elem);
}

// Returns an ArrayView of a pointer and size.
template<typename T>
ArrayView<T> make_array_view(const T* data, size_t size) {
    return ArrayView<T>(data, size);
}

// Returns an ArrayView of a range of elements.
template<typename T>
ArrayView<T> make_array_view(const T* begin, const T* end) {
    return ArrayView<T>(begin, end);
}

template<typename T>
ArrayView<T> make_array_view(const std::vector<T>& vec) {
    return vec;
}

template<typename T, size_t N>
ArrayView<T> make_array_view(const std::array<T, N>& arr) {
    return arr;
}

template<typename T, size_t N>
ArrayView<T> make_array_view(const T (&arr)[N]) {
    return arr;
}

template<typename T>
bool operator==(ArrayView<T> a, ArrayView<T> b) {
    return a.equals(b);
}

template<typename T>
bool operator!=(ArrayView<T> a, ArrayView<T> b) {
    return !a.equals(b);
}

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

using IntArrayView = ArrayView<int64_t>;

}// namespace aethermind

#endif//AETHERMIND_ARRAY_REF_H
