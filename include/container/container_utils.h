//
// Created by richard on 9/2/25.
//

#ifndef AETHERMIND_CONTAINER_UTILS_H
#define AETHERMIND_CONTAINER_UTILS_H

#include <iterator>
#include <optional>

namespace aethermind {

namespace details {

// template<typename T>
// constexpr bool compatible_with_any_v = std::is_same_v<T, Any> || TypeTraits<T>::storage_enabled;

template<typename Iter, typename T>
struct is_valid_iterator {
    static constexpr bool value = std::is_convertible_v<typename std::iterator_traits<Iter>::iterator_category, std::input_iterator_tag> &&
                                  (std::is_same_v<T, std::remove_cv_t<std::remove_reference_t<decltype(*std::declval<Iter>())>>> ||
                                   std::is_base_of_v<T, std::remove_cv_t<std::remove_reference_t<decltype(*std::declval<Iter>())>>>);
};

template<typename Iter, typename T>
struct is_valid_iterator<Iter, std::optional<T>> : is_valid_iterator<Iter, T> {};

// template<typename Iter>
// struct is_valid_iterator<Iter, Any> : std::true_type {};

template<typename Iter, typename T>
inline constexpr bool is_valid_iterator_v = is_valid_iterator<Iter, T>::value;


// IteratorAdapter is a wrapper around an iterator that converts the value
// type of the iterator to another type.
// \tparam Iter The type of the iterator.
// \tparam Converter The type of the converter.
template<typename Iter, typename Converter, typename Array>
class IteratorAdapter {
public:
    using value_type = Converter::value_type;
    using pointer = value_type*;
    using reference = value_type&;
    using iterator_category = std::iterator_traits<Iter>::iterator_category;
    using difference_type = std::iterator_traits<Iter>::difference_type;

    explicit IteratorAdapter(Array& arr, Iter iter) : arr_(arr), iter_(iter) {}

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
        return IteratorAdapter(arr_, iter_ + offset);
    }

    IteratorAdapter operator-(difference_type offset) const {
        return IteratorAdapter(arr_, iter_ - offset);
    }

    IteratorAdapter& operator+=(difference_type offset) {
        iter_ += offset;
        return *this;
    }

    IteratorAdapter& operator-=(difference_type offset) {
        iter_ -= offset;
        return *this;
    }

    template<typename T = IteratorAdapter,
             typename R = std::enable_if_t<std::is_same_v<iterator_category,
                                                          std::random_access_iterator_tag>,
                                           typename T::difference_type>>
    R operator-(const IteratorAdapter& other) const {
        return iter_ - other.iter_;
    }

    bool operator==(const IteratorAdapter& other) const {
        return iter_ == other.iter_;
    }

    bool operator!=(const IteratorAdapter& other) const {
        return !(*this == other);
    }

    decltype(auto) operator*() {
        // return Converter::convert(iter_);
        return Converter::convert(arr_, iter_);
    }

private:
    Array& arr_;
    Iter iter_;
};

template<typename Iter, typename Converter, typename Array>
class ReverseIteratorAdapter {
public:
    using value_type = Converter::value_type;
    using pointer = value_type*;
    using reference = value_type&;
    using iterator_category = std::iterator_traits<Iter>::iterator_category;
    using difference_type = std::iterator_traits<Iter>::difference_type;

    explicit ReverseIteratorAdapter(Array& arr, Iter iter) : arr_(arr), iter_(iter) {}

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
        return *this;
    }

    ReverseIteratorAdapter operator+(difference_type offset) const {
        return ReverseIteratorAdapter(arr_, iter_ - offset);
    }

    ReverseIteratorAdapter operator-(difference_type offset) const {
        return ReverseIteratorAdapter(arr_, iter_ + offset);
    }

    template<typename T = ReverseIteratorAdapter,
             typename R = std::enable_if_t<std::is_same_v<iterator_category,
                                                          std::random_access_iterator_tag>,
                                           typename T::difference_type>>
    R operator-(const ReverseIteratorAdapter& other) const {
        return other.iter_ - iter_;
    }

    bool operator==(const ReverseIteratorAdapter& other) const {
        return iter_ == other.iter_;
    }

    bool operator!=(const ReverseIteratorAdapter& other) const {
        return !(*this == other);
    }

    // auto&& operator*() {
    //     return Converter::convert(iter_);
    // }

    decltype(auto) operator*() {
        // return Converter::convert(iter_);
        return Converter::convert(arr_, iter_);
    }

private:
    Array& arr_;
    Iter iter_;
};


}// namespace details
}// namespace aethermind

#endif//AETHERMIND_CONTAINER_UTILS_H
