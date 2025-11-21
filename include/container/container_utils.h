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
// \tparam Container The value container.
template<typename Iter, typename Container>
class IteratorAdapter {
public:
    using traits_type = std::iterator_traits<Iter>;
    using value_type = traits_type::value_type;
    using pointer = traits_type::pointer;
    using reference = traits_type::reference;
    using iterator_category = traits_type::iterator_category;
    using difference_type = traits_type::difference_type;

    using Converter = Container::Converter;

    explicit IteratorAdapter(Container& arr, Iter iter) : arr_(arr), ptr_(&arr_), iter_(iter) {}

    explicit IteratorAdapter(Container* ptr, Iter iter) : ptr_(ptr), arr_(*ptr_), iter_(iter) {}

    template<typename Iter1,
             typename = std::enable_if_t<std::is_convertible_v<Iter1, Iter>>>
    IteratorAdapter(const IteratorAdapter<Iter1, Container>& other) : arr_(other.arr_), ptr_(&arr_), iter_(other.iter_) {}

    IteratorAdapter(const IteratorAdapter& other) : arr_(other.arr_), ptr_(&arr_), iter_(other.iter_) {}

    IteratorAdapter& operator=(const IteratorAdapter& other) {
        if (this != &other) {
            ptr_ = other.ptr_;
            iter_ = other.iter_;
        }
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

    // template<typename T = IteratorAdapter,
    //          typename R = std::enable_if_t<std::is_same_v<iterator_category,
    //                                                       std::random_access_iterator_tag>,
    //                                        typename T::difference_type>>
    // R operator-(const IteratorAdapter& other) const {
    //     return iter_ - other.iter_;
    // }

    const Iter& base() const noexcept {
        return iter_;
    }

    decltype(auto) operator*() {
        return Converter::convert(*ptr_, iter_);
    }

private:
    Container& arr_;
    Container* ptr_;
    Iter iter_;

    template<typename T1, typename T2>
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

template<typename Iter, typename Converter, typename Array>
class ReverseIteratorAdapter {
public:
    using traits_type = std::iterator_traits<Iter>;
    using value_type = traits_type::value_type;
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
