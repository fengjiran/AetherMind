//
// Created by richard on 9/2/25.
//

#ifndef AETHERMIND_CONTAINER_UTILS_H
#define AETHERMIND_CONTAINER_UTILS_H

#include <iterator>

namespace aethermind {
namespace details {

// IteratorAdapter is a wrapper around an iterator that converts the value
// type of the iterator to another type.
// \tparam Iter The type of the iterator.
// \tparam Converter The type of the converter.
template<typename Iter, typename Converter>
class IteratorAdapter {
public:
    using value_type = Converter::RetType;
    using pointer = Converter::RetType*;
    using reference = Converter::RetType&;
    using iterator_category = std::iterator_traits<Iter>::iterator_category;
    using difference_type = std::iterator_traits<Iter>::difference_type;

    explicit IteratorAdapter(Iter iter) : iter_(iter) {}

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
        return IteratorAdapter(iter_ + offset);
    }

    IteratorAdapter operator-(difference_type offset) const {
        return IteratorAdapter(iter_ - offset);
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

    const value_type operator*() const {
        return Converter::convert(*iter_);
    }

private:
    Iter iter_;
};

template<typename Iter, typename Converter>
class ReverseIteratorAdapter {
public:
    using value_type = Converter::RetType;
    using pointer = Converter::RetType*;
    using reference = Converter::RetType&;
    using iterator_category = std::iterator_traits<Iter>::iterator_category;
    using difference_type = std::iterator_traits<Iter>::difference_type;

    explicit ReverseIteratorAdapter(Iter iter) : iter_(iter) {}

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
        return ReverseIteratorAdapter(iter_ - offset);
    }

    ReverseIteratorAdapter operator-(difference_type offset) const {
        return ReverseIteratorAdapter(iter_ + offset);
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

    const value_type operator*() const {
        return Converter::convert(*iter_);
    }

private:
    Iter iter_;
};

}// namespace details
}// namespace aethermind

#endif//AETHERMIND_CONTAINER_UTILS_H
