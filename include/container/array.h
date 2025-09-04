//
// Created by 赵丹 on 2025/9/1.
//

#ifndef AETHERMIND_CONTAINER_ARRAY_IMPL_H
#define AETHERMIND_CONTAINER_ARRAY_IMPL_H

#include "any.h"
#include "container/container_utils.h"
#include "object.h"

#include <initializer_list>
#include <vector>

namespace aethermind {

class ArrayImpl : public Object {
public:
    using iterator = Any*;
    using const_iterator = const Any*;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using reference = Any&;
    using const_reference = const Any&;

    ArrayImpl() : start_(nullptr), size_(0), capacity_(0) {}

    ~ArrayImpl();

    // return the number of elements in the array.
    NODISCARD size_t size() const noexcept {
        return size_;
    }

    // return the number of elements that the array can hold.
    NODISCARD size_t capacity() const noexcept {
        return capacity_;
    }

    // return the mutable pointer to the first element in the array.
    NODISCARD iterator begin() noexcept {
        return static_cast<Any*>(start_);
    }

    // return the const pointer to the first element in the array.
    NODISCARD const_iterator begin() const noexcept {
        return static_cast<const Any*>(start_);
    }

    // return the mutable pointer to the element after the last element in the array.
    NODISCARD iterator end() noexcept {
        return begin() + size_;
    }

    // return the const pointer to the element after the last element in the array.
    NODISCARD const_iterator end() const noexcept {
        return begin() + size_;
    }

    // return the mutable reverse iterator to the first element in the array.
    NODISCARD reverse_iterator rbegin() noexcept {
        return reverse_iterator(end());
    }

    // return the const reverse iterator to the end element in the array
    NODISCARD const_reverse_iterator rbegin() const noexcept {
        return const_reverse_iterator(end());
    }

    NODISCARD reverse_iterator rend() noexcept {
        return reverse_iterator(begin());
    }

    NODISCARD const_reverse_iterator rend() const noexcept {
        return const_reverse_iterator(begin());
    }

    reference operator[](size_t idx) noexcept {
        return static_cast<Any*>(start_)[idx];
    }

    const_reference operator[](size_t idx) const noexcept {
        return static_cast<const Any*>(start_)[idx];
    }

    // shrink the array by delta elements.
    void ShrinkBy(int64_t delta);


private:
    void* start_;
    size_t size_;
    size_t capacity_;

    static constexpr size_t kInitSize = 4;
    static constexpr size_t kIncFactor = 2;

    template<typename T>
    friend class Array;
};

class ArrayImplNullType final : public ArrayImpl {
    static ArrayImplNullType singleton_;
    ArrayImplNullType() = default;

public:
    static constexpr ArrayImpl* singleton() noexcept {
        return &singleton_;
    }
};

template<>
struct GetNullType<ArrayImpl> {
    using type = ArrayImplNullType;
};

template<typename T>
class Array {
public:
    static_assert(details::compatible_with_any_v<T>, "T must be compatible with Any");

    struct Converter {
        using RetType = T;
        static T convert(const Any& elem) {
            if constexpr (std::is_same_v<T, Any>) {
                return elem;
            } else {
                return elem.cast<T>();
            }
        }
    };

    using iterator = details::IteratorAdapter<ArrayImpl::iterator, Converter>;
    using const_iterator = details::IteratorAdapter<ArrayImpl::const_iterator, Converter>;
    using reverse_iterator = details::ReverseIteratorAdapter<ArrayImpl::iterator, Converter>;
    using const_reverse_iterator = details::ReverseIteratorAdapter<ArrayImpl::const_iterator, Converter>;

    Array() = default;

    explicit Array(size_t n, const Any& value = Any());
    Array(const std::vector<T>&);   // NOLINT
    Array(std::initializer_list<T>);// NOLINT
    explicit Array(ObjectPtr<ArrayImpl> pimpl) : pimpl_(std::move(pimpl)) {}

    Array(const Array&) = default;
    Array(Array&&) noexcept = default;

    Array& operator=(const Array& other);
    Array& operator=(Array&& other) noexcept;


    NODISCARD bool defined() const noexcept {
        return pimpl_;
    }

    NODISCARD uint32_t use_count() const noexcept {
        return pimpl_.use_count();
    }

    NODISCARD bool unique() const noexcept {
        return use_count() == 1;
    }

    NODISCARD size_t size() const noexcept {
        return pimpl_->size();
    }

    NODISCARD size_t capacity() const noexcept {
        return pimpl_->capacity();
    }

    NODISCARD bool empty() const noexcept {
        return size() == 0;
    }

    iterator begin() noexcept {
        return iterator(pimpl_->begin());
    }

    const_iterator begin() const noexcept {
        return const_iterator(pimpl_->begin());
    }

    iterator end() noexcept {
        return iterator(pimpl_->end());
    }

    const_iterator end() const noexcept {
        return const_iterator(pimpl_->end());
    }

    reverse_iterator rbegin() noexcept {
        return reverse_iterator(pimpl_->end() - 1);
    }

    const_reverse_iterator rbegin() const noexcept {
        return const_reverse_iterator(pimpl_->end() - 1);
    }

    reverse_iterator rend() noexcept {
        return reverse_iterator(pimpl_->begin() - 1);
    }

    const_reverse_iterator rend() const noexcept {
        return const_reverse_iterator(pimpl_->begin() - 1);
    }

    const T front() const {
        if (empty()) {
            AETHERMIND_THROW(index_error) << "Cannot index an empty array.";
        }
        return *begin();
    }

    const T back() const {
        if (empty()) {
            AETHERMIND_THROW(index_error) << "Cannot index an empty array.";
        }
        return *(end() - 1);
    }

    const T operator[](int64_t i) const;

    void swap(Array& other) noexcept {
        std::swap(pimpl_, other.pimpl_);
    }

    // TODO: 实现 CopyOnWrite
    ObjectPtr<ArrayImpl> CopyOnWrite() {
        //
    }

    template<typename Iter>
    void assign();

private:
    ObjectPtr<ArrayImpl> pimpl_;

    void InitWithSize(size_t n, const Any& value);

    template<typename Iter, typename = std::enable_if_t<details::is_valid_iterator_v<Iter, T>>>
    void InitWithRange(Iter first, Iter last);

    ObjectPtr<ArrayImpl> CheckAndReallocate(size_t new_cap);
};

template<typename T>
Array<T>::Array(size_t n, const Any& value) {
    InitWithSize(n, value);
}

template<typename T>
Array<T>::Array(const std::vector<T>& other) {
    InitWithRange(other.begin(), other.end());
}

template<typename T>
Array<T>::Array(std::initializer_list<T> other) {
    InitWithRange(other.begin(), other.end());
}

template<typename T>
Array<T>& Array<T>::operator=(const Array& other) {
    Array(other).swap(*this);
    return *this;
}

template<typename T>
Array<T>& Array<T>::operator=(Array&& other) noexcept {
    Array(std::move(other)).swap(*this);
    return *this;
}

template<typename T>
const T Array<T>::operator[](int64_t i) const {
    if (empty()) {
        AETHERMIND_THROW(index_error) << "Cannot index an empty array.";
    }

    if (i < 0 || i >= size()) {
        AETHERMIND_THROW(index_error) << "the index out of range.";
    }

    return *(begin() + i);
}

template<typename T>
void Array<T>::InitWithSize(size_t n, const Any& value) {
    pimpl_ = make_array_object<ArrayImpl, Any>(n);
    pimpl_->start_ = reinterpret_cast<char*>(pimpl_.get()) + sizeof(ArrayImpl);
    pimpl_->size_ = 0;
    pimpl_->capacity_ = n;

    auto* p = pimpl_->begin();
    // To ensure exception safety, size is only incremented after the initialization succeeds
    size_t& i = pimpl_->size_;
    while (i < n) {
        new (p++) Any(value);
        ++i;
    }
}

template<typename T>
template<typename Iter, typename>
void Array<T>::InitWithRange(Iter first, Iter last) {
    auto n = std::distance(first, last);
    pimpl_ = make_array_object<ArrayImpl, Any>(n);
    pimpl_->start_ = reinterpret_cast<char*>(pimpl_.get()) + sizeof(ArrayImpl);
    pimpl_->size_ = 0;
    pimpl_->capacity_ = n;

    auto* p = pimpl_->begin();
    // To ensure exception safety, size is only incremented after the initialization succeeds
    size_t& i = pimpl_->size_;
    while (i < n) {
        new (p++) Any(*first++);
        ++i;
    }
}


}// namespace aethermind

#endif//AETHERMIND_CONTAINER_ARRAY_IMPL_H
