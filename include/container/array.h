//
// Created by 赵丹 on 2025/9/1.
//

#ifndef AETHERMIND_CONTAINER_ARRAY_IMPL_H
#define AETHERMIND_CONTAINER_ARRAY_IMPL_H

#include "any.h"
#include "object.h"
#include "container/container_utils.h"

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
    static_assert(compatible_with_any_v<T>, "T must be compatible with Any");

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

    explicit Array(size_t n, Any value = Any());

    NODISCARD bool defined() const noexcept {
        return impl_;
    }

    NODISCARD uint32_t use_count() const noexcept {
        return impl_.use_count();
    }

    NODISCARD bool unique() const noexcept {
        return use_count() == 1;
    }

    NODISCARD size_t size() const noexcept {
        return impl_->size();
    }

    NODISCARD size_t capacity() const noexcept {
        return impl_->capacity();
    }

    NODISCARD bool empty() const noexcept {
        return size() == 0;
    }

    iterator begin() noexcept {
        return iterator(impl_->begin());
    }

    const_iterator begin() const noexcept {
        return const_iterator(impl_->begin());
    }

    iterator end() noexcept {
        return iterator(impl_->end());
    }

    const_iterator end() const noexcept {
        return const_iterator(impl_->end());
    }

    reverse_iterator rbegin() noexcept {
        return reverse_iterator(impl_->end() - 1);
    }

    const_reverse_iterator rbegin() const noexcept {
        return const_reverse_iterator(impl_->end() - 1);
    }

    reverse_iterator rend() noexcept {
        return reverse_iterator(impl_->begin() - 1);
    }

    const_reverse_iterator rend() const noexcept {
        return const_reverse_iterator(impl_->begin() - 1);
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
        return *rbegin();
    }

    template<typename Iter>
    void assign();

private:
    ObjectPtr<ArrayImpl> impl_;
};

template<typename T>
Array<T>::Array(size_t n, Any value) : impl_(make_array_object<ArrayImpl, Any>(n)) {
    impl_->start_ = reinterpret_cast<char*>(impl_.get()) + sizeof(ArrayImpl);
    impl_->size_ = n;
    impl_->capacity_ = n;

    auto* p = impl_->begin();
    for (size_t i = 0; i < n; ++i) {
        new (p + i) Any(std::move(value));
    }
}

}// namespace aethermind

#endif//AETHERMIND_CONTAINER_ARRAY_IMPL_H
