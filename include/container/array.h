//
// Created by 赵丹 on 2025/9/1.
//

#ifndef AETHERMIND_CONTAINER_ARRAY_IMPL_H
#define AETHERMIND_CONTAINER_ARRAY_IMPL_H

#include "any.h"
#include "object.h"

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

    Array() = default;

    explicit Array(size_t n, Any value = Any());

    NODISCARD bool defined() const noexcept {
        return impl_;
    }

private:
    ObjectPtr<ArrayImpl> impl_;
};

}// namespace aethermind

#endif//AETHERMIND_CONTAINER_ARRAY_IMPL_H
