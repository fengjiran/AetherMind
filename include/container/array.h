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

    void clear();

private:
    void* start_;
    size_t size_;
    size_t capacity_;

    static constexpr size_t kInitSize = 4;
    static constexpr size_t kIncFactor = 2;

    static ArrayImpl* create(size_t n);

    // shrink the array by delta elements.
    void ShrinkBy(int64_t delta);

    void EnlargeBy(int64_t delta, const Any& value);

    void ConstructAtEnd(size_t n, const Any& value);

    template<typename Iter>
    void ConstructAtEnd(Iter first, Iter last) {
        auto* p = end();
        // placement new
        // To ensure exception safety, size is only incremented after the initialization succeeds
        for (auto it = first; it != last; ++it) {
            new (p++) Any(*it);
            ++size_;
        }
    }

    void MoveElemsRight(size_t dst, size_t src, size_t n);

    void MoveElemsLeft(size_t dst, size_t src, size_t n);

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

    explicit Array(size_t n, const Any& value = Any(T()))
        : pimpl_(ObjectPtr<ArrayImpl>::reclaim(ArrayImpl::create(n))) {
        pimpl_->ConstructAtEnd(n, value);
    }

    Array(const std::vector<T>& other) {//NOLINT
        pimpl_ = ObjectPtr<ArrayImpl>::reclaim(ArrayImpl::create(other.size()));
        pimpl_->ConstructAtEnd<>(other.begin(), other.end());
    }

    Array(std::initializer_list<T> other) {
        pimpl_ = ObjectPtr<ArrayImpl>::reclaim(ArrayImpl::create(other.size()));
        pimpl_->ConstructAtEnd<>(other.begin(), other.end());
    }

    explicit Array(ObjectPtr<ArrayImpl> pimpl) : pimpl_(std::move(pimpl)) {}

    Array(const Array&) = default;
    Array(Array&&) noexcept = default;

    Array& operator=(const Array& other) {
        Array(other).swap(*this);
        return *this;
    }

    Array& operator=(Array&& other) noexcept {
        Array(std::move(other)).swap(*this);
        return *this;
    }

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

    void push_back(const T& item) {
        CopyOnWrite(1);
        pimpl_->ConstructAtEnd(1, Any(item));
    }

    template<typename... Args>
    void emplace_back(Args&&... args) {
        CopyOnWrite(1);
        pimpl_->ConstructAtEnd(1, Any(T(std::forward<Args>(args)...)));
    }

    const T operator[](int64_t i) const {
        if (empty()) {
            AETHERMIND_THROW(index_error) << "Cannot index an empty array.";
        }

        if (i < 0 || i >= size()) {
            AETHERMIND_THROW(index_error) << "the index out of range.";
        }

        return *(begin() + i);
    }

    void swap(Array& other) noexcept {
        std::swap(pimpl_, other.pimpl_);
    }

    void clear() {
        CopyOnWrite();
        pimpl_->clear();
    }

    void pop_back() {
        if (empty()) {
            AETHERMIND_THROW(runtime_error) << "Cannot pop back an empty array.";
        }
        CopyOnWrite();
        pimpl_->ShrinkBy(1);
    }

    void resize(int64_t n) {
        if (n < 0) {
            AETHERMIND_THROW(value_error) << "Cannot resize an array to negative size.";
        }

        auto sz = size();
        if (sz < n) {
            CopyOnWrite(n - sz);
            pimpl_->EnlargeBy(n - sz, T());
        } else if (sz > n) {
            CopyOnWrite();
            pimpl_->ShrinkBy(sz - n);
        }
    }

    void reserve(int64_t n) {
        if (n > capacity()) {
            auto new_pimpl = ObjectPtr<ArrayImpl>::reclaim(ArrayImpl::create(n));
            auto* from = pimpl_->begin();
            auto* to = new_pimpl->begin();
            size_t& i = new_pimpl->size_;
            if (unique()) {
                while (i < size()) {
                    new (to++) Any(std::move(*from++));
                    ++i;
                }
                pimpl_ = std::move(new_pimpl);
            } else {
                while (i < size()) {
                    new (to++) Any(*from++);
                    ++i;
                }
                pimpl_ = new_pimpl;
            }
        }
    }

    void insert(iterator pos, const T& value) {
        size_t idx = std::distance(begin(), pos);
        size_t n = std::distance(pos, end());
        CopyOnWrite(1);
        pimpl_->EnlargeBy(1, T());
        pimpl_->MoveElemsRight(idx + 1, idx, n);
        new (pimpl_->begin() + idx) Any(value);
    }

    template<typename Iter>
    void insert(iterator pos, Iter first, Iter last) {
        static_assert(details::is_valid_iterator_v<Iter, T>, "Iter cannot be inserted into a Array<T>");
        if (first != last) {
            size_t idx = std::distance(begin(), pos);
            size_t n = std::distance(pos, end());
            size_t numel = std::distance(first, last);
            CopyOnWrite(numel);
            pimpl_->EnlargeBy(numel, T());
            pimpl_->MoveElemsRight(idx + numel, idx, n);

            auto* p = pimpl_->begin() + idx;
            for (int i = 0; i < numel; ++i) {
                new (p++) Any(std::move(*first++));
            }
        }
    }

private:
    ObjectPtr<ArrayImpl> pimpl_;
    void CopyOnWrite();
    void CopyOnWrite(size_t extra);
};

template<typename T>
void Array<T>::CopyOnWrite() {
    if (defined() && !unique()) {
        auto new_pimpl = ObjectPtr<ArrayImpl>::reclaim(ArrayImpl::create(capacity()));
        // copy to new ArrayImpl
        auto* from = pimpl_->begin();
        auto* to = new_pimpl->begin();
        size_t& i = new_pimpl->size_;
        while (i < size()) {
            new (to++) Any(*from++);
            ++i;
        }
        pimpl_ = new_pimpl;
    }
}

template<typename T>
void Array<T>::CopyOnWrite(size_t extra) {
    if (!defined()) {
        size_t n = std::max(extra, ArrayImpl::kInitSize);
        pimpl_ = ObjectPtr<ArrayImpl>::reclaim(ArrayImpl::create(n));
    } else if (unique()) {
        if (extra + size() > capacity()) {
            size_t n = std::max(capacity() * ArrayImpl::kIncFactor, extra + size());
            auto new_pimpl = ObjectPtr<ArrayImpl>::reclaim(ArrayImpl::create(n));
            // move to new ArrayImpl
            auto* from = pimpl_->begin();
            auto* to = new_pimpl->begin();
            size_t& i = new_pimpl->size_;
            while (i < size()) {
                new (to++) Any(std::move(*from++));
                ++i;
            }
            pimpl_ = std::move(new_pimpl);
        }
    } else {
        size_t new_cap = extra + size() > capacity() ? std::max(capacity() * ArrayImpl::kIncFactor, extra + size()) : capacity();
        auto new_pimpl = ObjectPtr<ArrayImpl>::reclaim(ArrayImpl::create(new_cap));
        // copy to new ArrayImpl
        auto* from = pimpl_->begin();
        auto* to = new_pimpl->begin();
        size_t& i = new_pimpl->size_;
        while (i < size()) {
            new (to++) Any(*from++);
            ++i;
        }
        pimpl_ = new_pimpl;
    }
}


}// namespace aethermind

#endif//AETHERMIND_CONTAINER_ARRAY_IMPL_H
