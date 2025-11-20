//
// Created by 赵丹 on 2025/9/1.
//

#ifndef AETHERMIND_CONTAINER_ARRAY_IMPL_H
#define AETHERMIND_CONTAINER_ARRAY_IMPL_H

#include "any.h"
#include "container/container_utils.h"

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

    ~ArrayImpl() override;

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

    static ObjectPtr<ArrayImpl> Create(size_t n);

    static ArrayImpl* CreateRawPtr(size_t n);

    void ConstructOneElemAtEnd(Any value);

    void ConstructAtEnd(size_t n, const Any& value);

    template<typename Iter>
    void ConstructAtEnd(Iter first, Iter last) {
        auto n = std::distance(first, last);
        CHECK(n <= capacity() - size());
        auto* p = end();
        // To ensure exception safety, size is only incremented after the initialization succeeds.
        for (auto it = first; it != last; ++it) {
            new (p++) Any(*it);
            ++size_;
        }
    }

    // shrink the array by delta elements.
    void ShrinkBy(int64_t delta);

    void EnlargeBy(int64_t delta, const Any& value);

    void MoveElemsRight(size_t dst, size_t src, size_t n);

    void MoveElemsLeft(size_t dst, size_t src, size_t n);

    template<typename T>
    friend class Array;
};

template<typename T>
class Array : public ObjectRef {
    static_assert(std::is_constructible_v<T>);
    class AnyProxy;
    class Converter;

public:
    using iterator = details::IteratorAdapter<ArrayImpl::iterator, Converter, Array>;
    using const_iterator = details::IteratorAdapter<ArrayImpl::const_iterator, Converter, const Array>;
    using reverse_iterator = details::ReverseIteratorAdapter<ArrayImpl::iterator, Converter, Array>;
    using const_reverse_iterator = details::ReverseIteratorAdapter<ArrayImpl::const_iterator, Converter, const Array>;

    Array() = default;

    explicit Array(size_t n, const T& value = T()) : pimpl_(ArrayImpl::Create(n)) {
        pimpl_->ConstructAtEnd(n, value);
    }

    Array(const std::vector<T>& other) : pimpl_(ArrayImpl::Create(other.size())) {//NOLINT
        pimpl_->ConstructAtEnd<>(other.begin(), other.end());                     // NOLINT
    }

    Array(std::initializer_list<T> other) : pimpl_(ArrayImpl::Create(other.size())) {
        pimpl_->ConstructAtEnd<>(other.begin(), other.end());// NOLINT
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
        return iterator(*this, pimpl_->begin());
    }

    const_iterator begin() const noexcept {
        return const_iterator(*this, pimpl_->begin());
    }

    iterator end() noexcept {
        return iterator(*this, pimpl_->end());
    }

    const_iterator end() const noexcept {
        return const_iterator(*this, pimpl_->end());
    }

    reverse_iterator rbegin() noexcept {
        return reverse_iterator(*this, pimpl_->end() - 1);
    }

    const_reverse_iterator rbegin() const noexcept {
        return const_reverse_iterator(*this, pimpl_->end() - 1);
    }

    reverse_iterator rend() noexcept {
        return reverse_iterator(*this, pimpl_->begin() - 1);
    }

    const_reverse_iterator rend() const noexcept {
        return const_reverse_iterator(*this, pimpl_->begin() - 1);
    }

    NODISCARD const Any& front() const {
        if (empty()) {
            AETHERMIND_THROW(IndexError) << "Cannot index an empty array.";
        }
        return *begin();
    }

    AnyProxy front() {
        if (empty()) {
            AETHERMIND_THROW(IndexError) << "Cannot index an empty array.";
        }
        // return AnyProxy(*this, 0);
        return {*this, 0};
    }

    NODISCARD const Any& back() const {
        if (empty()) {
            AETHERMIND_THROW(IndexError) << "Cannot index an empty array.";
        }
        return *(end() - 1);
    }

    AnyProxy back() {
        if (empty()) {
            AETHERMIND_THROW(IndexError) << "Cannot index an empty array.";
        }
        // return AnyProxy(*this, size() - 1);
        return {*this, size() - 1};
    }

    const Any& operator[](int64_t i) const {
        return *(begin() + i);
    }

    AnyProxy operator[](int64_t i) {
        return AnyProxy(*this, i);
    }

    NODISCARD const Any& at(int64_t i) const {
        if (empty()) {
            AETHERMIND_THROW(IndexError) << "Cannot index an empty array.";
        }

        if (i < 0 || i >= size()) {
            AETHERMIND_THROW(IndexError) << "the index out of range.";
        }

        return *(begin() + i);
    }

    AnyProxy at(int64_t i) {
        if (empty()) {
            AETHERMIND_THROW(IndexError) << "Cannot index an empty array.";
        }

        if (i < 0 || i >= size()) {
            AETHERMIND_THROW(IndexError) << "the index out of range.";
        }

        return AnyProxy(*this, i);
    }

    void push_back(const T& item) {
        COW(1);
        pimpl_->ConstructOneElemAtEnd(item);
    }

    template<typename... Args>
    void emplace_back(Args&&... args) {
        COW(1);
        pimpl_->ConstructOneElemAtEnd(T(std::forward<Args>(args)...));
    }

    void Set(int idx, T value) {
        if (idx < 0 || idx >= size()) {
            AETHERMIND_THROW(IndexError) << "indexing " << idx << " on an array of size " << size();
        }

        COW(0, true);
        *(pimpl_->begin() + idx) = std::move(value);
    }

    void swap(Array& other) noexcept {
        std::swap(pimpl_, other.pimpl_);
    }

    void clear() {
        COW(-size());
        pimpl_->clear();
    }

    void pop_back() {
        if (empty()) {
            AETHERMIND_THROW(RuntimeError) << "Cannot pop back an empty array.";
        }

        COW(-1);
        pimpl_->ShrinkBy(1);
    }

    void resize(int64_t n);

    void reserve(int64_t n);

    void insert(iterator pos, const T& value);

    template<typename Iter>
    void insert(iterator pos, Iter first, Iter last);

    void erase(iterator pos);
    void erase(iterator first, iterator last);

private:
    ObjectPtr<ArrayImpl> pimpl_;
    // Switch to a new container with the given capacity
    void SwitchContainer(size_t new_cap, bool copy_data = true);
    // Copy on write semantic
    void COW(int64_t delta, bool single_elem_inplace_change = false);
};

template<typename T>
void Array<T>::resize(int64_t n) {
    if (n < 0) {
        AETHERMIND_THROW(ValueError) << "Cannot resize an array to a negative size.";
    }

    const auto sz = size();
    COW(n - sz);
    if (sz < n) {
        pimpl_->ConstructAtEnd(n - sz, T());
    } else if (sz > n) {
        pimpl_->ShrinkBy(sz - n);
    }
}

template<typename T>
void Array<T>::reserve(int64_t n) {
    if (n > capacity()) {
        // auto new_pimpl = ObjectPtr<ArrayImpl>::reclaim(ArrayImpl::create(n));
        auto new_pimpl = ArrayImpl::Create(n);
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

template<typename T>
void Array<T>::insert(iterator pos, const T& value) {
    size_t idx = std::distance(begin(), pos);
    size_t n = std::distance(pos, end());
    COW(1);
    pimpl_->EnlargeBy(1, T());
    pimpl_->MoveElemsRight(idx + 1, idx, n);
    new (pimpl_->begin() + idx) Any(value);
}

template<typename T>
template<typename Iter>
void Array<T>::insert(iterator pos, Iter first, Iter last) {
    static_assert(details::is_valid_iterator_v<Iter, T>, "Iter cannot be inserted into a Array<T>");
    if (first != last) {
        size_t idx = std::distance(begin(), pos);
        size_t n = std::distance(pos, end());
        size_t numel = std::distance(first, last);
        COW(numel);
        pimpl_->EnlargeBy(numel, T());
        pimpl_->MoveElemsRight(idx + numel, idx, n);

        auto* p = pimpl_->begin() + idx;
        for (int i = 0; i < numel; ++i) {
            new (p++) Any(std::move(*first++));
        }
    }
}

template<typename T>
void Array<T>::erase(iterator pos) {
    if (!defined()) {
        AETHERMIND_THROW(runtime_error) << "Cannot erase an empty array.";
    }

    size_t idx = std::distance(begin(), pos);
    if (idx >= size()) {
        AETHERMIND_THROW(runtime_error) << "the index out of range.";
    }
    size_t n = std::distance(pos + 1, end());
    COW(-1);
    pimpl_->MoveElemsLeft(idx, idx + 1, n);
    pimpl_->ShrinkBy(1);
}

template<typename T>
void Array<T>::erase(iterator first, iterator last) {
    if (first == last) {
        return;
    }

    if (!defined()) {
        AETHERMIND_THROW(RuntimeError) << "Cannot erase an empty array.";
    }

    size_t begin_idx = std::distance(begin(), first);
    size_t end_idx = std::distance(begin(), last);
    size_t numel = std::distance(last, end());
    if (begin_idx >= end_idx) {
        AETHERMIND_THROW(index_error) << "cannot erase array in range [" << begin_idx << ", " << end_idx << ")";
    }

    if (begin_idx > size() || end_idx > size()) {
        AETHERMIND_THROW(index_error) << "the index out of range.";
    }

    COW(begin_idx - end_idx);
    pimpl_->MoveElemsLeft(begin_idx, end_idx, numel);
    pimpl_->ShrinkBy(end_idx - begin_idx);
}

template<typename T>
void Array<T>::SwitchContainer(size_t new_cap, bool copy_data) {
    auto new_pimpl = ArrayImpl::Create(new_cap);
    auto* src = pimpl_->begin();
    auto* dst = new_pimpl->begin();

    if (copy_data) {
        // copy to new ArrayImpl
        for (auto& i = new_pimpl->size_; i < size(); ++i) {
            new (dst++) Any(*src++);
        }
    } else {
        // move to new ArrayImpl
        for (auto& i = new_pimpl->size_; i < size(); ++i) {
            new (dst++) Any(std::move(*src++));
        }
    }
    pimpl_ = new_pimpl;
}

template<typename T>
void Array<T>::COW(int64_t delta, bool single_elem_inplace_change) {
    if (delta == 0) {// inplace
        if (single_elem_inplace_change) {
            if (!defined()) {
                AETHERMIND_THROW(RuntimeError) << "Cannot change an empty array.";
            }

            if (!unique()) {
                SwitchContainer(capacity());
            }
        }
    } else if (delta < 0) {// shrink the array
        if (!defined()) {
            AETHERMIND_THROW(RuntimeError) << "Cannot shrink an empty array.";
        }

        if (-delta > static_cast<int64_t>(size())) {
            AETHERMIND_THROW(RuntimeError) << "Cannot shrink the array by " << -delta << " elements.";
        }

        if (!unique()) {
            SwitchContainer(capacity());
        }
    } else {// expand the array
        const size_t new_size = static_cast<size_t>(delta) + size();
        if (!defined()) {
            size_t new_cap = std::max(new_size, ArrayImpl::kInitSize);
            SwitchContainer(new_cap);
        } else if (unique()) {
            if (new_size > capacity()) {
                size_t new_cap = std::max(new_size, capacity() * ArrayImpl::kIncFactor);
                SwitchContainer(new_cap, false);
            }
        } else {
            size_t new_cap = new_size > capacity() ? std::max(new_size, capacity() * ArrayImpl::kIncFactor)
                                                   : new_size;
            SwitchContainer(new_cap);
        }
    }
}

template<typename T>
class Array<T>::AnyProxy {
public:
    AnyProxy(Array& arr, size_t idx) : arr_(arr), idx_(idx) {}

    // template<typename U, typename = std::enable_if_t<std::is_convertible_v<U, T>>>
    // AnyProxy& operator=(U value) {
    //     arr_.COW(0, true);
    //     *(arr_.pimpl_->begin() + idx_) = T(std::move(value));
    //     return *this;
    // }

    AnyProxy& operator=(T value) {
        arr_.COW(0, true);
        *(arr_.pimpl_->begin() + idx_) = std::move(value);
        return *this;
    }

    friend bool operator==(const AnyProxy& lhs, const AnyProxy& rhs) {
        auto it_lhs = lhs.arr_.pimpl_->begin() + lhs.idx_;
        auto it_rhs = rhs.arr_.pimpl_->begin() + rhs.idx_;
        return *it_lhs == *it_rhs;
    }

    friend bool operator!=(const AnyProxy& lhs, const AnyProxy& rhs) {
        return !(lhs == rhs);
    }

    friend bool operator==(const AnyProxy& lhs, const Any& rhs) {
        auto it = lhs.arr_.pimpl_->begin() + lhs.idx_;
        return *it == rhs;
    }

    friend bool operator!=(const AnyProxy& lhs, const Any& rhs) {
        return !(lhs == rhs);
    }

    friend bool operator==(const Any& lhs, const AnyProxy& rhs) {
        auto it = rhs.arr_.pimpl_->begin() + rhs.idx_;
        return lhs == *it;
    }

    friend bool operator!=(const Any& lhs, const AnyProxy& rhs) {
        return !(lhs == rhs);
    }

private:
    Array& arr_;
    size_t idx_;
};

template<typename T>
class Array<T>::Converter {
public:
    using value_type = Any;
    // static value_type& convert(value_type* ptr) {
    //     return *ptr;
    // }
    //
    // static const value_type& convert(const value_type* ptr) {
    //     return *ptr;
    // }

    static const value_type& convert(const Array&, const value_type* ptr) {
        return *ptr;
    }

    static AnyProxy convert(Array& arr, value_type* ptr) {
        auto idx = std::distance(arr.pimpl_->begin(), ptr);
        return AnyProxy(arr, idx);
    }
};

}// namespace aethermind

namespace std {

template<typename T>
struct hash<aethermind::Array<T>> {
    size_t operator()(const aethermind::Array<T>& v) {
        size_t seed = 0;
        for (const auto& elem: v) {
            seed = aethermind::hash_combine(seed, aethermind::hash_details::simple_get_hash(elem));
        }
        return seed;
    }
};

}// namespace std

#endif//AETHERMIND_CONTAINER_ARRAY_IMPL_H
