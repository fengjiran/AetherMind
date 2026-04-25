// core.hpp - BasicStringCore implementation skeleton
// Part of AetherMind project, licensed under MIT License.
// See LICENSE.txt for details.
// SPDX-License-Identifier: MIT

#ifndef AETHERMIND_AMSTRING_CORE_HPP
#define AETHERMIND_AMSTRING_CORE_HPP

#include "allocator_support.hpp"
#include "char_algorithms.hpp"
#include "growth_policy.hpp"
#include "layout_policy.hpp"

#include <cstddef>
#include <memory>
#include <type_traits>

namespace aethermind {

// BasicStringCore - unified algorithm and lifecycle skeleton
// Combines LayoutPolicy, GrowthPolicy, and Allocator to implement string semantics
template<typename CharT,
         typename Traits = std::char_traits<CharT>,
         typename Allocator = std::allocator<CharT>,
         typename LayoutPolicy = DefaultLayoutPolicy<CharT>::type,
         typename GrowthPolicy = default_growth_policy>
class BasicStringCore {
public:
    using ValueType = CharT;
    using traits_type = Traits;
    using allocator_type = Allocator;
    using LayoutPolicyType = LayoutPolicy;
    using GrowthPolicyType = GrowthPolicy;

    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = CharT*;
    using const_pointer = const CharT*;

    using Storage = LayoutPolicy::Storage;
    using alloc_helper = allocator_traits_helper<Allocator>;
    using char_algo = char_algorithms<CharT, Traits>;

    static constexpr size_type npos = static_cast<size_type>(-1);

    BasicStringCore() noexcept {
        LayoutPolicy::InitEmpty(storage_);
    }

    explicit BasicStringCore(const allocator_type& a) noexcept
        : alloc_(a) {
        LayoutPolicy::InitEmpty(storage_);
    }

    BasicStringCore(const CharT* s, size_type n,
                    const allocator_type& a = allocator_type{})
        : alloc_(a) {
        if (n <= LayoutPolicy::kSmallCapacity) {
            LayoutPolicy::InitSmall(storage_, s, n);
        } else {
            const size_type cap = GrowthPolicy::min_heap_capacity(n);
            pointer ptr = alloc_helper::traits::allocate(alloc_, cap + 1);
            char_algo::copy(ptr, s, n);
            ptr[n] = char_algo::null_char();
            LayoutPolicy::InitExternal(storage_, ptr, n, cap);
        }
    }

    BasicStringCore(const BasicStringCore& other)
        : alloc_(alloc_helper::select_on_copy(other.alloc_)) {
        if (LayoutPolicy::is_small(other.storage_)) {
            LayoutPolicy::InitSmall(storage_,
                                    LayoutPolicy::data(other.storage_),
                                    LayoutPolicy::size(other.storage_));
        } else {
            const size_type sz = LayoutPolicy::size(other.storage_);
            const size_type cap = LayoutPolicy::capacity(other.storage_);
            pointer ptr = alloc_helper::traits::allocate(alloc_, cap + 1);
            char_algo::copy(ptr, LayoutPolicy::data(other.storage_), sz);
            ptr[sz] = char_algo::null_char();
            LayoutPolicy::InitExternal(storage_, ptr, sz, cap);
        }
    }

    BasicStringCore(BasicStringCore&& other) noexcept
        : alloc_(std::move(other.alloc_)) {
        if (LayoutPolicy::is_small(other.storage_)) {
            LayoutPolicy::InitSmall(storage_,
                                    LayoutPolicy::data(other.storage_),
                                    LayoutPolicy::size(other.storage_));
            LayoutPolicy::InitEmpty(other.storage_);
        } else {
            LayoutPolicy::InitExternal(storage_,
                                       LayoutPolicy::data(other.storage_),
                                       LayoutPolicy::size(other.storage_),
                                       LayoutPolicy::capacity(other.storage_));
            LayoutPolicy::InitEmpty(other.storage_);
        }
    }

    ~BasicStringCore() {
        if (LayoutPolicy::is_external(storage_)) {
            pointer ptr = LayoutPolicy::data(storage_);
            const size_type cap = LayoutPolicy::capacity(storage_);
            alloc_helper::traits::deallocate(alloc_, ptr, cap + 1);
        }
    }

    BasicStringCore& operator=(const BasicStringCore& other) {
        if (this == &other) {
            return *this;
        }

        const size_type other_sz = LayoutPolicy::size(other.storage_);
        const size_type other_cap = LayoutPolicy::capacity(other.storage_);
        const_pointer other_data = LayoutPolicy::data(other.storage_);

        if (other_sz <= LayoutPolicy::kSmallCapacity) {
            destroy_heap_if_needed();
            LayoutPolicy::InitSmall(storage_, other_data, other_sz);
        } else {
            pointer new_ptr = alloc_helper::traits::allocate(alloc_, other_cap + 1);
            char_algo::copy(new_ptr, other_data, other_sz);
            new_ptr[other_sz] = char_algo::null_char();

            destroy_heap_if_needed();
            LayoutPolicy::InitExternal(storage_, new_ptr, other_sz, other_cap);
        }

        if (alloc_helper::propagate_on_copy) {
            alloc_ = other.alloc_;
        }

        return *this;
    }

    BasicStringCore& operator=(BasicStringCore&& other) noexcept {
        if (this == &other) {
            return *this;
        }

        destroy_heap_if_needed();

        if (LayoutPolicy::is_small(other.storage_)) {
            LayoutPolicy::InitSmall(storage_,
                                    LayoutPolicy::data(other.storage_),
                                    LayoutPolicy::size(other.storage_));
            LayoutPolicy::InitEmpty(other.storage_);
        } else if (alloc_helper::is_always_equal || alloc_ == other.alloc_) {
            LayoutPolicy::InitExternal(storage_,
                                       LayoutPolicy::data(other.storage_),
                                       LayoutPolicy::size(other.storage_),
                                       LayoutPolicy::capacity(other.storage_));
            LayoutPolicy::InitEmpty(other.storage_);
        } else {
            const size_type sz = LayoutPolicy::size(other.storage_);
            const size_type cap = LayoutPolicy::capacity(other.storage_);
            pointer ptr = alloc_helper::traits::allocate(alloc_, cap + 1);
            char_algo::copy(ptr, LayoutPolicy::data(other.storage_), sz);
            ptr[sz] = char_algo::null_char();
            LayoutPolicy::InitExternal(storage_, ptr, sz, cap);
            other.destroy_heap_if_needed();
            LayoutPolicy::InitEmpty(other.storage_);
        }

        if (alloc_helper::propagate_on_move) {
            alloc_ = std::move(other.alloc_);
        }

        return *this;
    }

    const_pointer data() const noexcept {
        return LayoutPolicy::data(storage_);
    }

    pointer data() noexcept {
        return LayoutPolicy::data(storage_);
    }

    const_pointer c_str() const noexcept {
        return data();
    }

    size_type size() const noexcept {
        return LayoutPolicy::size(storage_);
    }

    size_type capacity() const noexcept {
        return LayoutPolicy::capacity(storage_);
    }

    bool empty() const noexcept {
        return size() == 0;
    }

    bool is_small() const noexcept {
        return LayoutPolicy::is_small(storage_);
    }

    bool is_external() const noexcept {
        return LayoutPolicy::is_external(storage_);
    }

    void clear() noexcept {
        destroy_heap_if_needed();
        LayoutPolicy::InitEmpty(storage_);
    }

    void reserve(size_type new_cap) {
        if (new_cap <= capacity()) {
            return;
        }

        reallocate_and_copy(new_cap);
    }

    void resize(size_type count) {
        resize(count, char_algo::null_char());
    }

    void resize(size_type count, CharT ch) {
        const size_type cur_sz = size();

        if (count < cur_sz) {
            pointer d = data();
            d[count] = char_algo::null_char();
            set_size(count);
        } else if (count > cur_sz) {
            if (count > capacity()) {
                reserve(count);
            }
            pointer d = data();
            char_algo::assign(d + cur_sz, count - cur_sz, ch);
            d[count] = char_algo::null_char();
            set_size(count);
        }
    }

    void shrink_to_fit() {
        const size_type sz = size();
        const size_type cap = capacity();

        if (cap <= LayoutPolicy::kSmallCapacity || sz == cap) {
            return;
        }

        if (sz <= LayoutPolicy::kSmallCapacity) {
            pointer old_ptr = LayoutPolicy::data(storage_);
            const size_type old_cap = cap;

            LayoutPolicy::InitSmall(storage_, old_ptr, sz);
            alloc_helper::traits::deallocate(alloc_, old_ptr, old_cap + 1);
        } else {
            reallocate_and_copy(sz);
        }
    }

    void assign(const CharT* s, size_type n) {
        if (n <= LayoutPolicy::kSmallCapacity) {
            destroy_heap_if_needed();
            LayoutPolicy::InitSmall(storage_, s, n);
        } else {
            const size_type req_cap = GrowthPolicy::min_heap_capacity(n);
            if (req_cap > capacity()) {
                destroy_heap_if_needed();
                pointer ptr = alloc_helper::traits::allocate(alloc_, req_cap + 1);
                char_algo::copy(ptr, s, n);
                ptr[n] = char_algo::null_char();
                LayoutPolicy::InitExternal(storage_, ptr, n, req_cap);
            } else {
                char_algo::copy(data(), s, n);
                data()[n] = char_algo::null_char();
                set_size(n);
            }
        }
    }

    void append(const CharT* s, size_type n) {
        if (n == 0) {
            return;
        }

        const size_type cur_sz = size();
        const size_type new_sz = cur_sz + n;

        if (new_sz > capacity()) {
            const size_type new_cap = GrowthPolicy::next_capacity(capacity(), new_sz);
            pointer new_ptr = alloc_helper::traits::allocate(alloc_, new_cap + 1);
            char_algo::copy(new_ptr, data(), cur_sz);
            char_algo::copy(new_ptr + cur_sz, s, n);
            new_ptr[new_sz] = char_algo::null_char();

            destroy_heap_if_needed();
            LayoutPolicy::InitExternal(storage_, new_ptr, new_sz, new_cap);
        } else {
            char_algo::copy(data() + cur_sz, s, n);
            data()[new_sz] = char_algo::null_char();
            set_size(new_sz);
        }
    }

    void swap(BasicStringCore& other) noexcept {
        using std::swap;
        swap(storage_, other.storage_);
        if (alloc_helper::propagate_on_swap) {
            swap(alloc_, other.alloc_);
        }
    }

    void check_invariants() const noexcept {
        LayoutPolicy::CheckInvariants(storage_);
    }

private:
    void destroy_heap_if_needed() noexcept {
        if (LayoutPolicy::is_external(storage_)) {
            pointer ptr = LayoutPolicy::data(storage_);
            const size_type cap = LayoutPolicy::capacity(storage_);
            alloc_helper::traits::deallocate(alloc_, ptr, cap + 1);
            LayoutPolicy::InitEmpty(storage_);
        }
    }

    void reallocate_and_copy(size_type new_cap) {
        const size_type sz = size();
        pointer new_ptr = alloc_helper::traits::allocate(alloc_, new_cap + 1);
        char_algo::copy(new_ptr, data(), sz);
        new_ptr[sz] = char_algo::null_char();

        destroy_heap_if_needed();
        LayoutPolicy::InitExternal(storage_, new_ptr, sz, new_cap);
    }

    void set_size(size_type new_size) noexcept {
        if (LayoutPolicy::is_small(storage_)) {
            LayoutPolicy::SetSmallSize(storage_, new_size);
        } else {
            LayoutPolicy::SetExternalSize(storage_, new_size);
        }
    }

    allocator_type& alloc_ref() noexcept { return alloc_; }
    const allocator_type& alloc_ref() const noexcept { return alloc_; }

    Storage storage_;
    [[no_unique_address]] allocator_type alloc_;
};

}// namespace aethermind

#endif// AETHERMIND_AMSTRING_CORE_HPP
