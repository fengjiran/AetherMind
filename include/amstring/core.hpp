// core.hpp - basic_string_core implementation skeleton
// Part of AetherMind project, licensed under MIT License.
// See LICENSE.txt for details.
// SPDX-License-Identifier: MIT

#ifndef AETHERMIND_AMSTRING_CORE_HPP
#define AETHERMIND_AMSTRING_CORE_HPP

#include "config.hpp"
#include "growth_policy.hpp"
#include "layout_policy.hpp"
#include "stable_layout_policy.hpp"
#include "allocator_support.hpp"
#include "char_algorithms.hpp"
#include "invariant.hpp"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace aethermind {

// basic_string_core - unified algorithm and lifecycle skeleton
// Combines LayoutPolicy, GrowthPolicy, and Allocator to implement string semantics
template<
    typename CharT,
    typename Traits = std::char_traits<CharT>,
    typename Allocator = std::allocator<CharT>,
    typename LayoutPolicy = stable_layout_policy<CharT>,
    typename GrowthPolicy = default_growth_policy
>
class basic_string_core {
public:
    using value_type = CharT;
    using traits_type = Traits;
    using allocator_type = Allocator;
    using layout_policy_type = LayoutPolicy;
    using growth_policy_type = GrowthPolicy;

    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = CharT*;
    using const_pointer = const CharT*;

    using storage_type = typename LayoutPolicy::storage_type;
    using alloc_helper = allocator_traits_helper<Allocator>;
    using char_algo = char_algorithms<CharT, Traits>;

    static constexpr size_type npos = static_cast<size_type>(-1);

public:
    basic_string_core() noexcept {
        LayoutPolicy::init_empty(storage_);
    }

    explicit basic_string_core(const allocator_type& a) noexcept
        : alloc_(a) {
        LayoutPolicy::init_empty(storage_);
    }

    basic_string_core(const CharT* s, size_type n,
                      const allocator_type& a = allocator_type{})
        : alloc_(a) {
        if (n <= LayoutPolicy::kSmallCapacity) {
            LayoutPolicy::init_small(storage_, s, n);
        } else {
            const size_type cap = GrowthPolicy::min_heap_capacity(n);
            pointer ptr = alloc_helper::traits::allocate(alloc_, cap + 1);
            char_algo::copy(ptr, s, n);
            ptr[n] = char_algo::null_char();
            LayoutPolicy::init_heap(storage_, ptr, n, cap);
        }
    }

    basic_string_core(const basic_string_core& other)
        : alloc_(alloc_helper::select_on_copy(other.alloc_)) {
        if (LayoutPolicy::is_small(other.storage_)) {
            LayoutPolicy::init_small(storage_,
                LayoutPolicy::data(other.storage_),
                LayoutPolicy::size(other.storage_));
        } else {
            const size_type sz = LayoutPolicy::size(other.storage_);
            const size_type cap = LayoutPolicy::capacity(other.storage_);
            pointer ptr = alloc_helper::traits::allocate(alloc_, cap + 1);
            char_algo::copy(ptr, LayoutPolicy::data(other.storage_), sz);
            ptr[sz] = char_algo::null_char();
            LayoutPolicy::init_heap(storage_, ptr, sz, cap);
        }
    }

    basic_string_core(basic_string_core&& other) noexcept
        : alloc_(std::move(other.alloc_)) {
        if (LayoutPolicy::is_small(other.storage_)) {
            LayoutPolicy::init_small(storage_,
                LayoutPolicy::data(other.storage_),
                LayoutPolicy::size(other.storage_));
            LayoutPolicy::init_empty(other.storage_);
        } else {
            LayoutPolicy::init_heap(storage_,
                LayoutPolicy::heap_ptr(other.storage_),
                LayoutPolicy::size(other.storage_),
                LayoutPolicy::capacity(other.storage_));
            LayoutPolicy::destroy_heap(other.storage_);
            LayoutPolicy::init_empty(other.storage_);
        }
    }

    ~basic_string_core() {
        if (LayoutPolicy::is_heap(storage_)) {
            pointer ptr = LayoutPolicy::heap_ptr(storage_);
            const size_type cap = LayoutPolicy::capacity(storage_);
            alloc_helper::traits::deallocate(alloc_, ptr, cap + 1);
        }
    }

    basic_string_core& operator=(const basic_string_core& other) {
        if (this == &other) {
            return *this;
        }

        const size_type other_sz = LayoutPolicy::size(other.storage_);
        const size_type other_cap = LayoutPolicy::capacity(other.storage_);
        const_pointer other_data = LayoutPolicy::data(other.storage_);

        if (other_sz <= LayoutPolicy::kSmallCapacity) {
            destroy_heap_if_needed();
            LayoutPolicy::init_small(storage_, other_data, other_sz);
        } else {
            pointer new_ptr = alloc_helper::traits::allocate(alloc_, other_cap + 1);
            char_algo::copy(new_ptr, other_data, other_sz);
            new_ptr[other_sz] = char_algo::null_char();

            destroy_heap_if_needed();
            LayoutPolicy::init_heap(storage_, new_ptr, other_sz, other_cap);
        }

        if (alloc_helper::propagate_on_copy) {
            alloc_ = other.alloc_;
        }

        return *this;
    }

    basic_string_core& operator=(basic_string_core&& other) noexcept {
        if (this == &other) {
            return *this;
        }

        destroy_heap_if_needed();

        if (LayoutPolicy::is_small(other.storage_)) {
            LayoutPolicy::init_small(storage_,
                LayoutPolicy::data(other.storage_),
                LayoutPolicy::size(other.storage_));
            LayoutPolicy::init_empty(other.storage_);
        } else if (alloc_helper::is_always_equal || alloc_ == other.alloc_) {
            LayoutPolicy::init_heap(storage_,
                LayoutPolicy::heap_ptr(other.storage_),
                LayoutPolicy::size(other.storage_),
                LayoutPolicy::capacity(other.storage_));
            LayoutPolicy::destroy_heap(other.storage_);
            LayoutPolicy::init_empty(other.storage_);
        } else {
            const size_type sz = LayoutPolicy::size(other.storage_);
            const size_type cap = LayoutPolicy::capacity(other.storage_);
            pointer ptr = alloc_helper::traits::allocate(alloc_, cap + 1);
            char_algo::copy(ptr, LayoutPolicy::data(other.storage_), sz);
            ptr[sz] = char_algo::null_char();
            LayoutPolicy::init_heap(storage_, ptr, sz, cap);
            other.destroy_heap_if_needed();
            LayoutPolicy::init_empty(other.storage_);
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

    void clear() noexcept {
        destroy_heap_if_needed();
        LayoutPolicy::init_empty(storage_);
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
            LayoutPolicy::set_size(storage_, count);
        } else if (count > cur_sz) {
            if (count > capacity()) {
                reserve(count);
            }
            pointer d = data();
            char_algo::assign(d + cur_sz, count - cur_sz, ch);
            d[count] = char_algo::null_char();
            LayoutPolicy::set_size(storage_, count);
        }
    }

    void shrink_to_fit() {
        const size_type sz = size();
        const size_type cap = capacity();

        if (cap <= LayoutPolicy::kSmallCapacity || sz == cap) {
            return;
        }

        if (sz <= LayoutPolicy::kSmallCapacity) {
            pointer old_ptr = LayoutPolicy::heap_ptr(storage_);
            const size_type old_cap = cap;

            LayoutPolicy::init_small(storage_, old_ptr, sz);
            alloc_helper::traits::deallocate(alloc_, old_ptr, old_cap + 1);
        } else {
            reallocate_and_copy(sz);
        }
    }

    void assign(const CharT* s, size_type n) {
        if (n <= LayoutPolicy::kSmallCapacity) {
            destroy_heap_if_needed();
            LayoutPolicy::init_small(storage_, s, n);
        } else {
            const size_type req_cap = GrowthPolicy::min_heap_capacity(n);
            if (req_cap > capacity()) {
                destroy_heap_if_needed();
                pointer ptr = alloc_helper::traits::allocate(alloc_, req_cap + 1);
                char_algo::copy(ptr, s, n);
                ptr[n] = char_algo::null_char();
                LayoutPolicy::init_heap(storage_, ptr, n, req_cap);
            } else {
                char_algo::copy(data(), s, n);
                data()[n] = char_algo::null_char();
                LayoutPolicy::set_size(storage_, n);
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
            LayoutPolicy::init_heap(storage_, new_ptr, new_sz, new_cap);
        } else {
            char_algo::copy(data() + cur_sz, s, n);
            data()[new_sz] = char_algo::null_char();
            LayoutPolicy::set_size(storage_, new_sz);
        }
    }

    void swap(basic_string_core& other) noexcept {
        using std::swap;
        swap(storage_, other.storage_);
        if (alloc_helper::propagate_on_swap) {
            swap(alloc_, other.alloc_);
        }
    }

    void check_invariants() const noexcept {
        LayoutPolicy::check_invariants(storage_);
    }

private:
    void destroy_heap_if_needed() noexcept {
        if (LayoutPolicy::is_heap(storage_)) {
            pointer ptr = LayoutPolicy::heap_ptr(storage_);
            const size_type cap = LayoutPolicy::capacity(storage_);
            alloc_helper::traits::deallocate(alloc_, ptr, cap + 1);
            LayoutPolicy::destroy_heap(storage_);
        }
    }

    void reallocate_and_copy(size_type new_cap) {
        const size_type sz = size();
        pointer new_ptr = alloc_helper::traits::allocate(alloc_, new_cap + 1);
        char_algo::copy(new_ptr, data(), sz);
        new_ptr[sz] = char_algo::null_char();

        destroy_heap_if_needed();
        LayoutPolicy::init_heap(storage_, new_ptr, sz, new_cap);
    }

    allocator_type& alloc_ref() noexcept { return alloc_; }
    const allocator_type& alloc_ref() const noexcept { return alloc_; }

private:
    storage_type storage_;
    [[no_unique_address]] allocator_type alloc_;
};

}// namespace aethermind

#endif// AETHERMIND_AMSTRING_CORE_HPP