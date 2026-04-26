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
#include <utility>

namespace aethermind {

// BasicStringCore - unified algorithm and lifecycle skeleton
// Combines LayoutPolicy, GrowthPolicy, and Allocator to implement string semantics
template<typename CharT,
         typename Traits = std::char_traits<CharT>,
         typename Allocator = std::allocator<CharT>,
         typename LayoutPolicy = DefaultLayoutPolicy<CharT>::type,
         typename GrowthPolicy = DefaultGrowthPolicy>
class BasicStringCore {
public:
    using ValueType = CharT;
    // using TraitsType = Traits;
    using AllocType = Allocator;
    using LayoutPolicyType = LayoutPolicy;
    using GrowthPolicyType = GrowthPolicy;

    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = CharT*;
    using const_pointer = const CharT*;

    using Storage = LayoutPolicy::Storage;
    using alloc_helper = AllocatorTraitsHelper<Allocator>;
    using char_algo = CharAlgorithm<CharT, Traits>;

    // static constexpr size_type npos = static_cast<size_type>(-1);

    BasicStringCore() noexcept {
        LayoutPolicy::InitEmpty(storage_);
    }

    explicit BasicStringCore(const AllocType& a) noexcept : alloc_(a) {
        LayoutPolicy::InitEmpty(storage_);
    }

    BasicStringCore(const CharT* src, size_type n, const AllocType& a = AllocType{}) : alloc_(a) {
        if (n <= LayoutPolicy::kSmallCapacity) {
            LayoutPolicy::InitSmall(storage_, src, n);
        } else {
            const size_type cap = GrowthPolicy::MinHeapCapacity(n);
            pointer ptr = alloc_helper::traits::allocate(alloc_, cap + 1);
            char_algo::copy(ptr, src, n);
            ptr[n] = char_algo::null_char();
            LayoutPolicy::InitExternal(storage_, ptr, n, cap);
        }
    }

    BasicStringCore(const BasicStringCore& other) : alloc_(alloc_helper::select_on_copy(other.alloc_)) {
        if (LayoutPolicy::is_small(other.storage_)) {
            LayoutPolicy::InitSmall(storage_, LayoutPolicy::data(other.storage_),
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

    BasicStringCore(BasicStringCore&& other) noexcept(std::is_nothrow_move_constructible_v<Allocator>)
        : alloc_(std::move(other.alloc_)) {
        MoveStorageFrom(std::move(other));
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

        if constexpr (alloc_helper::propagate_on_copy_assignment) {
            BasicStringCore tmp(other);
            DestroyHeapIfNeeded();
            alloc_ = other.alloc_;
            MoveStorageFrom(std::move(tmp));
        } else {
            BasicStringCore(other.data(), other.size(), alloc_).swap(*this);
        }

        CheckInvariants();
        return *this;
    }

    BasicStringCore& operator=(BasicStringCore&& other) noexcept((alloc_helper::propagate_on_move_assignment &&
                                                                  std::is_nothrow_move_assignable_v<Allocator>) ||
                                                                 (!alloc_helper::propagate_on_move_assignment &&
                                                                  alloc_helper::is_always_equal)) {
        if (this == &other) {
            return *this;
        }

        if constexpr (alloc_helper::propagate_on_move_assignment) {
            // Current external storage must be released by the current allocator
            // before allocator_ is replaced.
            DestroyHeapIfNeeded();
            alloc_ = std::move(other.alloc_);
            MoveStorageFrom(std::move(other));
        } else if constexpr (alloc_helper::is_always_equal) {
            // Any allocator instance can release memory allocated by any other
            // equivalent instance, so stealing external storage is safe.
            DestroyHeapIfNeeded();
            MoveStorageFrom(std::move(other));
        } else {
            // Equal non-propagating allocators can release each other's memory.
            if (alloc_ == other.alloc_) {
                DestroyHeapIfNeeded();
                MoveStorageFrom(std::move(other));
            } else {
                // Unequal non-propagating allocators cannot steal other's
                // external buffer. Reallocate using this->alloc_.
                const auto other_sz = other.size();
                const_pointer other_data = other.data();
                if (other_sz <= LayoutPolicy::kSmallCapacity) {
                    // No allocation needed for destination.
                    DestroyHeapIfNeeded();
                    LayoutPolicy::InitSmall(storage_, other_sz, other_data);

                    // Source must become a valid empty string.
                    // If source was External, release its buffer with source allocator.
                    other.DestroyHeapIfNeeded();
                    LayoutPolicy::InitEmpty(other.storage_);
                } else {
                    const auto new_cap = GrowthPolicy::MinHeapCapacity(other_sz);
                    pointer new_ptr = alloc_helper::traits::allocate(alloc_, new_cap + 1);
                    try {
                        char_algo::copy(new_ptr, other_data, other_sz);
                        new_ptr[other_sz] = char_algo::null_char();
                    } catch (...) {
                        alloc_helper::traits::deallocate(alloc_, new_ptr, new_cap + 1);
                        throw;
                    }
                    DestroyHeapIfNeeded();
                    LayoutPolicy::InitExternal(storage_, new_ptr, other_sz, new_cap);
                    // Source cannot be stolen because allocators are unequal and non-propagating.
                    // Release source external buffer with source allocator, then make source empty.
                    other.DestroyHeapIfNeeded();
                    LayoutPolicy::InitEmpty(other.storage_);
                }
            }
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

    AM_NODISCARD size_type size() const noexcept {
        return LayoutPolicy::size(storage_);
    }

    AM_NODISCARD size_type capacity() const noexcept {
        return LayoutPolicy::capacity(storage_);
    }

    AM_NODISCARD bool empty() const noexcept {
        return size() == 0;
    }

    AM_NODISCARD bool is_small() const noexcept {
        return LayoutPolicy::is_small(storage_);
    }

    AM_NODISCARD bool is_external() const noexcept {
        return LayoutPolicy::is_external(storage_);
    }

    void clear() noexcept {
        if (is_small()) {
            LayoutPolicy::SetSmallSize(storage_, 0);
        } else {
            LayoutPolicy::SetExternalSize(storage_, 0);
        }
    }

    void reserve(size_type new_cap) {
        if (new_cap <= capacity()) {
            return;
        }

        Reallocate(new_cap);
    }

    void resize(size_type n) {
        resize(n, char_algo::null_char());
    }

    void resize(size_type n, CharT ch) {
        const auto cur_sz = size();

        if (n < cur_sz) {
            pointer d = data();
            d[n] = char_algo::null_char();
            SetSize(n);
        } else if (n > cur_sz) {
            if (n > capacity()) {
                reserve(n);
            }
            pointer d = data();
            char_algo::assign(d + cur_sz, n - cur_sz, ch);
            d[n] = char_algo::null_char();
            SetSize(n);
        }
    }

    void shrink_to_fit() {
        const auto sz = size();
        const auto cap = capacity();

        if (cap <= LayoutPolicy::kSmallCapacity || sz == cap) {
            return;
        }

        if (sz <= LayoutPolicy::kSmallCapacity) {
            pointer old_ptr = LayoutPolicy::data(storage_);
            const auto old_cap = cap;

            LayoutPolicy::InitSmall(storage_, old_ptr, sz);
            alloc_helper::traits::deallocate(alloc_, old_ptr, old_cap + 1);
        } else {
            Reallocate(sz);
        }
    }

    void assign(const CharT* s, size_type n) {
        if (n <= LayoutPolicy::kSmallCapacity) {
            DestroyHeapIfNeeded();
            LayoutPolicy::InitSmall(storage_, s, n);
        } else {
            const size_type req_cap = GrowthPolicy::MinHeapCapacity(n);
            if (req_cap > capacity()) {
                DestroyHeapIfNeeded();
                pointer ptr = alloc_helper::traits::allocate(alloc_, req_cap + 1);
                char_algo::copy(ptr, s, n);
                ptr[n] = char_algo::null_char();
                LayoutPolicy::InitExternal(storage_, ptr, n, req_cap);
            } else {
                char_algo::copy(data(), s, n);
                data()[n] = char_algo::null_char();
                SetSize(n);
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
            const size_type new_cap = GrowthPolicy::NextCapacity(capacity(), new_sz);
            pointer new_ptr = alloc_helper::traits::allocate(alloc_, new_cap + 1);
            char_algo::copy(new_ptr, data(), cur_sz);
            char_algo::copy(new_ptr + cur_sz, s, n);
            new_ptr[new_sz] = char_algo::null_char();

            DestroyHeapIfNeeded();
            LayoutPolicy::InitExternal(storage_, new_ptr, new_sz, new_cap);
        } else {
            char_algo::copy(data() + cur_sz, s, n);
            data()[new_sz] = char_algo::null_char();
            SetSize(new_sz);
        }
    }

    void swap(BasicStringCore& other) noexcept {
        if (this == &other) {
            return;
        }

        if constexpr (alloc_helper::propagate_on_swap) {
            using std::swap;
            swap(alloc_, other.alloc_);
            SwapStorage(storage_, other.storage_);
        } else if constexpr (alloc_helper::is_always_equal) {
            SwapStorage(storage_, other.storage_);
        } else {
            if (alloc_ == other.alloc_) {
                SwapStorage(storage_, other.storage_);
            } else {
                // Strong guarantee for unequal non-propagating allocator path.
                // Each replacement is built using the destination object's allocator.
                BasicStringCore this_replacement(other.data(), other.size(), alloc_);
                BasicStringCore other_replacement(data(), size(), other.alloc_);
                SwapStorage(storage_, this_replacement.storage_);
                SwapStorage(other.storage_, other_replacement.storage_);
            }
        }
    }

    void CheckInvariants() const noexcept {
        LayoutPolicy::CheckInvariants(storage_);
    }

private:
    static void SwapStorage(Storage& lhs, Storage& rhs) noexcept {
        static_assert(std::is_trivially_copyable_v<Storage>,
                      "StorageType must be trivially copyable for raw storage swap.");
        alignas(Storage) std::byte tmp[sizeof(Storage)];
        std::memcpy(tmp, &lhs, sizeof(Storage));
        std::memcpy(&lhs, &rhs, sizeof(Storage));
        std::memcpy(&rhs, tmp, sizeof(Storage));
    }

    void DestroyHeapIfNeeded() noexcept {
        if (LayoutPolicy::is_external(storage_)) {
            pointer ptr = LayoutPolicy::data(storage_);
            const size_type cap = LayoutPolicy::capacity(storage_);
            alloc_helper::traits::deallocate(alloc_, ptr, cap + 1);
            LayoutPolicy::InitEmpty(storage_);
        }
    }

    void MoveStorageFrom(BasicStringCore&& other) noexcept {
        if (LayoutPolicy::is_small(other.storage_)) {
            LayoutPolicy::InitSmall(storage_, other.data(), other.size());
            LayoutPolicy::InitEmpty(other.storage_);
        } else {
            // External case:
            //
            // Precondition:
            // - this object has no owned external buffer, or it has already been released.
            // - this->alloc_ is allowed to deallocate other's external buffer.
            //
            // This is true for:
            // - move construction, because alloc_ is move-constructed from other.alloc_
            // - move assignment when allocator propagates
            // - move assignment when allocators are equal
            // - move assignment when allocator_traits::is_always_equal is true
            LayoutPolicy::InitExternal(storage_, other.data(), other.size(), other.capacity());
            LayoutPolicy::InitEmpty(other.storage_);
        }
    }

    void Reallocate(size_type new_cap) {
        const size_type sz = size();
        pointer new_ptr = alloc_helper::traits::allocate(alloc_, new_cap + 1);
        char_algo::copy(new_ptr, data(), sz);
        new_ptr[sz] = char_algo::null_char();

        DestroyHeapIfNeeded();
        LayoutPolicy::InitExternal(storage_, new_ptr, sz, new_cap);
    }

    void SetSize(size_type new_size) noexcept {
        if (LayoutPolicy::is_small(storage_)) {
            LayoutPolicy::SetSmallSize(storage_, new_size);
        } else {
            LayoutPolicy::SetExternalSize(storage_, new_size);
        }
    }

    AllocType& alloc_ref() noexcept {
        return alloc_;
    }

    const AllocType& alloc_ref() const noexcept {
        return alloc_;
    }

    Storage storage_;
    [[no_unique_address]] AllocType alloc_;
};

}// namespace aethermind

#endif// AETHERMIND_AMSTRING_CORE_HPP
