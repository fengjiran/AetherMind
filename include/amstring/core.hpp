// core.hpp - BasicStringCore implementation skeleton
// Part of AetherMind project, licensed under MIT License.
// See LICENSE.txt for details.
// SPDX-License-Identifier: MIT

#ifndef AETHERMIND_AMSTRING_CORE_HPP
#define AETHERMIND_AMSTRING_CORE_HPP

#include "aethermind/utils/overflow_check.h"
#include "allocator_support.hpp"
#include "char_algorithms.hpp"
#include "config.hpp"
#include "error.h"
#include "growth_policy.hpp"
#include "layout_policy.hpp"
#include "macros.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <string_view>
#include <type_traits>
#include <utility>

namespace aethermind {

/// Core implementation for small-string-optimized string storage.
///
/// Combines LayoutPolicy, GrowthPolicy, and Allocator to implement string
/// semantics with Small/Empty/External storage states.
///
/// **Ownership model**:
/// - Owns the heap buffer when in External state.
/// - Buffer is deallocated on destruction, assignment, or shrink_to_fit.
///
/// **Thread-safety**:
/// - Not thread-safe. External synchronization required for concurrent access.
///
/// **Major invariants**:
/// - `data()[size()]` is always the null terminator.
/// - `capacity() >= size()` always holds.
/// - Small capacity strings use inline storage; larger strings use heap.
/// - Empty state uses the inline empty representation (no allocation).
///
/// **Allocator propagation**:
/// - Follows allocator_traits propagation rules for copy/move/swap.
/// - Non-propagating allocators may require reallocation during assignment.
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

    BasicStringCore() noexcept(std::is_nothrow_default_constructible_v<AllocType>) {
        LayoutPolicy::InitEmpty(storage_);
    }

    explicit BasicStringCore(const AllocType& a) noexcept(std::is_nothrow_copy_constructible_v<AllocType>)
        : alloc_(a) {
        LayoutPolicy::InitEmpty(storage_);
    }

    BasicStringCore(const CharT* src, size_type n, const AllocType& a = AllocType{}) : alloc_(a) {
        if (n <= LayoutPolicy::kSmallCapacity) {
            LayoutPolicy::InitSmall(storage_, src, n);
        } else {
            const size_type cap = CheckedMinHeapCapacity(n);
            pointer ptr = alloc_helper::traits::allocate(alloc_, AllocationCountForCapacity(cap));
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
            pointer ptr = alloc_helper::traits::allocate(alloc_, AllocationCountForCapacity(cap));
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
        DestroyHeapIfNeeded();
    }

    BasicStringCore& operator=(const BasicStringCore& other) {
        if (this == &other) {
            return *this;
        }

        if constexpr (alloc_helper::propagate_on_copy_assignment) {
            BasicStringCore tmp(other.data(), other.size(), other.alloc_);
            DestroyHeapIfNeeded();
            LayoutPolicy::InitEmpty(storage_);
            alloc_ = other.alloc_;
            MoveStorageFrom(std::move(tmp));
        } else {
            BasicStringCore(other.data(), other.size(), alloc_).swap(*this);
        }

        CheckInvariants();
        return *this;
    }

    BasicStringCore& operator=(BasicStringCore&& other) noexcept(
            (alloc_helper::propagate_on_move_assignment &&
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
            LayoutPolicy::InitEmpty(storage_);
            alloc_ = std::move(other.alloc_);
            MoveStorageFrom(std::move(other));
        } else if constexpr (alloc_helper::is_always_equal) {
            // Any allocator instance can release memory allocated by any other
            // equivalent instance, so stealing external storage is safe.
            DestroyHeapIfNeeded();
            LayoutPolicy::InitEmpty(storage_);
            MoveStorageFrom(std::move(other));
        } else {
            // Equal non-propagating allocators can release each other's memory.
            if (alloc_ == other.alloc_) {
                DestroyHeapIfNeeded();
                LayoutPolicy::InitEmpty(storage_);
                MoveStorageFrom(std::move(other));
            } else {
                // Unequal non-propagating allocators cannot steal other's
                // external buffer. Reallocate using this->alloc_.
                const auto other_sz = other.size();
                const_pointer other_data = other.data();
                if (other_sz <= LayoutPolicy::kSmallCapacity) {
                    // No allocation needed for destination.
                    DestroyHeapIfNeeded();
                    LayoutPolicy::InitSmall(storage_, other_data, other_sz);

                    // Source must become a valid empty string.
                    // If source was External, release its buffer with source allocator.
                    other.DestroyHeapIfNeeded();
                    LayoutPolicy::InitEmpty(other.storage_);
                } else {
                    const auto new_cap = CheckedMinHeapCapacity(other_sz);

                    pointer new_ptr = alloc_helper::traits::allocate(alloc_, AllocationCountForCapacity(new_cap));
                    try {
                        char_algo::copy(new_ptr, other_data, other_sz);
                        new_ptr[other_sz] = char_algo::null_char();
                    } catch (...) {
                        alloc_helper::traits::deallocate(alloc_, new_ptr, AllocationCountForCapacity(new_cap));
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

    /// Returns a pointer to the null-terminated character array.
    ///
    /// The pointer is valid until the next mutating operation.
    /// For External state, the pointer points to the owned heap buffer.
    /// For Small state, the pointer points to inline storage.
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

    /// Clears content without releasing capacity.
    ///
    /// Postcondition: `size() == 0`, `capacity()` unchanged.
    /// External buffer remains allocated; terminator at `data()[0]`.
    void clear() noexcept {
        if (is_small()) {
            LayoutPolicy::SetSmallSize(storage_, 0);
        } else {
            LayoutPolicy::SetExternalSize(storage_, 0);
        }
    }

    /// Increases capacity to at least `new_cap`.
    ///
    /// Does nothing if `new_cap <= capacity()`.
    /// May reallocate and copy content for External state.
    /// Throws on allocation failure or capacity overflow.
    void reserve(size_type new_cap) {
        if (new_cap <= capacity()) {
            return;
        }

        Reallocate(new_cap);
    }

    /// Resizes to `n` characters.
    ///
    /// If `n < size()`, truncates and maintains terminator.
    /// If `n > size()`, expands and fills new characters with null or `ch`.
    /// May reallocate if `n > capacity()`.
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

    /// Reduces capacity to match size.
    ///
    /// For Small-fit content, moves back to inline storage and releases heap buffer.
    /// For External content, reallocates to `size()` capacity.
    /// For Large External, uses page-rounded capacity instead of exact size
    /// to maintain allocator efficiency.
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
            const size_type target_cap = IsLargeCapacity(cap) || IsLargeCapacity(sz) ? RoundUpCapacityToPage(sz) : sz;
            if (target_cap < cap) {
                Reallocate(target_cap);
            }
        }
    }

    /// Replaces content with `src[0..n)`.
    ///
    /// Uses copy-and-swap for strong exception guarantee.
    /// Existing capacity is released before allocation.
    void assign(const CharT* src, size_type n) {
        BasicStringCore(src, n, alloc_).swap(*this);
    }

    void assign(std::basic_string_view<CharT, Traits> sv) {
        assign(sv.data(), sv.size());
    }

    /// Replaces content with `count` copies of `ch`.
    ///
    /// Uses copy-and-swap for strong exception guarantee.
    void assign(size_type count, CharT ch) {
        BasicStringCore tmp(alloc_);
        tmp.append(count, ch);
        tmp.swap(*this);
    }

    void replace_range(size_type pos, size_type erased, const CharT* src, size_type src_count) {
        const size_type old_size = size();
        AM_CHECK(pos <= old_size);
        AM_CHECK(erased <= old_size - pos);
        AM_CHECK(src_count == 0 || src != nullptr);

        const size_type retained = old_size - erased;
        size_type new_size = 0;
        if (CheckOverflowAdd(retained, src_count, &new_size)) {
            ThrowCapacityError("replace");
        }
        EnsureExternalCapacity(new_size, "replace");

        if (erased == 0 && src_count == 0) {
            return;
        }

        if (src_count == 0) {
            EraseRangeFast(pos, erased);
            return;
        }

        if (!PointerInRange(src, data(), data() + old_size)) {
            if (src_count == erased) {
                ReplaceSameSizeFast(pos, src, src_count);
                return;
            }

            if (src_count < erased) {
                ReplaceShrinkFast(pos, erased, src, src_count);
                return;
            }

            if (new_size <= capacity()) {
                ReplaceGrowInplaceFast(pos, erased, src, src_count);
                return;
            }
        }

        ReplaceReallocateSlow(pos, erased, src, src_count, new_size);
    }

    /// Appends `src[0..n)` to the end.
    ///
    /// Handles self-aliasing safely: if `src` is within `[data(), data() + size())`,
    /// uses move semantics to preserve content before potential reallocation.
    /// Throws on allocation failure or size overflow.
    void append(const CharT* src, size_type n) {
        if (n == 0) {
            return;
        }

        const size_type cur_sz = size();
        const size_type new_sz = CheckedAddSize(cur_sz, n);
        pointer cur_data = data();
        const bool src_is_self_range = PointerInRange(src, cur_data, cur_data + cur_sz);
        const size_type src_offset = src_is_self_range ? static_cast<size_type>(src - cur_data) : 0;

        if (new_sz > capacity()) {
            const size_type new_cap = CheckedNextCapacity(capacity(), new_sz);
            pointer new_ptr = alloc_helper::traits::allocate(alloc_, AllocationCountForCapacity(new_cap));
            char_algo::copy(new_ptr, cur_data, cur_sz);
            const_pointer append_src = src_is_self_range ? new_ptr + src_offset : src;
            char_algo::copy(new_ptr + cur_sz, append_src, n);
            new_ptr[new_sz] = char_algo::null_char();

            DestroyHeapIfNeeded();
            LayoutPolicy::InitExternal(storage_, new_ptr, new_sz, new_cap);
        } else {
            pointer append_dst = cur_data + cur_sz;
            if (src_is_self_range) {
                char_algo::move(append_dst, src, n);
            } else {
                char_algo::copy(append_dst, src, n);
            }
            cur_data[new_sz] = char_algo::null_char();
            SetSize(new_sz);
        }
    }

    void append(std::basic_string_view<CharT, Traits> sv) {
        append(sv.data(), sv.size());
    }

    void append(size_type count, CharT ch) {
        if (count == 0) {
            return;
        }

        const size_type cur_sz = size();
        const size_type new_sz = CheckedAddSize(cur_sz, count);
        if (new_sz > capacity()) {
            Reallocate(CheckedNextCapacity(capacity(), new_sz));
        }

        pointer d = data();
        char_algo::assign(d + cur_sz, count, ch);
        d[new_sz] = char_algo::null_char();
        SetSize(new_sz);
    }

    void push_back(CharT ch) {
        append(&ch, 1);
    }

    /// Removes the last character.
    ///
    /// No-op if `size() == 0`.
    /// Maintains null terminator at `data()[size()]`.
    void pop_back() noexcept {
        const size_type current_size = size();
        if (current_size == 0) {
            return;
        }
        SetSize(current_size - 1);
    }

    /// Exchanges content and storage with `other`.
    ///
    /// noexcept when allocator propagates on swap or is always equal.
    /// For unequal non-propagating allocators, precondition violation triggers AM_CHECK.
    void swap(BasicStringCore& other) noexcept(SwapNoexcept()) {
        if constexpr (alloc_helper::propagate_on_swap) {
            using std::swap;
            swap(alloc_, other.alloc_);
            SwapStorage(storage_, other.storage_);
        } else {
            AM_CHECK(CanSwapStorageWith(other), "BasicStringCore::swap requires compatible allocators");
            SwapStorage(storage_, other.storage_);
        }
        CheckInvariants();
        other.CheckInvariants();
    }

    AllocType get_allocator() const noexcept(std::is_nothrow_copy_constructible_v<AllocType>) {
        return alloc_;
    }

    void CheckInvariants() const noexcept {
        LayoutPolicy::CheckInvariants(storage_);
    }

private:
    static constexpr bool SwapNoexcept() noexcept {
        if constexpr (alloc_helper::propagate_on_swap) {
            return std::is_nothrow_swappable_v<AllocType>;
        } else {
            // No allocation and no allocator swap.
            // Runtime allocator inequality is treated as precondition violation,
            // not as an allocating fallback path.
            return true;
        }
    }

    bool CanSwapStorageWith(const BasicStringCore& other) const noexcept {
        if constexpr (alloc_helper::propagate_on_swap || alloc_helper::is_always_equal) {
            return true;
        } else {
            return alloc_ == other.alloc_;
        }
    }

    static bool PointerInRange(const_pointer ptr, const_pointer begin, const_pointer end) noexcept {
        const std::less<const_pointer> less{};
        return !less(ptr, begin) && less(ptr, end);
    }

    AM_NORETURN static void ThrowCapacityError(const char* operation) {
        AM_THROW(out_of_range) << "BasicStringCore::" << operation << " exceeds maximum encodable capacity";
        AM_UNREACHABLE();
    }

    static size_type CheckedAddSize(size_type lhs, size_type rhs) {
        size_type result = 0;
        if (CheckOverflowAdd(lhs, rhs, &result)) {
            ThrowCapacityError("append");
        }
        EnsureExternalCapacity(result, "append");
        return result;
    }

    static void EnsureExternalCapacity(size_type capacity, const char* operation) {
        if (capacity > LayoutPolicy::max_external_capacity()) {
            ThrowCapacityError(operation);
        }
    }

    static size_type AllocationCountForCapacity(size_type capacity) {
        EnsureExternalCapacity(capacity, "allocate");
        size_type allocation_count = 0;
        if (CheckOverflowAdd(capacity, size_type{1}, &allocation_count)) {
            ThrowCapacityError("allocate");
        }
        return allocation_count;
    }

    static size_type CheckedMinHeapCapacity(size_type required) {
        EnsureExternalCapacity(required, "construct");
        const size_type capacity = GrowthPolicy::MinHeapCapacity(required);
        if (capacity < required) {
            ThrowCapacityError("construct");
        }
        EnsureExternalCapacity(capacity, "construct");
        return capacity;
    }

    static size_type CheckedNextCapacity(size_type old_capacity, size_type required) {
        EnsureExternalCapacity(required, "append");
        size_type capacity = IsLargeCapacity(old_capacity) ? LargeNextCapacity(old_capacity, required)
                                                           : GrowthPolicy::NextCapacity(old_capacity, required);
        if (capacity < required) {
            ThrowCapacityError("append");
        }
        EnsureExternalCapacity(capacity, "append");
        if (IsLargeCapacity(capacity)) {
            capacity = RoundUpCapacityToPage(capacity);
        }
        return capacity;
    }

    static size_type LargeNextCapacity(size_type old_capacity, size_type required) {
        size_type growth = old_capacity / config::kLargeGrowthFactorDenominator;
        if (growth == 0) {
            growth = 1;
        }

        size_type candidate = 0;
        if (CheckOverflowAdd(old_capacity, growth, &candidate)) {
            ThrowCapacityError("append");
        }

        return std::max(required, candidate);
    }

    static constexpr size_type LargeCapacityThreshold() noexcept {
        constexpr size_type kCharsForLargeThreshold = (config::kLargeThresholdBytes + sizeof(CharT) - 1) / sizeof(CharT);
        if constexpr (kCharsForLargeThreshold == 0) {
            return 0;
        } else {
            return kCharsForLargeThreshold - 1;
        }
    }

    static constexpr bool IsLargeCapacity(size_type capacity) noexcept {
        return capacity >= LargeCapacityThreshold();
    }

    static size_type RoundUpCapacityToPage(size_type capacity) {
        static_assert(config::kPageSizeBytes % sizeof(CharT) == 0,
                      "amstring page rounding requires page size to be divisible by CharT size");

        const size_type allocation_count = AllocationCountForCapacity(capacity);
        size_type allocation_bytes = 0;
        if (CheckOverflowMul(allocation_count, sizeof(CharT), &allocation_bytes)) {
            ThrowCapacityError("round_to_page");
        }

        const size_type remainder = allocation_bytes % config::kPageSizeBytes;
        if (remainder == 0) {
            return capacity;
        }

        size_type rounded_bytes = 0;
        if (CheckOverflowAdd(allocation_bytes, config::kPageSizeBytes - remainder, &rounded_bytes)) {
            ThrowCapacityError("round_to_page");
        }

        const size_type rounded_capacity = rounded_bytes / sizeof(CharT) - 1;
        EnsureExternalCapacity(rounded_capacity, "round_to_page");
        return rounded_capacity;
    }

    static void SwapStorage(Storage& lhs, Storage& rhs) noexcept {
        static_assert(std::is_trivially_copyable_v<Storage>,
                      "StorageType must be trivially copyable for raw storage swap.");
        if (std::addressof(lhs) == std::addressof(rhs)) {
            return;
        }

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
        EnsureExternalCapacity(new_cap, "reserve");
        const size_type sz = size();
        AM_CHECK(new_cap >= sz);
        pointer new_ptr = alloc_helper::traits::allocate(alloc_, AllocationCountForCapacity(new_cap));
        char_algo::copy(new_ptr, data(), sz);
        new_ptr[sz] = char_algo::null_char();

        DestroyHeapIfNeeded();
        LayoutPolicy::InitExternal(storage_, new_ptr, sz, new_cap);
    }

    void EraseRangeFast(size_type pos, size_type erased) noexcept {
        if (erased == 0) {
            return;
        }

        pointer d = data();
        const size_type old_size = size();
        const size_type erase_end = pos + erased;
        const size_type tail_with_terminator = old_size - erase_end + 1;
        char_algo::move(d + pos, d + erase_end, tail_with_terminator);
        SetSize(old_size - erased);
    }

    void ReplaceSameSizeFast(size_type pos, const CharT* src, size_type src_count) noexcept {
        if (src_count == 0) {
            return;
        }
        char_algo::copy(data() + pos, src, src_count);
    }

    void ReplaceShrinkFast(size_type pos, size_type erased, const CharT* src, size_type src_count) noexcept {
        pointer d = data();
        const size_type old_size = size();
        const size_type erase_end = pos + erased;
        const size_type new_size = old_size - erased + src_count;

        if (src_count != 0) {
            char_algo::copy(d + pos, src, src_count);
        }

        const size_type tail_with_terminator = old_size - erase_end + 1;
        char_algo::move(d + pos + src_count, d + erase_end, tail_with_terminator);
        SetSize(new_size);
    }

    void ReplaceGrowInplaceFast(size_type pos, size_type erased, const CharT* src, size_type src_count) noexcept {
        pointer d = data();
        const size_type old_size = size();
        const size_type erase_end = pos + erased;
        const size_type new_size = old_size - erased + src_count;

        const size_type tail_with_terminator = old_size - erase_end + 1;
        char_algo::move(d + pos + src_count, d + erase_end, tail_with_terminator);
        char_algo::copy(d + pos, src, src_count);
        SetSize(new_size);
    }

    void ReplaceReallocateSlow(size_type pos, size_type erased, const CharT* src, size_type src_count, size_type new_size) {
        BasicStringCore tmp(alloc_);
        tmp.reserve(new_size);
        tmp.append(data(), pos);
        tmp.append(src, src_count);
        tmp.append(data() + pos + erased, size() - pos - erased);
        tmp.swap(*this);
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
