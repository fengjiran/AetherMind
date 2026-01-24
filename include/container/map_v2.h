//
// Created by richard on 1/21/26.
//

#ifndef AETHERMIND_CONTAINER_MAP_V2_H
#define AETHERMIND_CONTAINER_MAP_V2_H

#include "any_utils.h"
#include "error.h"
#include "object.h"
#include "utils/hash.h"

#include <concepts>
#include <tuple>

namespace aethermind {

template<typename value_type, uint8_t BlockSize = MapMagicConstants::kSlotsPerBlock>
class HashTableBlock : public Object {
public:
    HashTableBlock() = default;
    ~HashTableBlock() override {
        for (size_t i = 0; i < BlockSize; ++i) {
            destroy(i);
        }
    }

    HashTableBlock(const HashTableBlock& other) noexcept(std::is_copy_constructible_v<value_type>)
        : storage_(other.storage_) {
        for (size_t i = 0; i < BlockSize; ++i) {
            if (other.IsConstructed(i)) {
                new (GetDataPtr(i)) value_type(*other.GetDataPtr(i));
            }
        }
    }


    HashTableBlock(HashTableBlock&&) = delete;
    HashTableBlock& operator=(const HashTableBlock&) = delete;
    HashTableBlock& operator=(HashTableBlock&&) = delete;

    const value_type* GetDataPtr(size_t slot_idx) const noexcept {
        return const_cast<HashTableBlock*>(this)->GetDataPtr(slot_idx);
    }

    value_type* GetDataPtr(size_t slot_idx) noexcept {
        AM_CHECK(slot_idx < BlockSize);
        auto* start = storage_.data() + FLAG_BYTES;
        return reinterpret_cast<value_type*>(start) + slot_idx;
    }

    // inplace construction, only run once when insert KV
    template<typename... Args>
    void emplace(size_t slot_idx, Args&&... args) noexcept(std::is_nothrow_constructible_v<value_type, Args...>) {
        destroy(slot_idx);
        new (GetDataPtr(slot_idx)) value_type(std::forward<Args>(args)...);
        SetFlag(slot_idx);
    }

    // destroy KV
    void destroy(size_t slot_idx) noexcept(std::is_nothrow_destructible_v<value_type>) {
        if (TestFlag(slot_idx)) {
            GetDataPtr(slot_idx)->~value_type();
            ResetFlag(slot_idx);
        }
    }

    NODISCARD bool IsConstructed(size_t slot_idx) const noexcept {
        return TestFlag(slot_idx);
    }

    NODISCARD bool IsUnConstructed(size_t slot_idx) const noexcept {
        return !TestFlag(slot_idx);
    }

private:
    static constexpr size_t PAIR_SIZE = sizeof(value_type);
    static constexpr size_t PAIR_ALIGN = alignof(value_type);
    static constexpr size_t FLAG_BYTES_RAW = (BlockSize + 7) / 8;
    static constexpr size_t FLAG_BYTES = (FLAG_BYTES_RAW + PAIR_ALIGN - 1) / PAIR_ALIGN * PAIR_ALIGN;
    static constexpr size_t TOTAL_SIZE = FLAG_BYTES + PAIR_SIZE * BlockSize;

    std::array<std::byte, TOTAL_SIZE> storage_{};

    NODISCARD static std::pair<size_t, size_t> GetFlagIdx(size_t slot_idx) noexcept {
        return {slot_idx / 8, slot_idx & 7};// slot_idx % 8
    }

    // test the slot is constructed
    // 0: Not construct
    // 1: Constructed
    NODISCARD bool TestFlag(size_t slot_idx) const noexcept {
        AM_CHECK(slot_idx < BlockSize);
        auto [b, p] = GetFlagIdx(slot_idx);
        return (storage_[b] & std::byte{1} << (7 - p)) != std::byte{0};
    }

    // Set the slot is constructed
    void SetFlag(size_t slot_idx) noexcept {
        AM_CHECK(slot_idx < BlockSize);
        auto [b, p] = GetFlagIdx(slot_idx);
        storage_[b] |= std::byte{1} << (7 - p);
    }

    // Set the slot is not constructed
    void ResetFlag(size_t slot_idx) noexcept {
        AM_CHECK(slot_idx < BlockSize);
        auto [b, p] = GetFlagIdx(slot_idx);
        storage_[b] &= ~(std::byte{1} << (7 - p));
    }
};

struct SlotInfo {
    std::byte meta;
    size_t prev;
    size_t next;

    SlotInfo() : meta(MapMagicConstants::kEmptySlot),
                 prev(MapMagicConstants::kInvalidIndex),
                 next(MapMagicConstants::kInvalidIndex) {}
};

template<typename K, typename V, typename Hasher = hash<K>>
class MapImplV2 : public Object {
public:
    using key_type = K;
    using mapped_type = V;
    using value_type = std::pair<const key_type, mapped_type>;
    using hasher = Hasher;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using reference = value_type&;
    using const_reference = const value_type&;
    using Constants = MapMagicConstants;

    // IteratorImpl is a base class for iterator and const_iterator
    template<bool IsConst>
    class Iterator;

    using iterator = Iterator<false>;
    using const_iterator = Iterator<true>;

    using Block = HashTableBlock<value_type>;
    struct Cursor;

    MapImplV2() = default;

    explicit MapImplV2(size_type n) {
        auto [fib_shift, slots] = CalculateSlotCount(n);
        const size_type block_num = CalculateBlockCount(slots);
        blocks_.reserve(block_num);
        for (size_type i = 0; i < block_num; ++i) {
            blocks_.push_back(make_object<Block>());
        }

        slot_infos_.resize(slots);
        slots_ = slots;
        fib_shift_ = fib_shift;
    }

    ~MapImplV2() override {
        reset();
    }

    MapImplV2(const MapImplV2&) = default;
    MapImplV2(MapImplV2&& other) noexcept = default;
    MapImplV2& operator=(const MapImplV2& other) {
        MapImplV2(other).swap(*this);
        return *this;
    }

    MapImplV2& operator=(MapImplV2&& other) noexcept {
        MapImplV2(std::move(other)).swap(*this);
        return *this;
    }

    NODISCARD size_type size() const noexcept {
        return size_;
    }

    NODISCARD size_type slots() const noexcept {
        return slots_;
    }

    iterator begin() {
        return {iter_list_head_, this};
    }

    iterator end() {
        return {Constants::kInvalidIndex, this};
    }

    const_iterator begin() const {
        return const_cast<MapImplV2*>(this)->begin();
    }

    const_iterator end() const {
        return const_cast<MapImplV2*>(this)->end();
    }

    iterator find(const key_type& key);

    const_iterator find(const key_type& key) const {
        return const_cast<MapImplV2*>(this)->find(key);
    }

    NODISCARD size_type count(const key_type& key) const {
        return find(key) != end();
    }

    // may be rehash
    std::pair<iterator, bool> insert(const value_type& value) {
        return emplace(value);
    }

    std::pair<iterator, bool> insert(value_type&& value) {
        return emplace(std::move(value));
    }

    template<typename Pair, typename... Args>
    std::pair<iterator, bool> emplace(Pair&& kv, Args&&... args);

    iterator erase(const_iterator pos);
    iterator erase(const_iterator first, const_iterator last);
    size_type erase(const key_type& key) {
        auto it = find(key);
        if (it != end()) {
            erase(it);
            return 1;
        }
        return 0;
    }

    NODISCARD value_type* GetDataPtr(size_type global_idx) const {
        AM_DCHECK(global_idx < slots_);
        AM_DCHECK(slot_infos_[global_idx].meta != Constants::kEmptySlot);
        auto block_idx = global_idx / Constants::kSlotsPerBlock;
        return blocks_[block_idx]->GetDataPtr(global_idx & Constants::kSlotsPerBlock - 1);
    }

    void reset() {
        size_ = 0;
        slots_ = 0;
        fib_shift_ = 63;
        version_ = 0;
        iter_list_head_ = Constants::kInvalidIndex;
        iter_list_tail_ = Constants::kInvalidIndex;
        std::vector<ObjectPtr<Block>>().swap(blocks_);
        std::vector<SlotInfo>().swap(slot_infos_);
        // compact
        // blocks_.clear();
        // blocks_.shrink_to_fit();
        // slot_infos_.clear();
        // slot_infos_.shrink_to_fit();
    }

    void swap(MapImplV2& other) noexcept {
        std::swap(size_, other.size_);
        std::swap(slots_, other.slots_);
        std::swap(iter_list_head_, other.iter_list_head_);
        std::swap(iter_list_tail_, other.iter_list_tail_);
        std::swap(fib_shift_, other.fib_shift_);
        blocks_.swap(other.blocks_);
        slot_infos_.swap(other.slot_infos_);
    }

    void clear() noexcept {
        reset();
    }

private:
    size_type size_{0};
    size_type slots_{0};
    // iterator version
    mutable size_type version_{0};
    // The head of iterator list
    size_type iter_list_head_{Constants::kInvalidIndex};
    // The tail of iterator list
    size_type iter_list_tail_{Constants::kInvalidIndex};
    // fib shift in Fibonacci hash
    uint32_t fib_shift_{Constants::kDefaultFibShift};
    // blocks ptr vector
    std::vector<ObjectPtr<Block>> blocks_;
    // all slot infos
    std::vector<SlotInfo> slot_infos_;

    // for rehash
    // bool is_rehashing_{false};
    // size_type rehashing_block_idx_{0};
    // std::vector<ObjectPtr<Block>> new_blocks_;
    // std::vector<SlotInfo> new_slot_infos_;

    void BlockCOW(size_type block_idx) {
        auto& block_ptr = blocks_[block_idx];
        if (!block_ptr.unique()) {
            block_ptr = make_object<Block>(*block_ptr);
        }
    }

    void rehash(size_type new_slots) {
        auto tmp = MapImplV2(new_slots);
        for (auto it = begin(); it != end(); ++it) {
            tmp.emplace(*it);
        }
        tmp.swap(*this);
        ++version_;
    }

    void grow() {
        rehash(std::max(static_cast<size_type>(16), slots() * Constants::kIncFactor));
    }

    template<typename... Args>
    std::pair<iterator, bool> emplace_first_attempt(Cursor target, Args&&... args);

    template<typename... Args>
    std::pair<iterator, bool> emplace_new_key(Cursor prev, Args&&... args);

    NODISCARD size_type GetNextIndexOf(size_type global_idx) const {
        if (global_idx == Constants::kInvalidIndex) {
            return global_idx;
        }
        AM_CHECK(global_idx < slots_);
        return slot_infos_[global_idx].next;
    }

    NODISCARD size_type GetPrevIndexOf(size_type global_idx) const {
        if (global_idx == Constants::kInvalidIndex) {
            return iter_list_tail_;
        }
        AM_CHECK(global_idx < slots_);
        return slot_infos_[global_idx].prev;
    }

    void IterListPushBack(size_type global_idx) {
        auto& slot = slot_infos_[global_idx];
        slot.prev = iter_list_tail_;
        slot.next = Constants::kInvalidIndex;

        if (iter_list_head_ == Constants::kInvalidIndex &&
            iter_list_tail_ == Constants::kInvalidIndex) {
            iter_list_head_ = global_idx;
        } else {
            slot_infos_[iter_list_tail_].next = global_idx;
        }

        iter_list_tail_ = global_idx;
    }

    // Remove the entry from iterator list.
    // This function is usually used before deletion,
    // and it does not change data content of the node.
    void IterListRemove(size_type global_idx) {
        // head
        auto& cur_slot = slot_infos_[global_idx];
        if (global_idx == iter_list_head_) {
            iter_list_head_ = cur_slot.next;
        } else {
            slot_infos_[cur_slot.prev].next = cur_slot.next;
        }

        // tail
        if (global_idx == iter_list_tail_) {
            iter_list_tail_ = cur_slot.prev;
        } else {
            slot_infos_[cur_slot.next].prev = cur_slot.prev;
        }
    }

    /*!
   * \brief Replace node src by dst in the iter list
   * \param src The source node
   * \param dst The destination node, must be empty
   * \note This function does not change data content of the nodes,
   *       which needs to be updated by the caller.
   */
    void IterListReplace(size_type src, size_type dst) {
        auto& src_slot = slot_infos_[src];
        auto& dst_slot = slot_infos_[dst];

        dst_slot.prev = src_slot.prev;
        dst_slot.next = src_slot.next;

        if (src == iter_list_head_) {
            iter_list_head_ = dst;
        } else {
            slot_infos_[src_slot.prev].next = dst;
        }

        if (src == iter_list_tail_) {
            iter_list_tail_ = dst;
        } else {
            slot_infos_[src_slot.next].prev = dst;
        }
    }

    NODISCARD Cursor CreateCursorFromHash(size_t hash_value) const {
        return {details::FibonacciHash(hash_value, fib_shift_), this};
    }

    // Whether the hash table is full.
    NODISCARD bool IsFull() const {
        return size() + 1 > static_cast<size_type>(static_cast<double>(slots()) * Constants::kMaxLoadFactor);
    }

    // Calculate the power-of-2 table size given the lower-bound of required capacity.
    // shift = 64 - log2(slots)
    static std::pair<uint32_t, size_type> CalculateSlotCount(size_type cap) {
        uint32_t shift = 64;
        size_t slots = 1;
        if (cap == 1) {
            return {shift, slots};
        }

        size_t c = cap - 1;
        while (c > 0) {
            --shift;
            slots <<= 1;
            c >>= 1;
        }
        AM_CHECK(slots >= cap);
        return {shift, slots};
    }

    static size_type CalculateBlockCount(size_type total_slots) {
        return (total_slots + Constants::kSlotsPerBlock - 1) / Constants::kSlotsPerBlock;
    }
};

template<typename K, typename V, typename Hasher>
struct MapImplV2<K, V, Hasher>::Cursor {
    Cursor() : global_idx_(0), owner_(nullptr) {}

    Cursor(size_t index, const MapImplV2* p) : global_idx_(index), owner_(p) {}

    NODISCARD size_t index() const {
        return global_idx_;
    }

    NODISCARD const MapImplV2* owner() const {
        return owner_;
    }

    void reset() {
        global_idx_ = 0;
        owner_ = nullptr;
    }

    NODISCARD std::byte& GetSlotMetadata() const {
        AM_DCHECK(!IsNone(), "The Cursor is none.");
        return const_cast<MapImplV2*>(owner())->slot_infos_[global_idx_].meta;
        // return owner()->slot_infos_[global_idx_].meta;
    }

    NODISCARD value_type& GetData() const {
        AM_DCHECK(!IsNone(), "The Cursor is none.");
        return *owner()->GetDataPtr(global_idx_);
    }

    NODISCARD const key_type& GetKey() const {
        return GetData().first;
    }

    NODISCARD mapped_type& GetValue() const {
        return GetData().second;
    }

    NODISCARD uint8_t GetOffsetIdx() const {
        return std::to_integer<uint8_t>(GetSlotMetadata() & Constants::kOffsetIdxMask);
    }

    NODISCARD bool IsIterListHead() const {
        AM_DCHECK(!IsNone(), "The Cursor is none.");
        return index() == owner()->iter_list_head_;
    }

    NODISCARD bool IsIterListTail() const {
        AM_DCHECK(!IsNone(), "The Cursor is none.");
        return index() == owner()->iter_list_tail_;
    }

    NODISCARD bool IsSlotEmpty() const {
        return GetSlotMetadata() == Constants::kEmptySlot;
    }

    NODISCARD bool IsSlotTombStone() const {
        return GetSlotMetadata() == Constants::kTombStoneSlot;
    }

    NODISCARD bool IsSlotHead() const {
        return (GetSlotMetadata() & Constants::kHeadFlagMask) == Constants::kHeadFlag;
    }

    NODISCARD bool IsSlotAlive() const {
        return !(IsSlotEmpty() || IsSlotTombStone());
    }

    void MarkSlotAsEmpty() const {
        GetSlotMetadata() = Constants::kEmptySlot;
    }

    void MarkSlotAsTombStone() const {
        GetSlotMetadata() = Constants::kTombStoneSlot;
    }

    template<typename... Args>
    void ConstructData(Args&&... args) const {
        AM_DCHECK(!IsNone(), "The Cursor is none.");
        AM_DCHECK(IsSlotEmpty() || IsSlotTombStone());
        const ObjectPtr<Block>& block = owner()->blocks_[global_idx_ / Constants::kSlotsPerBlock];
        block->emplace(global_idx_ & Constants::kSlotsPerBlock - 1, std::forward<Args>(args)...);
    }

    void DestroyData() const {
        AM_DCHECK(!IsNone(), "The Cursor is none.");
        const ObjectPtr<Block>& block = owner()->blocks_[global_idx_ / Constants::kSlotsPerBlock];
        block->destroy(global_idx_ & Constants::kSlotsPerBlock - 1);
    }

    // Set the entry's offset to its next entry.
    void SetNextSlotOffsetIndex(uint8_t offset_idx) const {
        AM_DCHECK(offset_idx < Constants::kNumOffsetDists);
        (GetSlotMetadata() &= Constants::kHeadFlagMask) |= std::byte{offset_idx};
    }

    // Whether the slot has the next slot on the linked list
    NODISCARD bool HasNextSlot() const {
        const auto idx = std::to_integer<uint8_t>(
                GetSlotMetadata() & Constants::kOffsetIdxMask);
        return Constants::NextProbePosOffset[idx] != 0;
    }

    // Move the current cursor to the next slot on the linked list
    bool MoveToNextSlot(std::optional<std::byte> meta_opt = std::nullopt) {
        std::byte meta = meta_opt ? meta_opt.value() : GetSlotMetadata();
        const auto idx = std::to_integer<uint8_t>(meta & Constants::kOffsetIdxMask);
        const auto offset = Constants::NextProbePosOffset[idx];
        if (offset == 0) {
            reset();
            return false;
        }

        // The probing will go to the next pos and round back to stay within
        // the correct range of the slots.
        // equal to (index_ + offset) % obj()->slots()
        auto t = global_idx_ + offset;
        global_idx_ = t >= owner()->slots() ? t & owner()->slots() - 1 : t;
        return true;
    }

    // Get the prev slot on the linked list
    NODISCARD Cursor FindPrevSlot() const {
        // start from the head of the linked list, which must exist
        auto cur = owner()->CreateCursorFromHash(hasher()(GetKey()));
        auto prev = cur;

        cur.MoveToNextSlot();
        while (index() != cur.index()) {
            prev = cur;
            cur.MoveToNextSlot();
        }

        return prev;
    }

    NODISCARD std::optional<std::pair<uint8_t, Cursor>> GetNextEmptySlot() const {
        for (uint8_t i = 1; i < Constants::kNumOffsetDists; ++i) {
            if (Cursor candidate(
                        index() + (Constants::NextProbePosOffset[i] & owner()->slots() - 1), owner());
                candidate.IsSlotEmpty() || candidate.IsSlotTombStone()) {
                return std::make_pair(i, candidate);
            }
        }
        return std::nullopt;
    }

    NODISCARD bool IsNone() const {
        return owner() == nullptr;
    }

private:
    // Index of entry on the array
    size_t global_idx_;
    // Pointer to the current MapImpl
    const MapImplV2* owner_;
};

template<typename K, typename V, typename Hasher>
template<bool IsConst>
class MapImplV2<K, V, Hasher>::Iterator {
public:
    using iterator_category = std::bidirectional_iterator_tag;
    using ContainerPtrType = std::conditional_t<IsConst, const MapImplV2*, MapImplV2*>;
    using value_type = MapImplV2<K, V, Hasher>::value_type;
    using pointer = std::conditional_t<IsConst, const value_type*, value_type*>;
    using reference = std::conditional_t<IsConst, const value_type&, value_type&>;
    using difference_type = std::ptrdiff_t;

    Iterator() noexcept : index_(0), version_(0), ptr_(nullptr) {}
    Iterator(size_type index, ContainerPtrType ptr) noexcept
        : index_(index), version_(ptr->version_), ptr_(ptr) {}

    // iterator can convert to const_iterator
    template<bool AlwaysFalse>
        requires(IsConst && !AlwaysFalse)
    Iterator(const Iterator<AlwaysFalse>& other) : index_(other.index()), version_(other.version()), ptr_(other.ptr()) {}//NOLINT

    NODISCARD size_type index() const noexcept {
        return index_;
    }

    NODISCARD ContainerPtrType ptr() const noexcept {
        return ptr_;
    }

    NODISCARD size_type version() const noexcept {
        return version_;
    }

    pointer operator->() const {
        Check();
        return ptr()->GetDataPtr(index_);
    }

    reference operator*() const {
        return *operator->();
    }

    Iterator& operator++() {
        Check();
        index_ = ptr()->GetNextIndexOf(index_);
        return *this;
    }

    Iterator& operator--() {
        Check();
        index_ = ptr()->GetPrevIndexOf(index_);
        return *this;
    }

    Iterator operator++(int) {
        Iterator tmp = *this;
        operator++();
        return tmp;
    }

    Iterator operator--(int) {
        Iterator tmp = *this;
        operator--();
        return tmp;
    }

    Iterator& operator+=(difference_type offset) {
        Check();
        if (offset < 0) {
            return operator-=(static_cast<difference_type>(-offset));
        }

        for (difference_type i = 0; i < offset; ++i) {
            index_ = ptr()->GetNextIndexOf(index_);
            if (index_ == Constants::kInvalidIndex) {
                break;
            }
        }
        return *this;
    }

    Iterator& operator-=(difference_type offset) {
        Check();
        if (offset < 0) {
            return operator+=(static_cast<difference_type>(-offset));
        }

        for (difference_type i = 0; i < offset; ++i) {
            index_ = ptr()->GetPrevIndexOf(index_);
            if (index_ == Constants::kInvalidIndex) {
                break;
            }
        }
        return *this;
    }

    Iterator operator+(difference_type offset) const {
        auto res = *this;
        res += offset;
        return res;
    }

    Iterator operator-(difference_type offset) const {
        auto res = *this;
        res -= offset;
        return res;
    }

    difference_type operator-(const Iterator& other) const {
        return static_cast<difference_type>(index()) - static_cast<difference_type>(other.index());
    }

    bool operator==(const Iterator& other) const {
        return index_ == other.index_ && version_ == other.version_ && ptr_ == other.ptr_;
    }

    bool operator!=(const Iterator& other) const {
        return !(*this == other);
    }


private:
    size_type index_;
    size_type version_;
    ContainerPtrType ptr_;

    void Check() const {
        CHECK(ptr_ != nullptr) << "Iterator pointer is nullptr.";
        CHECK(index_ <= ptr_->slots_) << "Iterator index is out of range.";
        CHECK(version_ == ptr_->version_) << "Iterator invalidated: container modified!";
    }
};

template<typename K, typename V, typename Hasher>
template<typename Pair, typename... Args>
std::pair<typename MapImplV2<K, V, Hasher>::iterator, bool> MapImplV2<K, V, Hasher>::emplace(Pair&& kv, Args&&... args) {
    auto global_idx = details::FibonacciHash(hasher()(kv.first), fib_shift_);
    bool is_first = true;

    while (true) {
        auto& info = slot_infos_[global_idx];
        if (is_first) {
            if ((info.meta & Constants::kHeadFlagMask) != Constants::kHeadFlag) {
                Cursor cur{global_idx, this};
                return emplace_first_attempt(cur, std::forward<Pair>(kv), std::forward<Args>(args)...);
            }
            is_first = false;
        }

        if (kv.first == GetDataPtr(global_idx)->first) {
            return {iterator(global_idx, this), false};
        }

        auto offset_idx = std::to_integer<uint8_t>(info.meta & Constants::kOffsetIdxMask);
        if (offset_idx == 0) {
            Cursor cur{global_idx, this};
            return emplace_new_key(cur, std::forward<Pair>(kv), std::forward<Args>(args)...);
        }

        auto t = global_idx + Constants::NextProbePosOffset[offset_idx];
        global_idx = t >= slots() ? t & slots() - 1 : t;
    }
}

template<typename K, typename V, typename Hasher>
template<typename... Args>
std::pair<typename MapImplV2<K, V, Hasher>::iterator, bool>
MapImplV2<K, V, Hasher>::emplace_first_attempt(Cursor target, Args&&... args) {
    if (IsFull()) {
        grow();
        return emplace(std::forward<Args>(args)...);
    }

    // Case 1: empty or tombstone
    if (target.IsSlotEmpty() || target.IsSlotTombStone()) {
        BlockCOW(target.index() / Constants::kSlotsPerBlock);
        target.ConstructData(std::forward<Args>(args)...);
        target.GetSlotMetadata() = Constants::kHeadFlag;
        IterListPushBack(target.index());
        ++size_;
        return {iterator(target.index(), this), true};
    }

    // Case 2: body of an irrelevant list
    // `target` is not the head of the linked list
    // move the original item of `target`
    // and construct new item on the position `target`
    // To make `target` empty, we
    // 1) find the previous element of `target` in the linked list
    // 2) move the linked list starting from `target`
    Cursor r = target;
    Cursor prev = target.FindPrevSlot();
    // after `target` is moved, we disallow writing to the slot
    bool is_first = true;
    std::byte r_meta;

    do {
        const auto empty_slot_info = prev.GetNextEmptySlot();
        if (!empty_slot_info) {
            grow();
            return emplace(std::forward<Args>(args)...);
        }

        BlockCOW(r.index() / Constants::kSlotsPerBlock);

        auto [offset_idx, empty] = empty_slot_info.value();
        BlockCOW(empty.index() / Constants::kSlotsPerBlock);
        // move `r` to `empty`, first move the data
        empty.ConstructData(std::move(r.GetData()));
        empty.GetSlotMetadata() = Constants::kTailFlag;
        // then move link list chain of r to empty
        // this needs to happen after NewTail so empty's prev/next get updated
        IterListReplace(r.index(), empty.index());
        r_meta = r.GetSlotMetadata();
        r.MarkSlotAsEmpty();
        if (is_first) {
            is_first = false;
            // the target must be marked as alive
            r.GetSlotMetadata() = r_meta;
        }
        // link `prev` to `empty`, and move forward
        prev.SetNextSlotOffsetIndex(offset_idx);
        prev = empty;
    } while (r.MoveToNextSlot(r_meta));

    target.ConstructData(std::forward<Args>(args)...);
    target.GetSlotMetadata() = Constants::kHeadFlag;
    IterListPushBack(target.index());
    ++size_;
    ++version_;
    return {iterator(target.index(), this), true};
}

template<typename K, typename V, typename Hasher>
template<typename... Args>
std::pair<typename MapImplV2<K, V, Hasher>::iterator, bool>
MapImplV2<K, V, Hasher>::emplace_new_key(Cursor prev, Args&&... args) {
    if (IsFull()) {
        grow();
        return emplace(std::forward<Args>(args)...);
    }

    const auto empty_slot_info = prev.GetNextEmptySlot();
    if (!empty_slot_info) {
        grow();
        return emplace(std::forward<Args>(args)...);
    }

    auto [offset_idx, empty] = empty_slot_info.value();
    BlockCOW(empty.index() / Constants::kSlotsPerBlock);
    empty.ConstructData(std::forward<Args>(args)...);
    empty.GetSlotMetadata() = Constants::kTailFlag;
    IterListPushBack(empty.index());
    prev.SetNextSlotOffsetIndex(offset_idx);
    ++size_;
    ++version_;
    return {iterator(empty.index(), this), true};
}

template<typename K, typename V, typename Hasher>
MapImplV2<K, V, Hasher>::iterator MapImplV2<K, V, Hasher>::erase(const_iterator pos) {
    if (pos == end()) {
        return end();
    }

    Cursor cur{pos.index(), this};
    if (cur.IsSlotEmpty() || cur.IsSlotTombStone()) {
        return end();
    }

    auto next_pos = pos + 1;
    if (cur.HasNextSlot()) {
        Cursor prev = cur;
        Cursor last = cur;
        last.MoveToNextSlot();
        while (last.HasNextSlot()) {
            prev = last;
            last.MoveToNextSlot();
        }

        IterListRemove(cur.index());
        IterListReplace(last.index(), cur.index());
        BlockCOW(cur.index() / Constants::kSlotsPerBlock);
        cur.ConstructData(last.GetData());
        last.MarkSlotAsTombStone();
        prev.SetNextSlotOffsetIndex(0);
    } else {// the last node
        if (!cur.IsSlotHead()) {
            // cut the link if there is any
            cur.FindPrevSlot().SetNextSlotOffsetIndex(0);
        }
        // unlink the node from iterator list
        IterListRemove(cur.index());
        cur.MarkSlotAsTombStone();
    }

    --size_;
    ++version_;
    return {next_pos.index(), this};
}

template<typename K, typename V, typename Hasher>
MapImplV2<K, V, Hasher>::iterator
MapImplV2<K, V, Hasher>::erase(const_iterator first, const_iterator last) {
    if (first == last) {
        return {first.index(), this};
    }

    if (first + 1 == last) {
        return erase(first);
    }

    if (first == begin() && last == end()) {
        clear();
        return end();
    }

    std::vector<std::pair<int, Cursor>> depth_in_chain;
    for (auto it = first; it != last; ++it) {
        Cursor cur{it.index(), it.ptr()};
        if (cur.IsSlotHead()) {
            depth_in_chain.emplace(0, cur);
        } else {
            Cursor root = CreateCursorFromHash(hasher()(cur.GetKey()));
            int depth = 0;
            while (root.MoveToNextSlot()) {
                ++depth;
                if (root == cur) {
                    break;
                }
            }
            depth_in_chain.emplace(depth, cur);
        }
    }

    std::sort(depth_in_chain.begin(), depth_in_chain.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    iterator res;
    for (auto it = depth_in_chain.rbegin(); it != depth_in_chain.rend(); ++it) {
        const_iterator pos{it.index(), it.ptr()};
        res = erase(pos);
    }

    return res;
}

template<typename K, typename V, typename Hasher>
MapImplV2<K, V, Hasher>::iterator MapImplV2<K, V, Hasher>::find(const key_type& key) {
    auto global_idx = details::FibonacciHash(hasher()(key), fib_shift_);
    bool is_first = true;
    while (true) {
        auto& info = slot_infos_[global_idx];
        if (is_first) {
            if ((info.meta & Constants::kHeadFlagMask) != Constants::kHeadFlag) {
                return end();
            }
            is_first = false;
        }

        if (key == GetDataPtr(global_idx)->first) {
            return {global_idx, this};
        }

        auto offset_idx = std::to_integer<uint8_t>(info.meta & Constants::kOffsetIdxMask);
        if (offset_idx == 0) {
            return end();
        }

        auto t = global_idx + Constants::NextProbePosOffset[offset_idx];
        global_idx = t >= slots() ? t & slots() - 1 : t;
    }
}


}// namespace aethermind

#endif//AETHERMIND_CONTAINER_MAP_V2_H
