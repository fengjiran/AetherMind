//
// Created by richard on 1/13/26.
//

#ifndef AETHERMIND_CONTAINER_MAP_V1_H
#define AETHERMIND_CONTAINER_MAP_V1_H

#include "any_utils.h"
#include "error.h"
#include "object.h"
#include "utils/hash.h"

#include <concepts>
#include <tuple>

namespace aethermind {

template<typename T, uint8_t BlockSize>
struct MapBlock : Object {
    std::array<std::byte, BlockSize + BlockSize * sizeof(T)> storage_;

    MapBlock() {// NOLINT
        for (uint8_t i = 0; i < BlockSize; ++i) {
            storage_[i] = MapMagicConstants::kEmptySlot;
        }
    }

    MapBlock(const MapBlock& other) {// NOLINT
        for (uint8_t i = 0; i < BlockSize; ++i) {
            if (other.storage_[i] != MapMagicConstants::kEmptySlot) {
                storage_[i] = other.storage_[i];
                new (GetEntryPtr(i)) T(*other.GetEntryPtr(i));
            }
        }
    }

    ~MapBlock() override {
        for (uint8_t i = 0; i < BlockSize; ++i) {
            if (storage_[i] != MapMagicConstants::kEmptySlot) {
                storage_[i] = MapMagicConstants::kEmptySlot;
                GetEntryPtr(i)->~Entry();
            }
        }
    }

    T* GetEntryPtr(uint8_t i) {
        return static_cast<T*>(static_cast<void*>(storage_.data() + BlockSize)) + i;
    }

    const T* GetEntryPtr(uint8_t i) const {
        return const_cast<MapBlock*>(this)->GetEntryPtr(i);
    }
};

template<typename K, typename V, typename Hasher>
class MapImpl : public Object {
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

    // IteratorImpl is a base class for iterator and const_iterator
    template<bool IsConst>
    class IteratorImpl;

    using iterator = IteratorImpl<false>;
    using const_iterator = IteratorImpl<true>;

    using Constants = MapMagicConstants;

    struct Entry;
    struct Cursor;
    using Block = MapBlock<Entry, Constants::kSlotsPerBlock>;

    MapImpl() : data_(nullptr), size_(0), slots_(0) {
        // blocks_.push_back(make_object<Block>());
    }

    explicit MapImpl(size_type n);

    MapImpl(const MapImpl&) = default;
    MapImpl(MapImpl&& other) noexcept
        : fib_shift_(other.fib_shift_), iter_list_head_(other.iter_list_head_), iter_list_tail_(other.iter_list_tail_),
          data_(other.data_), size_(other.size_), slots_(other.slots_) {
        other.data_ = nullptr;
        other.reset();
    }

    MapImpl& operator=(const MapImpl&) = default;
    MapImpl& operator=(MapImpl&& other) noexcept {
        MapImpl(std::move(other)).swap(*this);
        return *this;
    }

    ~MapImpl() override {
        reset();
    }


    iterator begin() {
        return {iter_list_head_, this};
    }

    iterator end() {
        return {kInvalidIndex, this};
    }

    const_iterator begin() const {
        return const_cast<MapImpl*>(this)->begin();
    }

    const_iterator end() const {
        return const_cast<MapImpl*>(this)->end();
    }

    iterator find(const key_type& key);
    // iterator find_(const key_type& key);

    const_iterator find(const key_type& key) const {
        return const_cast<MapImpl*>(this)->find(key);
    }

    NODISCARD size_type count(const key_type& key) const {
        return find(key) != end();
    }

    // may be rehash
    std::pair<iterator, bool> insert(value_type&& kv, bool assign = false);

    iterator erase(const_iterator pos);

private:
    // Max load factor of hash table
    static constexpr double kMaxLoadFactor = 0.75;
    // default fib shift
    static constexpr uint32_t kDefaultFibShift = 63;

    static constexpr size_type kIncFactor = 2;

    // Index indicator to indicate an invalid index.
    static constexpr size_type kInvalidIndex = std::numeric_limits<size_type>::max();

    // fib shift in Fibonacci hash
    uint32_t fib_shift_ = kDefaultFibShift;
    // The head of iterator list
    size_type iter_list_head_ = kInvalidIndex;
    // The tail of iterator list
    size_type iter_list_tail_ = kInvalidIndex;

    void* data_;
    size_type size_;
    size_type slots_;

    // std::vector<ObjectPtr<Block>> blocks_;// block level cow

    NODISCARD void* data() const noexcept {
        return data_;
    }

    NODISCARD size_type size() const noexcept {
        return size_;
    }

    NODISCARD size_type slots() const noexcept {
        return slots_;
    }

    void swap(MapImpl& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
        std::swap(slots_, other.slots_);
        std::swap(fib_shift_, other.fib_shift_);
        std::swap(iter_list_head_, other.iter_list_head_);
        std::swap(iter_list_tail_, other.iter_list_tail_);
    }

    void reset();

    NODISCARD Block* GetBlockByIndex(size_type block_idx) const {
        return static_cast<Block*>(data()) + block_idx;
    }

    NODISCARD Entry* GetEntryByIndex(size_type index) const {
        auto* block = GetBlockByIndex(index / Constants::kSlotsPerBlock);
        return block->GetEntryPtr(index & (Constants::kSlotsPerBlock - 1));
    }

    // Construct a ListNode from hash code if the position is head of list
    NODISCARD std::optional<Cursor> FindListHeadByHash(size_t hash_value) const {
        if (const auto head = CreateCursorFromHash(hash_value); head.IsHead()) {
            return head;
        }

        return std::nullopt;
    }

    NODISCARD value_type* GetDataPtr(size_t index) const {
        return &GetEntryByIndex(index)->data;
    }

    NODISCARD size_type GetNextIndexOf(size_type index) const {
        // keep at the end of iterator
        if (index == kInvalidIndex) {
            return index;
        }

        return GetEntryByIndex(index)->next;
    }

    NODISCARD size_type GetPrevIndexOf(size_type index) const {
        // this is the end iterator, we need to return tail.
        if (index == kInvalidIndex) {
            return iter_list_tail_;
        }

        return GetEntryByIndex(index)->prev;
    }

    NODISCARD Cursor CreateCursorFromHash(size_t hash_value) const {
        return {details::FibonacciHash(hash_value, fib_shift_), this};
    }

    // Whether the hash table is full.
    NODISCARD bool IsFull() const {
        return size() + 1 > static_cast<size_type>(static_cast<double>(slots()) * kMaxLoadFactor);
    }

    // Insert the entry into tail of iterator list.
    // This function does not change data content of the node.
    // or NodeListPushBack ?
    void IterListPushBack(Cursor node);

    // Remove the entry from iterator list.
    // This function is usually used before deletion,
    // and it does not change data content of the node.
    void IterListRemove(Cursor node);

    /*!
   * \brief Replace node src by dst in the iter list
   * \param src The source node
   * \param dst The destination node, must be empty
   * \note This function does not change data content of the nodes,
   *       which needs to be updated by the caller.
   */
    void IterListReplace(Cursor src, Cursor dst);

    /*!
   * \brief Spare an entry to be the head of a linked list.
   * As described in B3, during insertion, it is possible that the entire linked list does not
   * exist, but the slot of its head has been occupied by other linked lists. In this case, we need
   * to spare the slot by moving away the elements to another valid empty one to make insertion
   * possible.
   * \param target The given entry to be spared
   * \return The linked-list entry constructed as the head, if actual insertion happens
   */
    std::optional<Cursor> TryAllocateListHead(Cursor target);

    /*!
     * \brief Try to insert a key, or do nothing if already exists
     * \param kv The value pair
     * \param assign Whether to assign for existing key
     * \return The linked-list entry found or just constructed,indicating if actual insertion happens
     */
    std::pair<iterator, bool> TryInsertOrUpdate(value_type&& kv, bool assign = false);

    static size_type CalculateBlockCount(size_type total_slots) {
        return (total_slots + Constants::kSlotsPerBlock - 1) / Constants::kSlotsPerBlock;
    }

    // Calculate the power-of-2 table size given the lower-bound of required capacity.
    // shift = 64 - log2(slots)
    static std::pair<uint32_t, size_type> CalculateSlotCount(size_type cap);

    static ObjectPtr<MapImpl> CopyFrom(const MapImpl* src);

    template<typename, typename, typename>
    friend class MapV1;
};

template<typename K, typename V, typename Hasher>
struct MapImpl<K, V, Hasher>::Entry {
    value_type data;
    size_type prev;
    size_type next;

    Entry() : data(key_type{}, mapped_type{}), prev(kInvalidIndex), next(kInvalidIndex) {}
    Entry(key_type key, mapped_type value) : data(std::move(key), std::move(value)), prev(kInvalidIndex), next(kInvalidIndex) {}
    explicit Entry(const value_type& kv) : data(kv), prev(kInvalidIndex), next(kInvalidIndex) {}
    explicit Entry(value_type&& kv) : data(std::move(kv)), prev(kInvalidIndex), next(kInvalidIndex) {}
    Entry(value_type&& kv, size_type p, size_type n) : data(std::move(kv)), prev(p), next(n) {}

    void reset() {
        data.~value_type();
        prev = kInvalidIndex;
        next = kInvalidIndex;
    }
};

template<typename K, typename V, typename Hasher>
struct MapImpl<K, V, Hasher>::Cursor {
    Cursor() : index_(0), obj_(nullptr) {}

    Cursor(size_t index, const MapImpl* p) : index_(index), obj_(p) {}

    NODISCARD size_t index() const {
        return index_;
    }

    NODISCARD const MapImpl* obj() const {
        return obj_;
    }

    void reset() {
        index_ = 0;
        obj_ = nullptr;
    }

    NODISCARD bool IsIterListHead() const {
        AM_CHECK(!IsNone(), "The Cursor is none.");
        return index() == obj()->iter_list_head_;
    }

    NODISCARD bool IsIterListTail() const {
        AM_CHECK(!IsNone(), "The Cursor is none.");
        return index() == obj()->iter_list_tail_;
    }

    NODISCARD Block* GetBlock() const {
        AM_CHECK(!IsNone(), "The Cursor is none.");
        return obj()->GetBlockByIndex(index() / Constants::kSlotsPerBlock);
    }

    // NODISCARD ObjectPtr<Block> GetBlock_() const {
    //     CHECK(!IsNone()) << "The Cursor is none.";
    //     return obj()->blocks_[index() / Constants::kEntriesPerBlock];
    // }

    // Get metadata of an entry
    NODISCARD std::byte& GetSlotMetadata() const {
        // equal to index() % kEntriesPerBlock
        return GetBlock()->storage_[index() & (Constants::kSlotsPerBlock - 1)];
    }

    // NODISCARD std::byte& GetSlotMetadata_() const {
    //     return GetBlock_()->storage_[index() & (Constants::kEntriesPerBlock - 1)];
    // }

    // Get the entry ref
    NODISCARD Entry& GetEntry() const {
        AM_CHECK(!IsSlotEmpty(), "The entry is empty.");
        return *GetBlock()->GetEntryPtr(index() & (Constants::kSlotsPerBlock - 1));
    }

    // NODISCARD Entry& GetEntry_() const {
    //     CHECK(!IsSlotEmpty_()) << "The entry is empty.";
    //     return *GetBlock_()->GetEntryPtr(index() & (Constants::kEntriesPerBlock - 1));
    // }

    // Get KV
    NODISCARD value_type& GetData() const {
        return GetEntry().data;
    }

    // NODISCARD value_type& GetData_() const {
    //     return GetEntry_().data;
    // }

    NODISCARD const key_type& GetKey() const {
        return GetData().first;
    }

    // NODISCARD const key_type& GetKey_() const {
    //     return GetData_().first;
    // }

    NODISCARD mapped_type& GetValue() const {
        return GetData().second;
    }

    // NODISCARD mapped_type& GetValue_() const {
    //     return GetData_().second;
    // }

    NODISCARD bool IsNone() const {
        return obj() == nullptr;
    }

    NODISCARD bool IsSlotEmpty() const {
        return GetSlotMetadata() == Constants::kEmptySlot;
    }
    //
    // NODISCARD bool IsSlotEmpty_() const {
    //     return GetSlotMetadata_() == Constants::kEmptySlot;
    // }

    NODISCARD bool IsSlotProtected() const {
        return GetSlotMetadata() == Constants::kTombStoneSlot;
    }

    // NODISCARD bool IsSlotProtected_() const {
    //     return GetSlotMetadata_() == Constants::kProtectedSlot;
    // }

    NODISCARD bool IsHead() const {
        return (GetSlotMetadata() & Constants::kHeadFlagMask) == Constants::kHeadFlag;
    }

    // NODISCARD bool IsHead_() const {
    //     return (GetSlotMetadata_() & Constants::kHeadFlagMask) == Constants::kHeadFlag;
    // }

    void MarkSlotAsEmpty() const {
        GetSlotMetadata() = Constants::kEmptySlot;
    }

    // void MarkSlotAsEmpty_() const {
    //     GetSlotMetadata_() = Constants::kEmptySlot;
    // }

    void MarkSlotAsProtected() const {
        GetSlotMetadata() = Constants::kTombStoneSlot;
    }

    // void MarkSlotAsProtected_() const {
    //     GetSlotMetadata_() = Constants::kProtectedSlot;
    // }

    // Set the entry's offset to its next entry.
    void SetNextSlotOffsetIndex(uint8_t offset_idx) const {
        AM_CHECK(offset_idx < Constants::kNumOffsetDists);
        (GetSlotMetadata() &= Constants::kHeadFlagMask) |= std::byte{offset_idx};
    }

    void ConstructEntry(Entry&& entry) const {
        AM_CHECK(IsSlotEmpty());
        new (GetBlock()->GetEntryPtr(index() & (Constants::kSlotsPerBlock - 1))) Entry(std::move(entry));
    }

    // Destroy the item in the entry.
    void DestroyEntry() const {
        if (!IsSlotEmpty()) {
            GetEntry().~Entry();
            MarkSlotAsEmpty();
        }
    }

    void DestroyEntryData() const {
        GetData().~value_type();
    }

    // Construct a head of linked list inplace.
    void CreateHead(Entry&& entry) const {
        if (!IsSlotEmpty()) {
            DestroyEntry();
        }
        ConstructEntry(std::move(entry));
        GetSlotMetadata() = Constants::kHeadFlag;
    }

    // Construct a tail of linked list inplace
    void CreateTail(Entry&& entry) const {
        if (!IsSlotEmpty()) {
            DestroyEntry();
        }
        ConstructEntry(std::move(entry));
        GetSlotMetadata() = Constants::kTailFlag;
    }

    // Whether the slot has the next slot on the linked list
    NODISCARD bool HasNextSlot() const {
        const auto idx = std::to_integer<uint8_t>(GetSlotMetadata() & Constants::kOffsetIdxMask);
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
        auto t = index_ + offset;
        index_ = t >= obj()->slots() ? t & obj()->slots() - 1 : t;
        return true;
    }

    // Get the prev slot on the linked list
    NODISCARD Cursor FindPrevSlot() const {
        // start from the head of the linked list, which must exist
        auto cur = obj()->CreateCursorFromHash(hasher()(GetKey()));
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
            if (Cursor candidate((index() + Constants::NextProbePosOffset[i]) & (obj()->slots() - 1), obj());
                candidate.IsSlotEmpty()) {
                return std::make_pair(i, candidate);
            }
        }
        return std::nullopt;
    }

private:
    // Index of entry on the array
    size_t index_;
    // Pointer to the current DenseMapObj
    const MapImpl* obj_;
};

template<typename K, typename V, typename Hasher>
MapImpl<K, V, Hasher>::MapImpl(size_type n) : size_(0) {
    auto [fib_shift, slots] = CalculateSlotCount(n);
    const size_type block_num = CalculateBlockCount(slots);
    data_ = new Block[block_num];
    slots_ = slots;
    fib_shift_ = fib_shift;

    // blocks_.reserve(block_num);
    // for (size_type i = 0; i < block_num; ++i) {
    //     blocks_.push_back(make_object<Block>());
    // }
}

template<typename K, typename V, typename Hasher>
void MapImpl<K, V, Hasher>::reset() {
    if (data() != nullptr) {
        delete[] static_cast<Block*>(data());
        data_ = nullptr;
    }

    size_ = 0;
    slots_ = 0;
    fib_shift_ = 63;
    iter_list_head_ = kInvalidIndex;
    iter_list_tail_ = kInvalidIndex;
    // blocks_.clear();
}


// template<typename K, typename V, typename Hasher>
// MapImpl<K, V, Hasher>::iterator MapImpl<K, V, Hasher>::find_(const key_type& key) {
//     auto index = details::FibonacciHash(hasher()(key), fib_shift_);
//     bool is_first = true;
//     while (true) {
//         auto block_idx = index / Constants::kEntriesPerBlock;
//         auto inner_idx = index & (Constants::kEntriesPerBlock - 1);
//         auto meta = blocks_[block_idx]->storage_[inner_idx];
//         if (is_first) {
//             if ((meta & Constants::kHeadFlagMask) != Constants::kHeadFlag) {
//                 return end();
//             }
//             is_first = false;
//         }
//
//         if (key == blocks_[block_idx]->GetEntryPtr(inner_idx)->data.first) {
//             return {index, this};
//         }
//
//         auto offset_idx = std::to_integer<uint8_t>(meta & Constants::kOffsetIdxMask);
//         if (offset_idx == 0) {
//             return end();
//         }
//
//         auto offset = Constants::NextProbePosOffset[offset_idx];
//         auto t = index + offset;
//         index = t >= slots() ? t & slots() - 1 : t;
//     }
// }

template<typename K, typename V, typename Hasher>
MapImpl<K, V, Hasher>::iterator MapImpl<K, V, Hasher>::find(const key_type& key) {
    auto index = details::FibonacciHash(hasher()(key), fib_shift_);
    bool is_first = true;
    while (true) {
        auto block_idx = index / Constants::kSlotsPerBlock;
        auto inner_idx = index & (Constants::kSlotsPerBlock - 1);
        auto* block = GetBlockByIndex(block_idx);
        auto meta = block->storage_[inner_idx];
        if (is_first) {
            if ((meta & Constants::kHeadFlagMask) != Constants::kHeadFlag) {
                return end();
            }
            is_first = false;
        }

        if (key == block->GetEntryPtr(inner_idx)->data.first) {
            return {index, this};
        }

        auto offset_idx = std::to_integer<uint8_t>(meta & Constants::kOffsetIdxMask);
        if (offset_idx == 0) {
            return end();
        }

        auto offset = Constants::NextProbePosOffset[offset_idx];
        auto t = index + offset;
        index = t >= slots() ? t & slots() - 1 : t;
    }
}

template<typename K, typename V, typename Hasher>
MapImpl<K, V, Hasher>::iterator MapImpl<K, V, Hasher>::erase(const_iterator pos) {
    if (pos == end()) {
        return end();
    }

    auto next_pos = pos + 1;

    if (Cursor cur(pos.index(), this); cur.HasNextSlot()) {
        Cursor prev = cur;
        Cursor last = cur;
        last.MoveToNextSlot();
        while (last.HasNextSlot()) {
            prev = last;
            last.MoveToNextSlot();
        }

        // needs to first unlink node from the list
        IterListRemove(cur);
        // Move link chain of iter to last as we store last node to the new iter loc.
        IterListReplace(last, cur);

        auto cur_prev = cur.GetEntry().prev;
        auto cur_next = cur.GetEntry().next;
        auto cur_meta = cur.GetSlotMetadata();
        cur.DestroyEntry();
        cur.ConstructEntry(Entry{std::move(last.GetData()), cur_prev, cur_next});
        cur.GetSlotMetadata() = cur_meta;
        last.DestroyEntry();
        prev.SetNextSlotOffsetIndex(0);
    } else {// the last node
        if (!cur.IsHead()) {
            // cut the link if there is any
            cur.FindPrevSlot().SetNextSlotOffsetIndex(0);
        }
        // unlink the node from iterator list
        IterListRemove(cur);
        cur.DestroyEntry();
        cur.MarkSlotAsEmpty();
    }
    --this->size_;
    return {next_pos.index(), this};
}


template<typename K, typename V, typename Hasher>
void MapImpl<K, V, Hasher>::IterListPushBack(Cursor node) {
    node.GetEntry().prev = iter_list_tail_;
    node.GetEntry().next = kInvalidIndex;

    if (iter_list_head_ == kInvalidIndex && iter_list_tail_ == kInvalidIndex) {
        iter_list_head_ = node.index();
    } else {
        Cursor(iter_list_tail_, this).GetEntry().next = node.index();
    }

    iter_list_tail_ = node.index();
}

template<typename K, typename V, typename Hasher>
void MapImpl<K, V, Hasher>::IterListRemove(Cursor node) {
    // head
    if (node.IsIterListHead()) {
        iter_list_head_ = node.GetEntry().next;
    } else {
        Cursor prev_node(node.GetEntry().prev, this);
        prev_node.GetEntry().next = node.GetEntry().next;
    }

    // tail
    if (node.IsIterListTail()) {
        iter_list_tail_ = node.GetEntry().prev;
    } else {
        Cursor next_node(node.GetEntry().next, this);
        next_node.GetEntry().prev = node.GetEntry().prev;
    }
}

template<typename K, typename V, typename Hasher>
void MapImpl<K, V, Hasher>::IterListReplace(Cursor src, Cursor dst) {
    dst.GetEntry().prev = src.GetEntry().prev;
    dst.GetEntry().next = src.GetEntry().next;

    if (src.IsIterListHead()) {
        iter_list_head_ = dst.index();
    } else {
        Cursor prev_node(src.GetEntry().prev, this);
        prev_node.GetEntry().next = dst.index();
    }

    if (src.IsIterListTail()) {
        iter_list_tail_ = dst.index();
    } else {
        Cursor next_node(src.GetEntry().next, this);
        next_node.GetEntry().prev = dst.index();
    }
}

template<typename K, typename V, typename Hasher>
std::optional<typename MapImpl<K, V, Hasher>::Cursor> MapImpl<K, V, Hasher>::TryAllocateListHead(Cursor target) {
    // `target` is not the head of the linked list
    // move the original item of `target`
    // and construct new item on the position `target`
    // To make `target` empty, we
    // 1) find the previous element of `target` in the linked list
    // 2) move the linked list starting from `target`

    // move from the linked list after `r`
    Cursor r = target;
    // write to the tail of `prev`
    Cursor prev = target.FindPrevSlot();
    // after `target` is moved, we disallow writing to the slot
    bool is_first = true;
    std::byte r_meta;

    do {
        const auto empty_slot_info = prev.GetNextEmptySlot();
        if (!empty_slot_info) {
            return std::nullopt;
        }

        auto [offset_idx, empty] = empty_slot_info.value();

        // move `r` to `empty`
        // first move the data
        empty.CreateTail(Entry{std::move(r.GetData())});
        // then move link list chain of r to empty
        // this needs to happen after NewTail so empty's prev/next get updated
        IterListReplace(r, empty);
        // clear the metadata of `r`
        r_meta = r.GetSlotMetadata();
        r.DestroyEntry();
        if (is_first) {
            is_first = false;
            r.MarkSlotAsProtected();
        }

        // link `prev` to `empty`, and move forward
        prev.SetNextSlotOffsetIndex(offset_idx);
        prev = empty;
    } while (r.MoveToNextSlot(r_meta));// move `r` forward as well

    return target;
}

template<typename K, typename V, typename Hasher>
std::pair<typename MapImpl<K, V, Hasher>::iterator, bool> MapImpl<K, V, Hasher>::TryInsertOrUpdate(value_type&& kv, bool assign) {
    // The key is already in the hash table
    if (auto it = find(kv.first); it != end()) {
        if (assign) {
            Cursor cur{it.index(), it.ptr()};
            cur.GetValue() = std::move(kv.second);
            IterListRemove(cur);
            IterListPushBack(cur);
        }
        return {it, false};
    }

    // `node` can be:
    // 1) empty;
    // 2) body of an irrelevant list;
    // 3) head of the relevant list.
    auto node = CreateCursorFromHash(hasher()(kv.first));

    // Case 1: empty
    if (node.IsSlotEmpty()) {
        node.CreateHead(Entry{std::move(kv)});
        ++this->size_;
        IterListPushBack(node);
        return {iterator(node.index(), this), true};
    }

    // Case 2: body of an irrelevant list
    if (!node.IsHead()) {
        if (IsFull()) {
            return {end(), false};
        }

        if (auto target = TryAllocateListHead(node);
            target.has_value()) {
            target->CreateHead(Entry{std::move(kv)});
            ++this->size_;
            IterListPushBack(target.value());
            return {iterator(target->index(), this), true};
        }
        return {end(), false};
    }

    // Case 3: head of the relevant list
    // we iterate through the linked list until the end
    // make sure `node` is the prev element of `cur`
    Cursor cur = node;
    while (cur.MoveToNextSlot()) {
        // make sure `node` is the previous element of `cur`
        node = cur;
    }

    // `node` is the tail of the linked list
    // always check capacity before insertion
    if (IsFull()) {
        return {end(), false};
    }

    // find the next empty slot
    auto empty_slot_info = node.GetNextEmptySlot();
    if (!empty_slot_info) {
        return {end(), false};
    }
    auto [offset_idx, empty] = empty_slot_info.value();
    empty.CreateTail(Entry{std::move(kv)});
    // link `iter` to `empty`, and move forward
    node.SetNextSlotOffsetIndex(offset_idx);
    IterListPushBack(empty);
    ++this->size_;
    return {iterator(empty.index(), this), true};
}

template<typename K, typename V, typename Hasher>
std::pair<uint32_t, typename MapImpl<K, V, Hasher>::size_type>
MapImpl<K, V, Hasher>::CalculateSlotCount(size_type cap) {
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

template<typename K, typename V, typename Hasher>
ObjectPtr<MapImpl<K, V, Hasher>> MapImpl<K, V, Hasher>::CopyFrom(const MapImpl* src) {
    auto impl = make_object<MapImpl>(src->slots());
    auto block_num = CalculateBlockCount(src->slots());
    impl->size_ = src->size();
    impl->iter_list_head_ = src->iter_list_head_;
    impl->iter_list_tail_ = src->iter_list_tail_;

    auto* p = static_cast<Block*>(impl->data());
    for (size_t i = 0; i < block_num; ++i) {
        p[i] = *src->GetBlockByIndex(i);
    }

    return impl;
}

template<typename K, typename V, typename Hasher>
std::pair<typename MapImpl<K, V, Hasher>::iterator, bool> MapImpl<K, V, Hasher>::insert(value_type&& kv, bool assign) {
    if (auto [it, is_success] = TryInsertOrUpdate(std::move(kv), assign); it != end()) {
        return {it, is_success};
    }

    // Otherwise, start rehash
    auto new_impl = make_object<MapImpl>(slots() * kIncFactor);
    // need to insert in the same order as the original map
    size_t idx = iter_list_head_;
    while (idx != kInvalidIndex) {
        Cursor cur(idx, this);
        new_impl->TryInsertOrUpdate(std::move(cur.GetData()), assign);
        idx = cur.GetEntry().next;
        cur.DestroyEntry();
    }

    auto [pos, is_success] = new_impl->TryInsertOrUpdate(std::move(kv), assign);
    *this = std::move(*new_impl);
    return {iterator(pos.index(), this), is_success};
}

template<typename K, typename V, typename Hasher>
template<bool IsConst>
class MapImpl<K, V, Hasher>::IteratorImpl {
public:
    using iterator_category = std::bidirectional_iterator_tag;
    using ContainerPtrType = std::conditional_t<IsConst, const MapImpl*, MapImpl*>;
    using value_type = MapImpl<K, V, Hasher>::value_type;
    using pointer = std::conditional_t<IsConst, const value_type*, value_type*>;
    using reference = std::conditional_t<IsConst, const value_type&, value_type&>;
    using difference_type = std::ptrdiff_t;

    IteratorImpl() noexcept : index_(0), ptr_(nullptr) {}

    IteratorImpl(size_type index, ContainerPtrType ptr) noexcept : index_(index), ptr_(ptr) {}

    // iterator can convert to const_iterator
    template<bool AlwaysFalse>
        requires(IsConst && !AlwaysFalse)
    IteratorImpl(const IteratorImpl<AlwaysFalse>& other) : index_(other.index()), ptr_(other.ptr()) {}//NOLINT

    NODISCARD size_type index() const noexcept {
        return index_;
    }

    NODISCARD ContainerPtrType ptr() const noexcept {
        return ptr_;
    }

    pointer operator->() const {
        Check();
        return ptr()->GetDataPtr(index_);
    }

    reference operator*() const {
        return *operator->();
    }

    IteratorImpl& operator++() {
        Check();
        index_ = ptr()->GetNextIndexOf(index_);
        return *this;
    }

    IteratorImpl& operator--() {
        Check();
        index_ = ptr()->GetPrevIndexOf(index_);
        return *this;
    }

    IteratorImpl operator++(int) {
        IteratorImpl tmp = *this;
        operator++();
        return tmp;
    }

    IteratorImpl operator--(int) {
        IteratorImpl tmp = *this;
        operator--();
        return tmp;
    }

    IteratorImpl& operator+=(difference_type offset) {
        Check();
        if (offset < 0) {
            return operator-=(static_cast<difference_type>(-offset));
        }

        for (difference_type i = 0; i < offset; ++i) {
            index_ = ptr()->GetNextIndexOf(index_);
            if (index_ == kInvalidIndex) {
                break;
            }
        }
        return *this;
    }

    IteratorImpl& operator-=(difference_type offset) {
        Check();
        if (offset < 0) {
            return operator+=(static_cast<difference_type>(-offset));
        }

        for (difference_type i = 0; i < offset; ++i) {
            index_ = ptr()->GetPrevIndexOf(index_);
            if (index_ == kInvalidIndex) {
                break;
            }
        }
        return *this;
    }

    IteratorImpl operator+(difference_type offset) const {
        auto res = *this;
        res += offset;
        return res;
    }

    IteratorImpl operator-(difference_type offset) const {
        auto res = *this;
        res -= offset;
        return res;
    }

    difference_type operator-(const IteratorImpl& other) const {
        return static_cast<difference_type>(index()) - static_cast<difference_type>(other.index());
    }

    bool operator==(const IteratorImpl& other) const {
        return index_ == other.index_ && ptr_ == other.ptr_;
    }

    bool operator!=(const IteratorImpl& other) const {
        return !(*this == other);
    }

    void Check() const {
        AM_CHECK(ptr_ != nullptr, "Iterator pointer is nullptr.");
        AM_CHECK(index_ <= ptr_->slots(), "Iterator index is out of range.");
    }

private:
    size_type index_;
    ContainerPtrType ptr_;
};

template<typename K, typename V, typename Hasher = hash<K>>
class MapV1 : public ObjectRef {
public:
    using ContainerType = MapImpl<K, V, Hasher>;

    using key_type = ContainerType::key_type;
    using mapped_type = ContainerType::mapped_type;
    using value_type = ContainerType::value_type;
    using size_type = ContainerType::size_type;
    using hasher = Hasher;

    template<bool IsConst>
    class IteratorImpl;

    using iterator = IteratorImpl<false>;
    using const_iterator = IteratorImpl<true>;

    MapV1() : impl_(make_object<ContainerType>(16)) {}

    explicit MapV1(size_type n) : impl_(make_object<ContainerType>(std::max(n, static_cast<size_type>(16)))) {}

    template<typename Iter>
        requires requires(Iter t) {
            requires details::is_valid_iter<Iter>;
            { *t } -> std::convertible_to<value_type>;
        }
    MapV1(Iter first, Iter last) {
        auto _sz = std::distance(first, last);
        if (_sz > 0) {
            const auto size = static_cast<size_type>(_sz);
            impl_ = make_object<ContainerType>(std::max(size, static_cast<size_type>(16)));
            while (first != last) {
                impl_->insert(value_type(*first++));
                // impl_ = std::get<0>(ContainerType::insert(value_type(*first++), impl_));
            }
        }
    }

    MapV1(std::initializer_list<value_type> list) : MapV1(list.begin(), list.end()) {}

    MapV1(const MapV1&) = default;

    MapV1(MapV1&&) noexcept = default;

    MapV1& operator=(const MapV1& other) = default;

    MapV1& operator=(MapV1&& other) noexcept = default;

    MapV1& operator=(std::initializer_list<value_type> list) {
        MapV1(list).swap(*this);
        return *this;
    }

    NODISCARD size_type size() const noexcept {
        return impl_->size();
    }

    NODISCARD size_type slots() const noexcept {
        return impl_->slots();
    }

    NODISCARD bool empty() const noexcept {
        return size() == 0;
    }

    NODISCARD uint32_t use_count() const noexcept {
        return impl_.use_count();
    }

    NODISCARD bool unique() const noexcept {
        return use_count() == 1;
    }

    NODISCARD void* data() noexcept {
        return impl_->data();
    }

    iterator begin() noexcept {
        return iterator(impl_->begin());
    }

    iterator end() noexcept {
        return iterator(impl_->end());
    }

    const_iterator begin() const noexcept {
        return const_cast<MapV1*>(this)->begin();
    }

    const_iterator end() const noexcept {
        return const_cast<MapV1*>(this)->end();
    }

    iterator find(const key_type& key) {
        return iterator(impl_->find(key));
    }

    const_iterator find(const key_type& key) const {
        return const_cast<MapV1*>(this)->find(key);
    }

    bool contains(const key_type& key) const {
        return find(key) != end();
    }

    NODISCARD size_type count(const key_type& key) const {
        return contains(key) ? 1 : 0;
    }

    mapped_type& at(const key_type& key) {
        auto it = find(key);
        if (it == end()) {
            AETHERMIND_THROW(KeyError) << "Key does not exist";
        }

        return it->second;
    }

    const mapped_type& at(const key_type& key) const {
        return const_cast<MapV1*>(this)->at(key);
    }

    const mapped_type& operator[](const key_type& key) const {
        auto it = find(key);
        return it->second;
    }

    mapped_type& operator[](const key_type& key) {
        auto it = find(key);
        if (it == end()) {
            auto [iter, _] = insert(key, mapped_type{});
            return iter->second;
        }
        return it->second;
    }

    mapped_type& operator[](key_type&& key) {
        auto it = find(key);
        if (it == end()) {
            auto [iter, _] = insert(std::move(key), mapped_type{});
            return iter->second;
        }
        return it->second;
    }

    std::pair<iterator, bool> insert(value_type&& x) {
        return insert_(std::move(x), false);
    }

    std::pair<iterator, bool> insert(const value_type& x) {
        return insert_(value_type(x), false);
    }

    std::pair<iterator, bool> insert(const key_type& key, const mapped_type& value) {
        return insert_({key, value}, false);
    }

    std::pair<iterator, bool> insert(key_type&& key, mapped_type&& value) {
        return insert_({std::move(key), std::move(value)}, false);
    }

    template<typename Pair>
        requires(std::constructible_from<value_type, Pair &&> &&
                 !std::same_as<std::decay_t<Pair>, value_type>)
    std::pair<iterator, bool> insert(Pair&& x) {
        return insert_(value_type(std::forward<Pair>(x)), false);
    }

    template<typename Iter>
        requires details::is_valid_iter<Iter>
    void insert(Iter first, Iter last) {
        while (first != last) {
            insert(*first++);
        }
    }

    void insert(std::initializer_list<value_type> list) {
        insert(list.begin(), list.end());
    }

    template<typename Obj>
    std::pair<iterator, bool> insert_or_assign(key_type&& key, Obj&& obj) {
        return insert_({std::move(key), std::forward<Obj>(obj)}, true);
    }

    template<typename Obj>
    std::pair<iterator, bool> insert_or_assign(const key_type& key, Obj&& obj) {
        return insert_({key, std::forward<Obj>(obj)}, true);
    }

    template<typename... Args>
    std::pair<iterator, bool> emplace(Args&&... args) {
        return insert_(value_type(std::forward<Args>(args)...), false);
    }

    iterator erase(iterator pos) {
        return erase(const_iterator(pos));
        // if (pos == end()) {
        //     return end();
        // }
        // COW();
        // typename ContainerType::iterator it(pos.index(), pos.ptr());
        // return iterator(impl_->erase(it));
    }

    iterator erase(const_iterator pos) {
        // return erase(iterator(pos.index(), pos.ptr()));
        if (pos == end()) {
            return end();
        }

        COW();
        typename ContainerType::const_iterator it(pos.index(), pos.ptr());
        return iterator(impl_->erase(it));
    }

    size_type erase(const key_type& key) {
        auto it = find(key);
        if (it != end()) {
            erase(it);
            return 1;
        }
        return 0;
    }

    iterator erase(iterator first, iterator last) {
        if (first == last) {
            return first;
        }

        auto n = std::distance(first, last);
        iterator it = first;
        for (typename iterator::difference_type i = 0; i < n; ++i) {
            it = erase(it++);
        }
        return it;
    }

    iterator erase(const_iterator first, const_iterator last) {
        if (first == last) {
            return first;
        }

        auto n = std::distance(first, last);
        iterator it = first;
        for (typename iterator::difference_type i = 0; i < n; ++i) {
            it = erase(it++);
        }
        return it;
    }

    void clear() noexcept {
        impl_ = make_object<ContainerType>(ContainerType::Constants::kSlotsPerBlock);
    }

    void swap(MapV1& other) noexcept {
        std::swap(impl_, other.impl_);
    }

private:
    ObjectPtr<ContainerType> impl_;

    std::pair<iterator, bool> insert_(value_type&& x, bool assign) {
        if (!assign) {
            auto it = find(x.first);
            if (it != end()) {
                return {it, false};
            }
        }

        COW();
        // auto [impl, pos, is_success] = ContainerType::insert(std::move(x), impl_, assign);
        // impl_ = impl;
        auto [pos, is_success] = impl_->insert(std::move(x), assign);
        return {iterator(pos), is_success};
    }

    void COW() {
        if (!unique()) {
            impl_ = ContainerType::CopyFrom(impl_.get());
        }
    }
};

template<typename K, typename V, typename Hasher>
template<bool IsConst>
class MapV1<K, V, Hasher>::IteratorImpl {
public:
    using impl_iter_type = std::conditional_t<IsConst, typename ContainerType::const_iterator,
                                              typename ContainerType::iterator>;
    using ContainerPtrType = impl_iter_type::ContainerPtrType;
    using iterator_category = impl_iter_type::iterator_category;
    using value_type = impl_iter_type::value_type;
    using difference_type = impl_iter_type::difference_type;
    using pointer = impl_iter_type::pointer;
    using reference = impl_iter_type::reference;

    IteratorImpl() = default;

    IteratorImpl(size_type index, ContainerPtrType ptr) : iter_(index, ptr) {}

    explicit IteratorImpl(const impl_iter_type& iter) : iter_(iter) {}

    // iterator can convert to const_iterator
    template<bool AlwaysFalse>
        requires(IsConst && !AlwaysFalse)
    IteratorImpl(const IteratorImpl<AlwaysFalse>& other) : iter_(other.iter_) {}//NOLINT

    NODISCARD size_type index() const noexcept {
        return iter_.index();
    }

    NODISCARD ContainerPtrType ptr() const noexcept {
        return iter_.ptr();
    }

    pointer operator->() const {
        return iter_.operator->();
    }

    reference operator*() const {
        return iter_.operator*();
    }

    IteratorImpl& operator++() {
        ++iter_;
        return *this;
    }

    IteratorImpl& operator--() {
        --iter_;
        return *this;
    }

    IteratorImpl operator++(int) {
        IteratorImpl tmp = *this;
        operator++();
        return tmp;
    }

    IteratorImpl operator--(int) {
        IteratorImpl tmp = *this;
        operator--();
        return tmp;
    }

    IteratorImpl& operator+=(difference_type offset) {
        iter_ += offset;
        return *this;
    }

    IteratorImpl& operator-=(difference_type offset) {
        iter_ -= offset;
        return *this;
    }

    IteratorImpl operator+(difference_type offset) const {
        auto res = *this;
        res += offset;
        return res;
    }

    IteratorImpl operator-(difference_type offset) const {
        auto res = *this;
        res -= offset;
        return res;
    }

    difference_type operator-(const IteratorImpl& other) const {
        return iter_ - other.iter_;
    }

    bool operator==(const IteratorImpl& other) const {
        return iter_ == other.iter_;
    }

    bool operator!=(const IteratorImpl& other) const {
        return iter_ != other.iter_;
    }

private:
    impl_iter_type iter_;

    template<bool>
    friend class IteratorImpl;
};

}// namespace aethermind

#endif//AETHERMIND_CONTAINER_MAP_V1_H
