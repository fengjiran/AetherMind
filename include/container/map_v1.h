//
// Created by richard on 1/13/26.
//

#ifndef AETHERMIND_CONTAINER_MAP_H
#define AETHERMIND_CONTAINER_MAP_H

#include "any_utils.h"
#include "error.h"
#include "object.h"
#include "utils/hash.h"

#include <concepts>
#include <map>
#include <tuple>

namespace aethermind {

namespace details {

template<typename InputIter>
concept is_valid_iter_v1 = requires(InputIter t) {
    requires std::input_iterator<InputIter>;
    ++t;
    --t;
};

}// namespace details


struct MagicConstantsV1 {
    // 0b11111111 represent that the slot is empty
    static constexpr auto kEmptySlot = std::byte{0xFF};
    // 0b11111110 represent that the slot is protected
    static constexpr auto kProtectedSlot = std::byte{0xFE};
    // Number of probing choices available
    static constexpr int kNumOffsetDists = 126;
    // head flag
    static constexpr auto kHeadFlag = std::byte{0x00};
    // tail flag
    static constexpr auto kTailFlag = std::byte{0x80};
    // head flag mask
    static constexpr auto kHeadFlagMask = std::byte{0x80};
    // offset mask
    static constexpr auto kOffsetIdxMask = std::byte{0x7F};
    // default fib shift
    static constexpr uint32_t kDefaultFibShift = 63;

    // next probe position offset
    static const size_t NextProbePosOffset[kNumOffsetDists];
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

    MapImpl() : data_(nullptr), size_(0), slots_(0) {}
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

    const_iterator find(const key_type& key) const {
        return const_cast<MapImpl*>(this)->find(key);
    }

    NODISCARD size_type count(const key_type& key) const {
        return find(key) != end();
    }


private:
    // The number of elements in a memory block.
    static constexpr int kEntriesPerBlock = 16;
    // Max load factor of hash table
    static constexpr double kMaxLoadFactor = 0.75;
    // default fib shift
    static constexpr uint32_t kDefaultFibShift = 63;

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

    struct Entry;
    struct Block;
    struct Cursor;

    NODISCARD void* data() const noexcept {
        return data_;
    }

    NODISCARD size_type size() const noexcept {
        return size_;
    }

    NODISCARD size_type slots() const noexcept {
        return slots_;
    }

    void reset() {
        const size_type block_num = CalculateBlockCount(slots());
        for (size_t i = 0; i < block_num; ++i) {
            GetBlockByIndex(i)->~Block();
        }

        size_ = 0;
        slots_ = 0;
        fib_shift_ = 63;
    }

    NODISCARD Block* GetBlockByIndex(size_type block_idx) {
        return static_cast<Block*>(data_) + block_idx;
    }

    NODISCARD Entry* GetEntryByIndex(size_type index) {
        auto* block = GetBlockByIndex(index / kEntriesPerBlock);
        return block->GetEntryPtr(index & (kEntriesPerBlock - 1));
    }

    NODISCARD value_type* GetDataPtr(size_t index) {
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

    static size_type CalculateBlockCount(size_type total_slots) {
        return (total_slots + kEntriesPerBlock - 1) / kEntriesPerBlock;
    }

    // Calculate the power-of-2 table size given the lower-bound of required capacity.
    // shift = 64 - log2(slots)
    static std::pair<uint32_t, size_type> CalculateSlotCount(size_type cap);

    static ObjectPtr<MapImpl> Create(size_type n);
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

    ~Entry() {
        reset();
    }

    void reset() {
        data.~value_type();
        prev = kInvalidIndex;
        next = kInvalidIndex;
    }
};

template<typename K, typename V, typename Hasher>
struct MapImpl<K, V, Hasher>::Block {
    std::array<std::byte, kEntriesPerBlock + kEntriesPerBlock * sizeof(Entry)> storage_;

    Block() {// NOLINT
        for (uint8_t i = 0; i < kEntriesPerBlock; ++i) {
            storage_[i] = MagicConstants::kEmptySlot;
        }
    }

    Block(const Block& other) {// NOLINT
        for (uint8_t i = 0; i < kEntriesPerBlock; ++i) {
            if (other.storage_[i] != MagicConstants::kEmptySlot) {
                storage_[i] = other.storage_[i];
                new (GetEntryPtr(i)) Entry(*other.GetEntryPtr(i));
            }
        }
    }

    ~Block() {
        for (uint8_t i = 0; i < kEntriesPerBlock; ++i) {
            if (storage_[i] != MagicConstants::kEmptySlot) {
                storage_[i] = MagicConstants::kEmptySlot;
                GetEntryPtr(i)->~Entry();
            }
        }
    }

    Entry* GetEntryPtr(size_type i) {
        return static_cast<Entry*>(static_cast<void*>(storage_.data() + kEntriesPerBlock)) + i;
    }

    const Entry* GetEntryPtr(size_type i) const {
        return const_cast<Block*>(this)->GetEntryPtr(i);
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
        CHECK(!IsNone()) << "The Cursor is none.";
        return index() == obj()->iter_list_head_;
    }

    NODISCARD bool IsIterListTail() const {
        CHECK(!IsNone()) << "The Cursor is none.";
        return index() == obj()->iter_list_tail_;
    }

    NODISCARD Block* GetBlock() const {
        CHECK(!IsNone()) << "The Cursor is none.";
        return obj()->GetBlockByIndex(index() / kEntriesPerBlock);
    }

    // Get metadata of an entry
    NODISCARD std::byte& GetSlotMetadata() const {
        // equal to index() % kEntriesPerBlock
        return GetBlock()->storage_[index() & (kEntriesPerBlock - 1)];
    }

    // Get the entry ref
    NODISCARD Entry& GetEntry() const {
        CHECK(!IsNone()) << "The Cursor is none.";
        CHECK(!IsSlotEmpty()) << "The entry is empty.";
        return *GetBlock()->GetEntryPtr(index() & (kEntriesPerBlock - 1));
    }

    // Get KV
    NODISCARD value_type& GetData() const {
        return GetEntry().data;
    }

    NODISCARD const key_type& GetKey() const {
        return GetData().first;
    }

    NODISCARD mapped_type& GetValue() const {
        return GetData().second;
    }

    NODISCARD bool IsNone() const {
        return obj() == nullptr;
    }

    NODISCARD bool IsSlotEmpty() const {
        return GetSlotMetadata() == MagicConstants::kEmptySlot;
    }

    NODISCARD bool IsSlotProtected() const {
        return GetSlotMetadata() == MagicConstants::kProtectedSlot;
    }

    NODISCARD bool IsHead() const {
        return (GetSlotMetadata() & MagicConstants::kHeadFlagMask) == MagicConstants::kHeadFlag;
    }

    void MarkSlotAsEmpty() const {
        GetSlotMetadata() = MagicConstants::kEmptySlot;
    }

    void MarkSlotAsProtected() const {
        GetSlotMetadata() = MagicConstants::kProtectedSlot;
    }

    // Set the entry's offset to its next entry.
    void SetNextSlotOffsetIndex(uint8_t offset_idx) const {
        CHECK(offset_idx < MagicConstants::kNumOffsetDists);
        (GetSlotMetadata() &= MagicConstants::kHeadFlagMask) |= std::byte{offset_idx};
    }

    void ConstructEntry(Entry&& entry) const {
        CHECK(IsSlotEmpty());
        new (GetBlock()->GetEntryPtr(index() & (kEntriesPerBlock - 1))) Entry(std::move(entry));
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
        GetSlotMetadata() = MagicConstants::kHeadFlag;
    }

    // Construct a tail of linked list inplace
    void CreateTail(Entry&& entry) const {
        if (!IsSlotEmpty()) {
            DestroyEntry();
        }
        ConstructEntry(std::move(entry));
        GetSlotMetadata() = MagicConstants::kTailFlag;
    }

    // Whether the slot has the next slot on the linked list
    NODISCARD bool HasNextSlot() const {
        const auto idx = std::to_integer<uint8_t>(GetSlotMetadata() & MagicConstants::kOffsetIdxMask);
        return MagicConstants::NextProbePosOffset[idx] != 0;
    }

    // Move the current cursor to the next slot on the linked list
    bool MoveToNextSlot(std::optional<std::byte> meta_opt = std::nullopt) {
        std::byte meta = meta_opt ? meta_opt.value() : GetSlotMetadata();
        const auto idx = std::to_integer<uint8_t>(meta & MagicConstants::kOffsetIdxMask);
        const auto offset = MagicConstants::NextProbePosOffset[idx];
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
        for (uint8_t i = 1; i < MagicConstants::kNumOffsetDists; ++i) {
            if (Cursor candidate((index() + MagicConstants::NextProbePosOffset[i]) & (obj()->slots() - 1), obj());
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
    CHECK(slots >= cap);
    return {shift, slots};
}

template<typename K, typename V, typename Hasher>
MapImpl<K, V, Hasher>::iterator MapImpl<K, V, Hasher>::find(const key_type& key) {
    auto index = details::FibonacciHash(hasher()(key), fib_shift_);
    bool is_first = true;
    while (true) {
        auto block_idx = index / kEntriesPerBlock;
        auto inner_idx = index & (kEntriesPerBlock - 1);
        auto* block = GetBlockByIndex(block_idx);
        auto meta = block->storage_[inner_idx];
        if (is_first) {
            if ((meta & MagicConstants::kHeadFlagMask) != MagicConstants::kHeadFlag) {
                return end();
            }
            is_first = false;
        }

        if (key == block->GetEntryPtr(inner_idx)->data.first) {
            return {index, this};
        }

        auto offset_idx = std::to_integer<uint8_t>(meta & MagicConstants::kOffsetIdxMask);
        if (offset_idx == 0) {
            return end();
        }

        auto offset = MagicConstants::NextProbePosOffset[offset_idx];
        auto t = index + offset;
        index = t >= slots() ? t & slots() - 1 : t;
    }
}


template<typename K, typename V, typename Hasher>
ObjectPtr<MapImpl<K, V, Hasher>> MapImpl<K, V, Hasher>::Create(size_type n) {
    auto [fib_shift, slots] = CalculateSlotCount(n);
    const size_t block_num = CalculateBlockCount(slots);
    auto impl = make_array_object<MapImpl, Block>(block_num);
    impl->data_ = reinterpret_cast<char*>(impl.get()) + sizeof(MapImpl);
    impl->size_ = 0;
    impl->slots_ = slots;
    impl->fib_shift_ = fib_shift;
    impl->iter_list_head_ = kInvalidIndex;
    impl->iter_list_tail_ = kInvalidIndex;

    auto* p = static_cast<Block*>(impl->data());
    for (size_t i = 0; i < block_num; ++i) {
        new (p + i) Block;
    }

    return impl;
}

template<typename K, typename V, typename Hasher>
template<bool IsConst>
class MapImpl<K, V, Hasher>::IteratorImpl {
public:
    using iterator_category = std::bidirectional_iterator_tag;
    using ContainerPtrType = std::conditional_t<IsConst, const MapImpl*, MapImpl*>;
    using pointer = std::conditional_t<IsConst, const value_type*, value_type*>;
    using reference = std::conditional_t<IsConst, const value_type&, value_type&>;
    using difference_type = std::ptrdiff_t;

    IteratorImpl() noexcept : index_(0), ptr_(nullptr) {}

    IteratorImpl(size_t index, ContainerPtrType ptr) noexcept : index_(index), ptr_(ptr) {}

    // iterator can convert to const_iterator
    template<bool AlwaysFalse>
        requires(IsConst && !AlwaysFalse)
    IteratorImpl(const IteratorImpl<AlwaysFalse>& other) : index_(other.index()), ptr_(other.ptr()) {}//NOLINT

    NODISCARD size_t index() const noexcept {
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
        Check();
        auto res = *this;
        res += offset;
        return res;
    }

    IteratorImpl operator-(difference_type offset) const {
        Check();
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
        CHECK(ptr_ != nullptr) << "Iterator pointer is nullptr.";
        CHECK(index_ <= ptr_->slots()) << "Iterator index is out of range.";
    }

private:
    size_type index_;
    ContainerPtrType ptr_;
};

template<typename K, typename V, typename Hasher = hash<K>>
class MapV1 : public ObjectRef {
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

    NODISCARD size_type size() const noexcept {
        return size_;
    }

    NODISCARD size_type slots() const noexcept {
        return IsLocal() ? kThreshold : slots_;
    }

    NODISCARD bool empty() const noexcept {
        return size_ == 0;
    }

    NODISCARD pointer data() noexcept {
        return IsLocal() ? local_buffer_ : impl_->data();
    }

    NODISCARD bool IsLocal() const noexcept {
        return !impl_;
    }

private:
    static constexpr size_type kThreshold = 4;

    union {
        value_type local_buffer_[kThreshold];
        size_type slots_ = 0;
    };

    size_type size_ = 0;
    ObjectPtr<MapImpl<K, V, Hasher>> impl_;
};

}// namespace aethermind

#endif//AETHERMIND_CONTAINER_MAP_H
