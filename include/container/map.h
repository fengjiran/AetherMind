//
// Created by richard on 1/5/26.
//

#ifndef AETHERMIND_CONTAINER_MAP_OBJ_H
#define AETHERMIND_CONTAINER_MAP_OBJ_H

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
concept is_valid_iter = requires(InputIter t) {
    requires std::input_iterator<InputIter>;
    ++t;
    --t;
};

}// namespace details

template<typename Derived, typename K, typename V, typename Hasher>
class MapObj : public Object {
public:
    using key_type = K;
    using mapped_type = V;
    using value_type = std::pair<const key_type, mapped_type>;
    using size_type = size_t;

    // IteratorImpl is a base class for iterator and const_iterator
    template<bool IsConst>
    class IteratorImpl;

    using iterator = IteratorImpl<false>;
    using const_iterator = IteratorImpl<true>;

    MapObj() : data_(nullptr), size_(0), slots_(0) {}

    NODISCARD void* data() const {
        return data_;
    }

    NODISCARD size_type size() const {
        return size_;
    }

    NODISCARD size_type slots() const {
        return slots_;
    }

    NODISCARD bool empty() const {
        return size() == 0;
    }

    NODISCARD size_t count(const key_type& key) const {
        return GetDerivedPtr()->count_impl(key);
    }

    mapped_type& at(const key_type& key) {
        return GetDerivedPtr()->AtImpl(key);
    }

    NODISCARD const mapped_type& at(const key_type& key) const {
        return GetDerivedPtr()->AtImpl(key);
    }

    NODISCARD value_type* GetDataPtr(size_t idx) {
        return GetDerivedPtr()->GetDataPtrImpl(idx);
    }

    NODISCARD const value_type* GetDataPtr(size_t idx) const {
        return GetDerivedPtr()->GetDataPtrImpl(idx);
    }

    NODISCARD size_t GetNextIndexOf(size_t idx) const {
        return GetDerivedPtr()->GetNextIndexOfImpl(idx);
    }

    NODISCARD size_t GetPrevIndexOf(size_t idx) const {
        return GetDerivedPtr()->GetPrevIndexOfImpl(idx);
    }

    NODISCARD const_iterator begin() const {
        return GetDerivedPtr()->BeginImpl();
    }

    NODISCARD const_iterator end() const {
        return GetDerivedPtr()->EndImpl();
    }

    NODISCARD iterator begin() {
        return GetDerivedPtr()->BeginImpl();
    }

    NODISCARD iterator end() {
        return GetDerivedPtr()->EndImpl();
    }

    NODISCARD const_iterator find(const key_type& key) const {
        return GetDerivedPtr()->FindImpl(key);
    }

    NODISCARD iterator find(const key_type& key) {
        return GetDerivedPtr()->FindImpl(key);
    }

    NODISCARD iterator erase(iterator pos) {
        return GetDerivedPtr()->EraseImpl(pos);
    }

protected:
    void* data_;
    size_type size_;
    size_type slots_;

    static constexpr size_type kInitSize = 2; // Init map size
    static constexpr size_type kThreshold = 4;// The threshold of the small and dense map
    static constexpr size_type kIncFactor = 2;

    NODISCARD Derived* GetDerivedPtr() noexcept {
        return static_cast<Derived*>(this);
    }

    NODISCARD const Derived* GetDerivedPtr() const noexcept {
        return static_cast<const Derived*>(this);
    }

    // insert may be rehash
    static std::tuple<ObjectPtr<Object>, size_type, bool> insert(
            value_type&& kv, const ObjectPtr<Object>& old_impl, bool assign = false);
};

// in heap
template<typename K, typename V, typename Hasher = hash<K>>
class SmallMapObj : public MapObj<SmallMapObj<K, V, Hasher>, K, V, Hasher> {
public:
    using BaseType = MapObj<SmallMapObj, K, V, Hasher>;

    using key_type = K;
    using mapped_type = V;
    using value_type = std::pair<const key_type, mapped_type>;
    using size_type = size_t;
    using hasher = Hasher;

    using iterator = MapObj<SmallMapObj, K, V, Hasher>::iterator;
    using const_iterator = MapObj<SmallMapObj, K, V, Hasher>::const_iterator;

    SmallMapObj() {
        this->data_ = storage_.data();
        this->size_ = 0;
        this->slots_ = BaseType::kThreshold;
    }

    ~SmallMapObj() override {
        reset();
    }

private:
    std::array<std::byte, sizeof(value_type) * BaseType::kThreshold> storage_;

    NODISCARD iterator BeginImpl() {
        return {0, this};
    }

    NODISCARD iterator EndImpl() {
        return {this->size(), this};
    }

    NODISCARD const_iterator begin_impl() const {
        return const_cast<SmallMapObj*>(this)->BeginImpl();
    }

    NODISCARD const_iterator EndImpl() const {
        return const_cast<SmallMapObj*>(this)->EndImpl();
    }

    NODISCARD const_iterator FindImpl(const key_type& key) const;

    NODISCARD iterator FindImpl(const key_type& key);

    NODISCARD size_type count_impl(const key_type& key) const {
        return FindImpl(key) != EndImpl();
    }

    mapped_type& AtImpl(const key_type& key) {
        const auto iter = FindImpl(key);
        if (iter == EndImpl()) {
            AETHERMIND_THROW(KeyError) << "key is not exist.";
        }
        return iter->second;
    }

    NODISCARD const mapped_type& AtImpl(const key_type& key) const {
        return const_cast<SmallMapObj*>(this)->AtImpl(key);
    }

    NODISCARD value_type* GetDataPtrImpl(size_type index) {
        return static_cast<value_type*>(static_cast<void*>(storage_.data())) + index;
    }

    NODISCARD const value_type* GetDataPtrImpl(size_type index) const {
        return const_cast<SmallMapObj*>(this)->GetDataPtrImpl(index);
    }

    NODISCARD size_type GetNextIndexOfImpl(size_type idx) const {
        return idx + 1 < this->size() ? idx + 1 : this->size();
    }

    NODISCARD size_type GetPrevIndexOfImpl(size_type idx) const {
        return idx > 0 ? idx - 1 : this->size();
    }

    std::pair<iterator, bool> InsertImpl(value_type&& kv, bool assign = false);

    iterator EraseImpl(iterator pos);

    void reset();

    static ObjectPtr<SmallMapObj> Create() {
        return make_object<SmallMapObj>();
    }

    static ObjectPtr<SmallMapObj> CopyFrom(const SmallMapObj* src);

    template<typename, typename, typename, typename>
    friend class MapObj;
    template<typename, typename, typename>
    friend class DenseMapObj;
    template<typename, typename, typename>
    friend class Map;
};

template<typename K, typename V, typename Hasher>
void SmallMapObj<K, V, Hasher>::reset() {
    if (!this->empty()) {
        for (size_t i = 0; i < this->size(); ++i) {
            GetDataPtrImpl(i)->~value_type();
        }
        this->size_ = 0;
    }
}

template<typename K, typename V, typename Hasher>
SmallMapObj<K, V, Hasher>::iterator SmallMapObj<K, V, Hasher>::FindImpl(const key_type& key) {
    for (size_t i = 0; i < this->size(); ++i) {
        if (key == GetDataPtrImpl(i)->first) {
            return {i, this};
        }
    }

    return EndImpl();
}

template<typename K, typename V, typename Hasher>
SmallMapObj<K, V, Hasher>::const_iterator SmallMapObj<K, V, Hasher>::FindImpl(const key_type& key) const {
    return const_cast<SmallMapObj*>(this)->FindImpl(key);
}

template<typename K, typename V, typename Hasher>
std::pair<typename SmallMapObj<K, V, Hasher>::iterator, bool>
SmallMapObj<K, V, Hasher>::InsertImpl(value_type&& kv, bool assign) {
    if (const auto it = FindImpl(kv.first); it != EndImpl()) {
        if (assign) {
            it->second = std::move(kv.second);
        }
        return {it, false};
    }

    CHECK(this->size() < this->slots());
    new (GetDataPtrImpl(this->size())) value_type(std::move(kv));
    ++this->size_;
    return {iterator{this->size() - 1, this}, true};
}

template<typename K, typename V, typename Hasher>
SmallMapObj<K, V, Hasher>::iterator SmallMapObj<K, V, Hasher>::EraseImpl(iterator pos) {
    if (pos == EndImpl()) {
        return pos;
    }

    pos->~value_type();
    if (auto idx = pos.index(); idx < this->size() - 1) {
        while (idx < this->size() - 1) {
            new (GetDataPtrImpl(idx)) value_type(std::move(*GetDataPtrImpl(idx + 1)));
            ++idx;
        }
    }
    --this->size_;
    return pos;
}

template<typename K, typename V, typename Hasher>
ObjectPtr<SmallMapObj<K, V, Hasher>> SmallMapObj<K, V, Hasher>::CopyFrom(const SmallMapObj* src) {
    auto impl = Create();
    for (size_type i = 0; i < src->size(); ++i) {
        new (impl->GetDataPtrImpl(i)) value_type(*src->GetDataPtrImpl(i));
        ++impl->size_;
    }

    return impl;
}


/*
// 动态调整扩容因子，根据负载情况优化
static constexpr double kMinLoadFactor = 0.1;
static constexpr double kTargetLoadFactor = 0.7;

// 计算更精确的扩容大小
static size_type ComputeNewSlots(size_type current_slots, size_type new_elements) {
    double target_load = static_cast<double>(current_slots + new_elements) / kTargetLoadFactor;
    size_type new_slots = current_slots;
    while (new_slots < target_load) {
        new_slots <<= 1;
    }
    return new_slots;
}
*/

template<typename K, typename V, typename Hasher = hash<K>>
class DenseMapObj : public MapObj<DenseMapObj<K, V, Hasher>, K, V, Hasher> {
public:
    using key_type = K;
    using mapped_type = V;
    using value_type = std::pair<const key_type, mapped_type>;
    using size_type = size_t;
    using hasher = Hasher;

    using iterator = MapObj<DenseMapObj, K, V, Hasher>::iterator;
    using const_iterator = MapObj<DenseMapObj, K, V, Hasher>::const_iterator;

    DenseMapObj() = default;
    ~DenseMapObj() override {
        reset();
    }

private:
    // The number of elements in a memory block.
    static constexpr int kEntriesPerBlock = 16;
    // Max load factor of hash table
    static constexpr double kMaxLoadFactor = 0.99;
    // 0b11111111 representation of the metadata of an empty slot.
    static constexpr auto kEmptySlot = std::byte{0xFF};
    // 0b11111110 representation of the metadata of a protected slot.
    // static constexpr uint8_t kProtectedSlot = 0xFE;
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

    // Index indicator to indicate an invalid index.
    static constexpr size_type kInvalidIndex = std::numeric_limits<size_type>::max();
    static constexpr size_type NextProbePosOffset[kNumOffsetDists] = {
            // linear probing offset(0-15)
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            // Quadratic probing with triangle numbers, n(n+1)/2, n = 6 ~ 72
            // See also:
            // 1) https://en.wikipedia.org/wiki/Quadratic_probing
            // 2) https://fgiesen.wordpress.com/2015/02/22/triangular-numbers-mod-2n/
            // 3) https://github.com/skarupke/flat_hash_map
            21, 28, 36, 45, 55, 66, 78, 91, 105, 120,
            136, 153, 171, 190, 210, 231, 253, 276, 300, 325,
            351, 378, 406, 435, 465, 496, 528, 561, 595, 630,
            666, 703, 741, 780, 820, 861, 903, 946, 990, 1035,
            1081, 1128, 1176, 1225, 1275, 1326, 1378, 1431, 1485, 1540,
            1596, 1653, 1711, 1770, 1830, 1891, 1953, 2016, 2080, 2145,
            2211, 2278, 2346, 2415, 2485, 2556, 2628,
            // larger triangle numbers
            8515, 19110, 42778, 96141, 216153,
            486591, 1092981, 2458653, 5532801, 12442566,
            27993903, 62983476, 141717030, 318844378, 717352503,
            1614057336, 3631522476, 8170957530, 18384510628, 41364789378,
            93070452520, 209408356380, 471168559170, 1060128894105, 2385289465695,
            5366898840628, 12075518705635, 27169915244790, 61132312065111, 137547689707000,
            309482283181501, 696335127828753, 1566753995631385, 3525196511162271, 7931691992677701,
            17846306936293605, 40154190677507445, 90346928918121501, 203280589587557251,
            457381325854679626, 1029107982097042876, 2315492959180353330, 5209859154120846435};

    // fib shift in Fibonacci hash
    uint32_t fib_shift_ = kDefaultFibShift;
    // The head of iterator list
    size_type iter_list_head_ = kInvalidIndex;
    // The tail of iterator list
    size_type iter_list_tail_ = kInvalidIndex;

    struct Entry;
    struct Block;
    class Cursor;

    NODISCARD iterator BeginImpl() {
        return {iter_list_head_, this};
    }

    NODISCARD iterator EndImpl() {
        return {kInvalidIndex, this};
    }

    NODISCARD const_iterator BeginImpl() const {
        return const_cast<DenseMapObj*>(this)->BeginImpl();
    }

    NODISCARD const_iterator EndImpl() const {
        return const_cast<DenseMapObj*>(this)->EndImpl();
    }

    NODISCARD const_iterator FindImpl(const key_type& key) const {
        return const_cast<DenseMapObj*>(this)->FindImpl(key);
    }

    NODISCARD iterator FindImpl(const key_type& key) {
        const auto node = Search(key);
        return node.IsNone() ? EndImpl() : iterator(node.index(), this);
    }

    iterator EraseImpl(iterator pos);

    NODISCARD size_type count_impl(const key_type& key) const {
        return !Search(key).IsNone();
    }

    NODISCARD mapped_type& AtImpl(const key_type& key) {
        return At(key);
    }

    NODISCARD const mapped_type& AtImpl(const key_type& key) const {
        return At(key);
    }

    NODISCARD Cursor CreateCursorFromHash(size_t hash_value) const {
        return {details::FibonacciHash(hash_value, fib_shift_), this};
    }

    NODISCARD Block* GetBlockByIndex(size_t block_idx) const {
        return static_cast<Block*>(this->data()) + block_idx;
    }

    NODISCARD value_type* GetDataPtrImpl(size_t index) const {
        return &Cursor(index, this).GetData();
    }

    // Construct a ListNode from hash code if the position is head of list
    NODISCARD std::optional<Cursor> FindListHeadByHash(size_t hash_value) const {
        if (const auto head = CreateCursorFromHash(hash_value); head.IsHead()) {
            return head;
        }

        return std::nullopt;
    }

    NODISCARD size_type GetNextIndexOfImpl(size_type idx) const;

    NODISCARD size_type GetPrevIndexOfImpl(size_type idx) const;

    NODISCARD mapped_type& At(const key_type& key) const;

    void reset();

    /*!
   * \brief Search for the given key, throw exception if not exists
   * \param key The key
   * \return ListNode that associated with the key
   */
    NODISCARD Cursor Search(const key_type& key) const;

    // Whether the hash table is full.
    NODISCARD bool IsFull() const {
        return this->size() + 1 >
               static_cast<size_type>(static_cast<double>(this->slots()) * kMaxLoadFactor);
    }

    static size_type CalculateBlockCount(size_type total_slots) {
        return (total_slots + kEntriesPerBlock - 1) / kEntriesPerBlock;
    }

    // Calculate the power-of-2 table size given the lower-bound of required capacity.
    // shift = 64 - log2(slots)
    static std::pair<uint32_t, size_type> CalculateSlotCount(size_type cap);

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

    // may be rehash
    static std::tuple<ObjectPtr<Object>, iterator, bool> InsertImpl(
            value_type&& kv, const ObjectPtr<Object>& old_impl, bool assign = false);

    static ObjectPtr<DenseMapObj> Create(size_type n);

    static ObjectPtr<DenseMapObj> CopyFrom(const DenseMapObj* src);

    template<typename, typename, typename, typename>
    friend class MapObj;
    template<typename, typename, typename>
    friend class Map;
};

template<typename K, typename V, typename Hasher>
struct DenseMapObj<K, V, Hasher>::Entry {
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
struct DenseMapObj<K, V, Hasher>::Block {
    std::array<std::byte, kEntriesPerBlock + kEntriesPerBlock * sizeof(Entry)> storage_;

    Block() {// NOLINT
        for (uint8_t i = 0; i < kEntriesPerBlock; ++i) {
            storage_[i] = kEmptySlot;
        }
    }

    Block(const Block& other) {// NOLINT
        for (uint8_t i = 0; i < kEntriesPerBlock; ++i) {
            if (other.storage_[i] != kEmptySlot) {
                storage_[i] = other.storage_[i];
                new (GetEntryPtr(i)) Entry(*other.GetEntryPtr(i));
            }
        }
    }

    ~Block() {
        for (uint8_t i = 0; i < kEntriesPerBlock; ++i) {
            if (storage_[i] != kEmptySlot) {
                storage_[i] = kEmptySlot;
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
class DenseMapObj<K, V, Hasher>::Cursor {
public:
    Cursor() : index_(0), obj_(nullptr) {}

    Cursor(size_t index, const DenseMapObj* p) : index_(index), obj_(p) {}

    NODISCARD size_t index() const {
        return index_;
    }

    NODISCARD const DenseMapObj* obj() const {
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
        return GetBlock()->storage_[index() % kEntriesPerBlock];
    }

    // Get the entry ref
    NODISCARD Entry& GetEntry() const {
        CHECK(!IsNone()) << "The Cursor is none.";
        CHECK(!IsSlotEmpty()) << "The entry is empty.";
        return *GetBlock()->GetEntryPtr(index() % kEntriesPerBlock);
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
        return GetSlotMetadata() == kEmptySlot;
    }

    NODISCARD bool IsSlotProtected() const {
        return GetSlotMetadata() == kProtectedSlot;
    }

    NODISCARD bool IsHead() const {
        return (GetSlotMetadata() & kHeadFlagMask) == kHeadFlag;
    }

    void MarkSlotAsEmpty() const {
        GetSlotMetadata() = kEmptySlot;
    }

    void MarkSlotAsProtected() const {
        GetSlotMetadata() = kProtectedSlot;
    }

    // Set the entry's offset to its next entry.
    void SetNextSlotOffsetIndex(uint8_t offset_idx) const {
        CHECK(offset_idx < kNumOffsetDists);
        (GetSlotMetadata() &= kHeadFlagMask) |= std::byte{offset_idx};
    }

    void ConstructEntry(Entry&& entry) const {
        CHECK(IsSlotEmpty());
        new (GetBlock()->GetEntryPtr(index() % kEntriesPerBlock)) Entry(std::move(entry));
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
        GetSlotMetadata() = kHeadFlag;
    }

    // Construct a tail of linked list inplace
    void CreateTail(Entry&& entry) const {
        if (!IsSlotEmpty()) {
            DestroyEntry();
        }
        ConstructEntry(std::move(entry));
        GetSlotMetadata() = kTailFlag;
    }

    // Whether the slot has the next slot on the linked list
    NODISCARD bool HasNextSlot() const {
        const auto idx = std::to_integer<uint8_t>(GetSlotMetadata() & kOffsetIdxMask);
        return NextProbePosOffset[idx] != 0;
    }

    // Move the current cursor to the next slot on the linked list
    bool MoveToNextSlot(std::optional<std::byte> meta_opt = std::nullopt) {
        std::byte meta = meta_opt ? meta_opt.value() : GetSlotMetadata();
        const auto idx = std::to_integer<uint8_t>(meta & kOffsetIdxMask);
        const auto offset = NextProbePosOffset[idx];
        if (offset == 0) {
            reset();
            return false;
        }

        // The probing will go to the next pos and round back to stay within
        // the correct range of the slots.
        index_ = (index_ + offset) % obj()->slots();
        return true;
    }

    // Get the prev slot on the linked list
    NODISCARD Cursor FindPrevSlot() const {
        // start from the head of the linked list, which must exist
        auto cur = obj()->CreateCursorFromHash(get_hash(GetKey()));
        auto prev = cur;

        cur.MoveToNextSlot();
        while (index() != cur.index()) {
            prev = cur;
            cur.MoveToNextSlot();
        }

        return prev;
    }

    NODISCARD std::optional<std::pair<uint8_t, Cursor>> GetNextEmptySlot() const {
        for (uint8_t i = 1; i < kNumOffsetDists; ++i) {
            if (Cursor candidate((index() + NextProbePosOffset[i]) % obj()->slots(), obj());
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
    const DenseMapObj* obj_;
};

template<typename K, typename V, typename Hasher>
DenseMapObj<K, V, Hasher>::iterator DenseMapObj<K, V, Hasher>::EraseImpl(iterator pos) {
    if (pos == this->end()) {
        return this->end();
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
    return next_pos;
}

template<typename K, typename V, typename Hasher>
DenseMapObj<K, V, Hasher>::size_type DenseMapObj<K, V, Hasher>::GetNextIndexOfImpl(size_type idx) const {
    // keep at the end of iterator
    if (idx == kInvalidIndex) {
        return idx;
    }

    return Cursor(idx, this).GetEntry().next;
}

template<typename K, typename V, typename Hasher>
DenseMapObj<K, V, Hasher>::size_type DenseMapObj<K, V, Hasher>::GetPrevIndexOfImpl(size_type idx) const {
    // this is the end iterator, we need to return tail.
    if (idx == kInvalidIndex) {
        return iter_list_tail_;
    }

    return Cursor(idx, this).GetEntry().prev;
}

template<typename K, typename V, typename Hasher>
DenseMapObj<K, V, Hasher>::mapped_type& DenseMapObj<K, V, Hasher>::At(const key_type& key) const {
    const Cursor iter = Search(key);
    if (iter.IsNone()) {
        AETHERMIND_THROW(KeyError) << "Key not found";
    }

    return iter.GetValue();
}

template<typename K, typename V, typename Hasher>
void DenseMapObj<K, V, Hasher>::reset() {
    const size_t block_num = CalculateBlockCount(this->slots());
    for (size_t i = 0; i < block_num; ++i) {
        GetBlockByIndex(i)->~Block();
    }

    this->size_ = 0;
    this->slots_ = 0;
    fib_shift_ = 63;
}

template<typename K, typename V, typename Hasher>
DenseMapObj<K, V, Hasher>::Cursor DenseMapObj<K, V, Hasher>::Search(const key_type& key) const {
    if (this->empty()) {
        return {};
    }

    const auto head_opt = FindListHeadByHash(get_hash(key));
    if (!head_opt.has_value()) {
        return {};
    }

    auto node = *head_opt;
    do {
        if (key == node.GetKey()) {
            return node;
        }
    } while (node.MoveToNextSlot());

    return {};
}

template<typename K, typename V, typename Hasher>
std::pair<uint32_t, typename DenseMapObj<K, V, Hasher>::size_type> DenseMapObj<K, V, Hasher>::CalculateSlotCount(size_type cap) {
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
void DenseMapObj<K, V, Hasher>::IterListPushBack(Cursor node) {
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
void DenseMapObj<K, V, Hasher>::IterListRemove(Cursor node) {
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
void DenseMapObj<K, V, Hasher>::IterListReplace(Cursor src, Cursor dst) {
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
std::optional<typename DenseMapObj<K, V, Hasher>::Cursor>
DenseMapObj<K, V, Hasher>::TryAllocateListHead(Cursor target) {
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
std::pair<typename DenseMapObj<K, V, Hasher>::iterator, bool>
DenseMapObj<K, V, Hasher>::TryInsertOrUpdate(value_type&& kv, bool assign) {
    // The key is already in the hash table
    if (auto it = FindImpl(kv.first); it != EndImpl()) {
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
    auto node = CreateCursorFromHash(get_hash(kv.first));

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
            return {EndImpl(), false};
        }

        if (const auto target = TryAllocateListHead(node);
            target.has_value()) {
            target->CreateHead(Entry{std::move(kv)});
            ++this->size_;
            IterListPushBack(target.value());
            return {iterator(target->index(), this), true};
        }
        return {EndImpl(), false};
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
        return {EndImpl(), false};
    }

    // find the next empty slot
    auto empty_slot_info = node.GetNextEmptySlot();
    if (!empty_slot_info) {
        return {EndImpl(), false};
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
std::tuple<ObjectPtr<Object>, typename DenseMapObj<K, V, Hasher>::iterator, bool>
DenseMapObj<K, V, Hasher>::InsertImpl(value_type&& kv, const ObjectPtr<Object>& old_impl, bool assign) {
    auto* map = static_cast<DenseMapObj*>(old_impl.get());// NOLINT
    if (auto [it, is_success] = map->TryInsertOrUpdate(std::move(kv), assign); it != map->EndImpl()) {
        return {old_impl, it, is_success};
    }

    // Otherwise, start rehash
    auto new_impl = details::ObjectUnsafe::Downcast<DenseMapObj>(
            Create(map->slots() * DenseMapObj::kIncFactor));
    // need to insert in the same order as the original map
    size_t idx = map->iter_list_head_;
    while (idx != kInvalidIndex) {
        Cursor cur(idx, map);
        new_impl->TryInsertOrUpdate(std::move(cur.GetData()), assign);
        idx = cur.GetEntry().next;
        cur.DestroyEntry();
    }

    auto [pos, is_success] = new_impl->TryInsertOrUpdate(std::move(kv), assign);
    return {new_impl, pos, is_success};
}

template<typename K, typename V, typename Hasher>
ObjectPtr<DenseMapObj<K, V, Hasher>> DenseMapObj<K, V, Hasher>::Create(size_type n) {
    CHECK(n > DenseMapObj::kThreshold) << "The allocated size must be greate than the threshold of "
                                       << DenseMapObj::kThreshold
                                       << " when using SmallMapObj::Create";
    auto [fib_shift, slots] = CalculateSlotCount(n);
    const size_t block_num = CalculateBlockCount(slots);
    auto impl = make_array_object<DenseMapObj, Block>(block_num);
    impl->data_ = reinterpret_cast<char*>(impl.get()) + sizeof(DenseMapObj);
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
ObjectPtr<DenseMapObj<K, V, Hasher>> DenseMapObj<K, V, Hasher>::CopyFrom(const DenseMapObj* src) {
    auto impl = Create(src->slots());
    auto block_num = CalculateBlockCount(src->slots());
    impl->size_ = src->size();
    impl->iter_list_head_ = src->iter_list_head_;
    impl->iter_list_tail_ = src->iter_list_tail_;

    auto* p = static_cast<Block*>(impl->data());
    for (size_t i = 0; i < block_num; ++i) {
        new (p + i) Block(*src->GetBlockByIndex(i));
    }

    return impl;
}

template<typename Derived, typename K, typename V, typename Hasher>
std::tuple<ObjectPtr<Object>, typename MapObj<Derived, K, V, Hasher>::size_type, bool>
MapObj<Derived, K, V, Hasher>::insert(value_type&& kv, const ObjectPtr<Object>& old_impl, bool assign) {
    using SmallMapType = SmallMapObj<K, V, Hasher>;
    using DenseMapType = DenseMapObj<K, V, Hasher>;
    if constexpr (std::is_same_v<Derived, SmallMapType>) {
        auto* p = static_cast<SmallMapType*>(old_impl.get());//NOLINT
        const auto size = p->size();
        if (size < kThreshold) {
            auto [iter, is_success] = p->InsertImpl(std::move(kv), assign);
            return {old_impl, iter.index(), is_success};
        }

        ObjectPtr<Object> new_impl = DenseMapType::Create(size * kIncFactor);
        for (auto& iter: *p) {
            new_impl = std::get<0>(DenseMapType::InsertImpl(std::move(iter), new_impl));
        }
        auto [impl, iter, is_success] = DenseMapType::InsertImpl(std::move(kv), new_impl, assign);
        return {impl, iter.index(), is_success};
    } else {
        auto [impl, iter, is_success] = DenseMapType::InsertImpl(std::move(kv), old_impl, assign);
        return {impl, iter.index(), is_success};
    }
}

template<typename K, typename V, typename Hasher = hash<K>>
class Map : public ObjectRef {
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

    template<bool IsConst>
    class IteratorImpl;

    class ValueProxy;

    using iterator = IteratorImpl<false>;
    using const_iterator = IteratorImpl<true>;

    using SmallMapType = SmallMapObj<K, V, Hasher>;
    using DenseMapType = DenseMapObj<K, V, Hasher>;

    Map() : obj_(SmallMapType::Create()) {}

    explicit Map(size_type n) {
        if (n <= SmallMapType::kThreshold) {
            obj_ = SmallMapType::Create();
        } else {
            obj_ = DenseMapType::Create(n);
        }
    }

    template<typename Iter>
        requires requires(Iter t) {
            requires details::is_valid_iter<Iter>;
            { *t } -> std::convertible_to<value_type>;
        }
    Map(Iter first, Iter last) {
        auto _sz = std::distance(first, last);

        if (_sz <= 0) {
            obj_ = SmallMapType::Create();
        } else {
            const auto size = static_cast<size_type>(_sz);
            if (size <= SmallMapType::kThreshold) {
                obj_ = SmallMapType::Create();
                while (first != last) {
                    SmallMapType::insert(value_type(*first++), std::get<ObjectPtr<SmallMapType>>(obj_));
                }
            } else {
                obj_ = DenseMapType::Create(size);
                while (first != last) {
                    auto [t, ph0, ph1] = DenseMapType::insert(value_type(*first++),
                                                              std::get<ObjectPtr<DenseMapType>>(obj_));
                    obj_ = details::ObjectUnsafe::Downcast<DenseMapType>(t);
                }
            }
        }
    }

    Map(std::initializer_list<value_type> list) : Map(list.begin(), list.end()) {}

    Map(const Map& other) = default;

    Map(Map&& other) noexcept : obj_(std::move(other.obj_)) {//NOLINT
        // other.clear();
    }

    template<typename KU, typename VU>
        requires std::is_base_of_v<key_type, KU> && std::is_base_of_v<mapped_type, VU>
    Map(const Map<KU, VU>& other) : obj_(other.obj_) {}//NOLINT

    template<typename KU, typename VU>
        requires std::is_base_of_v<key_type, KU> && std::is_base_of_v<mapped_type, VU>
    Map(Map<KU, VU>&& other) noexcept : obj_(other.obj_) {//NOLINT
        other.clear();
    }

    Map& operator=(const Map& other) {
        Map(other).swap(*this);
        return *this;
    }

    Map& operator=(Map&& other) noexcept {
        Map(std::move(other)).swap(*this);
        return *this;
    }

    template<typename KU, typename VU>
        requires std::is_base_of_v<key_type, KU> && std::is_base_of_v<mapped_type, VU>
    Map& operator=(const Map<KU, VU>& other) {
        if (this != &other) {
            obj_ = other.obj_;
        }
        return *this;
    }

    template<typename KU, typename VU>
        requires std::is_base_of_v<key_type, KU> && std::is_base_of_v<mapped_type, VU>
    Map& operator=(Map<KU, VU>&& other) noexcept {
        if (this != &other) {
            obj_ = other.obj_;
            other.clear();
        }
        return *this;
    }

    Map& operator=(std::initializer_list<value_type> list) {
        Map(list).swap(*this);
        return *this;
    }

    NODISCARD size_type size() const noexcept {
        return std::visit([](const auto& arg) { return arg->size(); }, obj_);
    }

    NODISCARD size_type slots() const noexcept {
        return std::visit([](const auto& arg) { return arg->slots(); }, obj_);
    }

    NODISCARD bool empty() const noexcept {
        return size() == 0;
    }

    NODISCARD uint32_t use_count() const noexcept {
        return std::visit([](const auto& arg) { return arg->use_count(); }, obj_);
    }

    NODISCARD bool unique() const noexcept {
        return use_count() == 1;
    }

    NODISCARD size_type count(const key_type& key) const {
        return contains(key) ? 1 : 0;
    }

    iterator begin() noexcept {
        return std::visit([](const auto& arg) { return iterator(arg->begin()); }, obj_);
    }

    iterator end() noexcept {
        return std::visit([](const auto& arg) { return iterator(arg->end()); }, obj_);
    }

    const_iterator begin() const noexcept {
        return const_cast<Map*>(this)->begin();
    }

    const_iterator end() const noexcept {
        return const_cast<Map*>(this)->end();
    }

    iterator find(const key_type& key) {
        return std::visit([&](const auto& arg) { return iterator(arg->find(key)); }, obj_);
    }

    const_iterator find(const key_type& key) const {
        return const_cast<Map*>(this)->find(key);
    }

    bool contains(const key_type& key) const {
        return find(key) != end();
    }

    mapped_type& at(const key_type& key) {
        auto it = find(key);
        if (it == end()) {
            AETHERMIND_THROW(KeyError) << "Key does not exist";
        }

        return it->second;
    }

    const mapped_type& at(const key_type& key) const {
        return const_cast<Map*>(this)->at(key);
    }

    const mapped_type& operator[](const key_type& key) const {
        auto it = find(key);
        return it->second;
    }

    // mapped_type& operator[](const key_type& key) {
    //     auto it = find(key);
    //     if (it == end()) {
    //         auto [iter, _] = insert(key, mapped_type{});
    //         return iter->second;
    //     }
    //     return it->second;
    // }

    // mapped_type& operator[](key_type&& key) {
    //     auto it = find(key);
    //     if (it == end()) {
    //         auto [iter, _] = insert(std::move(key), mapped_type{});
    //         return iter->second;
    //     }
    //     return it->second;
    // }

    ValueProxy operator[](const key_type& key) {
        auto it = find(key);
        if (it == end()) {
            auto [iter, _] = insert(key, mapped_type{});
            return {*this, iter.index()};
        }
        return {*this, it.index()};
    }

    ValueProxy operator[](key_type&& key) {
        auto it = find(key);
        if (it == end()) {
            auto [iter, _] = insert(std::move(key), mapped_type{});
            return {*this, iter.index()};
        }
        return {*this, it.index()};
    }

    std::pair<iterator, bool> insert(value_type&& x) {
        return insert_impl(std::move(x), false);
    }

    std::pair<iterator, bool> insert(const value_type& x) {
        return insert_impl(value_type(x), false);
    }

    std::pair<iterator, bool> insert(const key_type& key, const mapped_type& value) {
        return insert_impl({key, value}, false);
    }

    std::pair<iterator, bool> insert(key_type&& key, mapped_type&& value) {
        return insert_impl({std::move(key), std::move(value)}, false);
    }

    template<typename Pair>
        requires(std::constructible_from<value_type, Pair &&> &&
                 !std::same_as<std::decay_t<Pair>, value_type>)
    std::pair<iterator, bool> insert(Pair&& x) {
        return insert_impl(value_type(std::forward<Pair>(x)), false);
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
        return insert_impl({std::move(key), std::forward<Obj>(obj)}, true);
    }

    template<typename Obj>
    std::pair<iterator, bool> insert_or_assign(const key_type& key, Obj&& obj) {
        return insert_impl({key, std::forward<Obj>(obj)}, true);
    }

    template<typename... Args>
    std::pair<iterator, bool> emplace(Args&&... args) {
        return insert_impl(value_type(std::forward<Args>(args)...), false);
    }

    iterator erase(iterator pos);
    iterator erase(const_iterator pos);
    size_type erase(const key_type& key);
    iterator erase(iterator first, iterator last);
    iterator erase(const_iterator first, const_iterator last);

    void clear() {
        obj_ = SmallMapType::Create();
    }

    void swap(Map& other) noexcept {
        std::swap(obj_, other.obj_);
    }

    NODISCARD bool IsSmallMap() const {
        return std::holds_alternative<ObjectPtr<SmallMapType>>(obj_);
    }

private:
    std::variant<ObjectPtr<SmallMapType>, ObjectPtr<DenseMapType>> obj_;

    NODISCARD SmallMapType* small_ptr() const {
        return std::get<ObjectPtr<SmallMapType>>(obj_).get();
    }

    NODISCARD DenseMapType* dense_ptr() const {
        return std::get<ObjectPtr<DenseMapType>>(obj_).get();
    }

    std::pair<iterator, bool> insert_impl(value_type&& x, bool assign);

    void COW() {
        if (!unique()) {
            if (IsSmallMap()) {
                obj_ = SmallMapType::CopyFrom(small_ptr());
            } else {
                obj_ = DenseMapType::CopyFrom(dense_ptr());
            }
        }
    }
};

template<typename K, typename V, typename Hasher>
std::pair<typename Map<K, V, Hasher>::iterator, bool> Map<K, V, Hasher>::insert_impl(value_type&& x, bool assign) {
    if (!assign) {
        auto it = find(x.first);
        if (it != end()) {
            return {it, false};
        }
    }

    COW();
    auto visitor = [&]<typename T>(const ObjectPtr<T>& arg) -> std::pair<iterator, bool> {
        auto [impl, idx, is_success] = T::insert(std::move(x), arg, assign);
        if constexpr (std::is_same_v<T, SmallMapType>) {
            if (auto* p = dynamic_cast<SmallMapType*>(impl.get())) {
                typename SmallMapType::iterator pos{idx, p};
                return {pos, is_success};
            }
        }
        typename DenseMapType::iterator pos{idx, static_cast<DenseMapType*>(impl.get())};
        obj_ = details::ObjectUnsafe::Downcast<DenseMapType>(impl);
        return {pos, is_success};
    };

    return std::visit(visitor, obj_);
}

template<typename K, typename V, typename Hasher>
Map<K, V, Hasher>::iterator Map<K, V, Hasher>::erase(const_iterator pos) {
    if (pos == end()) {
        return end();
    }

    COW();
    auto visitor = [&](const auto& arg) -> iterator {
        return arg->erase({pos.index(), arg.get()});
    };
    return std::visit(visitor, obj_);
}

template<typename K, typename V, typename Hasher>
Map<K, V, Hasher>::iterator Map<K, V, Hasher>::erase(iterator pos) {
    return erase(const_iterator(pos));
}

template<typename K, typename V, typename Hasher>
Map<K, V, Hasher>::size_type Map<K, V, Hasher>::erase(const key_type& key) {
    auto it = find(key);
    if (it != end()) {
        erase(it);
        return 1;
    }
    return 0;
}

template<typename K, typename V, typename Hasher>
Map<K, V, Hasher>::iterator Map<K, V, Hasher>::erase(iterator first, iterator last) {
    if (first == last) {
        return first;
    }

    auto n = std::distance(first, last);
    iterator it = first;
    for (difference_type i = 0; i < n; ++i) {
        it = erase(it++);
    }
    return it;
}

template<typename K, typename V, typename Hasher>
Map<K, V, Hasher>::iterator Map<K, V, Hasher>::erase(const_iterator first, const_iterator last) {
    if (first == last) {
        return first;
    }

    auto n = std::distance(first, last);
    iterator it = first;
    for (difference_type i = 0; i < n; ++i) {
        it = erase(it++);
    }
    return it;
}

template<typename Derived, typename K, typename V, typename Hasher>
template<bool IsConst>
class MapObj<Derived, K, V, Hasher>::IteratorImpl {
public:
    using iterator_category = std::bidirectional_iterator_tag;
    using ContainerPtrType = std::conditional_t<IsConst, const Derived*, Derived*>;
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
            if (index_ == ptr()->end().index()) {
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
            if (index_ == ptr()->end().index()) {
                break;
            }
        }
        return *this;
    }

    IteratorImpl operator+(difference_type offset) const {
        Check();
        if constexpr (std::is_same_v<Derived, SmallMapObj<K, V>>) {
            size_type new_idx = index() + offset;
            if (new_idx > ptr()->size()) {
                new_idx = ptr()->size();
            }
            return IteratorImpl(new_idx, ptr());
        } else {
            auto res = *this;
            res += offset;
            return res;
        }
    }

    IteratorImpl operator-(difference_type offset) const {
        Check();
        if constexpr (std::is_same_v<Derived, SmallMapObj<K, V>>) {
            size_type new_idx = index() - offset;
            if (new_idx > ptr()->size()) {
                new_idx = ptr()->size();
            }
            return IteratorImpl(new_idx, ptr());
        } else {
            auto res = *this;
            res -= offset;
            return res;
        }
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

template<typename K, typename V, typename Hasher>
template<bool IsConst>
class Map<K, V, Hasher>::IteratorImpl {
public:
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = std::pair<const K, V>;
    using pointer = std::conditional_t<IsConst, const value_type*, value_type*>;
    using reference = std::conditional_t<IsConst, const value_type&, value_type&>;
    using SmallIterType = std::conditional_t<IsConst, typename SmallMapType::const_iterator,
                                             typename SmallMapType::iterator>;
    using DenseIterType = std::conditional_t<IsConst, typename DenseMapType::const_iterator,
                                             typename DenseMapType::iterator>;

    IteratorImpl() = default;

    // iterator can convert to const_iterator
    template<bool AlwaysFalse>
        requires(IsConst && !AlwaysFalse)
    IteratorImpl(const IteratorImpl<AlwaysFalse>& other) {//NOLINT
        std::visit([&](const auto& iter) { iter_ = iter; }, other.iter_);
    }

    NODISCARD size_type index() const {
        return std::visit([](const auto& iter) { return iter.index(); }, iter_);
    }

    NODISCARD bool IsSmallMap() const {
        return std::holds_alternative<SmallIterType>(iter_);
    }

    IteratorImpl& operator++() {
        std::visit([](auto& iter) { ++iter; }, iter_);
        return *this;
    }

    IteratorImpl& operator--() {
        std::visit([](auto& iter) { --iter; }, iter_);
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

    IteratorImpl operator+(difference_type offset) const {
        return std::visit([&](const auto& iter) -> IteratorImpl { return iter + offset; }, iter_);
    }

    IteratorImpl operator-(difference_type offset) const {
        return std::visit([&](const auto& iter) -> IteratorImpl { return iter - offset; }, iter_);
    }

    IteratorImpl& operator+=(difference_type offset) {
        std::visit([&](auto& iter) -> IteratorImpl& { return iter += offset; }, iter_);
        return *this;
    }

    IteratorImpl& operator-=(difference_type offset) {
        std::visit([&](auto& iter) -> IteratorImpl& { return iter -= offset; }, iter_);
        return *this;
    }

    reference operator*() const {
        return std::visit([](auto& iter) -> reference { return *iter; }, iter_);
    }

    pointer operator->() const {
        return std::visit([](auto& iter) -> pointer { return iter.operator->(); }, iter_);
    }

    bool operator==(const IteratorImpl& other) const {
        if (IsSmallMap() != other.IsSmallMap()) {
            return false;
        }

        return std::visit([&]<typename T>(const T& it) { return it == std::get<T>(other.iter_); }, iter_);
    }

    bool operator!=(const IteratorImpl& other) const {
        return !(*this == other);
    }

    operator SmallIterType() const {//NOLINT
        return std::get<SmallIterType>(iter_);
    }

    operator DenseIterType() const {//NOLINT
        return std::get<DenseIterType>(iter_);
    }

private:
    std::variant<SmallIterType, DenseIterType> iter_;

    IteratorImpl(const SmallIterType& iter) : iter_(iter) {}// NOLINT

    IteratorImpl(const DenseIterType& iter) : iter_(iter) {}//NOLINT

    template<typename, typename, typename>
    friend class Map;
};

template<typename K, typename V, typename Hasher>
class Map<K, V, Hasher>::ValueProxy {
public:
    ValueProxy(Map& map, size_type idx) : map_(map), idx_(idx) {}

    ValueProxy& operator=(mapped_type x) {
        map_.COW();
        GetDataPtr()->second = std::move(x);
        return *this;
    }

    const mapped_type* operator->() const noexcept {
        return std::addressof(GetDataPtr()->second);
    }

    mapped_type* operator->() noexcept {
        return std::addressof(GetDataPtr()->second);
    }

    template<typename MemPtr>
    decltype(auto) operator->*(MemPtr&& mem_ptr) & noexcept {
        return get().*std::forward<MemPtr>(mem_ptr);
    }

    template<typename MemPtr>
    decltype(auto) operator->*(MemPtr&& mem_ptr) && noexcept {
        return get().*std::forward<MemPtr>(mem_ptr);
    }

    template<typename U = mapped_type>
        requires details::is_map<U>
    decltype(auto) operator[](const U::key_type& key) {
        return get()[key];
    }

    template<typename U = V>
        requires details::is_container<U>
    decltype(auto) operator[](U::size_type i) {
        return get()[i];
    }

    operator const mapped_type&() const noexcept {//NOLINT
        return GetDataPtr()->second;
    }

    operator mapped_type&() noexcept {//NOLINT
        return GetDataPtr()->second;
    }

    const mapped_type& get() const noexcept {
        return GetDataPtr()->second;
    }

    mapped_type& get() noexcept {
        return GetDataPtr()->second;
    }

    friend bool operator==(const ValueProxy& lhs, const ValueProxy& rhs) {
        return lhs.GetDataPtr()->second == rhs.GetDataPtr()->second;
    }

    friend bool operator!=(const ValueProxy& lhs, const ValueProxy& rhs) {
        return !(lhs == rhs);
    }

private:
    Map& map_;
    size_type idx_;

    value_type* GetDataPtr() const {
        return std::visit([&](const auto& arg) {
            return arg->GetDataPtr(idx_);
        },
                          map_.obj_);
    }
};
}// namespace aethermind

#endif//AETHERMIND_CONTAINER_MAP_OBJ_H
