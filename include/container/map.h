//
// Created by richard on 11/28/25.
//

#ifndef AETHERMIND_MAP_H
#define AETHERMIND_MAP_H

#include "any.h"

namespace aethermind {

template<typename Derived>
class MapImpl : public Object {
public:
    using key_type = Any;
    using value_type = Any;
    using KVType = std::pair<key_type, value_type>;

    class iterator;

    MapImpl() : data_(nullptr), size_(0), slots_(0) {}

    NODISCARD size_t size() const {
        return size_;
    }

    NODISCARD size_t slots() const {
        return slots_;
    }

    NODISCARD bool empty() const {
        return size() == 0;
    }

    NODISCARD size_t count(const key_type& key) const {
        return static_cast<const Derived*>(this)->count_impl(key);
    }

    value_type& at(const key_type& key) {
        return static_cast<Derived*>(this)->at_impl(key);
    }

    NODISCARD const value_type& at(const key_type& key) const {
        return static_cast<const Derived*>(this)->at_impl(key);
    }

    NODISCARD iterator begin() const {
        return static_cast<const Derived*>(this)->begin_impl();
    }

    NODISCARD iterator end() const {
        return static_cast<const Derived*>(this)->end_impl();
    }

    NODISCARD iterator find(const key_type& key) const {
        return static_cast<const Derived*>(this)->find_impl(key);
    }

    void erase(const iterator& pos) {
        return static_cast<Derived*>(this)->erase_impl(pos);
    }

    void erase(const key_type& key) {
        erase(find(key));
    }

protected:
    void* data_;
    size_t size_;
    size_t slots_;

    static constexpr size_t kInitSize = 2; // Init map size
    static constexpr size_t kThreshold = 4; // The threshold of the small and dense map

    template<typename Iter>
        requires requires(Iter t) {
            { *t } -> std::convertible_to<KVType>;
        }
    static ObjectPtr<Object> CreateFromRange(Iter first, Iter last);

    static ObjectPtr<Object> CopyFrom(const Object* src);

    // insert may be rehash
    static ObjectPtr<Object> insert(const KVType& kv, ObjectPtr<Object> old_impl);
};

template<typename Derived>
class MapImpl<Derived>::iterator {
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = KVType;
    using pointer = value_type*;
    using reference = value_type&;
    using difference_type = int64_t;

    iterator() : index_(0), ptr_(nullptr) {}

    iterator(size_t index, const MapImpl* ptr) : index_(index), ptr_(ptr) {}

    NODISCARD size_t index() const {
        return index_;
    }

    pointer operator->() const {
        const auto* p = static_cast<const Derived*>(ptr_);
        return p->DeRefIter(index_);
    }

    reference operator*() const {
        return *operator->();
    }

    iterator& operator++() {
        const auto* p = static_cast<const Derived*>(ptr_);
        index_ = p->IncIter(index_);
        return *this;
    }

    iterator& operator--() {
        const auto* p = static_cast<const Derived*>(ptr_);
        index_ = p->DecIter(index_);
        return *this;
    }

    iterator operator++(int) {
        iterator tmp = *this;
        operator++();
        return tmp;
    }

    iterator operator--(int) {
        iterator tmp = *this;
        operator--();
        return tmp;
    }

    bool operator==(const iterator& other) const {
        return index_ == other.index_ && ptr_ == other.ptr_;
    }

    bool operator!=(const iterator& other) const {
        return !(*this == other);
    }

protected:
    size_t index_;      // The position in the array.
    const MapImpl* ptr_;// The container it pointer to.


    friend class SmallMapImpl;
    friend class DenseMapImpl;
};

class SmallMapImpl : public MapImpl<SmallMapImpl> {
public:
    SmallMapImpl() = default;
    ~SmallMapImpl() override;

private:

    NODISCARD iterator begin_impl() const {
        return {0, this};
    }

    NODISCARD iterator end_impl() const {
        return {size(), this};
    }

    NODISCARD iterator find_impl(const key_type& key) const;

    NODISCARD size_t count_impl(const key_type& key) const {
        return find(key).index() < size();
    }

    value_type& at_impl(const key_type& key);

    NODISCARD const value_type& at_impl(const key_type& key) const;

    void erase_impl(const iterator& pos);

    NODISCARD KVType* DeRefIter(size_t index) const {
        return static_cast<KVType*>(data_) + index;
    }

    NODISCARD size_t IncIter(size_t index) const {
        return index + 1 < size_ ? index + 1 : size_;
    }

    NODISCARD size_t DecIter(size_t index) const {
        return index > 0 ? index - 1 : size_;
    }

    static ObjectPtr<SmallMapImpl> Create(size_t n = kInitSize);

    static ObjectPtr<SmallMapImpl> CopyFromImpl(const SmallMapImpl* src);

    template<typename Iter>
    static ObjectPtr<SmallMapImpl> CreateFromRangeImpl(Iter first, Iter last) {
        const auto n = std::distance(first, last);
        auto impl = Create(n);
        auto* ptr = static_cast<KVType*>(impl->data_);
        while (first != last) {
            new (ptr++) KVType(*first++);
        }
        return impl;
    }

    static ObjectPtr<Object> InsertMaybeRehash(const KVType& kv, ObjectPtr<Object> old_impl);

    template<typename Derived>
    friend class MapImpl;
    friend class DenseMapImpl;
    template<typename K, typename V>
    friend class Map;
};


/*! \brief A specialization of hash map that implements the idea of array-based hash map.
 * Another reference implementation can be found [1].
 *
 * A. Overview
 *
 * DenseMapObj did several improvements over traditional separate chaining hash,
 * in terms of cache locality, memory footprints and data organization.
 *
 * A1. Implicit linked list. For better cache locality, instead of using linked list
 * explicitly for each bucket, we store list data into a single array that spans contiguously
 * in memory, and then carefully design access patterns to make sure most of them fall into
 * a single cache line.
 *
 * A2. 1-byte metadata. There is only 1 byte overhead for each slot in the array to indexing and
 * traversal. This can be divided in 3 parts.
 * 1) Reserved code: (0b11111111)_2 indicates a slot is empty; (0b11111110)_2 indicates protected,
 * which means the slot is empty but not allowed to be written.
 * 2) If not empty or protected, the highest bit is used to indicate whether data in the slot is
 * head of a linked list.
 * 3) The rest 7 bits are used as the "next pointer" (i.e. pointer to the next element). On 64-bit
 * architecture, an ordinary pointer can take up to 8 bytes, which is not acceptable overhead when
 * dealing with 16-byte ObjectRef pairs. Based on a commonly noticed fact that the lists are
 * relatively short (length <= 3) in hash maps, we follow [1]'s idea that only allows the pointer to
 * be one of the 126 possible values, i.e. if the next element of i-th slot is (i + x)-th element,
 * then x must be one of the 126 pre-defined values.
 *
 * A3. Data blocking. We organize the array in the way that every 16 elements forms a data block.
 * The 16-byte metadata of those 16 elements is stored together, followed by the real data, i.e.
 * 16 key-value pairs.
 *
 * B. Implementation details
 *
 * B1. Power-of-2 table size and Fibonacci Hashing. We use power-of-two as table size to avoid
 * modulo for more efficient arithmetics. To make the hash-to-slot mapping distribute more evenly,
 * we use the Fibonacci Hashing [2] trick.
 *
 * B2. Traverse a linked list in the array.
 * 1) List head. Assume Fibonacci Hashing maps a given key to slot i, if metadata at slot i
 * indicates that it is list head, then we found the head; otherwise the list is empty. No probing
 * is done in this procedure. 2) Next element. To find the next element of a non-empty slot i, we
 * look at the last 7 bits of the metadata at slot i. If they are all zeros, then it is the end of
 * the list; otherwise, we know that the next element is (i + candidates[the-last-7-bits]).
 *
 * B3. InsertMaybeReHash an element. Following B2, we first traverse the linked list to see if this
 * element is in the linked list, and if not, we put it at the end by probing the next empty
 * position in one of the 126 candidate positions. If the linked list does not even exist, but the
 * slot for list head has been occupied by another linked list, we should find this intruder another
 * place.
 *
 * B4. Quadratic probing with triangle numbers. In open address hashing, it is provable that probing
 * with triangle numbers can traverse power-of-2-sized table [3]. In our algorithm, we follow the
 * suggestion in [1] that also use triangle numbers for "next pointer" as well as sparing for list
 * head.
 *
 * [1] https://github.com/skarupke/flat_hash_map
 * [2] https://programmingpraxis.com/2018/06/19/fibonacci-hash/
 * [3] https://fgiesen.wordpress.com/2015/02/22/triangular-numbers-mod-2n/
 */
class DenseMapImpl : public MapImpl<DenseMapImpl> {
public:
    DenseMapImpl() = default;
    ~DenseMapImpl() override {
        reset();
    }

private:
    struct Entry;
    struct Block;
    class ListNode;

    // The number of elements in a memory block.
    static constexpr int kBlockSize = 16;
    // Max load factor of hash table
    static constexpr double kMaxLoadFactor = 0.99;
    // 0b11111111 representation of the metadata of an empty slot.
    static constexpr uint8_t kEmptySlot = 0xFF;
    // 0b11111110 representation of the metadata of a protected slot.
    static constexpr uint8_t kProtectedSlot = 0xFE;
    // Number of probing choices available
    static constexpr int kNumOffsetDists = 126;
    // Index indicator to indicate an invalid index.
    static constexpr size_t kInvalidIndex = std::numeric_limits<size_t>::max();

    static const size_t NextProbePosOffset[kNumOffsetDists];

    // fib shift in Fibonacci hash
    uint32_t fib_shift_ = 63;
    // The head of iterator list
    size_t iter_list_head_ = kInvalidIndex;
    // The tail of iterator list
    size_t iter_list_tail_ = kInvalidIndex;

    NODISCARD iterator begin_impl() const {
        return {iter_list_head_, this};
    }

    NODISCARD iterator end_impl() const {
        return {kInvalidIndex, this};
    }

    NODISCARD iterator find_impl(const key_type& key) const;

    NODISCARD size_t count_impl(const key_type& key) const;

    NODISCARD value_type& at_impl(const key_type& key) {
        return At(key);
    }

    NODISCARD const value_type& at_impl(const key_type& key) const {
        return At(key);
    }

    void erase_impl(const iterator& pos);

    NODISCARD Block* GetBlock(size_t block_idx) const;

    NODISCARD ListNode IndexFromHash(size_t hash_value) const;

    NODISCARD KVType* DeRefIter(size_t index) const;

    // Construct a ListNode from hash code if the position is head of list
    NODISCARD ListNode GetListHead(size_t hash_value) const;

    NODISCARD size_t IncIter(size_t index) const;

    NODISCARD size_t DecIter(size_t index) const;

    NODISCARD value_type& At(const key_type& key) const;

    /*!
   * \brief Search for the given key, throw exception if not exists
   * \param key The key
   * \return ListNode that associated with the key
   */
    NODISCARD ListNode Search(const key_type& key) const;

    // Whether the hash table is full.
    NODISCARD bool IsFull() const {
        return size() + 1 > static_cast<size_t>(static_cast<double>(slots()) * kMaxLoadFactor);
    }

    void reset();

    // Insert the entry into tail of iterator list.
    // This function does not change data content of the node.
    // or NodeListPushBack ?
    void IterListPushBack(ListNode node);

    // Unlink the entry from iterator list.
    // This function is usually used before deletion,
    // and it does not change data content of the node.
    void IterListUnlink(ListNode node);

    /*!
   * \brief Replace node src by dst in the iter list
   * \param src The source node
   * \param dst The destination node, must be empty
   * \note This function does not change data content of the nodes,
   *       which needs to be updated by the caller.
   */
    void IterListReplaceNodeBy(ListNode src, ListNode dst);

    /*!
   * \brief Spare an entry to be the head of a linked list.
   * As described in B3, during insertion, it is possible that the entire linked list does not
   * exist, but the slot of its head has been occupied by other linked lists. In this case, we need
   * to spare the slot by moving away the elements to another valid empty one to make insertion
   * possible.
   * \param target The given entry to be spared
   * \param key The indexing key
   * \return The linked-list entry constructed as the head, if actual insertion happens
   */
    // bool TrySpareListHead(ListNode target, const key_type& key, ListNode* result);
    std::optional<ListNode> TrySpareListHead(ListNode target, const key_type& key);

    /*!
   * \brief Try to insert a key, or do nothing if already exists
   * \param key The indexing key
   * \return The linked-list entry found or just constructed,indicating if actual insertion happens
   */
    // bool TryInsert(const key_type& key, ListNode* result);
    std::optional<ListNode> TryInsert(const key_type& key);

    static ObjectPtr<Object> InsertMaybeRehash(const KVType& kv, ObjectPtr<Object> old_impl);

    static size_t ComputeBlockNum(size_t slots) {
        return (slots + kBlockSize - 1) / kBlockSize;
    }

    // Calculate the power-of-2 table size given the lower-bound of required capacity.
    // shift = 64 - log2(slots)
    static std::pair<uint32_t, size_t> ComputeSlotNum(size_t cap);

    static ObjectPtr<DenseMapImpl> Create(uint32_t fib_shift, size_t slots);

    static ObjectPtr<DenseMapImpl> CopyFromImpl(const DenseMapImpl* src);

    template<typename Derived>
    friend class MapImpl;
    template<typename K, typename V>
    friend class Map;
};

template<typename Derived>
template<typename Iter>
    requires requires(Iter t) {
        { *t } -> std::convertible_to<std::pair<Any, Any>>;
    }
ObjectPtr<Object> MapImpl<Derived>::CreateFromRange(Iter first, Iter last) {
    const int64_t _cap = std::distance(first, last);
    if (_cap <= 0) {
        return SmallMapImpl::Create();
    }

    auto cap = static_cast<size_t>(_cap);
    if (cap < kThreshold) {
        if (cap < kInitSize) {
            return SmallMapImpl::CreateFromRangeImpl(first, last);
        }

        // need to insert to avoid duplicate keys
        ObjectPtr<Object> impl = SmallMapImpl::Create(cap);
        while (first != last) {
            impl = SmallMapImpl::InsertMaybeRehash(KVType(*first++), impl);
        }
        return impl;
    }

    auto [fib_shift, slots] = DenseMapImpl::ComputeSlotNum(cap);
    ObjectPtr<Object> impl = DenseMapImpl::Create(fib_shift, slots);
    while (first != last) {
        impl = DenseMapImpl::InsertMaybeRehash(KVType(*first++), impl);
    }
    return impl;
}

template<typename Derived>
ObjectPtr<Object> MapImpl<Derived>::CopyFrom(const Object* src) {
    const auto* p = static_cast<const Derived*>(src);
    if constexpr (std::is_same_v<Derived, SmallMapImpl>) {
        return SmallMapImpl::CopyFromImpl(p);
    } else {
        return DenseMapImpl::CopyFromImpl(p);
    }
}

template<typename Derived>
ObjectPtr<Object> MapImpl<Derived>::insert(const KVType& kv, ObjectPtr<Object> old_impl) {
    if constexpr (std::is_same_v<Derived, SmallMapImpl>) {
        auto* p = static_cast<SmallMapImpl*>(old_impl.get());//NOLINT
        if (p->slots() <= kThreshold) {
            if (p->size() < kThreshold) {
                return SmallMapImpl::InsertMaybeRehash(kv, old_impl);
            }

            const auto new_impl = CreateFromRange(p->begin(), p->end());
            return DenseMapImpl::InsertMaybeRehash(kv, new_impl);
        }
        return old_impl;
    }

    return DenseMapImpl::InsertMaybeRehash(kv, old_impl);
}

template<typename K, typename V>
class Map : public ObjectRef {
public:
    Map() : impl_(SmallMapImpl::Create()) {}

    NODISCARD size_t size() const {
        return IsSmallMap() ? small_ptr()->size() : dense_ptr()->size();
    }

    NODISCARD size_t slots() const {
        return IsSmallMap() ? small_ptr()->slots() : dense_ptr()->slots();
    }

    NODISCARD bool empty() const {
        return size() == 0;
    }

    void insert(const K& key, const V& value) {
        std::pair<Any, Any> data(key, value);
        impl_ = IsSmallMap() ? SmallMapImpl::insert(data, impl_) : DenseMapImpl::insert(data, impl_);
    }

    V at(const K& key) const {
        return IsSmallMap() ? small_ptr()->at(key).template cast<V>() : dense_ptr()->at(key).template cast<V>();
    }

    V operator[](const K& key) const {
        return at(key);
    }

    NODISCARD bool IsSmallMap() const {
        return dynamic_cast<SmallMapImpl*>(impl_.get()) != nullptr;
    }

private:
    ObjectPtr<Object> impl_;

    NODISCARD SmallMapImpl* small_ptr() const {
        return static_cast<SmallMapImpl*>(impl_.get()); //NOLINT
    }

    NODISCARD DenseMapImpl* dense_ptr() const {
        return static_cast<DenseMapImpl*>(impl_.get()); //NOLINT
    }
};

}// namespace aethermind

#endif//AETHERMIND_MAP_H
