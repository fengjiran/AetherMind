//
// Created by richard on 11/28/25.
//

#ifndef AETHERMIND_CONTAINER_MAP_H
#define AETHERMIND_CONTAINER_MAP_H

#include "any.h"

#include <concepts>
#include <tuple>

namespace aethermind {

namespace details {

template<typename InputIter>
concept is_valid_iter = requires(InputIter t) {
    requires std::convertible_to<typename std::iterator_traits<InputIter>::iterator_category,
                                 std::input_iterator_tag>;
    { *t } -> std::convertible_to<std::pair<Any, Any>>;
};

}// namespace details

template<typename Derived>
class MapImpl : public Object {
public:
    using key_type = Any;
    using mapped_type = Any;
    using KVType = std::pair<key_type, mapped_type>;

    class iterator;
    class const_iterator;

    MapImpl() : data_(nullptr), size_(0), slots_(0) {}

    NODISCARD void* data() const {
        return data_;
    }

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

    mapped_type& at(const key_type& key) {
        return static_cast<Derived*>(this)->at_impl(key);
    }

    NODISCARD const mapped_type& at(const key_type& key) const {
        return static_cast<const Derived*>(this)->at_impl(key);
    }

    NODISCARD const_iterator begin() const {
        return static_cast<const Derived*>(this)->begin_impl();
    }

    NODISCARD const_iterator end() const {
        return static_cast<const Derived*>(this)->end_impl();
    }

    NODISCARD iterator begin() {
        return static_cast<Derived*>(this)->begin_impl();
    }

    NODISCARD iterator end() {
        return static_cast<Derived*>(this)->end_impl();
    }

    NODISCARD const_iterator find(const key_type& key) const {
        return static_cast<const Derived*>(this)->find_impl(key);
    }

    NODISCARD iterator find(const key_type& key) {
        return static_cast<Derived*>(this)->find_impl(key);
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
    static constexpr size_t kThreshold = 4;// The threshold of the small and dense map
    static constexpr size_t kIncFactor = 2;

    static ObjectPtr<Object> Create(size_t n = kInitSize);

    template<typename Iter>
        requires details::is_valid_iter<Iter>
    static ObjectPtr<Object> CreateFromRange(Iter first, Iter last);

    static ObjectPtr<Object> CopyFrom(const Object* src);

    // insert may be rehash
    static ObjectPtr<Object> insert_or_assign(const KVType& kv, const ObjectPtr<Object>& old_impl);

    static std::tuple<ObjectPtr<Object>, size_t, bool> insert(KVType&& kv, const ObjectPtr<Object>& old_impl);
};

template<typename Derived>
class MapImpl<Derived>::iterator {
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = KVType;
    using pointer = value_type*;
    using reference = value_type&;
    using difference_type = std::ptrdiff_t;

    iterator() : index_(0), ptr_(nullptr) {}

    iterator(size_t index, Derived* ptr) : index_(index), ptr_(ptr) {}

    NODISCARD size_t index() const {
        return index_;
    }

    NODISCARD Derived* ptr() const {
        return ptr_;
    }

    pointer operator->() const {
        return ptr_->GetDataPtr(index_);
    }

    reference operator*() const {
        return *operator->();
    }

    iterator& operator++() {
        index_ = ptr_->IncIter(index_);
        return *this;
    }

    iterator& operator--() {
        index_ = ptr_->DecIter(index_);
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
    size_t index_;// The position in the array.
    Derived* ptr_;

    friend class SmallMapImpl;
    friend class DenseMapImpl;
};

template<typename Derived>
class MapImpl<Derived>::const_iterator {
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = KVType;
    using const_pointer = const value_type*;
    using const_reference = const value_type&;
    using difference_type = std::ptrdiff_t;

    const_iterator() : index_(0), ptr_(nullptr) {}

    const_iterator(size_t index, const Derived* ptr) : index_(index), ptr_(ptr) {}

    const_iterator(const iterator& iter) : index_(iter.index()), ptr_(iter.ptr()) {}//NOLINT

    NODISCARD size_t index() const {
        return index_;
    }

    const_pointer operator->() const {
        return ptr_->GetDataPtr(index_);
    }

    const_reference operator*() const {
        return *operator->();
    }

    const_iterator& operator++() {
        index_ = ptr_->IncIter(index_);
        return *this;
    }

    const_iterator& operator--() {
        index_ = ptr_->DecIter(index_);
        return *this;
    }

    const_iterator operator++(int) {
        const_iterator tmp = *this;
        operator++();
        return tmp;
    }

    const_iterator operator--(int) {
        const_iterator tmp = *this;
        operator--();
        return tmp;
    }

    bool operator==(const const_iterator& other) const {
        return index_ == other.index_ && ptr_ == other.ptr_;
    }

    bool operator!=(const const_iterator& other) const {
        return !(*this == other);
    }

protected:
    size_t index_;      // The position in the array.
    const Derived* ptr_;// The container it pointer to.

    friend class SmallMapImpl;
    friend class DenseMapImpl;
};

class SmallMapImpl : public MapImpl<SmallMapImpl> {
public:
    SmallMapImpl() = default;
    ~SmallMapImpl() override = default;

private:
    NODISCARD iterator begin_impl() {
        return {0, this};
    }

    NODISCARD iterator end_impl() {
        return {size(), this};
    }

    NODISCARD const_iterator begin_impl() const {
        return const_cast<SmallMapImpl*>(this)->begin_impl();
    }

    NODISCARD const_iterator end_impl() const {
        return const_cast<SmallMapImpl*>(this)->end_impl();
    }

    NODISCARD const_iterator find_impl(const key_type& key) const;

    NODISCARD iterator find_impl(const key_type& key);

    NODISCARD size_t count_impl(const key_type& key) const {
        return find(key).index() < size();
    }

    mapped_type& at_impl(const key_type& key);

    NODISCARD const mapped_type& at_impl(const key_type& key) const;

    void erase_impl(const iterator& pos);

    // GetDataPtr
    NODISCARD KVType* GetDataPtr(size_t index) const {
        return static_cast<KVType*>(data()) + index;
    }

    NODISCARD size_t IncIter(size_t index) const {
        return index + 1 < size_ ? index + 1 : size_;
    }

    NODISCARD size_t DecIter(size_t index) const {
        return index > 0 ? index - 1 : size_;
    }

    static ObjectPtr<SmallMapImpl> CreateImpl(size_t n = kInitSize);

    static ObjectPtr<SmallMapImpl> CopyFromImpl(const SmallMapImpl* src);

    template<typename Iter>
    static ObjectPtr<SmallMapImpl> CreateFromRangeImpl(Iter first, Iter last) {
        const auto n = std::distance(first, last);
        auto impl = Create(n);
        auto* ptr = static_cast<KVType*>(static_cast<SmallMapImpl*>(impl.get())->data());
        while (first != last) {
            new (ptr++) KVType(*first++);
        }
        return impl;
    }

    static ObjectPtr<Object> InsertOrAssignImpl(const KVType& kv, ObjectPtr<Object> old_impl);
    static std::tuple<ObjectPtr<Object>, iterator, bool> InsertOrAssignImpl_(
            KVType&& kv, const ObjectPtr<Object>& old_impl);

    static std::tuple<ObjectPtr<Object>, iterator, bool> InsertImpl(
            KVType&& kv, const ObjectPtr<Object>& old_impl);

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
    class Cursor;

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

    NODISCARD iterator begin_impl() {
        return {iter_list_head_, this};
    }

    NODISCARD iterator end_impl() {
        return {kInvalidIndex, this};
    }

    NODISCARD const_iterator begin_impl() const {
        return const_cast<DenseMapImpl*>(this)->begin_impl();
    }

    NODISCARD const_iterator end_impl() const {
        return const_cast<DenseMapImpl*>(this)->end_impl();
    }

    NODISCARD const_iterator find_impl(const key_type& key) const;

    NODISCARD iterator find_impl(const key_type& key);

    NODISCARD size_t count_impl(const key_type& key) const;

    NODISCARD mapped_type& at_impl(const key_type& key) {
        return At(key);
    }

    NODISCARD const mapped_type& at_impl(const key_type& key) const {
        return At(key);
    }

    void erase_impl(const iterator& pos);

    NODISCARD Block* GetBlock(size_t block_idx) const;

    NODISCARD Cursor GetCursorFromHash(size_t hash_value) const;

    NODISCARD KVType* GetDataPtr(size_t index) const;

    // Construct a ListNode from hash code if the position is head of list
    NODISCARD std::optional<Cursor> GetListHead(size_t hash_value) const;

    NODISCARD size_t IncIter(size_t index) const;

    NODISCARD size_t DecIter(size_t index) const;

    NODISCARD mapped_type& At(const key_type& key) const;

    /*!
   * \brief Search for the given key, throw exception if not exists
   * \param key The key
   * \return ListNode that associated with the key
   */
    NODISCARD Cursor Search(const key_type& key) const;

    // Whether the hash table is full.
    NODISCARD bool IsFull() const {
        return size() + 1 > static_cast<size_t>(static_cast<double>(slots()) * kMaxLoadFactor);
    }

    void reset();

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
   * \param key The indexing key
   * \return The linked-list entry constructed as the head, if actual insertion happens
   */
    // bool TrySpareListHead(ListNode target, const key_type& key, ListNode* result);
    std::optional<Cursor> TrySpareListHeadOrAssign(Cursor target, const key_type& key);
    std::optional<Cursor> TrySpareListHead(Cursor target, KVType&& kv);

    /*!
   * \brief Try to insert a key, or do nothing if already exists
   * \param key The indexing key
   * \return The linked-list entry found or just constructed,indicating if actual insertion happens
   */
    std::optional<Cursor> TryInsertOrAssign(const key_type& key);

    std::pair<iterator, bool> TryInsert(KVType&& kv);

    // may be rehash
    static ObjectPtr<Object> InsertOrAssignImpl(const KVType& kv, ObjectPtr<Object> old_impl);

    static std::tuple<ObjectPtr<Object>, iterator, bool> InsertImpl(
            KVType&& kv, const ObjectPtr<Object>& old_impl);

    static size_t ComputeBlockNum(size_t slots) {
        return (slots + kBlockSize - 1) / kBlockSize;
    }

    // Calculate the power-of-2 table size given the lower-bound of required capacity.
    // shift = 64 - log2(slots)
    static std::pair<uint32_t, size_t> ComputeSlotNum(size_t cap);

    static ObjectPtr<DenseMapImpl> CreateImpl(size_t n);

    static ObjectPtr<DenseMapImpl> CopyFromImpl(const DenseMapImpl* src);

    template<typename Derived>
    friend class MapImpl;
    template<typename K, typename V>
    friend class Map;
};

template<typename Derived>
ObjectPtr<Object> MapImpl<Derived>::Create(size_t n) {
    if (n <= kInitSize) {
        return SmallMapImpl::CreateImpl(kInitSize);
    }

    if (n <= kThreshold) {
        return SmallMapImpl::CreateImpl(n);
    }
    return DenseMapImpl::CreateImpl(n);
}

template<typename Derived>
template<typename Iter>
    requires details::is_valid_iter<Iter>
ObjectPtr<Object> MapImpl<Derived>::CreateFromRange(Iter first, Iter last) {
    const int64_t _sz = std::distance(first, last);
    if (_sz <= 0) {
        return Create();
    }

    const auto size = static_cast<size_t>(_sz);
    if (size <= kThreshold) {
        // need to insert to avoid duplicate keys
        ObjectPtr<Object> impl = Create(size);
        while (first != last) {
            impl = SmallMapImpl::InsertOrAssignImpl(KVType(*first++), impl);
        }
        return impl;
    }

    ObjectPtr<Object> impl = Create(size);
    while (first != last) {
        impl = DenseMapImpl::InsertOrAssignImpl(KVType(*first++), impl);
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
ObjectPtr<Object> MapImpl<Derived>::insert_or_assign(const KVType& kv, const ObjectPtr<Object>& old_impl) {
    if constexpr (std::is_same_v<Derived, SmallMapImpl>) {
        auto* p = static_cast<SmallMapImpl*>(old_impl.get());//NOLINT
        const auto size = p->size();
        if (size < kThreshold) {
            return SmallMapImpl::InsertOrAssignImpl(kv, old_impl);
        }

        ObjectPtr<Object> new_impl = DenseMapImpl::Create(size * kIncFactor);
        for (const auto& iter: *p) {
            new_impl = DenseMapImpl::InsertOrAssignImpl(KVType(iter), new_impl);
        }
        return DenseMapImpl::InsertOrAssignImpl(kv, new_impl);
    } else {
        return DenseMapImpl::InsertOrAssignImpl(kv, old_impl);
    }
}

template<typename Derived>
std::tuple<ObjectPtr<Object>, size_t, bool> MapImpl<Derived>::insert(KVType&& kv, const ObjectPtr<Object>& old_impl) {
    if constexpr (std::is_same_v<Derived, SmallMapImpl>) {
        auto* p = static_cast<SmallMapImpl*>(old_impl.get());//NOLINT
        const auto size = p->size();
        if (size < kThreshold) {
            auto [impl, iter, is_success] = SmallMapImpl::InsertImpl(std::move(kv), old_impl);
            return {impl, iter.index(), is_success};
        }

        ObjectPtr<Object> new_impl = DenseMapImpl::Create(size * kIncFactor);
        for (auto& iter: *p) {
            new_impl = std::get<0>(DenseMapImpl::InsertImpl(KVType(std::move(iter)), new_impl));
        }
        auto [impl, iter, is_success] = DenseMapImpl::InsertImpl(std::move(kv), new_impl);
        return {impl, iter.index(), is_success};
    } else {
        auto [impl, iter, is_success] = DenseMapImpl::InsertImpl(std::move(kv), old_impl);
        return {impl, iter.index(), is_success};
    }
}

template<typename K, typename V>
class Map : public ObjectRef {
public:
    using key_type = K;
    using mapped_type = V;
    using value_type = std::pair<key_type, mapped_type>;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using reference = value_type&;
    using const_reference = const value_type&;

    Map() : impl_(SmallMapImpl::Create()) {}

    explicit Map(size_type n) : impl_(SmallMapImpl::Create(n)) {}

    Map(std::initializer_list<value_type> list) : Map(list.begin(), list.end()) {}

    Map(const Map& other) = default;

    Map(Map&& other) noexcept : impl_(other.impl_) {//NOLINT
        other.clear();
    }

    template<typename Iter>
    Map(Iter first, Iter last) : impl_(SmallMapImpl::CreateFromRange(first, last)) {}

    template<typename KU, typename VU>
        requires std::is_base_of_v<key_type, KU> && std::is_base_of_v<mapped_type, VU>
    Map(const Map<KU, VU>& other) : impl_(other.impl_) {}//NOLINT

    template<typename KU, typename VU>
        requires std::is_base_of_v<key_type, KU> && std::is_base_of_v<mapped_type, VU>
    Map(Map<KU, VU>&& other) noexcept : impl_(other.impl_) {//NOLINT
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
            impl_ = other.impl_;
        }
        return *this;
    }

    template<typename KU, typename VU>
        requires std::is_base_of_v<key_type, KU> && std::is_base_of_v<mapped_type, VU>
    Map& operator=(Map<KU, VU>&& other) noexcept {
        if (this != &other) {
            impl_ = other.impl_;
            other.clear();
        }
        return *this;
    }

    Map& operator=(std::initializer_list<value_type> list) {
        Map(list).swap(*this);
        return *this;
    }

    class iterator;
    class const_iterator;

    NODISCARD size_type size() const noexcept {
        return IsSmallMap() ? small_ptr()->size() : dense_ptr()->size();
    }

    NODISCARD size_type slots() const noexcept {
        return IsSmallMap() ? small_ptr()->slots() : dense_ptr()->slots();
    }

    NODISCARD bool empty() const noexcept {
        return size() == 0;
    }

    NODISCARD uint32_t use_count() const noexcept {
        return IsSmallMap() ? small_ptr()->use_count() : dense_ptr()->use_count();
    }

    NODISCARD bool unique() const noexcept {
        return use_count() == 1;
    }

    NODISCARD size_type count() const {
        return IsSmallMap() ? small_ptr()->count() : dense_ptr()->count();
    }

    std::pair<iterator, bool> insert(value_type&& data) {
        auto it = find(data.first);
        if (it != end()) {
            return {it, false};
        }

        if (!unique()) {
            COW();
        }

        auto [impl, idx, is_success] = IsSmallMap() ? SmallMapImpl::insert(std::move(data), impl_)
                                                    : DenseMapImpl::insert(std::move(data), impl_);

        impl_ = impl;
        if (IsSmallMap()) {
            SmallMapImpl::iterator pos(idx, small_ptr());
            return {pos, is_success};
        }

        DenseMapImpl::iterator pos(idx, dense_ptr());
        return {pos, is_success};
    }

    std::pair<iterator, bool> insert(const value_type& x) {
        return insert(value_type(x));
    }

    std::pair<iterator, bool> insert(const key_type& key, const mapped_type& value) {
        return insert(value_type(key, value));
    }

    std::pair<iterator, bool> insert(key_type&& key, mapped_type&& value) {
        return insert(value_type(std::move(key), std::move(value)));
    }

    std::pair<iterator, bool> insert(const std::pair<Any, Any>& x) {
        return insert(x.first.cast<key_type>(), x.second.cast<mapped_type>());
    }

    template<typename Pair>
        requires(std::constructible_from<value_type, Pair &&> &&
                 !std::same_as<std::decay_t<Pair>, value_type>)
    std::pair<iterator, bool> insert(Pair&& x) {
        return insert(value_type(std::forward<Pair>(x)));
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

    void erase(const key_type& key) {
        COW();
        IsSmallMap() ? small_ptr()->erase(key) : dense_ptr()->erase(key);
    }

    Any& at(const key_type& key) {
        return IsSmallMap() ? small_ptr()->at(key) : dense_ptr()->at(key);
    }

    const Any& at(const key_type& key) const {
        return IsSmallMap() ? small_ptr()->at(key) : dense_ptr()->at(key);
    }

    Any& operator[](const key_type& key) {
        return at(key);
    }

    const Any& operator[](const key_type& key) const {
        return at(key);
    }

    iterator begin() noexcept {
        return IsSmallMap() ? iterator(small_ptr()->begin()) : iterator(dense_ptr()->begin());
    }

    iterator end() noexcept {
        return IsSmallMap() ? iterator(small_ptr()->end()) : iterator(dense_ptr()->end());
    }

    const_iterator begin() const noexcept {
        return IsSmallMap() ? const_iterator(small_ptr()->begin()) : const_iterator(dense_ptr()->begin());
    }

    const_iterator end() const noexcept {
        return IsSmallMap() ? const_iterator(small_ptr()->end()) : const_iterator(dense_ptr()->end());
    }

    void clear() {
        impl_ = SmallMapImpl::Create();
    }

    iterator find(const key_type& key) {
        return IsSmallMap() ? iterator(small_ptr()->find(key)) : iterator(dense_ptr()->find(key));
    }

    const_iterator find(const key_type& key) const {
        return IsSmallMap() ? iterator(small_ptr()->find(key)) : iterator(dense_ptr()->find(key));
    }

    NODISCARD bool IsSmallMap() const {
        return dynamic_cast<SmallMapImpl*>(impl_.get()) != nullptr;
    }

    void swap(Map& other) noexcept {
        impl_.swap(other.impl_);
    }

private:
    ObjectPtr<Object> impl_;

    NODISCARD SmallMapImpl* small_ptr() const {
        return static_cast<SmallMapImpl*>(impl_.get());//NOLINT
    }

    NODISCARD DenseMapImpl* dense_ptr() const {
        return static_cast<DenseMapImpl*>(impl_.get());//NOLINT
    }

    void COW() {
        if (!unique()) {
            impl_ = IsSmallMap() ? SmallMapImpl::CopyFrom(impl_.get())
                                 : DenseMapImpl::CopyFrom(impl_.get());
        }
    }
};

template<typename K, typename V>
class Map<K, V>::iterator {
public:
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = SmallMapImpl::KVType;
    using pointer = value_type*;
    using reference = value_type&;

    iterator() = default;

    iterator& operator++() {
        if (is_small_map_) {
            ++iter_.small_iter;
        } else {
            ++iter_.dense_iter;
        }
        return *this;
    }

    iterator& operator--() {
        if (is_small_map_) {
            --iter_.small_iter;
        } else {
            --iter_.dense_iter;
        }
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

    reference operator*() const {
        return is_small_map_ ? iter_.small_iter.operator*() : iter_.dense_iter.operator*();
    }

    pointer operator->() const {
        return is_small_map_ ? iter_.small_iter.operator->() : iter_.dense_iter.operator->();
    }

    bool operator==(const iterator& other) const {
        if (is_small_map_ && other.is_small_map_) {
            return iter_.small_iter == other.iter_.small_iter;
        }

        if (!is_small_map_ && !other.is_small_map_) {
            return iter_.dense_iter == other.iter_.dense_iter;
        }

        return false;
    }

    bool operator!=(const iterator& other) const {
        return !(*this == other);
    }

private:
    union Iter {
        SmallMapImpl::iterator small_iter;
        DenseMapImpl::iterator dense_iter;

        Iter(const SmallMapImpl::iterator& iter) : small_iter(iter) {}//NOLINT
        Iter(const DenseMapImpl::iterator& iter) : dense_iter(iter) {}//NOLINT
    };

    Iter iter_;
    bool is_small_map_{true};

    iterator(const SmallMapImpl::iterator& iter) : iter_(iter), is_small_map_(true) {} //NOLINT
    iterator(const DenseMapImpl::iterator& iter) : iter_(iter), is_small_map_(false) {}//NOLINT

    template<typename, typename>
    friend class Map;
};

// sizeof = 24
template<typename K, typename V>
class Map<K, V>::const_iterator {
public:
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = SmallMapImpl::KVType;
    using const_pointer = const value_type*;
    using const_reference = const value_type&;

    const_iterator() = default;

    const_iterator& operator++() {
        if (is_small_map_) {
            ++iter_.small_iter;
        } else {
            ++iter_.dense_iter;
        }
        return *this;
    }

    const_iterator& operator--() {
        if (is_small_map_) {
            --iter_.small_iter;
        } else {
            --iter_.dense_iter;
        }
        return *this;
    }

    const_iterator operator++(int) {
        const_iterator tmp = *this;
        operator++();
        return tmp;
    }

    const_iterator operator--(int) {
        const_iterator tmp = *this;
        operator--();
        return tmp;
    }

    const_reference operator*() const {
        return is_small_map_ ? iter_.small_iter.operator*() : iter_.dense_iter.operator*();
    }

    const_pointer operator->() const {
        return is_small_map_ ? iter_.small_iter.operator->() : iter_.dense_iter.operator->();
    }

    bool operator==(const const_iterator& other) const {
        if (is_small_map_ && other.is_small_map_) {
            return iter_.small_iter == other.iter_.small_iter;
        }

        if (!is_small_map_ && !other.is_small_map_) {
            return iter_.dense_iter == other.iter_.dense_iter;
        }

        return false;
    }

    bool operator!=(const const_iterator& other) const {
        return !(*this == other);
    }

private:
    union Iter {
        SmallMapImpl::const_iterator small_iter;
        DenseMapImpl::const_iterator dense_iter;

        Iter(const SmallMapImpl::const_iterator& iter) : small_iter(iter) {}//NOLINT
        Iter(const DenseMapImpl::const_iterator& iter) : dense_iter(iter) {}//NOLINT
    };

    Iter iter_;
    bool is_small_map_{true};

    const_iterator(const SmallMapImpl::const_iterator& iter) : iter_(iter), is_small_map_(true) {} //NOLINT
    const_iterator(const DenseMapImpl::const_iterator& iter) : iter_(iter), is_small_map_(false) {}//NOLINT

    template<typename, typename>
    friend class Map;
};

}// namespace aethermind

#endif//AETHERMIND_CONTAINER_MAP_H
