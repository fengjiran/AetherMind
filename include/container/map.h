//
// Created by richard on 11/28/25.
//

#ifndef AETHERMIND_CONTAINER_MAP_H
#define AETHERMIND_CONTAINER_MAP_H

#include "any.h"
#include "map.h"

#include <concepts>
#include <tuple>

namespace aethermind {

namespace details {

template<typename InputIter>
concept is_valid_iter = requires(InputIter t) {
    requires std::input_iterator<InputIter>;
    { *t } -> std::convertible_to<std::pair<Any, Any>>;
    ++t;
    --t;
};

}// namespace details

template<typename Derived>
class MapImpl : public Object {
public:
    using key_type = Any;
    using mapped_type = Any;
    using KVType = std::pair<key_type, mapped_type>;

    // IteratorImpl is a base class for iterator and const_iterator
    template<bool IsConst>
    class IteratorImpl;

    using iterator = IteratorImpl<false>;
    using const_iterator = IteratorImpl<true>;

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
        return GetDerivedPtr()->count_impl(key);
    }

    mapped_type& at(const key_type& key) {
        return GetDerivedPtr()->at_impl(key);
    }

    NODISCARD const mapped_type& at(const key_type& key) const {
        return GetDerivedPtr()->at_impl(key);
    }

    NODISCARD KVType* GetDataPtr(size_t idx) const {
        return GetDerivedPtr()->GetDataPtrImpl(idx);
    }

    NODISCARD size_t GetNextIndexOf(size_t idx) const {
        return GetDerivedPtr()->GetNextIndexOfImpl(idx);
    }

    NODISCARD size_t GetPrevIndexOf(size_t idx) const {
        return GetDerivedPtr()->GetPrevIndexOfImpl(idx);
    }

    NODISCARD const_iterator begin() const {
        return GetDerivedPtr()->begin_impl();
    }

    NODISCARD const_iterator end() const {
        return GetDerivedPtr()->end_impl();
    }

    NODISCARD iterator begin() {
        return GetDerivedPtr()->begin_impl();
    }

    NODISCARD iterator end() {
        return GetDerivedPtr()->end_impl();
    }

    NODISCARD const_iterator find(const key_type& key) const {
        return GetDerivedPtr()->find_impl(key);
    }

    NODISCARD iterator find(const key_type& key) {
        return GetDerivedPtr()->find_impl(key);
    }

    NODISCARD iterator erase(iterator pos);

protected:
    void* data_;
    size_t size_;
    size_t slots_;

    static constexpr size_t kInitSize = 2; // Init map size
    static constexpr size_t kThreshold = 4;// The threshold of the small and dense map
    static constexpr size_t kIncFactor = 2;

    NODISCARD Derived* GetDerivedPtr() noexcept {
        return static_cast<Derived*>(this);
    }

    NODISCARD const Derived* GetDerivedPtr() const noexcept {
        return static_cast<const Derived*>(this);
    }

    static ObjectPtr<Object> Create(size_t n = kInitSize);

    template<details::is_valid_iter Iter>
    static ObjectPtr<Object> CreateFromRange(Iter first, Iter last);

    static ObjectPtr<Object> CopyFrom(const Object* src);

    // insert may be rehash
    static std::tuple<ObjectPtr<Object>, size_t, bool> insert(KVType&& kv, const ObjectPtr<Object>& old_impl, bool assign = false);
};

template<typename Derived>
template<bool IsConst>
class MapImpl<Derived>::IteratorImpl {
public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = KVType;
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

    IteratorImpl operator+(difference_type offset) const {
        Check();
        if (offset < 0) {
            return operator-(static_cast<difference_type>(-offset));
        }

        auto sz = index();
        for (difference_type i = 0; i < offset && sz != ptr()->end().index(); ++i) {
            sz = ptr()->GetNextIndexOf(sz);
        }
        return IteratorImpl(sz, ptr());
    }

    IteratorImpl operator-(difference_type offset) const {
        Check();
        if (offset < 0) {
            return operator+(static_cast<difference_type>(-offset));
        }

        auto sz = index();
        for (difference_type i = 0; i < offset && sz != ptr()->end().index(); ++i) {
            sz = ptr()->GetPrevIndexOf(sz);
        }
        return IteratorImpl(sz, ptr());
    }

    IteratorImpl& operator+=(difference_type offset) {
        Check();
        if (offset < 0) {
            return operator-=(static_cast<difference_type>(-offset));
        }

        for (difference_type i = 0; i < offset && index_ != ptr()->end().index(); ++i) {
            index_ = ptr()->GetNextIndexOf(index_);
        }
        return *this;
    }

    IteratorImpl& operator-=(difference_type offset) {
        Check();
        if (offset < 0) {
            return operator+=(static_cast<difference_type>(-offset));
        }

        for (difference_type i = 0; i < offset && index_ != ptr()->end().index(); ++i) {
            index_ = ptr()->GetPrevIndexOf(index_);
        }
        return *this;
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
        // check index out of range
    }

private:
    size_t index_;
    ContainerPtrType ptr_;
};

class SmallMapImpl : public MapImpl<SmallMapImpl> {
public:
    SmallMapImpl() = default;
    ~SmallMapImpl() override {
        reset();
    }

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

    iterator erase_impl(iterator pos);

    NODISCARD KVType* GetDataPtrImpl(size_t index) const {
        return static_cast<KVType*>(data()) + index;
    }

    NODISCARD size_t GetNextIndexOfImpl(size_t idx) const {
        return idx + 1 < size_ ? idx + 1 : size_;
    }

    NODISCARD size_t GetPrevIndexOfImpl(size_t idx) const {
        return idx > 0 ? idx - 1 : size_;
    }

    void reset();

    static ObjectPtr<SmallMapImpl> CreateImpl(size_t n = kInitSize);

    static ObjectPtr<SmallMapImpl> CopyFromImpl(const SmallMapImpl* src);

    static std::tuple<ObjectPtr<Object>, iterator, bool> InsertImpl(
            KVType&& kv, const ObjectPtr<Object>& old_impl, bool assign = false);

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

    void erase_impl1(const iterator& pos);
    iterator erase_impl(iterator pos);

    NODISCARD Block* GetBlock(size_t block_idx) const;

    NODISCARD Cursor GetCursorFromHash(size_t hash_value) const;

    NODISCARD KVType* GetDataPtrImpl(size_t index) const;

    // Construct a ListNode from hash code if the position is head of list
    NODISCARD std::optional<Cursor> GetListHead(size_t hash_value) const;

    NODISCARD size_t GetNextIndexOfImpl(size_t idx) const;

    NODISCARD size_t GetPrevIndexOfImpl(size_t idx) const;

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
   * \param kv The value pair
   * \return The linked-list entry constructed as the head, if actual insertion happens
   */
    std::optional<Cursor> TrySpareListHead(Cursor target, KVType&& kv);

    /*!
   * \brief Try to insert a key, or do nothing if already exists
   * \param kv The value pair
   * \param assign Whether to assign for existing key
   * \return The linked-list entry found or just constructed,indicating if actual insertion happens
   */
    std::pair<iterator, bool> TryInsert(KVType&& kv, bool assign = false);

    // may be rehash
    static std::tuple<ObjectPtr<Object>, iterator, bool> InsertImpl(
            KVType&& kv, const ObjectPtr<Object>& old_impl, bool assign = false);

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
template<details::is_valid_iter Iter>
ObjectPtr<Object> MapImpl<Derived>::CreateFromRange(Iter first, Iter last) {
    const int64_t _sz = std::distance(first, last);
    if (_sz <= 0) {
        return Create();
    }

    const auto size = static_cast<size_t>(_sz);
    ObjectPtr<Object> impl = Create(size);
    if (size <= kThreshold) {
        while (first != last) {
            impl = std::get<0>(SmallMapImpl::insert(KVType(*first++), impl));
        }
    } else {
        while (first != last) {
            impl = std::get<0>(DenseMapImpl::insert(KVType(*first++), impl));
        }
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
std::tuple<ObjectPtr<Object>, size_t, bool> MapImpl<Derived>::insert(KVType&& kv, const ObjectPtr<Object>& old_impl, bool assign) {
    if constexpr (std::is_same_v<Derived, SmallMapImpl>) {
        auto* p = static_cast<SmallMapImpl*>(old_impl.get());//NOLINT
        const auto size = p->size();
        if (size < kThreshold) {
            auto [impl, iter, is_success] = SmallMapImpl::InsertImpl(std::move(kv), old_impl, assign);
            return {impl, iter.index(), is_success};
        }

        ObjectPtr<Object> new_impl = DenseMapImpl::Create(size * kIncFactor);
        for (auto& iter: *p) {
            new_impl = std::get<0>(DenseMapImpl::InsertImpl(std::move(iter), new_impl));
        }
        auto [impl, iter, is_success] = DenseMapImpl::InsertImpl(std::move(kv), new_impl, assign);
        return {impl, iter.index(), is_success};
    } else {
        auto [impl, iter, is_success] = DenseMapImpl::InsertImpl(std::move(kv), old_impl, assign);
        return {impl, iter.index(), is_success};
    }
}

template<typename Derived>
MapImpl<Derived>::iterator MapImpl<Derived>::erase(iterator pos) {
    return static_cast<Derived*>(this)->erase_impl(pos);
}

template<typename K, typename V>
class Map : public ObjectRef {
public:
    using key_type = K;
    using mapped_type = V;
    using value_type = std::pair<const key_type, mapped_type>;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using reference = value_type&;
    using const_reference = const value_type&;

    template<bool IsConst>
    class IteratorImpl;

    using iterator = IteratorImpl<false>;
    using const_iterator = IteratorImpl<true>;

    class PairProxy;

    Map() : obj_(details::ObjectUnsafe::Downcast<SmallMapImpl>(SmallMapImpl::Create())) {}

    explicit Map(size_type n) {
        auto tmp = SmallMapImpl::Create(n);
        if (n <= SmallMapImpl::kThreshold) {
            obj_ = details::ObjectUnsafe::Downcast<SmallMapImpl>(tmp);
        } else {
            obj_ = details::ObjectUnsafe::Downcast<DenseMapImpl>(tmp);
        }
    }

    Map(std::initializer_list<value_type> list) : Map(list.begin(), list.end()) {}

    Map(const Map& other) = default;

    Map(Map&& other) noexcept : obj_(other.obj_) {//NOLINT
        other.clear();
    }

    template<typename Iter>
    Map(Iter first, Iter last) {
        auto tmp = SmallMapImpl::CreateFromRange(first, last);
        auto n = std::distance(first, last);
        if (n <= SmallMapImpl::kThreshold) {
            obj_ = details::ObjectUnsafe::Downcast<SmallMapImpl>(tmp);
        } else {
            obj_ = details::ObjectUnsafe::Downcast<DenseMapImpl>(tmp);
        }
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

    // std::pair<iterator, bool> insert(const SmallMapImpl::KVType& x) {
    //     return insert_impl({x.first.cast<key_type>(), x.second.cast<mapped_type>()}, false);
    // }

    // static_assert(std::convertible_to<std::pair<Any, Any>&, std::pair<const int, int>>);

    template<typename Pair>
        requires(std::constructible_from<value_type, Pair &&> &&
                 !std::same_as<std::decay_t<Pair>, value_type>)
    std::pair<iterator, bool> insert(Pair&& x) {
        return insert_impl(value_type(std::forward<Pair>(x)), false);
    }

    template<typename T = Any>
        requires requires(T t) {
            t.template cast<key_type>();
            t.template cast<mapped_type>();
        }
    std::pair<iterator, bool> insert(const std::pair<T, T>& x) {
        return insert_impl({x.first.template cast<key_type>(),
                            x.second.template cast<mapped_type>()},
                           false);
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
    iterator erase(iterator first, iterator last);
    iterator erase(const_iterator first, const_iterator last);

    size_type erase(const key_type& key) {
        auto it = find(key);
        if (it != end()) {
            erase(it);
            return 1;
        }
        return 0;
    }

    bool contains(const key_type& key) const {
        return find(key) != end();
    }

    PairProxy at(const key_type& key) {
        auto it = find(key);
        if (it == end()) {
            AETHERMIND_THROW(KeyError) << "Key does not exist";
        }
        return {*this, it.index()};
    }

    const Any& at(const key_type& key) const {
        return std::visit([&](const auto& arg) -> const Any& { return arg->at(key); }, obj_);
    }

    PairProxy operator[](const key_type& key) {
        return at(key);
    }

    const Any& operator[](const key_type& key) const {
        return at(key);
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

    void clear() {
        obj_ = details::ObjectUnsafe::Downcast<SmallMapImpl>(SmallMapImpl::Create());
    }

    iterator find(const key_type& key) {
        return std::visit([&](const auto& arg) { return iterator(arg->find(key)); }, obj_);
    }

    const_iterator find(const key_type& key) const {
        return const_cast<Map*>(this)->find(key);
    }

    NODISCARD bool IsSmallMap() const {
        return std::holds_alternative<ObjectPtr<SmallMapImpl>>(obj_);
    }

    void swap(Map& other) noexcept {
        std::swap(obj_, other.obj_);
    }

private:
    std::variant<ObjectPtr<SmallMapImpl>, ObjectPtr<DenseMapImpl>> obj_;

    NODISCARD SmallMapImpl* small_ptr() const {
        return std::get<ObjectPtr<SmallMapImpl>>(obj_).get();
    }

    NODISCARD DenseMapImpl* dense_ptr() const {
        return std::get<ObjectPtr<DenseMapImpl>>(obj_).get();
    }

    std::pair<iterator, bool> insert_impl(value_type&& x, bool assign) {
        if (!assign) {
            auto it = find(x.first);
            if (it != end()) {
                return {it, false};
            }
        }

        COW();
        auto [impl, idx, is_success] =
                IsSmallMap() ? SmallMapImpl::insert(std::move(x), std::get<ObjectPtr<SmallMapImpl>>(obj_), assign)
                             : DenseMapImpl::insert(std::move(x), std::get<ObjectPtr<DenseMapImpl>>(obj_), assign);

        if (dynamic_cast<SmallMapImpl*>(impl.get())) {
            obj_ = details::ObjectUnsafe::Downcast<SmallMapImpl>(impl);
            SmallMapImpl::iterator pos(idx, small_ptr());
            return {pos, is_success};
        }

        obj_ = details::ObjectUnsafe::Downcast<DenseMapImpl>(impl);
        DenseMapImpl::iterator pos(idx, dense_ptr());
        return {pos, is_success};
    }

    void COW() {
        if (!unique()) {
            const auto* p = std::visit([](const auto& arg) -> Object* { return arg.get(); }, obj_);
            if (IsSmallMap()) {
                obj_ = details::ObjectUnsafe::Downcast<SmallMapImpl>(SmallMapImpl::CopyFrom(p));
            } else {
                obj_ = details::ObjectUnsafe::Downcast<DenseMapImpl>(DenseMapImpl::CopyFrom(p));
            }
        }
    }

    // cow wrapper
    template<typename Operator>
    decltype(auto) WithCOW(Operator&& op) {
        if (!unique()) {
            const auto* p = std::visit([](const auto& arg) -> Object* { return arg.get(); }, obj_);
            if (IsSmallMap()) {
                obj_ = details::ObjectUnsafe::Downcast<SmallMapImpl>(SmallMapImpl::CopyFrom(p));
            } else {
                obj_ = details::ObjectUnsafe::Downcast<DenseMapImpl>(DenseMapImpl::CopyFrom(p));
            }
        }
        return op();
    }
};

template<typename K, typename V>
template<bool IsConst>
class Map<K, V>::IteratorImpl {
public:
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = SmallMapImpl::KVType;
    using pointer = std::conditional_t<IsConst, const value_type*, value_type*>;
    using reference = std::conditional_t<IsConst, const value_type&, value_type&>;
    using SmallIterType = std::conditional_t<IsConst, SmallMapImpl::const_iterator, SmallMapImpl::iterator>;
    using DenseIterType = std::conditional_t<IsConst, DenseMapImpl::const_iterator, DenseMapImpl::iterator>;

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
        return std::visit([&](const auto& iter) { return iter + offset; }, iter_);
    }

    IteratorImpl operator-(difference_type offset) const {
        return std::visit([&](const auto& iter) { return iter - offset; }, iter_);
    }

    IteratorImpl& operator+=(difference_type offset) {
        std::visit([&](auto& iter) { iter += offset; }, iter_);
        return *this;
    }

    IteratorImpl& operator-=(difference_type offset) {
        std::visit([&](auto& iter) { iter -= offset; }, iter_);
        return *this;
    }

    reference operator*() const {
        return std::visit([](auto& iter) -> reference { return *iter; }, iter_);
    }

    pointer operator->() const {
        return std::visit([](auto& iter) -> pointer { return iter.operator->(); }, iter_);
    }

    bool operator==(const IteratorImpl& other) const {
        if (IsSmallMap() && other.IsSmallMap()) {
            return std::get<SmallIterType>(iter_) == std::get<SmallIterType>(other.iter_);
        }

        if (!IsSmallMap() && !other.IsSmallMap()) {
            return std::get<DenseIterType>(iter_) == std::get<DenseIterType>(other.iter_);
        }

        return false;
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

    template<typename, typename>
    friend class Map;
};

template<typename K, typename V>
class Map<K, V>::PairProxy {
public:
    PairProxy(Map& map, size_type idx) : map_(map), idx_(idx) {}

    // PairProxy& operator=(value_type x) {
    //     map_.COW();
    //     auto* p = GetRawDataPtr();
    //     *p = std::move(x);
    //     return *this;
    // }

    PairProxy& operator=(mapped_type x) {
        map_.COW();
        auto* p = GetRawDataPtr();
        p->second = std::move(x);
        return *this;
    }

    template<typename U = V>
        requires details::is_map<U>
    decltype(auto) operator[](const U::key_type& key) {
        auto* p = GetRawDataPtr();
        return p->second.template operator[]<U>(key);
    }

    template<typename U = V>
        requires details::is_container<U>
    decltype(auto) operator[](U::size_type i) {
        auto* p = GetRawDataPtr();
        return p->second.template operator[]<U>(i);
    }

    friend bool operator==(const PairProxy& lhs, const PairProxy& rhs) {
        return *lhs.GetRawDataPtr() == *rhs.GetRawDataPtr();
    }

    friend bool operator!=(const PairProxy& lhs, const PairProxy& rhs) {
        return !(lhs == rhs);
    }

    friend bool operator==(const PairProxy& lhs, const Any& rhs) {
        return lhs.GetRawDataPtr()->second == rhs;
    }

    friend bool operator!=(const PairProxy& lhs, const Any& rhs) {
        return !(rhs == lhs);
    }

    friend bool operator==(const Any& lhs, const PairProxy& rhs) {
        return rhs == lhs;
    }

    friend bool operator!=(const Any& lhs, const PairProxy& rhs) {
        return !(lhs == rhs);
    }

private:
    Map& map_;
    size_type idx_;

    NODISCARD DenseMapImpl::KVType* GetRawDataPtr() const {
        return std::visit([&](const auto& arg) { return arg->GetDataPtr(idx_); },
                          map_.obj_);
    }
};

template<typename K, typename V>
Map<K, V>::iterator Map<K, V>::erase(iterator pos) {
    return erase(const_iterator(pos));
}

template<typename K, typename V>
Map<K, V>::iterator Map<K, V>::erase(const_iterator pos) {
    if (pos == end()) {
        return end();
    }

    COW();
    auto visitor = [&](const auto& arg) -> iterator {
        return arg->erase({pos.index(), arg.get()});
    };
    return std::visit(visitor, obj_);
}

template<typename K, typename V>
Map<K, V>::iterator Map<K, V>::erase(const_iterator first, const_iterator last) {
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

template<typename K, typename V>
Map<K, V>::iterator Map<K, V>::erase(iterator first, iterator last) {
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

}// namespace aethermind

#endif//AETHERMIND_CONTAINER_MAP_H
