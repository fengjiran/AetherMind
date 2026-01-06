//
// Created by richard on 1/5/26.
//

#ifndef AETHERMIND_CONTAINER_MAP_OBJ_H
#define AETHERMIND_CONTAINER_MAP_OBJ_H

#include <concepts>
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

template<typename K, typename V, typename Derived>
class MapObjBase : public Object {
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

    MapObjBase() : data_(nullptr), size_(0), slots_(0) {}

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
    static std::tuple<ObjectPtr<Object>, size_t, bool> insert(
            value_type&& kv, const ObjectPtr<Object>& old_impl, bool assign = false);
};

template<typename K, typename V, typename Derived>
template<bool IsConst>
class MapObjBase<K, V, Derived>::IteratorImpl {
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

template<typename K, typename V>
class SmallMapObj : public MapObjBase<K, V, SmallMapObj<K, V>> {
public:
    using key_type = K;
    using mapped_type = V;
    using value_type = std::pair<const key_type, mapped_type>;
    using size_type = size_t;

    using iterator = MapObjBase<K, V, SmallMapObj>::iterator;
    using const_iterator = MapObjBase<K, V, SmallMapObj>::const_iterator;

    SmallMapObj() = default;
    ~SmallMapObj() override {
        reset();
    }

private:
    NODISCARD iterator begin_impl() {
        return {0, this};
    }

    NODISCARD iterator end_impl() {
        return {this->size(), this};
    }

    NODISCARD const_iterator begin_impl() const {
        return const_cast<SmallMapObj*>(this)->begin_impl();
    }

    NODISCARD const_iterator end_impl() const {
        return const_cast<SmallMapObj*>(this)->end_impl();
    }

    NODISCARD const_iterator find_impl(const key_type& key) const;

    NODISCARD iterator find_impl(const key_type& key);

    NODISCARD size_t count_impl(const key_type& key) const {
        return this->find(key).index() < this->size();
    }

    mapped_type& at_impl(const key_type& key) {
        const auto iter = this->find(key);
        if (iter == this->end()) {
            AETHERMIND_THROW(KeyError) << "key is not exist.";
        }
        return iter->second;
    }

    NODISCARD const mapped_type& at_impl(const key_type& key) const {
        return const_cast<SmallMapObj*>(this)->at_impl(key);
    }

    NODISCARD value_type* GetDataPtrImpl(size_t index) const {
        return static_cast<value_type*>(this->data()) + index;
    }

    NODISCARD size_type GetNextIndexOfImpl(size_type idx) const {
        return idx + 1 < this->size() ? idx + 1 : this->size();
    }

    NODISCARD size_type GetPrevIndexOfImpl(size_type idx) const {
        return idx > 0 ? idx - 1 : this->size();
    }

    void reset();

    static ObjectPtr<SmallMapObj> Create(size_type n = SmallMapObj::kInitSize);

    static ObjectPtr<SmallMapObj> CopyFrom(const SmallMapObj* src);

    static std::tuple<ObjectPtr<Object>, iterator, bool> InsertImpl(
            value_type&& kv, const ObjectPtr<Object>& old_impl, bool assign = false);

    template<details::is_valid_iter Iter>
    static ObjectPtr<SmallMapObj> CreateFromRange(Iter first, Iter last);

    template<typename, typename, typename>
    friend class MapObjBase;
    template<typename T1, typename T2>
    friend class DenseMapObj;
};

template<typename K, typename V>
void SmallMapObj<K, V>::reset() {
    auto* p = static_cast<value_type*>(this->data());
    for (size_t i = 0; i < this->size(); ++i) {
        p[i].~value_type();
    }

    this->size_ = 0;
    this->slots_ = 0;
}

template<typename K, typename V>
SmallMapObj<K, V>::iterator SmallMapObj<K, V>::find_impl(const key_type& key) {
    const auto* p = static_cast<value_type*>(this->data());
    for (size_t i = 0; i < this->size(); ++i) {
        if (key == p[i].first) {
            return {i, this};
        }
    }

    return this->end();
}

template<typename K, typename V>
SmallMapObj<K, V>::const_iterator SmallMapObj<K, V>::find_impl(const key_type& key) const {
    return const_cast<SmallMapObj*>(this)->find_impl(key);
}

template<typename K, typename V>
ObjectPtr<SmallMapObj<K, V>> SmallMapObj<K, V>::Create(size_type n) {
    CHECK(n <= SmallMapObj::kThreshold)
            << "The allocated size must be less equal to the threshold of " << SmallMapObj::kThreshold
            << " when using SmallMapObj::Create";
    if (n < SmallMapObj::kInitSize) {
        n = SmallMapObj::kInitSize;
    }
    auto impl = make_array_object<SmallMapObj, value_type>(n);
    impl->data_ = reinterpret_cast<char*>(impl.get()) + sizeof(SmallMapObj);
    impl->size_ = 0;
    impl->slots_ = n;
    return impl;
}

template<typename K, typename V>
ObjectPtr<SmallMapObj<K, V>> SmallMapObj<K, V>::CopyFrom(const SmallMapObj* src) {
    const auto* first = static_cast<value_type*>(src->data());
    auto impl = Create(src->slots());
    auto* dst = static_cast<value_type*>(impl->data());
    for (size_t i = 0; i < src->size(); ++i) {
        new (dst + i) value_type(*first++);
        ++impl->size_;
    }

    return impl;
}

template<typename K, typename V>
std::tuple<ObjectPtr<Object>, typename SmallMapObj<K, V>::iterator, bool>
SmallMapObj<K, V>::InsertImpl(value_type&& kv, const ObjectPtr<Object>& old_impl, bool assign) {
    auto* map = static_cast<SmallMapObj*>(old_impl.get());//NOLINT
    if (const auto it = map->find(kv.first); it != map->end()) {
        if (assign) {
            it->second = std::move(kv.second);
        }
        return {old_impl, it, false};
    }

    if (map->size() < map->slots()) {
        auto* p = static_cast<value_type*>(map->data()) + map->size();
        new (p) value_type(std::move(kv));
        ++map->size_;
        return {old_impl, iterator(map->size() - 1, map), true};
    }

    const size_t new_cap =
            std::min(SmallMapObj::kThreshold,
                     std::max(SmallMapObj::kIncFactor * map->slots(), SmallMapObj::kInitSize));
    auto new_impl = details::ObjectUnsafe::Downcast<SmallMapObj>(Create(new_cap));
    auto* src = static_cast<value_type*>(map->data());
    auto* dst = static_cast<value_type*>(new_impl->data());
    for (size_t i = 0; i < map->size(); ++i) {
        new (dst++) value_type(std::move(*src++));
        ++new_impl->size_;
    }
    new (dst) value_type(std::move(kv));
    ++new_impl->size_;
    iterator pos(new_impl->size() - 1, new_impl.get());
    return {new_impl, pos, true};
}

template<typename K, typename V>
class DenseMapObj : public MapObjBase<K, V, DenseMapObj<K, V>> {
public:
    using key_type = K;
    using mapped_type = V;
    using value_type = std::pair<const key_type, mapped_type>;
    using size_type = size_t;

    using iterator = MapObjBase<K, V, DenseMapObj>::iterator;
    using const_iterator = MapObjBase<K, V, DenseMapObj>::const_iterator;

    DenseMapObj() = default;
    ~DenseMapObj() override {
        reset();
    }

    // private:
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
    static constexpr size_type kInvalidIndex = std::numeric_limits<size_type>::max();
    static constexpr size_type NextProbePosOffset[kNumOffsetDists] = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            // Quadratic probing with triangle numbers. See also:
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
    uint32_t fib_shift_ = 63;
    // The head of iterator list
    size_type iter_list_head_ = kInvalidIndex;
    // The tail of iterator list
    size_type iter_list_tail_ = kInvalidIndex;

    struct Entry;
    struct Block;
    class Cursor;

    NODISCARD iterator begin_impl() {
        return {iter_list_head_, this};
    }

    NODISCARD iterator end_impl() {
        return {kInvalidIndex, this};
    }

    NODISCARD const_iterator begin_impl() const {
        return const_cast<DenseMapObj*>(this)->begin_impl();
    }

    NODISCARD const_iterator end_impl() const {
        return const_cast<DenseMapObj*>(this)->end_impl();
    }

    NODISCARD const_iterator find_impl(const key_type& key) const {
        return const_cast<DenseMapObj*>(this)->find_impl(key);
    }

    NODISCARD iterator find_impl(const key_type& key) {
        const auto node = Search(key);
        return node.IsNone() ? this->end() : iterator(node.index(), this);
    }

    NODISCARD size_type count_impl(const key_type& key) const {
        return !Search(key).IsNone();
    }

    NODISCARD mapped_type& at_impl(const key_type& key) {
        return At(key);
    }

    NODISCARD const mapped_type& at_impl(const key_type& key) const {
        return At(key);
    }

    NODISCARD Cursor GetCursorFromHash(size_t hash_value) const {
        return {details::FibonacciHash(hash_value, fib_shift_), this};
    }

    NODISCARD Block* GetBlock(size_t block_idx) const {
        return static_cast<Block*>(this->data()) + block_idx;
    }

    NODISCARD value_type* GetDataPtrImpl(size_t index) const {
        return &Cursor(index, this).GetData();
    }

    // Construct a ListNode from hash code if the position is head of list
    NODISCARD std::optional<Cursor> GetListHead(size_t hash_value) const {
        if (const auto head = GetCursorFromHash(hash_value); head.IsHead()) {
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

    static size_type ComputeBlockNum(size_type slots) {
        return (slots + kBlockSize - 1) / kBlockSize;
    }

    // Calculate the power-of-2 table size given the lower-bound of required capacity.
    // shift = 64 - log2(slots)
    static std::pair<uint32_t, size_type> ComputeSlotNum(size_type cap);

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
    std::optional<Cursor> TrySpareListHead(Cursor target, value_type&& kv);

    /*!
     * \brief Try to insert a key, or do nothing if already exists
     * \param kv The value pair
     * \param assign Whether to assign for existing key
     * \return The linked-list entry found or just constructed,indicating if actual insertion happens
     */
    std::pair<iterator, bool> TryInsert(value_type&& kv, bool assign = false);

    // may be rehash
    static std::tuple<ObjectPtr<Object>, iterator, bool> InsertImpl(
            value_type&& kv, const ObjectPtr<Object>& old_impl, bool assign = false);

    static ObjectPtr<DenseMapObj> Create(size_type n);

    static ObjectPtr<DenseMapObj> CopyFrom(const DenseMapObj* src);

    template<details::is_valid_iter Iter>
    static ObjectPtr<DenseMapObj> CreateFromRange(Iter first, Iter last);

    template<typename, typename, typename>
    friend class MapObjBase;
};

template<typename K, typename V>
struct DenseMapObj<K, V>::Entry {
    value_type data{};
    size_type prev = kInvalidIndex;
    size_type next = kInvalidIndex;

    Entry() = default;
    Entry(key_type key, mapped_type value) : data(std::move(key), std::move(value)) {}
    explicit Entry(const value_type& kv) : data(kv) {}
    explicit Entry(value_type&& kv) : data(std::move(kv)) {}

    ~Entry() {
        reset();
    }

    void reset() {
        // data.first.reset();
        // data.second.reset();
        prev = kInvalidIndex;
        next = kInvalidIndex;
    }
};

template<typename K, typename V>
struct DenseMapObj<K, V>::Block {
    uint8_t bytes[kBlockSize + kBlockSize * sizeof(Entry)];

    Block() {// NOLINT
        auto* data = reinterpret_cast<Entry*>(bytes + kBlockSize);
        for (uint8_t i = 0; i < kBlockSize; ++i) {
            bytes[i] = kEmptySlot;
            new (data + i) Entry;
        }
    }

    Block(const Block& other) {// NOLINT
        const auto* src = reinterpret_cast<const Entry*>(other.bytes + kBlockSize);
        auto* dst = reinterpret_cast<Entry*>(bytes + kBlockSize);
        for (uint8_t i = 0; i < kBlockSize; ++i) {
            bytes[i] = other.bytes[i];
            new (dst + i) Entry(src[i]);
        }
    }

    ~Block() {
        auto* data = reinterpret_cast<Entry*>(bytes + kBlockSize);
        for (uint8_t i = 0; i < kBlockSize; ++i) {
            bytes[i] = kEmptySlot;
            data[i].~Entry();
        }
    }
};

template<typename K, typename V>
class DenseMapObj<K, V>::Cursor {
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
        return index() == obj()->iter_list_head_;
    }

    NODISCARD bool IsIterListTail() const {
        return index() == obj()->iter_list_tail_;
    }

    NODISCARD Block* GetBlock() const {
        return obj()->GetBlock(index() / kBlockSize);
    }

    // Get metadata of an entry
    NODISCARD uint8_t& GetMeta() const {
        return GetBlock()->bytes[index() % kBlockSize];
    }

    // Get an entry ref
    NODISCARD Entry& GetEntry() const {
        auto* p = reinterpret_cast<Entry*>(GetBlock()->bytes + kBlockSize);
        return p[index() % kBlockSize];
    }

    // Get KV
    NODISCARD value_type& GetData() const {
        return GetEntry().data;
    }

    NODISCARD key_type& GetKey() const {
        return GetData().first;
    }

    NODISCARD mapped_type& GetValue() const {
        return GetData().second;
    }

    NODISCARD bool IsNone() const {
        return obj() == nullptr;
    }

    NODISCARD bool IsEmpty() const {
        return GetMeta() == kEmptySlot;
    }

    NODISCARD bool IsProtected() const {
        return GetMeta() == kProtectedSlot;
    }

    NODISCARD bool IsHead() const {
        return (GetMeta() & 0x80) == 0x00;
    }

    void SetEmpty() const {
        GetMeta() = kEmptySlot;
    }

    void SetProtected() const {
        GetMeta() = kProtectedSlot;
    }

    // Set the entry's offset to its next entry.
    void SetOffsetIdx(uint8_t offset_idx) const {
        CHECK(offset_idx < kNumOffsetDists);
        (GetMeta() &= 0x80) |= offset_idx;
    }

    // Destroy the item in the entry.
    void DestructData() const {
        GetEntry().~Entry();
    }

    // Construct a head of linked list inplace.
    void CreateHead(const Entry& entry) const {
        GetMeta() = 0x00;
        GetEntry() = entry;
    }

    // Construct a tail of linked list inplace
    void CreateTail(const Entry& entry) const {
        GetMeta() = 0x80;
        GetEntry() = entry;
    }

    // Whether the slot has the next slot on the linked list
    NODISCARD bool HasNextSlot() const {
        return NextProbePosOffset[GetMeta() & 0x7F] != 0;
    }

    // Move the current cursor to the next slot on the linked list
    bool MoveToNextSlot(std::optional<uint8_t> meta_opt = std::nullopt) {
        uint8_t meta = meta_opt ? meta_opt.value() : GetMeta();
        const auto offset = NextProbePosOffset[meta & 0x7F];
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
        auto cur = obj()->GetCursorFromHash(AnyHash()(GetKey()));
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
                candidate.IsEmpty()) {
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

template<typename K, typename V>
DenseMapObj<K, V>::size_type DenseMapObj<K, V>::GetNextIndexOfImpl(size_type idx) const {
    // keep at the end of iterator
    if (idx == kInvalidIndex) {
        return idx;
    }

    return Cursor(idx, this).GetEntry().next;
}

template<typename K, typename V>
DenseMapObj<K, V>::size_type DenseMapObj<K, V>::GetPrevIndexOfImpl(size_type idx) const {
    // this is the end iterator, we need to return tail.
    if (idx == kInvalidIndex) {
        return iter_list_tail_;
    }

    return Cursor(idx, this).GetEntry().prev;
}

template<typename K, typename V>
DenseMapObj<K, V>::mapped_type& DenseMapObj<K, V>::At(const key_type& key) const {
    const Cursor iter = Search(key);
    if (iter.IsNone()) {
        AETHERMIND_THROW(KeyError) << "Key not found";
    }

    return iter.GetValue();
}

template<typename K, typename V>
void DenseMapObj<K, V>::reset() {
    const size_t block_num = ComputeBlockNum(this->slots());
    auto* p = static_cast<Block*>(this->data());
    for (size_t i = 0; i < block_num; ++i) {
        p[i].~Block();
    }

    this->size_ = 0;
    this->slots_ = 0;
    fib_shift_ = 63;
}


template<typename K, typename V>
DenseMapObj<K, V>::Cursor DenseMapObj<K, V>::Search(const key_type& key) const {
    if (this->empty()) {
        return {};
    }

    const auto head_opt = GetListHead(AnyHash()(key));
    if (!head_opt.has_value()) {
        return {};
    }

    auto node = *head_opt;
    while (!node.IsNone()) {
        if (key == node.GetKey()) {
            return node;
        }
        node.MoveToNextSlot();
    }

    return {};
}

template<typename K, typename V>
std::pair<uint32_t, typename DenseMapObj<K, V>::size_type> DenseMapObj<K, V>::ComputeSlotNum(size_type cap) {
    uint32_t shift = 64;
    size_t slots = 1;
    size_t c = cap;
    while (c > 0) {
        --shift;
        slots <<= 1;
        c >>= 1;
    }
    CHECK(slots > cap);

    if (slots < 2 * cap) {
        --shift;
        slots <<= 1;
    }

    return {shift, slots};
}


template<typename K, typename V>
void DenseMapObj<K, V>::IterListPushBack(Cursor node) {
    node.GetEntry().prev = iter_list_tail_;
    node.GetEntry().next = kInvalidIndex;

    if (iter_list_head_ == kInvalidIndex && iter_list_tail_ == kInvalidIndex) {
        iter_list_head_ = node.index();
    } else {
        Cursor(iter_list_tail_, this).GetEntry().next = node.index();
    }

    iter_list_tail_ = node.index();
}

template<typename K, typename V>
void DenseMapObj<K, V>::IterListRemove(Cursor node) {
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

template<typename K, typename V>
void DenseMapObj<K, V>::IterListReplace(Cursor src, Cursor dst) {
    dst.GetEntry().prev = src.GetEntry().prev;
    dst.GetEntry().next = src.GetEntry().next;

    if (src.IsIterListHead()) {
        iter_list_head_ = dst.index();
    } else {
        Cursor prev_node(dst.GetEntry().prev, this);
        prev_node.GetEntry().next = dst.index();
    }

    if (src.IsIterListTail()) {
        iter_list_tail_ = dst.index();
    } else {
        Cursor next_node(dst.GetEntry().next, this);
        next_node.GetEntry().prev = dst.index();
    }
}

template<typename K, typename V>
std::optional<typename DenseMapObj<K, V>::Cursor>
DenseMapObj<K, V>::TrySpareListHead(Cursor target, value_type&& kv) {
    // `target` is not the head of the linked list
    // move the original item of `target` (if any)
    // and construct new item on the position `target`
    // To make `target` empty, we
    // 1) find `w` the previous element of `target` in the linked list
    // 2) copy the linked list starting from `r = target`
    // 3) paste them after `w`

    // read from the linked list after `r`
    Cursor r = target;
    // write to the tail of `w`
    Cursor w = target.FindPrevSlot();
    // after `target` is moved, we disallow writing to the slot
    bool is_first = true;
    uint8_t r_meta;

    do {
        const auto empty_slot_info = w.GetNextEmptySlot();
        if (!empty_slot_info) {
            return std::nullopt;
        }

        uint8_t offset_idx = empty_slot_info->first;
        Cursor empty = empty_slot_info->second;

        // move `r` to `empty`
        // first move the data over
        empty.CreateTail(Entry(std::move(r.GetData())));
        // then move link list chain of r to empty
        // this needs to happen after NewTail so empty's prev/next get updated
        IterListReplace(r, empty);
        // clear the metadata of `r`
        r_meta = r.GetMeta();
        if (is_first) {
            is_first = false;
            r.SetProtected();
        } else {
            r.SetEmpty();
        }
        // link `w` to `empty`, and move forward
        w.SetOffsetIdx(offset_idx);
        w = empty;
    } while (r.MoveToNextSlot(r_meta));// move `r` forward as well

    // finally, we have done moving the linked list
    // fill data_ into `target`
    target.CreateHead(Entry(std::move(kv)));
    ++this->size_;
    return target;
}

template<typename K, typename V>
std::pair<typename DenseMapObj<K, V>::iterator, bool>
DenseMapObj<K, V>::TryInsert(value_type&& kv, bool assign) {
    // The key is already in the hash table
    if (auto it = this->find(kv.first); it != this->end()) {
        if (assign) {
            Cursor cur(it.index(), it.ptr());
            cur.GetValue() = std::move(kv.second);
            IterListRemove(cur);
            IterListPushBack(cur);
        }
        return {it, false};
    }

    // required that `iter` to be the head of a linked list through which we can iterator.
    // `iter` can be:
    // 1) empty;
    // 2) body of an irrelevant list;
    // 3) head of the relevant list.
    auto node = GetCursorFromHash(AnyHash()(kv.first));

    // Case 1: empty
    if (node.IsEmpty()) {
        node.CreateHead(Entry(std::move(kv)));
        ++this->size_;
        IterListPushBack(node);
        return {iterator(node.index(), this), true};
    }

    // Case 2: body of an irrelevant list
    if (!node.IsHead()) {
        if (IsFull()) {
            return {this->end(), false};
        }

        if (const auto target = TrySpareListHead(node, std::move(kv)); target.has_value()) {
            IterListPushBack(target.value());
            return {iterator(target->index(), this), true};
        }
        return {this->end(), false};
    }

    // Case 3: head of the relevant list
    // we iterate through the linked list until the end
    // make sure `iter` is the prev element of `cur`
    Cursor cur = node;
    while (cur.MoveToNextSlot()) {
        // make sure `node` is the previous element of `cur`
        node = cur;
    }

    // `node` is the tail of the linked list
    // always check capacity before insertion
    if (IsFull()) {
        return {this->end(), false};
    }

    // find the next empty slot
    auto empty_slot_info = node.GetNextEmptySlot();
    if (!empty_slot_info) {
        return {this->end(), false};
    }

    uint8_t offset_idx = empty_slot_info->first;
    Cursor empty = empty_slot_info->second;
    empty.CreateTail(Entry(std::move(kv)));
    // link `iter` to `empty`, and move forward
    node.SetOffsetIdx(offset_idx);
    IterListPushBack(empty);
    ++this->size_;
    return {iterator(node.index(), this), true};
}

template<typename K, typename V>
std::tuple<ObjectPtr<Object>, typename DenseMapObj<K, V>::iterator, bool>
DenseMapObj<K, V>::InsertImpl(value_type&& kv, const ObjectPtr<Object>& old_impl, bool assign) {
    auto* map = static_cast<DenseMapObj*>(old_impl.get());// NOLINT
    if (auto [it, is_success] = map->TryInsert(std::move(kv), assign); it != map->end()) {
        return {old_impl, it, is_success};
    }

    // Otherwise, start rehash
    auto new_impl = details::ObjectUnsafe::Downcast<DenseMapObj>(
            Create(map->slots() * DenseMapObj::kIncFactor));
    // need to insert in the same order as the original map
    size_t idx = map->iter_list_head_;
    while (idx != kInvalidIndex) {
        Cursor cur(idx, map);
        new_impl->TryInsert(std::move(cur.GetData()), assign);
        idx = cur.GetEntry().next;
    }

    auto [pos, is_success] = new_impl->TryInsert(std::move(kv), assign);
    return {new_impl, pos, is_success};
}


template<typename K, typename V>
ObjectPtr<DenseMapObj<K, V>> DenseMapObj<K, V>::Create(size_type n) {
    CHECK(n > DenseMapObj::kThreshold) << "The allocated size must be greate than the threshold of "
                                       << DenseMapObj::kThreshold
                                       << " when using SmallMapObj::Create";
    auto [fib_shift, slots] = ComputeSlotNum(n);
    const size_t block_num = ComputeBlockNum(slots);
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

template<typename K, typename V>
ObjectPtr<DenseMapObj<K, V>> DenseMapObj<K, V>::CopyFrom(const DenseMapObj* src) {
    auto impl = Create(src->slots());
    auto block_num = ComputeBlockNum(src->slots());
    impl->size_ = src->size();
    impl->iter_list_head_ = src->iter_list_head_;
    impl->iter_list_tail_ = src->iter_list_tail_;

    auto* p = static_cast<Block*>(impl->data());
    for (size_t i = 0; i < block_num; ++i) {
        new (p + i) Block(*src->GetBlock(i));
    }

    return impl;
}

template<typename K, typename V, typename Derived>
std::tuple<ObjectPtr<Object>, size_t, bool>
MapObjBase<K, V, Derived>::insert(value_type&& kv, const ObjectPtr<Object>& old_impl, bool assign) {
    if constexpr (std::is_same_v<Derived, SmallMapObj<K, V>>) {
        auto* p = static_cast<SmallMapObj<K, V>*>(old_impl.get());//NOLINT
        const auto size = p->size();
        if (size < kThreshold) {
            auto [impl, iter, is_success] = SmallMapObj<K, V>::InsertImpl(std::move(kv), old_impl, assign);
            return {impl, iter.index(), is_success};
        }

        ObjectPtr<Object> new_impl = DenseMapObj<K, V>::Create(size * kIncFactor);
        for (auto& iter: *p) {
            new_impl = std::get<0>(DenseMapObj<K, V>::InsertImpl(std::move(iter), new_impl));
        }
        auto [impl, iter, is_success] = DenseMapObj<K, V>::InsertImpl(std::move(kv), new_impl, assign);
        return {impl, iter.index(), is_success};
    } else {
        auto [impl, iter, is_success] = DenseMapObj<K, V>::InsertImpl(std::move(kv), old_impl, assign);
        return {impl, iter.index(), is_success};
    }
}


}// namespace aethermind

#endif//AETHERMIND_CONTAINER_MAP_OBJ_H
