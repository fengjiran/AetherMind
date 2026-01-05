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
    CHECK(n <= SmallMapObj::kThreshold);
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

private:
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

    static ObjectPtr<DenseMapObj> Create(size_type n);

    static ObjectPtr<DenseMapObj> CopyFromImpl(const DenseMapObj* src);

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
ObjectPtr<DenseMapObj<K, V>> DenseMapObj<K, V>::Create(size_type n) {
    CHECK(n > DenseMapObj::kThreshold);
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
ObjectPtr<DenseMapObj<K, V>> DenseMapObj<K, V>::CopyFromImpl(const DenseMapObj* src) {
    const auto* first = static_cast<value_type*>(src->data());
    auto impl = Create(src->slots());
    auto block_num = ComputeBlockNum(src->slots());
    // auto impl = make_array_object<DenseMapObj, Block>(block_num);
    // impl->data_ = reinterpret_cast<char*>(impl.get()) + sizeof(DenseMapObj);
    impl->size_ = src->size();
    // impl->slots_ = src->slots();
    // impl->fib_shift_ = src->fib_shift_;
    impl->iter_list_head_ = src->iter_list_head_;
    impl->iter_list_tail_ = src->iter_list_tail_;

    auto* p = static_cast<Block*>(impl->data());
    for (size_t i = 0; i < block_num; ++i) {
        new (p + i) Block(*src->GetBlock(i));
    }

    return impl;
}


}// namespace aethermind

#endif//AETHERMIND_CONTAINER_MAP_OBJ_H
